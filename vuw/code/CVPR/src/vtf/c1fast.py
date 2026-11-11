"""K29 fast path for the learned looming estimate (C1, k28_ttrain.TempHead) and the Res18x3 range refiners (v3a / own)
on the Jetson. Decision rules, weights and thresholds are unchanged; what changes:

crop64_fast   nusc_crop.crop64 with the constant border applied to the patch only (the reference pads the WHOLE image
              for every crop). The patch pixels are the same, so the 64x64 output is bit-identical (checked by
              `check_crop64`).
Bank          per pipeline: the 64x64 crop and the C1 encoder feature of (frame id, exact float64 box bytes), kept for
              the frames f, f - 1, f - 2. C1 reads the crops of the last three frames, so each crop is cut and encoded
              once; the current-frame crops are shared with the refiner.
Encoder       the C1 image encoder (flip, bilinear 64 -> 128, normalisation, ResNet-18): 'trt16' / 'trt32' static
              TensorRT engines per batch bucket (engines/c1enc_<p>_b<b>.engine, build_k29.sh), 't32' torch FP32 eager,
              't16g' torch FP16 with one CUDA graph per bucket. Batches are padded to BUCKETS (chunks of 32 above);
              padding rows are discarded.
Head          the C1 MLP (torch FP32, the checkpoint's own nn.Sequential).
C1Fast        one encoder call per lift for the crops not yet in the bank, one MLP call per lift; chains whose object
              cannot have a braking demand (outside the corridor or beyond max_range on the lift's own z / lat) are
              skipped: their ttc stays +inf (exact, docs/paper/handoff/K29/STATUS.md).
RefFast       = scripts/58 RefinerBatch (same features, normalisation, dynamic FP16 engine) with the batch padded to
              REF_BUCKETS and the crops from the Bank.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

S, CTXF = 64, 1.5
BUCKETS = (1, 2, 4, 8, 16, 32)
REF_BUCKETS = (1, 2, 4, 8, 16)
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}


def crop64_fast(img, box):
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * CTXF, (y2 - y1) * CTXF
    side = max(w, h, 8.0)
    X1, Y1, X2, Y2 = int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)
    H, W = img.shape[:2]
    ys, ye, xs, xe = max(Y1, 0), min(Y2, H), max(X1, 0), min(X2, W)
    if Y1 >= 0 and X1 >= 0 and Y2 <= H and X2 <= W:
        patch = img[Y1:Y2, X1:X2]
    else:
        patch = np.full((Y2 - Y1, X2 - X1, img.shape[2]), 114, img.dtype)
        if ye > ys and xe > xs:
            patch[ys - Y1:ye - Y1, xs - X1:xe - X1] = img[ys:ye, xs:xe]
    return cv2.resize(patch, (S, S), interpolation=cv2.INTER_AREA)


def check_crop64(img, boxes):
    """number of boxes whose crop64_fast differs from the reference crop64 (must be 0)"""
    from nusc_crop import crop64
    return int(sum(not np.array_equal(crop64(img, b), crop64_fast(img, b)) for b in boxes))


POOL_MIN, _POOL = 8, []


def _pool():
    if not _POOL:
        from concurrent.futures import ThreadPoolExecutor
        _POOL.append(ThreadPoolExecutor(6))
    return _POOL[0]


def _key(fid, box):
    return fid, np.asarray(box, np.float64).tobytes()


class Bank:
    def __init__(self, cap=512, dim=512):
        self.cap, self.dim, self.feat = cap, dim, None             # feature rows allocated on the first encode, grown x2
        self.row, self.crops, self.free = {}, {}, list(range(cap - 1, -1, -1))
        self.fid, self.pending = None, []                     # pending: CUDA events of speculative encodes (overlap)

    def wait(self):
        for ev in self.pending:
            torch.cuda.current_stream().wait_event(ev)
        self.pending.clear()

    def new_frame(self, fid):
        self.wait()
        self.fid = fid
        for k in [k for k in self.row if k[0] < fid - 2]:
            self.free.append(self.row.pop(k))
        for k in [k for k in self.crops if k[0] < fid - 2]:
            del self.crops[k]

    def alloc(self, n):
        if self.feat is None:
            self.feat = torch.empty((self.cap, self.dim), dtype=torch.float32, device="cuda")
        while len(self.free) < n:
            new = torch.empty((2 * self.cap, self.dim), dtype=torch.float32, device="cuda")
            new[:self.cap] = self.feat
            self.free = list(range(2 * self.cap - 1, self.cap - 1, -1)) + self.free
            self.feat, self.cap = new, 2 * self.cap
        return [self.free.pop() for _ in range(n)]

    def reset(self):
        self.free += list(self.row.values())
        self.row.clear(); self.crops.clear()

    def crop(self, fid, img, box):
        k = _key(fid, box)
        c = self.crops.get(k)
        if c is None:
            c = self.crops[k] = crop64_fast(img, box)
        return c

    def crop_many(self, items):
        """[(fid, img, box)] -> crops; the missing ones are cut in a thread pool when there are >= POOL_MIN of them
        (cv2 releases the GIL; the result is the same as cutting them one by one)"""
        ks = [_key(f, b) for f, _, b in items]
        todo = [i for i, k in enumerate(ks) if k not in self.crops]
        if len(todo) >= POOL_MIN:
            for i, c in zip(todo, _pool().map(lambda i: crop64_fast(items[i][1], items[i][2]), todo)):
                self.crops[ks[i]] = c
        return [self.crops[k] if k in self.crops else self.crop(*it) for k, it in zip(ks, items)]


class _TRTStatic:
    def __init__(self, path, b, inp="crop", out="feat"):
        import tensorrt as trt
        self.rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.eng = self.rt.deserialize_cuda_engine(Path(path).read_bytes())
        self.ctx = self.eng.create_execution_context()
        names = [self.eng.get_binding_name(i) for i in range(self.eng.num_bindings)]
        self.ctx.set_binding_shape(names.index(inp), (b, 3, S, S))
        self.inp = torch.zeros((b, 3, S, S), dtype=torch.float32, device="cuda")
        self.out = torch.empty(tuple(self.ctx.get_binding_shape(names.index(out))), dtype=torch.float32, device="cuda")
        self.bind = [int((self.inp if n == inp else self.out).data_ptr()) for n in names]

    def run(self):
        self.ctx.execute_async_v2(self.bind, torch.cuda.current_stream().cuda_stream)


class Encoder:
    def __init__(self, kind, eng_dir, head_ckpt):
        self.kind = kind
        if kind in ("trt16", "trt32"):
            p = "fp16" if kind == "trt16" else "fp32"
            self.r = {b: _TRTStatic(Path(eng_dir) / f"c1enc_{p}_b{b}.engine", b) for b in BUCKETS}
        else:
            import importlib.util
            spec = importlib.util.spec_from_file_location("c1x", Path(eng_dir) / "export_c1_encoder.py")
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            enc = m.Enc()
            enc.load_state_dict({k: v for k, v in head_ckpt["state"].items() if not k.startswith("head.")})
            enc = enc.cuda().eval()
            if kind == "t32":
                self.enc = enc
            else:
                enc = enc.half()
                self.enc = enc                                     # keep the weights alive: the graphs read their memory
                self.g = {}
                for b in BUCKETS:
                    inp = torch.zeros((b, 3, S, S), dtype=torch.float16, device="cuda")
                    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s), torch.no_grad():
                        for _ in range(3):
                            enc(inp)
                    torch.cuda.current_stream().wait_stream(s)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g), torch.no_grad():
                        out = enc(inp)
                    self.g[b] = (g, inp, out)

    def __call__(self, x):
        """x: (n, 3, 64, 64) float32 in [0, 1] on the GPU -> (n, 512) float32"""
        n = x.shape[0]
        if self.kind == "t32":
            with torch.no_grad():
                return torch.cat([self.enc(x[k:k + 32]) for k in range(0, n, 32)]) if n > 32 else self.enc(x)
        out = torch.empty((n, 512), dtype=torch.float32, device="cuda")
        for k0 in range(0, n, 32):
            xc = x[k0:k0 + 32]
            m = xc.shape[0]
            b = next(bb for bb in BUCKETS if bb >= m)
            if self.kind == "t16g":
                g, inp, o = self.g[b]
                inp[:m].copy_(xc)
                if m < b:
                    inp[m:].zero_()
                g.replay()
                out[k0:k0 + m].copy_(o[:m])
            else:
                r = self.r[b]
                r.inp[:m].copy_(xc)
                if m < b:
                    r.inp[m:].zero_()
                r.run()
                out[k0:k0 + m].copy_(r.out[:m])
        return out


class Head:
    def __init__(self, head_ckpt):
        n = head_ckpt["n_feat"]
        self.mlp = torch.nn.Sequential(torch.nn.Linear(3 * 512 + n, 256), torch.nn.ReLU(inplace=True), torch.nn.Linear(256, 1))
        self.mlp.load_state_dict({k[5:]: v for k, v in head_ckpt["state"].items() if k.startswith("head.")})
        self.mlp = self.mlp.cuda().eval()

    def __call__(self, e, f):
        with torch.no_grad():
            return self.mlp(torch.cat([e, f], 1))[:, 0]


class C1Fast:
    """rh (the head's output, clipped to [0, 2]) for the chains of one lift call; pruned chains are left out."""

    def __init__(self, enc, head, hmu, hsd, pp=None, prune=True):
        self.enc, self.head, self.hmu, self.hsd, self.pp, self.prune = enc, head, hmu, hsd, pp, prune
        self.stats = {"chains": 0, "pruned": 0, "encoded": 0, "calls": 0, "spec_encoded": 0, "spec_calls": 0}
        self.side = None

    def spec(self, bank, fid, img, boxes):
        """overlap (STATUS e): encode the current-frame crops of `boxes` on a second CUDA stream while the caller goes on
        (the guard's grown-crop calls); C1Fast.__call__ waits for it before reading the bank. Unused features are dropped
        with their frame."""
        need, seen = [], set()
        for b in boxes:
            k = _key(fid, b)
            if k not in bank.row and k not in seen:
                seen.add(k); need.append((fid, img, b))
        if not need:
            return
        cr = np.stack(bank.crop_many(need))
        if self.side is None:
            self.side = torch.cuda.Stream()
        self.side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side):
            x = torch.from_numpy(cr).cuda().permute(0, 3, 1, 2).float() / 255.0
            feats = self.enc(x)
            rows = bank.alloc(len(need))
            bank.feat[torch.tensor(rows, device="cuda")] = feats
            ev = torch.cuda.Event()
            ev.record(self.side)
        for (f_, _, b), r in zip(need, rows):
            bank.row[_key(f_, b)] = r
        bank.pending.append(ev)
        self.stats["spec_encoded"] += len(need); self.stats["spec_calls"] += 1

    def keep(self, chn, z, lo, hi):
        if not self.prune:
            return np.ones(len(chn), bool)
        from rap import planner as P
        idx = np.array([c[0] for c in chn])
        z, lo, hi = (np.asarray(a, np.float64)[idx] for a in (z, lo, hi))
        return P.in_corridor(lo, hi, self.pp.corridor_half_w) & (z <= self.pp.max_range)

    def __call__(self, bank, fid, img, img1, img2, X, chn, s1, s2, fe, keep):
        sel = np.flatnonzero(keep)
        self.stats["chains"] += len(chn); self.stats["pruned"] += len(chn) - len(sel)
        if not len(sel):
            return {}
        keys, need, seen = [], [], set()
        for n in sel:
            i, j1, j2 = chn[n]
            for f_, im, b in ((fid, img, X[i]), (fid - 1, img1, s1[j1]), (fid - 2, img2, s2[j2])):
                k = _key(f_, b)
                keys.append(k)
                if k not in bank.row and k not in seen:
                    seen.add(k); need.append((k, im, b))
        bank.wait()
        if need:
            cr = np.stack(bank.crop_many([(k[0], im, b) for k, im, b in need]))
            x = torch.from_numpy(cr).cuda().permute(0, 3, 1, 2).float() / 255.0
            feats = self.enc(x)
            rows = bank.alloc(len(need))
            for (k, _, _), r in zip(need, rows):
                bank.row[k] = r
            bank.feat[torch.tensor(rows, device="cuda")] = feats
            self.stats["encoded"] += len(need); self.stats["calls"] += 1
        e = bank.feat.index_select(0, torch.tensor([bank.row[k] for k in keys], device="cuda")).reshape(len(sel), 3 * 512)
        fin = torch.from_numpy(((np.array([fe[n] for n in sel], np.float32) - self.hmu) / self.hsd).astype(np.float32)).cuda()
        rh = np.clip(self.head(e, fin).float().cpu().numpy(), 0, 2)
        return dict(zip(sel.tolist(), rh))


class RefFast:
    """scripts/58 RefinerBatch with the batch padded to REF_BUCKETS and the crops from the Bank (bind() first)."""

    def __init__(self, tag, eng_dir, predicted_geometry):
        import tensorrt as trt
        self.rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.eng = self.rt.deserialize_cuda_engine((Path(eng_dir) / f"refiner_{tag}_dyn.engine").read_bytes())
        self.ctx = self.eng.create_execution_context()
        self.names = [self.eng.get_binding_name(i) for i in range(self.eng.num_bindings)]
        z = np.load(Path(eng_dir) / f"refiner_{tag}_norm.npz")
        self.fmu, self.fsd = z["fmu"], z["fsd"]
        self.pg = predicted_geometry
        self.buf = {b: (torch.zeros((b, 3, S, S), dtype=torch.float32, device="cuda"),
                        torch.zeros((b, len(self.fmu)), dtype=torch.float32, device="cuda"),
                        torch.empty(b, dtype=torch.float32, device="cuda")) for b in REF_BUCKETS}
        self.cur, self.bank, self.fid = None, None, None

    def bind(self, bank, fid):
        self.bank, self.fid = bank, fid
        return self

    def __call__(self, img, items, cam, R, t, W, H):
        if not items:
            return []
        boxes = np.array([it[0] for it in items], float)
        g = self.pg({"xyxy": boxes, "coarse": np.array([it[2] for it in items])}, None, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
        F = []
        for j, (box, conf, coarse, zs) in enumerate(items):
            x1, y1, x2, y2 = box
            zg, zh = float(g["z_ground"][j]), float(g["z_height"][j])
            F.append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(conf), float(CLS[str(coarse)]),
                      float(zs), zg, zh, float(np.log(max(zg, .5) / max(zh, .5))), 1.0])
        F = ((np.array(F, np.float32) - self.fmu) / self.fsd).astype(np.float32)
        cr = np.stack(self.bank.crop_many([(self.fid, img, it[0]) for it in items]))
        x = torch.from_numpy(cr).cuda().permute(0, 3, 1, 2).float() / 255.0
        fe = torch.from_numpy(F).cuda()
        res = []
        for k0 in range(0, len(items), REF_BUCKETS[-1]):
            m = min(len(items) - k0, REF_BUCKETS[-1])
            b = next(bb for bb in REF_BUCKETS if bb >= m)
            ci, fi, oi = self.buf[b]
            ci[:m].copy_(x[k0:k0 + m]); fi[:m].copy_(fe[k0:k0 + m])
            if m < b:
                ci[m:].zero_(); fi[m:].zero_()
            if self.cur != b:
                self.ctx.set_binding_shape(self.names.index("crop"), (b, 3, S, S))
                self.ctx.set_binding_shape(self.names.index("feats"), (b, len(self.fmu)))
                self.cur = b
            bufs = {"crop": ci, "feats": fi, "log_ratio": oi}
            self.ctx.execute_async_v2([int(bufs[nm].data_ptr()) for nm in self.names], torch.cuda.current_stream().cuda_stream)
            res.append(oi[:m].cpu().numpy())
        return list(np.exp(np.concatenate(res).astype(np.float64)))
