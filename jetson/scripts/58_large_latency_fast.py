#!/usr/bin/env python
"""K11 — faster implementation of ours on the Jetson (large frames), decision-identical by construction where stated,
with a parity check against the K5 follow-up run (scripts/57 --gpupost --dyncrops) on the same timed frames.

Same frames, whole-frame pipelines, cost and lift as scripts/55 / 57 (imported unchanged). Ours, step by step (flags):
  base         = 57 --gpupost --dyncrops without the per-stage breakdown syncs (CUDA syncs only at each pipeline's
                 start and end; every pipeline ends on host values)
  --refbatch   all support objects of a frame refined in ONE engine call (dynamic-batch refiner engines
                 engines/refiner_{v1,v2}_vuw_dyn.engine; per-object features / crops exactly as 55's Refiner)
  --ctx C      NEW CONFIGURATION (not decision-identical): support crops with context C instead of 2.0 (vtf.speculative.CTX)
  --mask_y0 Y  ego mask (K10/K13): detections lying >= 50 % in the image rows >= Y get range / ttc = +inf before the lift
               for EVERY pipeline (no demand, never support, no crop on them); AV2 hood: Y = 1830
  --fmargin F  NEW CONFIGURATION: the field crop enlarged by F x its size on each side before detection
               (fewer truncated boxes -> fewer sequential guard calls); evaluated on the DEV split first (VUW)
Output (run folder large_latency_fast_<ds>): latency.csv (the K5 format), per-frame pickle with actions, parity.csv
(share of timed frames whose ours / ours_noref cost equals the reference run's; G on the timed frames of both).
usage: 58_large_latency_fast.py --dataset nusc|av2 [--refbatch] [--ctx 2.0] [--fmargin 0] --ref_run <57 run folder>
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("k5", ROOT / "scripts" / "55_large_latency.py")
K = importlib.util.module_from_spec(_s)
_s.loader.exec_module(K)
from vtf import gpupost as GP                                                    # noqa: E402
from vtf.gpuprep import letterbox                                                # noqa: E402
from vtf.trtdyn import DynTRT                                                    # noqa: E402

P, SP, THR, Mode, ENG = K.P, K.SP, K.THR, K.Mode, K.ENG
STATS = {"calls": 0}


def sync():
    torch.cuda.synchronize()


class Crops:
    """detect_many: each crop on a ceil32(crop x scale) canvas with the frame letterbox (the VUW geometry), crops of
    equal canvas batched (<= engine max batch), GPU candidate selection + lean post-processing (bit-identical)."""

    def __init__(self, G, scale):
        self.G, self.s = G, scale
        self.eng = DynTRT(ENG / "yolov8s_dyn.engine")

    def __call__(self, crops):
        groups = {}
        for i, cr in enumerate(crops):
            x1, y1 = int(cr[0]), int(cr[1])
            x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
            sub = self.G.x[:, y1:y2, x1:x2]
            nh = min(1024, max(32, int(math.ceil(sub.shape[1] * self.s / 32)) * 32))
            nw = min(1024, max(32, int(math.ceil(sub.shape[2] * self.s / 32)) * 32))
            xb, r, pad = letterbox(sub, nh, nw)
            groups.setdefault((nh, nw), []).append((i, xb, r, pad, (y2 - y1, x2 - x1, 3), x1, y1))
        out = [None] * len(crops)
        for (nh, nw), items in groups.items():
            mode = Mode(f"c{nh}x{nw}", nh, nw)
            for k0 in range(0, len(items), self.eng.max_batch):
                chunk = items[k0:k0 + self.eng.max_batch]
                raw = self.eng(torch.stack([it[1] for it in chunk]))
                STATS["calls"] += 1
                STATS["tpx"] = STATS.get("tpx", 0) + len(chunk) * nh * nw
                cols = GP.host_columns(raw, mode, THR)
                for k, (i, _, r, pad, shp, x1, y1) in enumerate(chunk):
                    d = GP.post_lean(cols[k], mode, r, pad, shp, min_conf=THR)
                    kk = d["conf"] >= THR
                    out[i] = {"xyxy": d["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                              "conf": d["conf"][kk].astype(float), "coarse": d["coarse"][kk]}
        return out


class RefinerBatch:
    """55's Refiner (same features, crops, normalisation) for all support objects in one dynamic-batch engine call."""

    def __init__(self, tag):
        import tensorrt as trt
        self.rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.eng = self.rt.deserialize_cuda_engine((ENG / f"refiner_{tag}_dyn.engine").read_bytes())
        self.ctx = self.eng.create_execution_context()
        self.names = [self.eng.get_binding_name(i) for i in range(self.eng.num_bindings)]
        z = np.load(ENG / f"refiner_{tag}_norm.npz")
        self.fmu, self.fsd = z["fmu"], z["fsd"]

    def __call__(self, img, items, cam, R, t, W, H):
        """items: [(box, conf, coarse, z_scene)] -> [correction factor]"""
        from nusc_crop import crop64
        if not items:
            return []
        boxes = np.array([it[0] for it in items], float)
        g = K.predicted_geometry({"xyxy": boxes, "coarse": np.array([it[2] for it in items])}, None, cam.calib, cam.cam_h,
                                 cam.dt, cam_to_ego=(R, t))
        F = []
        for j, (box, conf, coarse, zs) in enumerate(items):
            x1, y1, x2, y2 = box
            zg, zh = float(g["z_ground"][j]), float(g["z_height"][j])
            F.append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(conf), float(K.CLS[str(coarse)]),
                      float(zs), zg, zh, float(np.log(max(zg, .5) / max(zh, .5))), 1.0])
        F = ((np.array(F, np.float32) - self.fmu) / self.fsd).astype(np.float32)
        n = len(items)
        crop = torch.from_numpy(np.stack([crop64(img, it[0]) for it in items])).permute(0, 3, 1, 2).float().cuda() / 255.0
        feats = torch.from_numpy(F).cuda()
        out = torch.empty(n, dtype=torch.float32, device="cuda")
        bufs = {"crop": crop.contiguous(), "feats": feats.contiguous(), "log_ratio": out}
        self.ctx.set_binding_shape(self.names.index("crop"), tuple(crop.shape))
        self.ctx.set_binding_shape(self.names.index("feats"), tuple(feats.shape))
        self.ctx.execute_v2([int(bufs[nm].data_ptr()) for nm in self.names])
        STATS["calls"] += 1
        return list(np.exp(out.cpu().numpy().astype(np.float64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["av2", "nusc"])
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--max_units", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--refbatch", action="store_true")
    ap.add_argument("--ctx", type=float, default=2.0)
    ap.add_argument("--fmargin", type=float, default=0.0)
    ap.add_argument("--ref_run", default="")
    ap.add_argument("--mask_y0", type=float, default=0.0)
    args = ap.parse_args()
    run = K.runmeta.new_run(f"large_latency_fast_{args.dataset}" + ("_masked" if args.mask_y0 else ""), vars(args))
    if args.mask_y0:
        _pg = K.predicted_geometry

        def masked_pg(dets, *a, **k):
            g = _pg(dets, *a, **k)
            b = np.asarray(dets["xyxy"], float).reshape(-1, 4)
            if not len(b):
                return g
            m = (b[:, 3] - np.maximum(b[:, 1], args.mask_y0)).clip(0, None) / np.maximum(b[:, 3] - b[:, 1], 1e-6) >= 0.5
            if m.any():
                g = dict(g)
                for key in ("z", "z_ground", "z_height", "ttc"):
                    if key in g:
                        arr = np.asarray(g[key], float).copy(); arr[m] = np.inf; g[key] = arr
            return g
        K.predicted_geometry = masked_pg
    SP.CTX = args.ctx
    if args.fmargin > 0:
        _fc = SP.field_crop

        def field_crop(*a, **k):
            c = _fc(*a, **k)
            if c is None:
                return None
            w, h = c[2] - c[0], c[3] - c[1]
            W_, H_ = a[3], a[4]
            return np.array([max(c[0] - args.fmargin * w, 0), max(c[1] - args.fmargin * h, 0),
                             min(c[2] + args.fmargin * w, W_), min(c[3] + args.fmargin * h, H_)])
        SP.field_crop = field_crop
    det = K.FastDetector(str(Path(K.MODELS) / "yolov8s.pt"), backend="trt", engine_dir=str(ENG), engine_stem="yolov8s_crop")
    if args.dataset == "av2":
        M = {"cheap": Mode("320x256", 320, 256), "u640": Mode("640x512", 640, 512), "u960": Mode("960x768", 960, 768),
             "u1280": Mode("1280x992", 1280, 992), "u1536": Mode("1536x1184", 1536, 1184), "u2048 (native)": Mode("2048x1568", 2048, 1568)}
        vc, scale, gen, rtag, NATIVE = 0.0, 960 / 2048, K.frames_av2(args), "v2_vuw", 1550 * 2048
    else:
        M = {"cheap": Mode("192x320", 192, 320), "u640": Mode("384x640", 384, 640), "u960": Mode("576x960", 576, 960),
             "u1280": Mode("736x1280", 736, 1280), "u1600 (native)": Mode("928x1600", 928, 1600)}
        vc, scale, gen, rtag, NATIVE = 10.0, 0.6, K.frames_nusc(args), "v1_vuw", 1600 * 900
    ref1 = K.Refiner(rtag)
    refb = RefinerBatch(rtag) if args.refbatch else None
    G = K.GPUDetect()
    many = Crops(G, scale)
    G.x, G.shape = torch.full((3, 2048, 2048), K.PAD, device="cuda"), (2048, 2048, 3)
    many([[0, 0, 64, 64]] * 3 + [[0, 0, 400, 400], [0, 0, 1200, 300], [0, 0, 300, 1200], [0, 0, 1000, 1000]])
    pp, cp = P.PlannerParams(), P.CostParams()

    def gdetect(m):
        xb, r, pad = letterbox(G.x, m.net_h, m.net_w)
        raw = det.forward(xb[None], m)
        return GP.post_lean(GP.host_columns(raw, m, THR)[0], m, r, pad, G.shape, min_conf=THR)
    rows, hist, cur, prev_act = [], {}, None, None
    for unit, fr, path, v, g, cam, (R, t) in gen:
        if unit != cur:
            hist, cur, prev_act = {}, unit, None
        img = cv2.imread(path)
        H, W = img.shape[:2]
        a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
        rec = {"unit": unit, "frame": fr}

        def lift(xyxy, coarse, pv):
            gk = K.predicted_geometry({"xyxy": xyxy, "coarse": coarse}, pv, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
            return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

        def act(tt):
            return P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)

        def J(a):
            return P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]

        def keep(d):
            k = d["conf"] >= THR
            return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}
        tc = None
        for nm, m in M.items():
            sync(); t0 = time.perf_counter()
            G.frame(img)
            d = keep(gdetect(m))
            tt = lift(d["xyxy"], d["coarse"], hist.get(nm))
            a = act(tt)
            sync(); rec[f"ms_{nm}"] = 1e3 * (time.perf_counter() - t0)
            rec[f"a_{nm}"], rec[f"J_{nm}"], rec[f"px_{nm}"] = a, J(a), m.net_h * m.net_w / NATIVE
            hist[nm] = d
            if nm == "cheap":
                tc = tt
        # ours
        STATS["calls"], STATS["tpx"] = 0, 0
        sync(); t0 = time.perf_counter()
        G.frame(img)
        d = keep(gdetect(M["cheap"])); STATS["calls"] += 1
        d["mono"] = lift(d["xyxy"], d["coarse"], hist.get("c_ours"))
        scene, crops, rounds = SP.run(d, v, cam, W, H, lambda cr: many([cr])[0],
                                      lambda full: lift(full["xyxy"], full["coarse"], hist.get("ours")), pp, v_close=vc,
                                      max_rounds=3, guard=True, detect_many=many, batch_round=True)
        a_noref = act(scene["mono"])
        t_loop = time.perf_counter()
        refined = [np.asarray(a, float).copy() for a in scene["mono"]]
        sup = [p_ for p_ in SP.support(scene["mono"], v, pp) if scene["from_full"][p_]]
        if refb is not None:
            corr = refb(img, [(scene["xyxy"][p_], scene["conf"][p_], scene["coarse"][p_], scene["mono"][0][p_]) for p_ in sup], cam, R, t, W, H)
        else:
            corr = [ref1(img, scene["xyxy"][p_], scene["conf"][p_], scene["coarse"][p_], scene["mono"][0][p_], cam, R, t, W, H) for p_ in sup]
            STATS["calls"] += len(sup)
        for p_, c in zip(sup, corr):
            for a in refined[:3]:
                a[p_] *= c
        a_ours = act(tuple(refined))
        sync(); t_end = time.perf_counter()
        rec.update({"ms_ours_noref": 1e3 * (t_loop - t0), "ms_ours": 1e3 * (t_end - t0), "a_ours_noref": a_noref,
                    "a_ours": a_ours, "J_ours_noref": J(a_noref), "J_ours": J(a_ours), "calls_ours": STATS["calls"],
                    "n_crops": len(crops), "rounds": rounds, "n_refined": len(sup),
                    "px_ours": (M["cheap"].net_h * M["cheap"].net_w + STATS["tpx"]) / NATIVE})
        hist["ours"] = {"xyxy": scene["xyxy"]}
        hist["c_ours"] = {"xyxy": d["xyxy"], "coarse": d["coarse"]}
        prev_act = act(tc)
        rec["warm"] = len(rows) < args.warmup
        rows.append(rec)
        if len(rows) % 400 == 0:
            print(f"  {len(rows)} frames", flush=True)
        if args.max_frames and len(rows) >= args.max_frames:
            break
    d = pd.DataFrame(rows)
    d.to_pickle(run / "large_latency_fast.pkl")
    dd = d[~d.warm]
    L = dd.J_cheap.to_numpy()
    uq = dd.unit.unique(); idx = [np.flatnonzero(dd.unit.to_numpy() == u) for u in uq]
    rng = np.random.default_rng(0); boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    Vo = L - dd.J_ours.to_numpy()
    out = []
    for nm in list(M) + ["ours_noref", "ours"]:
        ms = dd[f"ms_{nm}"]; V = L - dd[f"J_{nm}"].to_numpy()
        r = {"pipeline": nm, "frames": len(dd), "G": round(V.sum() / L.sum(), 3) if nm != "cheap" else 0.0, "mean_ms": ms.mean(),
             "median_ms": ms.median(), "p95_ms": ms.quantile(.95), "p99_ms": ms.quantile(.99),
             "engine_calls_per_frame": dd.calls_ours.mean() if nm.startswith("ours") else 1.0,
             "network_px_mean": dd["px_ours" if nm.startswith("ours") else f"px_{nm}"].mean(),
             "network_px_p95": dd["px_ours" if nm.startswith("ours") else f"px_{nm}"].quantile(.95)}
        if nm != "ours":
            dr = [(Vo[b].sum() - V[b].sum()) / L[b].sum() for b in boots]
            r["ci_lo_ours_minus"], r["ci_hi_ours_minus"] = np.percentile(dr, 2.5), np.percentile(dr, 97.5)
        out.append(r)
    res = pd.DataFrame(out); res.to_csv(run / "latency.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"{args.dataset}: {len(dd)} timed frames; crops/frame {dd.n_crops.mean():.2f}; flags refbatch={args.refbatch} ctx={args.ctx} fmargin={args.fmargin}")
    print(res.round(3).to_string(index=False))
    if args.ref_run:
        rf = pd.read_pickle(Path(args.ref_run) / "large_latency.pkl")
        m = dd.merge(rf[~rf.warm][["unit", "frame", "J_ours", "J_ours_noref", "J_cheap"]], on=["unit", "frame"], suffixes=("", "_ref"))
        par = {"frames_matched": len(m), "identical_cost_ours": float(np.isclose(m.J_ours, m.J_ours_ref).mean()),
               "identical_cost_ours_noref": float(np.isclose(m.J_ours_noref, m.J_ours_noref_ref).mean()),
               "G_ours": (m.J_cheap.sum() - m.J_ours.sum()) / m.J_cheap.sum(),
               "G_ours_ref_run": (m.J_cheap_ref.sum() - m.J_ours_ref.sum()) / m.J_cheap_ref.sum(),
               "identical_cost_cheap": float(np.isclose(m.J_cheap, m.J_cheap_ref).mean())}
        pd.DataFrame([par]).to_csv(run / "parity.csv", index=False)
        print("parity:", {k: round(v, 4) for k, v in par.items()})


if __name__ == "__main__":
    main()
