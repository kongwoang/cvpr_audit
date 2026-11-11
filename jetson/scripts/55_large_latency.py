#!/usr/bin/env python
"""K5 — Jetson wall-clock on LARGE frames: AV2 ring_front_center (1550x2048; the 10 val logs copied to ~/datasets/av2)
and nuScenes CAM_FRONT (1600x900; the scenes whose images are on the Jetson = DEEP's 85 old scenes, NOT new test).

Fast path (scripts/54): clocks locked (jetson_clocks), GPU letterbox / crop resize, the crops of a round and each guard
growth step in one batched TensorRT call, one host copy per call; image decode excluded; every pipeline timed from its
own frame upload to its decision; same process, engines warmed, the first --warmup frames not reported.
Pipelines: cheap; u640 / u960 / u1280 / [u1536] / native whole frame; ours (AV2: v4-lite = v_close 0, crops at
960 / 2048; nuScenes: v3 = v_close 10, crops at 0.6; guard, merged-scene history, <= 3 rounds) without and with the
frozen refiner (AV2 v2, nuScenes v1: the VUW checkpoints exported to TensorRT, dataset flag 1.0, the scene's lift range
as feature — as in the VUW evaluations). Crops are resized by exactly the configured scale into the smallest canvas
engine that holds them (96x288 batch 1/2/4/8; 288x96, 192x576, 576x192, 320x320, 288x864, 640x640, 384x1248 batch 1).
Per frame and pipeline: G inputs (DEEP's braking cost), latency, engine calls; for ours a breakdown (draft pass, crop
pre-processing, detector calls of the rounds, guard re-detection calls, lift, field + support + merge = the rest of the
loop, refiner) measured with CUDA syncs at the stage boundaries; board power sampled at ~100 Hz (INA3221 rails)
and attributed to each pipeline's time segment.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("e2e", ROOT / "scripts" / "40_e2e_v2.py")
E = importlib.util.module_from_spec(_s)
_s.loader.exec_module(E)
P, SP, runmeta, THR = E.P, E.SP, E.runmeta, E.THR
from rap import power                                                            # noqa: E402
from rap.detect import Mode                                                      # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from rap.paths import MODELS                                                     # noqa: E402
from rap.trt import TRTModule                                                    # noqa: E402
from vtf.fastpost import FastDetector                                            # noqa: E402
from vtf.gpuprep import PAD, GPUDetect                                           # noqa: E402

ENG = runmeta.ROOT / "engines"
CANVAS = [(96, 288), (288, 96), (192, 576), (576, 192), (320, 320), (288, 864), (640, 640), (384, 1248)]
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
STATS = {}                                       # per-pipeline counters of the current frame


def sync():
    torch.cuda.synchronize()


class Counted:
    """wraps an engine call: counts calls and synchronised time into STATS[key]."""

    def __init__(self, f, key):
        self.f, self.key = f, key

    def __call__(self, *a, **k):
        sync(); t = time.perf_counter()
        out = self.f(*a, **k)
        sync()
        s = STATS.setdefault(self.key, [0, 0.0]); s[0] += 1; s[1] += time.perf_counter() - t
        return out


class Refiner:
    """frozen VUW refiner (v1 / v2) as a TensorRT engine; features exactly as nusc_full_eval / av2_eval_v4."""

    def __init__(self, tag):
        import tensorrt as trt
        self.rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.eng = self.rt.deserialize_cuda_engine((ENG / f"refiner_{tag}.engine").read_bytes())
        self.ctx = self.eng.create_execution_context()
        self.buf = [torch.empty(tuple(self.eng.get_binding_shape(i)), dtype=torch.float32, device="cuda") for i in range(self.eng.num_bindings)]
        self.names = [self.eng.get_binding_name(i) for i in range(self.eng.num_bindings)]
        z = np.load(ENG / f"refiner_{tag}_norm.npz")
        self.fmu, self.fsd = z["fmu"], z["fsd"]

    def __call__(self, img, box, conf, coarse, z_scene, cam, R, t, W, H):
        from nusc_crop import crop64
        g = predicted_geometry({"xyxy": box[None].astype(float), "coarse": np.array([coarse])}, None, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
        x1, y1, x2, y2 = box
        zg, zh = float(g["z_ground"][0]), float(g["z_height"][0])
        f = np.array([[x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(conf), float(CLS[str(coarse)]),
                       float(z_scene), zg, zh, float(np.log(max(zg, .5) / max(zh, .5))), 1.0]], np.float32)
        f = (f - self.fmu) / self.fsd
        self.buf[self.names.index("crop")].copy_(torch.from_numpy(crop64(img, box)).permute(2, 0, 1)[None].float() / 255.0)
        self.buf[self.names.index("feats")].copy_(torch.from_numpy(f.astype(np.float32)))
        self.ctx.execute_v2([int(b.data_ptr()) for b in self.buf])
        return float(np.exp(self.buf[self.names.index("log_ratio")].item()))


def fixed_scale(x, s, net_h, net_w):
    h, w = x.shape[1:]
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    top, left = int(round((net_h - nh) / 2 - 0.1)), int(round((net_w - nw) / 2 - 0.1))
    out = torch.full((3, net_h, net_w), PAD, device=x.device, dtype=x.dtype)
    out[:, top:top + nh, left:left + nw] = F.interpolate(x[None], size=(nh, nw), mode="bilinear", align_corners=False)[0]
    return out, s, (left, top)


class ScaledCrops:
    """detect_many: crops at a fixed scale, grouped by canvas; the 96x288 group in batched engines.
    Stage ("round" / "guard") is set by the instrumented guarded_batch below; tensor pixels are accumulated."""

    def __init__(self, G, det, scale, batch):
        self.G, self.det, self.s, self.batch = G, det, scale, batch
        self.modes = {c: Mode(f"{c[0]}x{c[1]}", c[0], c[1]) for c in CANVAS if (ENG / f"yolov8s_crop_{c[0]}x{c[1]}.engine").exists()}
        self.stage = "round"

    def canvas(self, h, w):
        fits = [c for c in self.modes if c[0] >= h * self.s and c[1] >= w * self.s]
        return min(fits, key=lambda c: c[0] * c[1]) if fits else max(self.modes, key=lambda c: c[0] * c[1])

    def __call__(self, crops):
        sync(); t0 = time.perf_counter()
        groups = {}
        for i, cr in enumerate(crops):
            x1, y1 = int(cr[0]), int(cr[1])
            x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
            sub = self.G.x[:, y1:y2, x1:x2]
            c = self.canvas(y2 - y1, x2 - x1)
            s = min(self.s, c[0] / (y2 - y1), c[1] / (x2 - x1))
            xb, r, pad = fixed_scale(sub, s, *c)
            groups.setdefault(c, []).append((i, xb, r, pad, (y2 - y1, x2 - x1, 3), x1, y1))
        sync()
        st = STATS.setdefault("crop_prep", [0, 0.0]); st[0] += 1; st[1] += time.perf_counter() - t0
        out = [None] * len(crops)
        for c, items in groups.items():
            mode = self.modes[c]
            engines = self.batch if c == (96, 288) else {1: None}
            sizes = sorted(engines)
            k0 = 0
            while k0 < len(items):
                b = next((q for q in sizes if q >= len(items) - k0), sizes[-1])
                chunk = items[k0:k0 + b]
                xb = torch.stack([it[1] for it in chunk] + [torch.full_like(chunk[0][1], PAD)] * (b - len(chunk)))
                call = Counted(self.det.raw_forward if b == 1 else engines[b], f"det_{self.stage}")
                raw = (call(xb, mode) if b == 1 else call(xb))[: len(chunk)].float().cpu().numpy()
                px = STATS.setdefault(f"tpx_{self.stage}", [0, 0.0]); px[1] += b * c[0] * c[1]
                for k, (i, _, r, pad, shp, x1, y1) in enumerate(chunk):
                    d = self.det.postprocess_np(raw[k], mode, r, pad, shp, min_conf=THR)
                    kk = d["conf"] >= THR
                    out[i] = {"xyxy": d["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                              "conf": d["conf"][kk].astype(float), "coarse": d["coarse"][kk]}
                k0 += len(chunk)
        return out


_ORIG_GB = SP.guarded_batch


def instrumented_guarded_batch(detect_many, *a, **k):
    """same semantics as speculative.guarded_batch; marks the first detect_many call of each round as 'round' and the
    guard's growth steps as 'guard' (runtime wrapper in this measurement script only; the frozen module is unchanged)."""
    sc = detect_many

    def dm(crs):
        r = sc(crs)
        sc.stage = "guard"
        return r
    inner = _ORIG_GB(dm, *a, **k)

    def det_all(crops):
        sc.stage = "round"
        return inner(crops)
    return det_all


SP.guarded_batch = instrumented_guarded_batch


def frames_av2(args):
    from vtf import av2 as AV
    root = Path.home() / "datasets" / "av2" / "sensor" / "val"
    for log in sorted(p.name for p in root.iterdir() if (p / "annotations.feather").exists()):
        L = AV.AV2Log(root, log)
        geom, speeds = L.geometry(), L.speeds()
        R, t = L.transform("ego")
        for fr in range(len(L.frame_ts)):
            if Path(L.image_path(fr)).exists():
                g = geom[geom["frame"] == fr] if len(geom) else geom
                yield log, fr, str(L.image_path(fr)), float(speeds[fr]), (g[(g["y2"] - g["y1"]) >= 10] if len(g) else g), L, (R, t)


def frames_nusc(args):
    from rap.cache import Camera
    from rap.nusc import NuScenesDB, make_adapter
    from rap.paths import NUSCENES_TRAINVAL
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    n_sc = 0
    for sc in db.scenes:
        s = db.scene_name(sc)
        toks = db.samples(sc)
        if not all(Path(db.image_path(tk)).exists() for tk in toks):
            continue
        try:
            cam = Camera("nuScenes", s)
        except Exception:                                                        # noqa: BLE001
            continue
        geom, speeds = ad.geometry(s), ad.speeds(s)
        R, t = cam.transform("ego")
        for fr, tk in enumerate(toks):
            g = geom[geom["frame"] == fr]
            yield s, fr, str(db.image_path(tk)), float(speeds[fr]) if fr < len(speeds) else float(speeds[-1]), \
                g[(g["y2"] - g["y1"]) >= 10], cam, (R, t)
        n_sc += 1
        if args.max_units and n_sc >= args.max_units:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["av2", "nusc"])
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--max_units", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=200)
    args = ap.parse_args()
    run = runmeta.new_run(f"large_latency_{args.dataset}", vars(args))
    det = FastDetector(str(Path(MODELS) / "yolov8s.pt"), backend="trt", engine_dir=str(ENG), engine_stem="yolov8s_crop")
    if args.dataset == "av2":
        M = {"cheap": Mode("320x256", 320, 256), "u640": Mode("640x512", 640, 512), "u960": Mode("960x768", 960, 768),
             "u1280": Mode("1280x992", 1280, 992), "u1536": Mode("1536x1184", 1536, 1184), "u2048 (native)": Mode("2048x1568", 2048, 1568)}
        vc, scale, gen, rtag, NATIVE = 0.0, 960 / 2048, frames_av2(args), "v2_vuw", 1550 * 2048
    else:
        M = {"cheap": Mode("192x320", 192, 320), "u640": Mode("384x640", 384, 640), "u960": Mode("576x960", 576, 960),
             "u1280": Mode("736x1280", 736, 1280), "u1600 (native)": Mode("928x1600", 928, 1600)}
        vc, scale, gen, rtag, NATIVE = 10.0, 0.6, frames_nusc(args), "v1_vuw", 1600 * 900
    missing = [k for k, m in M.items() if not (ENG / f"yolov8s_crop_{m.name}.engine").exists()]
    M = {k: m for k, m in M.items() if k not in missing}
    print("whole-frame pipelines:", list(M), "| missing engines (not run):", missing, flush=True)
    det.raw_forward = det.forward
    det.forward = Counted(det.raw_forward, "det_frame")
    batch = {1: None}
    for b in (2, 4, 8):
        if (ENG / f"yolov8s_crop_b{b}_96x288.engine").exists():
            batch[b] = TRTModule(ENG / f"yolov8s_crop_b{b}_96x288.engine")
    ref = Refiner(rtag)
    G = GPUDetect()
    many = ScaledCrops(G, det, scale, batch)
    G.x, G.shape = torch.full((3, 2048, 2048), PAD, device="cuda"), (2048, 2048, 3)
    many([[0, 0, 64, 64]] * 9 + [[0, 0, 400, 400], [0, 0, 1200, 300], [0, 0, 300, 1200], [0, 0, 600, 200], [0, 0, 200, 600]])
    pp, cp = P.PlannerParams(), P.CostParams()
    # power sampler (all INA3221 rails summed), ~100 Hz
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            pw = power.read_power_mw() if power.AVAILABLE else {}
            samples.append((time.perf_counter(), sum(pw.values())))
            time.sleep(0.01)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    time.sleep(3.0)
    idle_mw = float(np.mean([p for _, p in samples])) if samples else float("nan")
    rows, hist, cur, prev_act = [], {}, None, None
    for unit, fr, path, v, g, cam, (R, t) in gen:
        if unit != cur:
            hist, cur, prev_act = {}, unit, None
        img = cv2.imread(path)
        H, W = img.shape[:2]
        a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
        rec = {"unit": unit, "frame": fr}

        def lift(xyxy, coarse, pv):
            gk = predicted_geometry({"xyxy": xyxy, "coarse": coarse}, pv, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
            return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

        def J(tt):
            return P.decision_cost(P.discrete_action(P.required_decel(*tt, v, pp)[0], pp), a_gt, prev_act, pp, cp)["J"]

        def keep(d):
            k = d["conf"] >= THR
            return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}

        def timed(name, fn):
            STATS.clear(); sync(); t0 = time.perf_counter()
            out = fn()
            sync(); t1 = time.perf_counter()
            rec[f"ms_{name}"] = 1e3 * (t1 - t0)
            rec[f"calls_{name}"] = sum(v_[0] for k_, v_ in STATS.items() if k_.startswith("det_"))
            rec[f"seg_{name}"] = (t0, t1)
            return out

        def whole(nm, m):
            def fn():
                G.frame(img)
                d = keep(G.detect(det, m, THR))
                return d, lift(d["xyxy"], d["coarse"], hist.get(nm))
            d, tt = timed(nm, fn)
            rec[f"J_{nm}"] = J(tt)
            rec[f"px_{nm}"] = m.net_h * m.net_w / NATIVE
            hist[nm] = d
            return d, tt
        dc, tc = whole("cheap", M["cheap"])
        for nm, m in M.items():
            if nm != "cheap":
                whole(nm, m)
        # ours: draft -> loop -> refiner, with stage timing
        bd = {}

        def ours():
            G.frame(img)
            d = keep(G.detect(det, M["cheap"], THR))
            sync(); bd["t_draft"] = time.perf_counter()
            tl = [0.0]

            def lf(full):
                sync(); a = time.perf_counter()
                o = lift(full["xyxy"], full["coarse"], hist.get("ours"))
                tl[0] += time.perf_counter() - a
                return o
            d["mono"] = lift(d["xyxy"], d["coarse"], hist.get("c_ours"))
            scene, crops, rounds = SP.run(d, v, cam, W, H, lambda cr: many([cr])[0], lf, pp, v_close=vc, max_rounds=3,
                                          guard=True, detect_many=many, batch_round=True)
            sync(); bd["t_loop"] = time.perf_counter(); bd["lift"] = tl[0]
            return d, scene, crops, rounds
        STATS.clear(); sync(); t0 = time.perf_counter()
        d_o, scene, crops, rounds = ours()
        t_loop = bd["t_loop"]
        refined = [np.asarray(a, float).copy() for a in scene["mono"]]
        n_ref = 0
        for p_ in SP.support(scene["mono"], v, pp):
            if scene["from_full"][p_]:
                corr = ref(img, scene["xyxy"][p_], scene["conf"][p_], scene["coarse"][p_], scene["mono"][0][p_], cam, R, t, W, H)
                for a in refined[:3]:
                    a[p_] *= corr
                n_ref += 1
        sync(); t_end = time.perf_counter()
        st = {k: v_ for k, v_ in STATS.items()}
        draft_det = st.get("det_frame", [0, 0.0])
        rec.update({
            "ms_ours_noref": 1e3 * (t_loop - t0), "ms_ours": 1e3 * (t_end - t0), "seg_ours": (t0, t_end),
            "J_ours_noref": J(scene["mono"]), "J_ours": J(tuple(refined)),
            "calls_ours": sum(v_[0] for k_, v_ in st.items() if k_.startswith("det_")) + n_ref,
            "n_crops": len(crops), "rounds": rounds, "n_refined": n_ref,
            "px_ours": (M["cheap"].net_h * M["cheap"].net_w + st.get("tpx_round", [0, 0.0])[1] + st.get("tpx_guard", [0, 0.0])[1]) / NATIVE,
            "bd_draft_ms": 1e3 * (bd["t_draft"] - t0),
            "bd_crop_prep_ms": 1e3 * st.get("crop_prep", [0, 0.0])[1],
            "bd_det_round_ms": 1e3 * st.get("det_round", [0, 0.0])[1], "bd_det_round_calls": st.get("det_round", [0, 0.0])[0],
            "bd_det_guard_ms": 1e3 * st.get("det_guard", [0, 0.0])[1], "bd_det_guard_calls": st.get("det_guard", [0, 0.0])[0],
            "bd_lift_ms": 1e3 * bd["lift"], "bd_refiner_ms": 1e3 * (t_end - t_loop)})
        rec["bd_field_support_merge_ms"] = rec["ms_ours_noref"] - rec["bd_draft_ms"] - rec["bd_crop_prep_ms"] - \
            rec["bd_det_round_ms"] - rec["bd_det_guard_ms"] - rec["bd_lift_ms"]
        hist["ours"] = {"xyxy": scene["xyxy"]}
        hist["c_ours"] = {"xyxy": d_o["xyxy"], "coarse": d_o["coarse"]}
        prev_act = P.discrete_action(P.required_decel(*tc, v, pp)[0], pp)
        rec["warm"] = len(rows) < args.warmup
        rows.append(rec)
        if len(rows) % 200 == 0:
            print(f"  {len(rows)} frames", flush=True)
        if args.max_frames and len(rows) >= args.max_frames:
            break
    stop.set(); th.join()
    d = pd.DataFrame(rows)
    S = np.array(samples)
    names = list(M) + ["ours_noref", "ours"]

    def energy(seg):
        if not len(S):
            return np.nan
        a, b = seg
        m = (S[:, 0] >= a) & (S[:, 0] <= b)
        p = S[m, 1].mean() if m.any() else np.interp((a + b) / 2, S[:, 0], S[:, 1])
        return p * (b - a)                       # mW * s = mJ
    for nm in names:
        seg_col = "seg_ours" if nm.startswith("ours") else f"seg_{nm}"
        if nm == "ours_noref":
            d["mJ_ours_noref"] = [energy((s[0], s[0] + ms / 1e3)) for s, ms in zip(d.seg_ours, d.ms_ours_noref)]
        else:
            d[f"mJ_{nm}"] = [energy(s) for s in d[seg_col]]
    d.drop(columns=[c for c in d.columns if c.startswith("seg_")]).to_pickle(run / "large_latency.pkl")
    dd = d[~d.warm]
    L = dd.J_cheap.to_numpy()
    uq = dd.unit.unique(); idx = [np.flatnonzero(dd.unit.to_numpy() == u) for u in uq]
    rng = np.random.default_rng(0); boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    Vo = L - dd.J_ours.to_numpy()
    out = []
    for nm in names:
        ms = dd[f"ms_{nm}"]
        V = L - dd[f"J_{nm}"].to_numpy()
        r = {"pipeline": nm, "frames": len(dd), "G": round(V.sum() / L.sum(), 3) if nm != "cheap" else 0.0,
             "mean_ms": ms.mean(), "median_ms": ms.median(), "p95_ms": ms.quantile(0.95), "p99_ms": ms.quantile(0.99),
             "engine_calls_per_frame": dd["calls_ours" if nm.startswith("ours") else f"calls_{nm}"].mean(),
             "network_px_mean": dd["px_ours" if nm.startswith("ours") else f"px_{nm}"].mean(),
             "network_px_p95": dd["px_ours" if nm.startswith("ours") else f"px_{nm}"].quantile(0.95),
             "energy_mJ_per_frame": dd[f"mJ_{nm}"].mean(), "idle_board_mW": idle_mw}
        if nm != "ours":
            dr = [(Vo[b].sum() - V[b].sum()) / L[b].sum() for b in boots]
            r["ci_lo_ours_minus"], r["ci_hi_ours_minus"] = np.percentile(dr, 2.5), np.percentile(dr, 97.5)
        out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(run / "latency.csv", index=False)
    bcols = [c for c in dd.columns if c.startswith("bd_")]
    bdk = pd.DataFrame({"stage": bcols, "mean": [dd[c].mean() for c in bcols], "p95": [dd[c].quantile(0.95) for c in bcols]})
    bdk.to_csv(run / "breakdown.csv", index=False)
    w = dd.loc[dd.n_crops.idxmax()]
    worst = {"unit": w.unit, "frame": int(w.frame), "n_crops": int(w.n_crops), "rounds": int(w.rounds), "ms_ours": w.ms_ours,
             "ms_ours_noref": w.ms_ours_noref, "px_ours": w.px_ours}
    pd.DataFrame([worst]).to_csv(run / "worst_frame.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"{args.dataset}: {len(dd)} timed frames ({args.warmup} warm-up excluded), {dd.unit.nunique()} units; crops/frame "
          f"{dd.n_crops.mean():.2f}; idle board power {idle_mw:.0f} mW")
    print(res.round(3).to_string(index=False)); print(bdk.round(2).to_string(index=False)); print("worst:", worst)


if __name__ == "__main__":
    main()
