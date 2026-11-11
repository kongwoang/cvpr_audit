#!/usr/bin/env python
"""K28 — Jetson latency of the current configurations (v5.1-S; the K28 B verifiers; the final configuration in phase
3), with the whole-frame pipelines plain and "+ components" (T4+T2_1 lift on their own history + the Res18x3 refiner),
and the bounded loop. Machinery of scripts/58 / 59 (imported): GPU letterbox / candidate selection, the dynamic crop
engine, the dynamic-batch refiner engine, 59's bounded_run / OneCall. New here: the robust looming lift (T4+T2_1 of
k27_dev.ttc_ss, copied verbatim below) on each pipeline's same-scale history, the Res18x3 refiner engines
(engines/refiner_<tag>_dyn.engine from export_refiners_v3.py), the verifier's dynamic engine <verifier>_dyn.engine and
whole-frame engines <wf_det>_crop_<h>x<w>.engine.
Pipelines (each timed per frame from the image upload to the action; CUDA syncs only at start / end):
  cheap                    YOLOv8s at 320 (the draft), its lift on the previous draft
  <wf>                     whole frame, lift on its previous detections (K24 "plain"), no refiner
  <wf>+comp                whole frame + T4+T2_1 lift + refiner
  ours / ours_noref        v5.1-S: draft + speculative.run (v_close, guard, <= 3 rounds, crops of equal canvas batched,
                           batch_round) with the T4+T2_1 lift on its own history (final scene's crop boxes) + refiner
  ours_bN                  59's bounded loop (at most N calls, one batched call each) with the same lift + refiner
Per frame: ms, engine calls, tensor px, actions, cost. --ref_dir: VUW per-frame results (k24 / k28 pickles) to check
decision parity of ours with the evaluated implementation on the timed frames (key J_<ref_key>).
usage: 60_k28_latency.py --dataset nusc|av2 [--verifier yolov8s] [--refiner v3a_vuw] [--wf_det yolov8s]
                         [--wf u640,u960,...] [--bounds 2,3] [--mask_y0 1830] [--max_frames n] [--tag name]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import math
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("k5", ROOT / "scripts" / "55_large_latency.py")
K = importlib.util.module_from_spec(_s); _s.loader.exec_module(K)
_f = importlib.util.spec_from_file_location("k11", ROOT / "scripts" / "58_large_latency_fast.py")
KF = importlib.util.module_from_spec(_f); _f.loader.exec_module(KF)
_b = importlib.util.spec_from_file_location("k18b", ROOT / "scripts" / "59_bounded_latency.py")
KB = importlib.util.module_from_spec(_b); _b.loader.exec_module(KB)
from rap.mono import box_iou, ttc_from_scale                                     # noqa: E402
from vtf import gpupost as GP                                                    # noqa: E402
from vtf.gpuprep import letterbox                                                # noqa: E402
from vtf.trtdyn import DynTRT                                                    # noqa: E402

P, SP, THR, Mode, ENG = K.P, K.SP, K.THR, K.Mode, K.ENG
STATS = KF.STATS
KB.STATS = STATS


def sync():
    torch.cuda.synchronize()


# ---------------------------------------------------------------- k27_dev.assoc_idx / ttc_ss (verbatim, without T3)
def assoc_idx(curr, prev, thr=0.3):
    out = -np.ones(len(curr), int)
    if len(prev) == 0 or len(curr) == 0:
        return out
    iou = box_iou(curr, prev)
    taken = set()
    for ci in np.argsort(-iou.max(axis=1)):
        for pj in np.argsort(-iou[ci]):
            if iou[ci, pj] < thr:
                break
            if pj not in taken:
                taken.add(int(pj)); out[ci] = pj
                break
    return out


def ttc_ss(X, s1, s2, sc, dt, gate=None, t4=False):
    X = np.asarray(X, float).reshape(-1, 4)
    n = len(X)
    h = X[:, 3] - X[:, 1]
    P1, src = np.full((n, 4), np.nan), np.zeros(n, int)
    j1 = assoc_idx(X, s1) if s1 is not None and len(s1) else -np.ones(n, int)
    if (j1 >= 0).any():
        P1[j1 >= 0], src[j1 >= 0] = s1[j1[j1 >= 0]], 1
    h1 = P1[:, 3] - P1[:, 1]
    if not t4:
        ttc = ttc_from_scale(h, h1, dt)
        if gate is not None:
            with np.errstate(invalid="ignore"):
                ttc[~((h - h1) * sc > gate)] = np.inf
        return ttc, src
    P2 = np.full((n, 4), np.nan)
    if s1 is not None and len(s1) and s2 is not None and len(s2):
        a12 = assoc_idx(s1, s2)
        for i in np.flatnonzero(src == 1):
            if a12[j1[i]] >= 0:
                P2[i] = s2[a12[j1[i]]]
    h2 = P2[:, 3] - P2[:, 1]
    ttc = np.full(n, np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.isfinite(h1) & np.isfinite(h2) & (h > 1) & (h1 > 1) & (h2 > 1)
        r = np.median(np.stack([(h1 / h2 - 1) / dt, (h / h1 - 1) / dt, (h / h2 - 1) / (2 * dt)]), 0)
        ex = ok & (r > 1e-3 / dt)
        ttc[ex] = np.minimum(1 / r[ex], 1e3)
        if gate is not None:
            ttc[~((h - h2) * sc > gate)] = np.inf
    return ttc, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["av2", "nusc"])
    ap.add_argument("--verifier", default="yolov8s")
    ap.add_argument("--refiner", default="v3a_vuw")
    ap.add_argument("--wf_det", default="yolov8s")
    ap.add_argument("--wf", default="")
    ap.add_argument("--bounds", default="2,3")
    ap.add_argument("--v_close", type=float, default=0.0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--max_units", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--mask_y0", type=float, default=0.0)
    ap.add_argument("--ref_dir", default="")
    ap.add_argument("--ref_key", default="v51S|v3a")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    run = K.runmeta.new_run(f"k28_latency_{args.dataset}" + (f"_{args.tag}" if args.tag else ""), vars(args))
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
    det_s = K.FastDetector(str(Path(K.MODELS) / "yolov8s.pt"), backend="trt", engine_dir=str(ENG), engine_stem="yolov8s_crop")
    det_w = det_s if args.wf_det == "yolov8s" else K.FastDetector(str(Path(K.MODELS) / f"{args.wf_det}.pt"), backend="trt",
                                                                  engine_dir=str(ENG), engine_stem=f"{args.wf_det}_crop")
    if args.dataset == "av2":
        WF = {"u640": (640, 512), "u960": (960, 768), "u1280": (1280, 992), "u1536": (1536, 1184), "native": (2048, 1568)}
        CH, scale, gen, NATIVE = Mode("320x256", 320, 256), 960 / 2048, K.frames_av2(args), 1550 * 2048
        MS = {"u640": 640 / 2048, "u960": 960 / 2048, "u1280": 1280 / 2048, "u1536": 1536 / 2048, "native": 1.0}
    else:
        WF = {"u448": (256, 448), "u640": (384, 640), "u768": (448, 768), "u960": (576, 960), "u1280": (736, 1280),
              "native": (928, 1600)}
        CH, scale, gen, NATIVE = Mode("192x320", 192, 320), 0.6, K.frames_nusc(args), 1600 * 900
        MS = {"u448": 0.28, "u640": 0.4, "u768": 0.48, "u960": 0.6, "u1280": 0.8, "native": 1.0}
    keep_wf = args.wf.split(",") if args.wf else [k for k in WF if k not in ("u448", "u768")]
    WF = {k: v for k, v in WF.items() if k in keep_wf}
    M = {nm: Mode(f"{h}x{w}", h, w) for nm, (h, w) in WF.items()}
    missing = [m.name for m in M.values() if not (ENG / f"{args.wf_det}_crop_{m.name}.engine").exists()]
    assert not missing, ("missing whole-frame engines", args.wf_det, missing)
    refb = KF.RefinerBatch(args.refiner)
    G = K.GPUDetect()
    many = KF.Crops(G, scale)
    if args.verifier != "yolov8s":
        many.eng = DynTRT(ENG / f"{args.verifier}_dyn.engine")
    one = KB.OneCall(G, scale, many.eng)
    G.x, G.shape = torch.full((3, 2048, 2048), K.PAD, device="cuda"), (2048, 2048, 3)
    many([[0, 0, 64, 64]] * 2 + [[0, 0, 400, 400], [0, 0, 1000, 1000]])
    bounds = [int(b) for b in args.bounds.split(",") if b]
    pp, cp = P.PlannerParams(), P.CostParams()

    def gdetect(dt_, m):
        xb, r, pad = letterbox(G.x, m.net_h, m.net_w)
        raw = dt_.forward(xb[None], m)
        return GP.post_lean(GP.host_columns(raw, m, THR)[0], m, r, pad, G.shape, min_conf=THR)
    rows, hist, cur, prev_act, imgs = [], {}, None, None, 0
    for unit, fr, path, v, g, cam, (R, t) in gen:
        if args.max_units and unit != cur and len({r["unit"] for r in rows}) >= args.max_units:
            break
        if unit != cur:
            hist, cur, prev_act = {}, unit, None
        img = cv2.imread(path)
        H, W = img.shape[:2]
        dt = float(cam.dt)
        a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
        rec = {"unit": unit, "frame": fr}

        def lift(xyxy, coarse, pv):
            gk = K.predicted_geometry({"xyxy": xyxy, "coarse": coarse}, pv, cam.calib, cam.cam_h, dt, cam_to_ego=(R, t))
            return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

        def lift_t4(name, sc):
            h = hist.get(name, {})

            def f(full):
                z, lo, hi, _ = lift(full["xyxy"], full["coarse"], None)
                tt, _ = ttc_ss(full["xyxy"], h.get("s1"), h.get("s2"), sc, dt, 1.0, True)
                return z, lo, hi, tt
            return f

        def push(name, xyxy):
            h = hist.setdefault(name, {})
            h["s2"] = h.get("s1")
            h["s1"] = np.asarray(xyxy, float).reshape(-1, 4)

        def act(tt):
            return P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)

        def J(a):
            return P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]

        def keep(d):
            k = d["conf"] >= THR
            return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}

        def refine(scene):
            refined = [np.asarray(a, float).copy() for a in scene["mono"]]
            sup = [p_ for p_ in SP.support(scene["mono"], v, pp) if scene["from_full"][p_]]
            corr = refb(img, [(scene["xyxy"][p_], scene["conf"][p_], scene["coarse"][p_], scene["mono"][0][p_]) for p_ in sup], cam, R, t, W, H)
            for p_, c in zip(sup, corr):
                for a in refined[:3]:
                    a[p_] *= c
            return tuple(refined), len(sup)
        # ---- cheap draft
        sync(); t0 = time.perf_counter()
        G.frame(img)
        d = keep(gdetect(det_s, CH))
        tc = lift(d["xyxy"], d["coarse"], hist.get("cheap_prev"))
        a_c = act(tc)
        sync(); rec["ms_cheap"] = 1e3 * (time.perf_counter() - t0)
        rec.update({"a_cheap": a_c, "J_cheap": J(a_c), "px_cheap": CH.net_h * CH.net_w / NATIVE})
        # ---- whole frame, plain and + components
        for nm, m in M.items():
            sync(); t0 = time.perf_counter()
            G.frame(img)
            dw = keep(gdetect(det_w, m))
            tt = lift(dw["xyxy"], dw["coarse"], hist.get(f"{nm}_prev"))
            a = act(tt)
            sync(); rec[f"ms_{nm}"] = 1e3 * (time.perf_counter() - t0)
            rec[f"a_{nm}"], rec[f"J_{nm}"], rec[f"px_{nm}"] = a, J(a), m.net_h * m.net_w / NATIVE
            sync(); t0 = time.perf_counter()
            G.frame(img)
            dw2 = keep(gdetect(det_w, m))
            sc_ = {"xyxy": dw2["xyxy"], "conf": dw2["conf"], "coarse": dw2["coarse"], "mono": lift_t4(f"{nm}+comp", MS[nm])(dw2),
                   "from_full": np.ones(len(dw2["conf"]), bool)}
            tt2, _ = refine(sc_)
            a2 = act(tt2)
            sync(); rec[f"ms_{nm}+comp"] = 1e3 * (time.perf_counter() - t0)
            rec[f"a_{nm}+comp"], rec[f"J_{nm}+comp"], rec[f"px_{nm}+comp"] = a2, J(a2), m.net_h * m.net_w / NATIVE
            hist[f"{nm}_prev"] = dw
            push(f"{nm}+comp", dw2["xyxy"])
        # ---- ours (v5.1-S) and the bounded variants
        for nm, N in [("ours", None)] + [(f"ours_b{b}", b) for b in bounds]:
            STATS["calls"], STATS["tpx"], STATS["exec"], STATS["tpx_b"] = 0, 0, 0, 0
            sync(); t0 = time.perf_counter()
            G.frame(img)
            dd = keep(gdetect(det_s, CH)); STATS["calls"] += 1
            dd["mono"] = lift(dd["xyxy"], dd["coarse"], hist.get("cheap_prev"))
            if N is None:
                scene, crops, rounds = SP.run(dd, v, cam, W, H, lambda cr: many([cr])[0], lift_t4(nm, scale), pp,
                                              v_close=args.v_close, max_rounds=3, guard=True, detect_many=many, batch_round=True)
                info = {"n_crops": len(crops), "rounds": rounds}
            else:
                scene, calls, hit = KB.bounded_run(dd, v, cam, W, H, one, lift_t4(nm, scale), pp, args.v_close, N)
                info = {"calls_loop": calls, "bound_hit": hit}
            a_noref = act(scene["mono"])
            t_loop = time.perf_counter()
            tt, n_ref = refine(scene)
            a_o = act(tt)
            sync(); t_end = time.perf_counter()
            px = (CH.net_h * CH.net_w + (STATS["tpx"] if N is None else STATS["tpx_b"])) / NATIVE
            rec.update({f"ms_{nm}_noref": 1e3 * (t_loop - t0), f"ms_{nm}": 1e3 * (t_end - t0), f"a_{nm}_noref": a_noref,
                        f"a_{nm}": a_o, f"J_{nm}_noref": J(a_noref), f"J_{nm}": J(a_o), f"px_{nm}": px, f"px_{nm}_noref": px,
                        f"calls_{nm}": STATS["calls"] + STATS["exec"], f"n_refined_{nm}": n_ref,
                        **{f"{k}_{nm}": v_ for k, v_ in info.items()}})
            ff = np.asarray(scene["from_full"], bool)
            push(nm, np.asarray(scene["xyxy"], float).reshape(-1, 4)[ff])
        hist["cheap_prev"] = {"xyxy": d["xyxy"], "coarse": d["coarse"]}
        prev_act = a_c
        rec["warm"] = len(rows) < args.warmup
        rows.append(rec)
        if len(rows) % 200 == 0:
            print(f"  {len(rows)} frames", flush=True)
        if args.max_frames and len(rows) >= args.max_frames:
            break
    d = pd.DataFrame(rows)
    d.to_pickle(run / "k28_latency.pkl")
    dd = d[~d.warm].reset_index(drop=True)
    L = dd.J_cheap.to_numpy()
    uq = dd.unit.unique(); idx = [np.flatnonzero(dd.unit.to_numpy() == u) for u in uq]
    rng = np.random.default_rng(0); boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    Vo = L - dd.J_ours.to_numpy()
    names = ["cheap"] + [x for nm in M for x in (nm, f"{nm}+comp")] + ["ours_noref", "ours"] + [x for b in bounds for x in (f"ours_b{b}_noref", f"ours_b{b}")]
    out = []
    for nm in names:
        ms = dd[f"ms_{nm}"]; V = L - dd[f"J_{nm}"].to_numpy()
        r = {"pipeline": nm, "verifier": args.verifier if nm.startswith("ours") else (args.wf_det if nm != "cheap" else "yolov8s"),
             "refiner": args.refiner if (nm.startswith("ours") and not nm.endswith("noref")) or nm.endswith("+comp") else "none",
             "frames": len(dd), "G": V.sum() / L.sum() if nm != "cheap" else 0.0, "mean_ms": ms.mean(), "median_ms": ms.median(),
             "p95_ms": ms.quantile(.95), "p99_ms": ms.quantile(.99), "max_ms": ms.max(), "tensor_px_mean": dd[f"px_{nm}"].mean(),
             "tensor_px_p95": dd[f"px_{nm}"].quantile(.95)}
        if nm != "ours":
            dr = [(Vo[b].sum() - V[b].sum()) / L[b].sum() for b in boots]
            r["ci_lo_ours_minus"], r["ci_hi_ours_minus"] = np.percentile(dr, 2.5), np.percentile(dr, 97.5)
        out.append(r)
    res = pd.DataFrame(out); res.to_csv(run / "latency.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"{args.dataset}: {len(dd)} timed frames; verifier {args.verifier}, refiner {args.refiner}, wf {args.wf_det}")
    print(res.round(3).to_string(index=False))
    if args.ref_dir:
        rf = []
        for f in sorted(glob.glob(str(Path(args.ref_dir) / "*.pkl"))):
            rf += pickle.load(open(f, "rb"))
        rf = pd.DataFrame(rf)
        m = dd.merge(rf[["unit", "frame", f"J_{args.ref_key}", "J_cheap"]], on=["unit", "frame"], suffixes=("", "_ref"))
        par = {"frames_matched": len(m), "identical_cost_ours": float(np.isclose(m.J_ours, m[f"J_{args.ref_key}"]).mean()),
               "identical_cost_cheap": float(np.isclose(m.J_cheap, m.J_cheap_ref).mean()),
               "G_ours_timed": (m.J_cheap.sum() - m.J_ours.sum()) / max(m.J_cheap.sum(), 1e-9),
               "G_ref_same_frames": (m.J_cheap_ref.sum() - m[f"J_{args.ref_key}"].sum()) / max(m.J_cheap_ref.sum(), 1e-9)}
        pd.DataFrame([par]).to_csv(run / "parity.csv", index=False)
        print("parity:", {k: round(v, 4) for k, v in par.items()})


if __name__ == "__main__":
    main()
