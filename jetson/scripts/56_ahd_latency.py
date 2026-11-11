#!/usr/bin/env python
"""K7 streaming input — Jetson wall-clock per frame on Argoverse-HD val (Argoverse 1.1 ring_front_center 1920x1200 @ 30 Hz,
24 sequences; images copied to ~/datasets/argoverse1), for the streaming-AP (sAP) simulation on VUW.

Same fast path and instrumentation as scripts/55_large_latency.py (imported, unchanged): TensorRT FP16 YOLOv8s, clocks
locked, every pipeline timed from its own frame upload to its decision (lift included), CUDA syncs at the boundaries.
Pipelines = those of remote/A_method/argoverse_hd_eval.py: cheap 224x320, uniform 640 / 960 / 1280 / 1600 / 1920 (native),
ours = AV2's frozen configuration (v4-lite: v_close 0, crops at 960 / 1920 = 0.5, guard, merged-scene history, <= 3 rounds)
without and with refiner v2 (TensorRT). Output: per-frame latency table (unit = sequence id, frame = fid) + summary;
a warm-up pass over the first --warmup frames of the first sequence is run and discarded before the recorded pass, so
every frame of every sequence has a warm latency (the streaming simulation needs all frames).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("k5", ROOT / "scripts" / "55_large_latency.py")
K = importlib.util.module_from_spec(_s)
_s.loader.exec_module(K)
P, SP, runmeta, THR, sync, STATS = K.P, K.SP, K.runmeta, K.THR, K.sync, K.STATS
Mode, predicted_geometry, ENG = K.Mode, K.predicted_geometry, K.ENG
HD = Path.home() / "datasets" / "argoverse_hd" / "Argoverse-HD" / "annotations"
AV1 = Path.home() / "datasets" / "argoverse1" / "argoverse-tracking" / "val"
UNIF = (640, 960, 1280, 1600, 1920)


def seqs():
    d = json.loads((HD / "val.json").read_text())
    by = {}
    for im in d["images"]:
        by.setdefault(im["sid"], []).append(im)
    out = []
    for sid, ims in sorted(by.items()):
        ims = sorted(ims, key=lambda x: x["fid"])
        out.append((sid, d["seq_dirs"][sid].split("/")[1], [i["name"] for i in ims], [i["id"] for i in ims]))
    return out


def main():
    import time
    from vtf import av1 as A1
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=200)
    args = ap.parse_args()
    run = runmeta.new_run("ahd_latency", vars(args))
    det = K.FastDetector(str(Path(K.MODELS) / "yolov8s.pt"), backend="trt", engine_dir=str(ENG), engine_stem="yolov8s_crop")
    M = {"cheap": Mode("224x320", 224, 320)}
    for s in UNIF:
        h = int(math.ceil(s * 1200 / 1920 / 32)) * 32
        M[f"u{s}"] = Mode(f"{h}x{s}", h, s)
    missing = [k for k, m in M.items() if not (ENG / f"yolov8s_crop_{m.name}.engine").exists()]
    assert not missing, f"missing engines {missing} (engines/build_ahd.sh)"
    NATIVE = 1920 * 1200
    det.raw_forward = det.forward
    det.forward = K.Counted(det.raw_forward, "det_frame")
    batch = {1: None}
    for b in (2, 4, 8):
        if (ENG / f"yolov8s_crop_b{b}_96x288.engine").exists():
            batch[b] = K.TRTModule(ENG / f"yolov8s_crop_b{b}_96x288.engine")
    ref = K.Refiner("v2_vuw")
    G = K.GPUDetect()
    many = K.ScaledCrops(G, det, 960 / 1920, batch)
    G.x, G.shape = torch.full((3, 2048, 2048), K.PAD, device="cuda"), (2048, 2048, 3)
    many([[0, 0, 64, 64]] * 9 + [[0, 0, 400, 400], [0, 0, 1200, 300], [0, 0, 300, 1200], [0, 0, 600, 200], [0, 0, 200, 600]])
    pp, cp = P.PlannerParams(), P.CostParams()
    rows = []
    S = seqs()[: args.max_seqs or None]
    plan = [(S[0], args.warmup, False)] + [(s_, None, True) for s_ in S]
    for (sid, log, names, ids), nmax, record in plan:
        names, ids = (names[:nmax], ids[:nmax]) if nmax else (names, ids)
        L = A1.AV1Log(AV1, log, names)
        geom, speeds = L.geometry(), L.speeds()
        gtf = L.gt_frames()
        R, t = L.transform("ego")
        W, H = L.W, L.H
        hist, prev_act = {}, None
        for fr, iid in enumerate(ids):
            img = cv2.imread(str(L.image_path(fr)))
            if img is None:
                continue
            v = float(speeds[fr])
            g = geom[geom["frame"] == fr] if len(geom) else geom
            g = g[(g["y2"] - g["y1"]) >= 10] if len(g) else g
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
            rec = {"sid": sid, "frame": fr, "image_id": iid, "gt": fr in gtf}

            def lift(xyxy, coarse, pv):
                gk = predicted_geometry({"xyxy": xyxy, "coarse": coarse}, pv, L.calib, L.cam_h, L.dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def J(tt):
                return P.decision_cost(P.discrete_action(P.required_decel(*tt, v, pp)[0], pp), a_gt, prev_act, pp, cp)["J"]

            def keep(d):
                k_ = d["conf"] >= THR
                return {"xyxy": d["xyxy"][k_].astype(float), "conf": d["conf"][k_].astype(float), "coarse": d["coarse"][k_]}

            def whole(nm, m):
                STATS.clear(); sync(); t0 = time.perf_counter()
                G.frame(img)
                d = keep(G.detect(det, m, THR))
                tt = lift(d["xyxy"], d["coarse"], hist.get(nm))
                sync(); rec[f"ms_{nm}"] = 1e3 * (time.perf_counter() - t0)
                rec[f"J_{nm}"], rec[f"px_{nm}"] = J(tt), m.net_h * m.net_w / NATIVE
                hist[nm] = d
                return tt
            tc = whole("cheap", M["cheap"])
            for nm, m in M.items():
                if nm != "cheap":
                    whole(nm, m)
            STATS.clear(); sync(); t0 = time.perf_counter()
            G.frame(img)
            d = keep(G.detect(det, M["cheap"], THR))
            d["mono"] = lift(d["xyxy"], d["coarse"], hist.get("c_ours"))
            scene, crops, rounds = SP.run(d, v, L, W, H, lambda cr: many([cr])[0],
                                          lambda full: lift(full["xyxy"], full["coarse"], hist.get("ours")), pp, v_close=0.0,
                                          max_rounds=3, guard=True, detect_many=many, batch_round=True)
            sync(); t_loop = time.perf_counter()
            refined = [np.asarray(a, float).copy() for a in scene["mono"]]
            for p_ in SP.support(scene["mono"], v, pp):
                if scene["from_full"][p_]:
                    corr = ref(img, scene["xyxy"][p_], scene["conf"][p_], scene["coarse"][p_], scene["mono"][0][p_], L, R, t, W, H)
                    for a in refined[:3]:
                        a[p_] *= corr
            sync(); t_end = time.perf_counter()
            rec.update({"ms_ours_noref": 1e3 * (t_loop - t0), "ms_ours": 1e3 * (t_end - t0),
                        "J_ours_noref": J(scene["mono"]), "J_ours": J(tuple(refined)), "n_crops": len(crops), "rounds": rounds,
                        "calls_ours": sum(v_[0] for k_, v_ in STATS.items() if k_.startswith("det_")),
                        "px_ours": (M["cheap"].net_h * M["cheap"].net_w + STATS.get("tpx_round", [0, 0.0])[1] +
                                    STATS.get("tpx_guard", [0, 0.0])[1]) / NATIVE})
            hist["ours"] = {"xyxy": scene["xyxy"]}
            hist["c_ours"] = {"xyxy": d["xyxy"], "coarse": d["coarse"]}
            prev_act = P.discrete_action(P.required_decel(*tc, v, pp)[0], pp)
            if not record:
                continue
            rows.append(rec)
            if len(rows) % 500 == 0:
                print(f"  {len(rows)} frames (seq {sid})", flush=True)
                pd.DataFrame(rows).to_csv(run / "ahd_latency_per_frame.csv", index=False)
    d = pd.DataFrame(rows)
    d.to_csv(run / "ahd_latency_per_frame.csv", index=False)
    dd = d
    names = list(M) + ["ours_noref", "ours"]
    gt = dd[dd["gt"].astype(bool)]
    Lc = gt.J_cheap.sum()
    out = [{"pipeline": nm, "frames": len(dd), "mean_ms": dd[f"ms_{nm}"].mean(), "median_ms": dd[f"ms_{nm}"].median(),
            "p95_ms": dd[f"ms_{nm}"].quantile(.95), "p99_ms": dd[f"ms_{nm}"].quantile(.99),
            "G_jetson_parity": round((Lc - gt[f"J_{nm}"].sum()) / Lc, 3)} for nm in names]
    res = pd.DataFrame(out)
    res.to_csv(run / "ahd_latency_summary.csv", index=False)
    print(f"{len(dd)} timed frames, {dd.sid.nunique()} sequences; crops/frame {dd.n_crops.mean():.2f}")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
