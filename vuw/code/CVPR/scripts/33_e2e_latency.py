#!/usr/bin/env python
"""B2 step 2: end-to-end wall-clock of the deployed pipelines on the Jetson (TensorRT FP16), and the decision value of
the deployed speculative pipeline (braking, KITTI mono).

Per frame (image decode excluded, identical for all pipelines), each pipeline = preprocessing + TensorRT inference +
NMS + monocular lift (+ the speculative extras):
  cheap        YOLOv8s 96x320
  full640      YOLOv8s 192x640 alone
  full960      YOLOv8s 288x960 alone
  cascade640   cheap, then full640 on every frame (DEEP's escalation, uniform)
  speculative  cheap -> closed-form DSF + pivots (vtf.dsf) -> each crop letterboxed into a 96x288 engine at up to the
               960 scale -> lift -> fuse -> decision
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from rap import planner as P
from rap.cache import Camera
from rap.decision import KittiAdapter, _apply_range_source
from rap.detect import MODES, Mode, TwoFidelityDetector
from rap.mono import predicted_geometry
from rap.paths import KITTI_IMAGES, MODELS
from vtf import dsf, runmeta

R_GRID = np.array([12.5, 17.5, 22.5, 27.5, 35.0, 45.0, 55.0, 70.0])
Y_GRID = np.array([-6.0, -4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0])
OBJ_W, OBJ_H, CTX, THR = 1.8, 1.6, 2.0, 0.25
DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()
    run = runmeta.new_run("e2e_latency_KITTI", vars(args))
    w = str(Path(MODELS) / "yolov8s.pt")
    base = TwoFidelityDetector(w, backend="trt", engine_dir=str(Path(MODELS) / "engines"), engine_stem="yolov8s")
    crop = TwoFidelityDetector(w, backend="trt", engine_dir=str(runmeta.ROOT / "engines"), engine_stem="yolov8s_crop")
    m_cheap, m_full = MODES["cheap_320"], MODES["full_640"]
    m_960, m_canvas = Mode("288x960", 288, 960), Mode("96x288", 96, 288)
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[(dv.track == "KITTI") & (dv.geometry == "mono") & (dv.system == "brake")].copy()
    dv["frame"] = dv.frame.astype(int)
    seqs = sorted(set(dv[dv.split == args.split].seq)) if args.split != "all" else sorted(set(dv.seq))

    def detect(det, img, mode):
        d, _ = det.detect(img, mode)
        sync()
        return d

    def lift(d, prev, cam, R, t, keep_thr=THR):
        k = d["conf"] >= keep_thr
        dd = {"xyxy": d["xyxy"][k].astype(float), "coarse": d["coarse"][k]}
        return dd, predicted_geometry(dd, prev, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))

    rows, n = [], 0
    for s in seqs:
        geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s)
        cam = Camera("KITTI", s)
        R, t = cam.transform("ego")
        P2 = cam.calib.P2
        fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
        prev = {"cheap": None, "full640": None, "full960": None}
        prev_act = None
        frames = sorted(int(f) for f in dv[dv.seq == s].frame)
        for fr in frames:
            img = cv2.imread(str(Path(KITTI_IMAGES) / s / f"{fr:06d}.png"))
            H, W = img.shape[:2]
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            rec = {"seq": s, "frame": fr}
            # ---- baselines
            for name, det, mode in (("full640", base, m_full), ("full960", crop, m_960)):
                t0 = time.perf_counter()
                d = detect(det, img, mode)
                dd, gg = lift(d, prev[name], cam, R, t)
                rec[f"ms_{name}"] = 1e3 * (time.perf_counter() - t0)
                prev[name] = dd
                tt_ = tuple(np.asarray(gg[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))
                rec[f"J_{name}_trt"] = P.decision_cost(P.discrete_action(P.required_decel(*tt_, v, pp)[0], pp),
                                                       a_gt, prev_act, pp, cp)["J"]
            # ---- speculative (its first stage is the cheap pipeline)
            t0 = time.perf_counter()
            d = detect(base, img, m_cheap)
            dc, gc = lift(d, prev["cheap"], cam, R, t)
            t_cheap = time.perf_counter()
            mono_t = tuple(np.asarray(gc[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))
            sens = dsf.insertion_grid(mono_t, v, R_GRID, Y_GRID, OBJ_W, pp)
            crops = []
            if sens.any():
                rr, yy = np.meshgrid(R_GRID, Y_GRID, indexing="ij")
                rr, yy = rr[sens], yy[sens]
                vb = cy + fy * cam.cam_h / rr
                uc, hw = cx - fx * yy / rr, fx * OBJ_W / (2 * rr)
                crops.append([max((uc - hw).min(), 0), max((vb - fy * OBJ_H / rr).min(), 0),
                              min((uc + hw).max(), W), min(vb.max(), H)])
            for k in dsf.brake_pivots(mono_t, v, pp):
                x1, y1, x2, y2 = dc["xyxy"][k]
                cw, ch = (x2 - x1) * CTX, (y2 - y1) * CTX
                crops.append([max((x1 + x2) / 2 - cw / 2, 0), max((y1 + y2) / 2 - ch / 2, 0),
                              min((x1 + x2) / 2 + cw / 2, W), min((y1 + y2) / 2 + ch / 2, H)])
            t_field = time.perf_counter()
            boxes, coarse = [], []
            for cr in crops:
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                dk = detect(crop, img[y1:y2, x1:x2], m_canvas)
                keep = dk["conf"] >= THR
                boxes.append(dk["xyxy"][keep].astype(float) + np.array([x1, y1, x1, y1], float))
                coarse.append(dk["coarse"][keep])
            t_crops = time.perf_counter()
            if crops:
                bx = np.concatenate(boxes)
                dk = {"xyxy": bx, "coarse": np.concatenate(coarse)}
                gk = predicted_geometry(dk, prev["cheap"], cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t)) if len(bx) else None
                u, w_ = (dc["xyxy"][:, 0] + dc["xyxy"][:, 2]) / 2, (dc["xyxy"][:, 1] + dc["xyxy"][:, 3]) / 2
                inside = np.zeros(len(u), bool)
                for cr in crops:
                    inside |= (u >= cr[0]) & (u <= cr[2]) & (w_ >= cr[1]) & (w_ <= cr[3])
                zk = tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc")) if gk else tuple(np.zeros(0) for _ in range(4))
                fused = tuple(np.r_[a[~inside], b] for a, b in zip(mono_t, zk))
            else:
                fused = mono_t
            a_spec = P.discrete_action(P.required_decel(*fused, v, pp)[0], pp)
            t_end = time.perf_counter()
            a_c = P.discrete_action(P.required_decel(*mono_t, v, pp)[0], pp)
            rec.update({"ms_cheap": 1e3 * (t_cheap - t0), "ms_field": 1e3 * (t_field - t_cheap),
                        "ms_crops": 1e3 * (t_crops - t_field), "ms_fuse": 1e3 * (t_end - t_crops),
                        "ms_speculative": 1e3 * (t_end - t0), "n_crops": len(crops),
                        "J_c_trt": P.decision_cost(a_c, a_gt, prev_act, pp, cp)["J"],
                        "J_spec_trt": P.decision_cost(a_spec, a_gt, prev_act, pp, cp)["J"]})
            prev["cheap"], prev_act = dc, a_c
            rows.append(rec)
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
        if args.max_frames and n >= args.max_frames:
            break
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_pickle(run / "e2e.pkl")
    t = d.iloc[20:]                                   # warm-up excluded
    lat = {k: (float(t[k].median()), float(t[k].mean()), float(t[k].quantile(0.95)))
           for k in ("ms_cheap", "ms_full640", "ms_full960", "ms_speculative", "ms_field", "ms_crops", "ms_fuse")}
    lat["ms_cascade640 (cheap+full640)"] = (float((t.ms_cheap + t.ms_full640).median()),
                                            float((t.ms_cheap + t.ms_full640).mean()),
                                            float((t.ms_cheap + t.ms_full640).quantile(0.95)))
    res = pd.DataFrame(lat, index=["median", "mean", "p95"]).T
    Lc = float(d.J_c_trt.sum())
    print(res.round(2).to_string())
    print(f"crops per frame {d.n_crops.mean():.2f}; decision value, all on the TensorRT branch (braking, KITTI mono, "
          f"{args.split}): uniform 640 {(d.J_c_trt - d.J_full640_trt).sum() / Lc:.4f}, uniform 960 "
          f"{(d.J_c_trt - d.J_full960_trt).sum() / Lc:.4f}, speculative {(d.J_c_trt - d.J_spec_trt).sum() / Lc:.4f}")
    res.to_csv(run / "latency.csv")


if __name__ == "__main__":
    main()
