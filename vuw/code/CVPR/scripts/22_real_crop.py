#!/usr/bin/env python
"""D3-real: run YOLOv8s on the DSF crop itself (no proxy), at the full mode's scale, and score the decision.

For every frame with a DSF crop (from the latest `region_KITTI_<geometry>` run), the crop of the original image is
detected with the same weights at the full mode's pixel scale (640 / 1242 of the image width), boxes are shifted back
to image coordinates, lifted with DEEP's monocular lift (ego frame), and replace the cheap detections whose centre lies
in the crop.  The braking controller and Planner B are scored with DEEP's shared history.  Compared with the proxy
(cached full-frame detections inside the crop) and with uniform frame escalation.
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rap import planner as P
from rap import planner_b as B
from rap.cache import Camera, DetCache
from rap.decision import KittiAdapter, _apply_range_source
from rap.detect import Mode, TwoFidelityDetector
from rap.mono import predicted_geometry
from rap.paths import CACHE, KITTI_IMAGES, MODELS
from rap.risk import RiskConfig
from vtf import runmeta

SCALE = 640 / 1242
DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"


def in_crop(xyxy, crop):
    if crop is None or not len(xyxy):
        return np.zeros(len(xyxy), bool)
    u, v = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
    return (u >= crop[0]) & (u <= crop[2]) & (v >= crop[1]) & (v <= crop[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["mono", "oracle"], default="mono")
    ap.add_argument("--split", default="all", choices=["all", "test"])
    args = ap.parse_args()
    run = runmeta.new_run(f"realcrop_KITTI_{args.geometry}", vars(args))
    rg = pd.read_pickle(sorted(glob.glob(str(runmeta.RAW / f"*_region_KITTI_{args.geometry}" / "region__*.pkl")))[-1])
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[(dv.track == "KITTI") & (dv.geometry == args.geometry)].copy()
    dv["frame"] = dv.frame.astype(int)
    split = dict(zip(dv.seq, dv.split))
    det = TwoFidelityDetector(str(Path(MODELS) / "yolov8s.pt"), backend="torch")
    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()
    pb, cb = B.PARAMS_B["static_obstacles"], B.COSTS_B["default"]
    rng = np.random.default_rng(0)
    crops = {(r.seq, int(r.frame)): (r.brake_crop, r.traj_crop) for r in rg.itertuples()}
    seqs = sorted(set(rg.seq)) if args.split == "all" else sorted(s for s in set(rg.seq) if split[s] == "test")
    rows = []
    for s in seqs:
        geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s)
        cam = Camera("KITTI", s)
        c = DetCache(Path(CACHE) / "det" / "cheap_320" / f"{s}.npz")
        f = DetCache(Path(CACHE) / "det" / "full_640" / f"{s}.npz")
        prev_act = prev_plan = None
        prev_det = None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            truth = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc"))
            dc = c.det(i)
            kc = dc["conf"] >= cfg.thr("cheap")
            geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
            bxc = dc["xyxy"][kc].astype(float)
            cons_c = tuple(np.asarray(a, float) for a in _apply_range_source(bxc, geo_c, g, args.geometry, rng, 0.0))
            img = None
            rec = {"seq": s, "frame": fr}
            for name, crop in (("brake", crops[(s, fr)][0]), ("traj", crops[(s, fr)][1])):
                if name == "brake":
                    decide = lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp)       # noqa: E731
                    cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]               # noqa: E731
                else:
                    decide = lambda t, pu=prev_plan: B.plan(*t, v, pu, pb, cb)                     # noqa: E731
                    cost = lambda a, pu=prev_plan: B.executed_cost(a, *truth, v, pu, pb, cb)["J"]  # noqa: E731
                J_c = cost(decide(cons_c))
                rec[f"{name}_J_c"] = J_c
                if crop is None:
                    rec[f"{name}_J_real"] = J_c
                    rec[f"{name}_crop_px"] = 0
                    continue
                if img is None:
                    img = cv2.imread(str(Path(KITTI_IMAGES) / s / f"{fr:06d}.png"))
                x1, y1, x2, y2 = (int(math.floor(crop[0])), int(math.floor(crop[1])),
                                  int(math.ceil(crop[2])), int(math.ceil(crop[3])))
                x2, y2 = max(x2, x1 + 8), max(y2, y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * SCALE / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * SCALE / 32)) * 32)
                d, _ = det.detect(sub, Mode(f"crop_{nh}x{nw}", nh, nw))
                keep = d["conf"] >= cfg.thr("full")
                dd = {k: (np.asarray(d[k])[keep] if hasattr(d[k], "__len__") and len(d[k]) == len(d["conf"]) else d[k])
                      for k in d}
                dd["xyxy"] = dd["xyxy"].astype(float) + np.array([x1, y1, x1, y1], float)
                R, t = cam.transform("ego")
                gk = predicted_geometry(dd, prev_det, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                cons_k = tuple(np.asarray(a, float) for a in
                               _apply_range_source(dd["xyxy"], gk, g, args.geometry, rng, 0.0))
                out = in_crop(bxc, crop)
                fused = tuple(np.r_[a[~out], b] for a, b in zip(cons_c, cons_k))
                rec[f"{name}_J_real"] = cost(decide(fused))
                rec[f"{name}_crop_px"] = int(nh * nw)
            prev_act = P.discrete_action(P.required_decel(*cons_c, v, pp)[0], pp)
            prev_plan = B.plan(*cons_c, v, prev_plan, pb, cb)
            prev_det = {"xyxy": dc["xyxy"][kc]}
            rows.append(rec)
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_pickle(run / "realcrop.pkl")
    out = []
    for name in ("brake", "traj"):
        ref = dv[dv.system == name][["seq", "frame", "split", "V", "J_cheap"]]
        m = d.merge(ref, on=["seq", "frame"]).merge(rg[["seq", "frame", f"{name}_J_r", f"{name}_area"]], on=["seq", "frame"])
        assert np.allclose(m[f"{name}_J_c"], m.J_cheap, atol=1e-9)
        for sp in ("all", "test"):
            q = m if sp == "all" else m[m.split == "test"]
            if not len(q):
                continue
            Lc = float(q.J_cheap.sum())
            out.append({"cell": f"KITTI {args.geometry} {name}", "split": sp, "n": len(q),
                        "G_all_full": float(q.V.sum()) / Lc,
                        "G_region_proxy": float((q[f"{name}_J_c"] - q[f"{name}_J_r"]).sum()) / Lc,
                        "G_region_real_crop": float((q[f"{name}_J_c"] - q[f"{name}_J_real"]).sum()) / Lc,
                        "mean_crop_net_pixels_vs_full640": float(q[f"{name}_crop_px"].mean() / (192 * 640))})
    res = pd.DataFrame(out)
    res.to_csv(run / "realcrop_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
