#!/usr/bin/env python
"""D5/D6 (exploratory): decisions of the KITTI consumers under every cached perception mode.

Modes: YOLOv8s at 320/384/512/640/960 and RT-DETR-l at 320/480/640.  Every mode's loss is charged against the
previous action of the all-YOLOv8s-320 run (DEEP's shared history), so J_a - J_b is the decision value of switching
one input from mode a to mode b.  Also records each mode's misses and false positives against the reference boxes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap import planner_b as B
from rap.cache import DetCache
from rap.decision import KittiAdapter, _apply_range_source
from rap.mono import box_iou
from rap.paths import CACHE
from rap.risk import RiskConfig
from vtf import runmeta

MODES = {"Y320": ("det", "cheap_320"), "Y384": ("det", "cheap_384"), "Y512": ("det512", "cheap_512"),
         "Y640": ("det", "full_640"), "Y960": ("det", "full_960"), "R320": ("rtdetr_kitti", "rt_cheap_320"),
         "R480": ("rtdetr_kitti_mid", "rt_mid_480"), "R640": ("rtdetr_kitti", "rt_full_640")}
BASE = "Y320"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["mono", "oracle"], required=True)
    ap.add_argument("--thr", type=float, default=0.25, help="one operating threshold for every mode")
    ap.add_argument("--mode_thr", default="", help="per-mode overrides, e.g. R640=0.5,R320=0.5")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    thr = {m: args.thr for m in MODES}
    thr.update({k: float(v) for k, v in (x.split("=") for x in args.mode_thr.split(",") if x)})
    run = runmeta.new_run(f"multimode{args.tag}_{args.geometry}", vars(args))
    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()
    pb, cb = B.PARAMS_B["static_obstacles"], B.COSTS_B["default"]
    rng = np.random.default_rng(0)
    seqs = [p.stem for p in sorted((Path(CACHE) / "det" / "cheap_320").glob("*.npz"))]
    rows = []
    for s in seqs:
        geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s)
        caches = {m: DetCache(Path(CACHE) / d / mode / f"{s}.npz") for m, (d, mode) in MODES.items()}
        base = caches[BASE]
        prev_act = prev_plan = None
        for i, fr in enumerate(base.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            truth = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc"))
            rec = {"seq": s, "frame": fr, "v_ego": v}
            acts = {}
            for m, c in caches.items():
                idx = c.index[fr]
                d, geo = c.det(idx), c.geo(idx)
                k = d["conf"] >= thr[m]
                bx = d["xyxy"][k].astype(float)
                z, lo, hi, tt = _apply_range_source(bx, {kk: vv[k] for kk, vv in geo.items()}, g, args.geometry, rng, 0.0)
                a = P.discrete_action(P.required_decel(z, lo, hi, tt, v, pp)[0], pp)
                pl = B.plan(z, lo, hi, tt, v, prev_plan, pb, cb)
                acts[m] = (a, pl)
                rec[f"act_{m}"] = a
                rec[f"J_{m}"] = P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]
                rec[f"plan_{m}"] = pl
                rec[f"JB_{m}"] = B.executed_cost(pl, *truth, v, prev_plan, pb, cb)["J"]
                if len(gt) and len(bx):
                    iou = box_iou(gt, bx)
                    rec[f"fn_{m}"] = int((iou.max(1) < 0.5).sum())
                    rec[f"fp_{m}"] = int((iou.max(0) < 0.5).sum())
                else:
                    rec[f"fn_{m}"], rec[f"fp_{m}"] = len(gt), len(bx)
                rec[f"n_{m}"] = int(k.sum())
            prev_act, prev_plan = acts[BASE]
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_pickle(run / f"multimode__KITTI__{args.geometry}.pkl")
    print(f"  {len(df)} frames", flush=True)


if __name__ == "__main__":
    main()
