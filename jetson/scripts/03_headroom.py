#!/usr/bin/env python
"""Exploratory (post hoc, NOT pre-registered): decision-pivot verification against frame escalation, oracle headroom.

Per frame and consumer (braking controller; Planner B on KITTI), the loss under:
  cheap, full                          DEEP's two branches (shared action history)
  {cheap,full}_pK{1,2}                 up to K decision pivots verified by an oracle verifier
  {cheap,full}_pK1_e{eps}_s{sigma}     the same with verifier errors (existence flip rate eps, range noise sigma)
  {cheap,full}_all                     every detection verified (misses remain)
Writes `headroom__<dataset>__<geometry>.pkl`.  Every quantity is an oracle bound on what verification could buy.
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap import planner_b as B
from rap.cache import DetCache
from rap.decision import KittiAdapter
from rap.paths import CACHE, NUSCENES_TRAINVAL
from rap.risk import RiskConfig
from vtf import flips, runmeta
from vtf import headroom as HR

NOISE = [(0.05, 0.05), (0.10, 0.10), (0.20, 0.20)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["KITTI", "nuScenes"], required=True)
    ap.add_argument("--geometry", choices=["mono", "oracle"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify_iou", type=float, default=0.5, help="IoU at which the oracle verifier calls a box real")
    args = ap.parse_args()
    HR.MATCH_IOU = args.verify_iou
    tag = "" if args.verify_iou == 0.5 else f"_iou{args.verify_iou}"
    run = runmeta.new_run(f"headroom{tag}_{args.dataset}_{args.geometry}", vars(args))
    if args.dataset == "KITTI":
        det_dir, cm, fm, adapter, with_traj = Path(CACHE) / "det", "cheap_320", "full_640", KittiAdapter, True
    else:
        from rap.nusc import NuScenesDB, make_adapter
        det_dir, cm, fm = Path(CACHE) / "nusc_det_tv", "ns_cheap_320", "ns_full_640"
        adapter, with_traj = make_adapter(NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")), False
    seqs = [p.stem for p in sorted((det_dir / cm).glob("*.npz"))][: args.limit]
    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()
    pb, cb = B.PARAMS_B["static_obstacles"], B.COSTS_B["default"]
    rng0 = np.random.default_rng(0)
    rows = []
    for s in seqs:
        geom, speeds = adapter.geometry(s), adapter.speeds(s)
        c, f = DetCache(det_dir / cm / f"{s}.npz"), DetCache(det_dir / fm / f"{s}.npz")
        prev_act = prev_plan = None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            j = f.index[fr]
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            br = {"cheap": flips.Branch(c.det(i), c.geo(i), cfg.thr("cheap"), g, args.geometry, rng0),
                  "full": flips.Branch(f.det(j), f.geo(j), cfg.thr("full"), g, args.geometry, rng0)}
            rec = {"seq": s, "frame": fr}
            rng = np.random.default_rng(zlib.crc32(f"{s}:{fr}".encode()))
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            truth = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc"))
            consumers = {"brake": (lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp),
                                   lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"])}
            if with_traj:
                pu = prev_plan
                consumers["traj"] = (lambda t: B.plan(*t, v, pu, pb, cb),
                                     lambda a: B.executed_cost(a, *truth, v, pu, pb, cb)["J"])
            for name, (decide, cost) in consumers.items():
                for tag, b in br.items():
                    rec[f"{name}_J_{tag}"] = cost(decide(b.cons_raw))
                    for K in (1, 2):
                        S, n = HR.refine(b.xyxy, b.cons, g, decide, v, pp, K)
                        rec[f"{name}_J_{tag}_pK{K}"] = cost(decide(S.cons()))
                        rec[f"{name}_n_{tag}_pK{K}"] = n
                    for eps, sig in NOISE:
                        S, _ = HR.refine(b.xyxy, b.cons, g, decide, v, pp, 1, rng, eps, sig)
                        rec[f"{name}_J_{tag}_pK1_e{eps}_s{sig}"] = cost(decide(S.cons()))
                    rec[f"{name}_J_{tag}_all"] = cost(decide(HR.verify_all(b.xyxy, b.cons, g).cons()))
            prev_act = P.discrete_action(P.required_decel(*br["cheap"].cons_raw, v, pp)[0], pp)
            if with_traj:
                prev_plan = B.plan(*br["cheap"].cons_raw, v, prev_plan, pb, cb)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_pickle(run / f"headroom__{args.dataset}__{args.geometry}.pkl")
    print(f"  {len(df)} frames written", flush=True)


if __name__ == "__main__":
    main()
