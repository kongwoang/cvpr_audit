#!/usr/bin/env python
"""D35 (exploratory): amortise an escalation over time — carry the full mode's extra objects forward.

Escalate every k-th frame (phase-shifted per sequence).  On a non-escalated frame t, the perceived set is the cheap set
plus the full-only objects of the last escalated frame t0 that the cheap mode still does not see (no cheap detection at
IoU >= 0.3 with the old box), carried for at most k-1 frames with their range reduced by the ego's travel
v * (t - t0) * dt (static-world assumption).  Braking controller, shared history.  Compared at the same escalation rate
1/k with random frame escalation and with escalating every frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import KittiAdapter
from rap.mono import box_iou
from rap.paths import CACHE, NUSCENES_TRAINVAL
from rap.risk import RiskConfig
from vtf import flips, runmeta

DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["KITTI", "nuScenes"], required=True)
    ap.add_argument("--geometry", choices=["mono", "oracle"], required=True)
    args = ap.parse_args()
    run = runmeta.new_run(f"reuse_{args.dataset}_{args.geometry}", vars(args))
    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()
    rng0 = np.random.default_rng(0)
    if args.dataset == "KITTI":
        det_dir, cm, fm, adapter = Path(CACHE) / "det", "cheap_320", "full_640", KittiAdapter
    else:
        from rap.nusc import NuScenesDB, make_adapter
        det_dir, cm, fm = Path(CACHE) / "nusc_det_tv", "ns_cheap_320", "ns_full_640"
        adapter = make_adapter(NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval"))
    seqs = [p.stem for p in sorted((det_dir / cm).glob("*.npz"))]
    KS = (2, 3, 5)
    rows = []
    for s in seqs:
        geom, speeds = adapter.geometry(s), adapter.speeds(s)
        dt = Camera(args.dataset, s).dt
        c, f = DetCache(det_dir / cm / f"{s}.npz"), DetCache(det_dir / fm / f"{s}.npz")
        prev_act = None
        carry = {k: None for k in KS}          # (t0, boxes, cons) of the last escalation, per k
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            j = f.index[fr]
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            cheap = flips.Branch(c.det(i), c.geo(i), cfg.thr("cheap"), g, args.geometry, rng0)
            full = flips.Branch(f.det(j), f.geo(j), cfg.thr("full"), g, args.geometry, rng0)
            cost = lambda t: P.decision_cost(P.discrete_action(P.required_decel(*t, v, pp)[0], pp), a_gt, prev_act,  # noqa: E731
                                             pp, cp)["J"]
            rec = {"seq": s, "frame": fr, "J_c": cost(cheap.cons_raw), "J_f": cost(full.cons_raw)}
            pairs, added, _ = flips.match(full, cheap)
            for k in KS:
                esc = (i % k) == 0
                rec[f"esc_k{k}"] = esc
                if esc:
                    rec[f"J_k{k}"] = rec["J_f"]
                    carry[k] = (i, full.xyxy[added], tuple(a[added] for a in full.cons))
                else:
                    t0, bx, cons = carry[k] if carry[k] is not None else (i, np.zeros((0, 4)), tuple(np.zeros(0) for _ in range(4)))
                    if len(bx) and len(cheap):
                        seen = box_iou(bx, cheap.xyxy).max(1) >= 0.3
                    else:
                        seen = np.zeros(len(bx), bool)
                    keep = ~seen
                    shift = v * (i - t0) * dt
                    z = np.maximum(cons[0][keep] - shift, 0.5)
                    extra = (z, cons[1][keep], cons[2][keep], cons[3][keep])
                    rec[f"J_k{k}"] = cost(tuple(np.r_[a, b] for a, b in zip(cheap.cons, extra)))
            prev_act = P.discrete_action(P.required_decel(*cheap.cons_raw, v, pp)[0], pp)
            rows.append(rec)
    d = pd.DataFrame(rows)
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[(dv.track == args.dataset) & (dv.geometry == args.geometry) & (dv.system == "brake")].copy()
    dv["frame"] = dv.frame.astype(int)
    d = d.merge(dv[["seq", "frame", "split", "V", "J_cheap"]], on=["seq", "frame"])
    assert np.allclose(d.J_c - d.J_f, d.V, atol=1e-9)
    d.to_pickle(run / "reuse.pkl")
    out = []
    for sp in ("all", "test"):
        e = d if sp == "all" else d[d.split == "test"]
        Lc = float(e.J_cheap.sum())
        r = {"cell": f"{args.dataset} {args.geometry} brake", "split": sp, "G_all_full": float(e.V.sum()) / Lc}
        for k in KS:
            r[f"G_every{k}_plus_carry"] = float((e.J_c - e[f"J_k{k}"]).sum()) / Lc
            r[f"G_every{k}_no_carry"] = float(e.V[e[f"esc_k{k}"]].sum()) / Lc
            r[f"G_random_rate_1/{k}"] = float(e.V.sum()) / k / Lc
        out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(run / "reuse_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).T.to_string(), flush=True)


if __name__ == "__main__":
    main()
