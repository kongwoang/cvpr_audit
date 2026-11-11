#!/usr/bin/env python
"""D7 (exploratory): stakes-proportional evidence — a detection that would change the decision needs more confidence.

Rule, per branch and frame (deployable: detections, mono lift, the consumer): start from the detections with
conf >= t_base; a detection whose removal alone would change the decision (a pivot) is admitted only if its confidence
is >= t_pivot, iterated until the decision's pivots all pass.  (t_base, t_pivot) are chosen on the non-test units by
realised decision value of the full branch against all-cheap; the rule is applied to both branches, and reported on
the test split.  t_pivot = t_base is DEEP's pipeline.
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
from rap.paths import CACHE, NUSCENES_TRAINVAL
from rap.risk import RiskConfig
from vtf import runmeta

T_BASE = [0.25]
T_PIVOT = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"


def admit(conf, cons, decide, t_pivot, max_iter=5):
    """Indices kept after removing low-confidence pivots one at a time."""
    keep = np.ones(len(conf), bool)
    for _ in range(max_iter):
        idx = np.flatnonzero(keep)
        if not len(idx):
            break
        t = tuple(a[idx] for a in cons)
        base = decide(t)
        weak = [k for k, i in enumerate(idx) if conf[i] < t_pivot
                and decide(tuple(np.delete(a, k) for a in t)) != base]
        if not weak:
            break
        k = min(weak, key=lambda k: conf[idx[k]])
        keep[idx[k]] = False
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["KITTI", "nuScenes"], required=True)
    ap.add_argument("--geometry", choices=["mono", "oracle"], required=True)
    args = ap.parse_args()
    run = runmeta.new_run(f"stakes_{args.dataset}_{args.geometry}", vars(args))
    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()
    pb, cb = B.PARAMS_B["static_obstacles"], B.COSTS_B["default"]
    rng = np.random.default_rng(0)
    if args.dataset == "KITTI":
        det_dir, cm, fm, adapter, with_traj = Path(CACHE) / "det", "cheap_320", "full_640", KittiAdapter, True
    else:
        from rap.nusc import NuScenesDB, make_adapter
        det_dir, cm, fm = Path(CACHE) / "nusc_det_tv", "ns_cheap_320", "ns_full_640"
        adapter, with_traj = make_adapter(NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")), False
    seqs = [p.stem for p in sorted((det_dir / cm).glob("*.npz"))]
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
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            truth = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc"))
            rec = {"seq": s, "frame": fr}
            br = {}
            for tag, cache, idx in (("c", c, i), ("f", f, j)):
                d, geo = cache.det(idx), cache.geo(idx)
                k = d["conf"] >= min(T_BASE)
                geo_k = {kk: vv[k] for kk, vv in geo.items()}
                cons = tuple(np.asarray(a, float) for a in
                             _apply_range_source(d["xyxy"][k].astype(float), geo_k, g, args.geometry, rng, 0.0))
                br[tag] = (d["conf"][k].astype(float), cons)
            consumers = {"brake": (lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp),
                                   lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"])}
            if with_traj:
                pu = prev_plan
                consumers["traj"] = (lambda t: B.plan(*t, v, pu, pb, cb),
                                     lambda a: B.executed_cost(a, *truth, v, pu, pb, cb)["J"])
            for name, (decide, cost) in consumers.items():
                for tp in T_PIVOT:
                    for tag in ("c", "f"):
                        conf, cons = br[tag]
                        keep = admit(conf, cons, decide, tp)
                        rec[f"{name}_J_{tag}_{tp}"] = cost(decide(tuple(a[keep] for a in cons)))
            prev_act = P.discrete_action(P.required_decel(*br["c"][1], v, pp)[0], pp)
            if with_traj:
                prev_plan = B.plan(*br["c"][1], v, prev_plan, pb, cb)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_pickle(run / f"stakes__{args.dataset}__{args.geometry}.pkl")

    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[dv.track == args.dataset].copy()
    dv["frame"] = dv.frame.astype(int)
    out = []
    for name in (["brake", "traj"] if with_traj else ["brake"]):
        ref = dv[(dv.geometry == args.geometry) & (dv.system == name)][["seq", "frame", "split", "J_cheap", "J_full"]]
        d = df.merge(ref, on=["seq", "frame"])
        assert np.allclose(d[f"{name}_J_c_0.25"], d.J_cheap, atol=1e-9)
        assert np.allclose(d[f"{name}_J_f_0.25"], d.J_full, atol=1e-9)
        nt, te = d[d.split != "test"], d[d.split == "test"]
        # choose t_pivot on non-test: (a) for the full branch, (b) for the cheap branch, by realised loss
        best_f = min(T_PIVOT, key=lambda t: nt[f"{name}_J_f_{t}"].sum())
        best_c = min(T_PIVOT, key=lambda t: nt[f"{name}_J_c_{t}"].sum())
        Lc = float(te.J_cheap.sum())
        out.append({"cell": f"{args.dataset} {args.geometry} {name}", "t_pivot_full": best_f, "t_pivot_cheap": best_c,
                    "G_full (DEEP)": float((te.J_cheap - te.J_full).sum()) / Lc,
                    "G_full + stakes rule": float((te.J_cheap - te[f"{name}_J_f_{best_f}"]).sum()) / Lc,
                    "G_cheap + stakes rule (no escalation)": float((te.J_cheap - te[f"{name}_J_c_{best_c}"]).sum()) / Lc,
                    **{f"G_full t={t}": float((te.J_cheap - te[f"{name}_J_f_{t}"]).sum()) / Lc for t in T_PIVOT}})
    res = pd.DataFrame(out)
    res.to_csv(run / f"stakes_summary__{args.dataset}__{args.geometry}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).T.to_string(), flush=True)


if __name__ == "__main__":
    main()
