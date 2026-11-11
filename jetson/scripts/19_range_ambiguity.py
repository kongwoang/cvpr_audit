#!/usr/bin/env python
"""D39 (exploratory): trigger a pivot's range refinement only when the decision is ambiguous over its range cues.

The monocular lift fuses two range cues per box, the ground-plane cue `z_ground` and the class-height cue `z_height`.
For the cheap decision's pivot, recompute the braking decision with the pivot at each cue; if the decisions differ
(or differ from the fused one), the decision is range-ambiguous and the pivot is sent to a refiner (oracle here:
reference range; noisy: multiplicative error s).  Existence is not verified.  Mono geometry only (the lift is the
consumer's geometry).
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap.cache import DetCache
from rap.decision import KittiAdapter
from rap.mono import box_iou
from rap.paths import CACHE, NUSCENES_TRAINVAL
from rap.risk import RiskConfig
from vtf import flips, runmeta
from vtf import headroom as HR

DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["KITTI", "nuScenes"], default="KITTI")
    args = ap.parse_args()
    run = runmeta.new_run(f"range_ambiguity_{args.dataset}", vars(args))
    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()
    rng0 = np.random.default_rng(0)
    if args.dataset == "KITTI":
        det_dir, cm, adapter = Path(CACHE) / "det", "cheap_320", KittiAdapter
    else:
        from rap.nusc import NuScenesDB, make_adapter
        det_dir, cm = Path(CACHE) / "nusc_det_tv", "ns_cheap_320"
        adapter = make_adapter(NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval"))
    seqs = [p.stem for p in sorted((det_dir / cm).glob("*.npz"))]
    rows = []
    for s in seqs:
        geom, speeds = adapter.geometry(s), adapter.speeds(s)
        c = DetCache(det_dir / cm / f"{s}.npz")
        prev_act = None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            cheap = flips.Branch(c.det(i), c.geo(i), cfg.thr("cheap"), g, "mono", rng0)
            decide = lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp)       # noqa: E731
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]               # noqa: E731
            base_t = cheap.cons
            a0 = decide(base_t)
            rec = {"seq": s, "frame": fr, "J_c": cost(a0), "amb": 0, "has_pivot": 0}
            S = HR.Obj(cheap.xyxy, cheap.cons)
            piv = HR.pivot_of(S, decide, v, pp) if len(cheap) else None
            J_ref = J_ref_all = J_noisy = rec["J_c"]
            if piv is not None:
                rec["has_pivot"] = 1
                acts = set()
                for cue in ("z_ground", "z_height"):
                    t = [a.copy() for a in base_t]
                    zc = float(cheap.mono[cue][piv])
                    if np.isfinite(zc):
                        scale = zc / max(float(t[0][piv]), 1e-3)
                        t[0][piv] = zc
                        t[1][piv] *= scale
                        t[2][piv] *= scale
                        acts.add(decide(tuple(t)))
                amb = len(acts | {a0}) > 1
                rec["amb"] = int(amb)
                # oracle range refinement of the pivot (if it matches a reference box); existence untouched
                gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
                t_ref = [a.copy() for a in base_t]
                t_noisy = [a.copy() for a in base_t]
                if len(gt):
                    iou = box_iou(cheap.xyxy[piv:piv + 1], gt)[0]
                    if iou.max() >= 0.5:
                        b = int(np.argmax(iou))
                        rng = np.random.default_rng(zlib.crc32(f"{s}:{fr}".encode()))
                        sc = float(np.clip(1 + rng.normal(0, 0.1), 0.3, 3))
                        for tt, k in ((t_ref, 1.0), (t_noisy, sc)):
                            tt[0][piv] = float(g["long_near"][b]) * k
                            tt[1][piv] = float(g["lat_min"][b]) * k
                            tt[2][piv] = float(g["lat_max"][b]) * k
                            tt[3][piv] = float(g["ttc"][b])
                J_ref_all = cost(decide(tuple(t_ref)))
                J_ref = J_ref_all if amb else rec["J_c"]
                J_noisy = cost(decide(tuple(t_noisy))) if amb else rec["J_c"]
            rec.update({"J_ref_if_amb": J_ref, "J_ref_always": J_ref_all, "J_noisy_if_amb": J_noisy})
            prev_act = a0
            rows.append(rec)
    d = pd.DataFrame(rows)
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[(dv.track == args.dataset) & (dv.geometry == "mono") & (dv.system == "brake")].copy()
    dv["frame"] = dv.frame.astype(int)
    d = d.merge(dv[["seq", "frame", "split", "V", "J_cheap"]], on=["seq", "frame"])
    assert np.allclose(d.J_c, d.J_cheap, atol=1e-9)
    d.to_pickle(run / "range_ambiguity.pkl")
    out = []
    for sp in ("all", "test"):
        e = d if sp == "all" else d[d.split == "test"]
        Lc = float(e.J_cheap.sum())
        out.append({"cell": f"{args.dataset} mono brake", "split": sp, "G_all_full": float(e.V.sum()) / Lc,
                    "frames_with_pivot": float(e.has_pivot.mean()), "frames_ambiguous": float(e.amb.mean()),
                    "G_refine_pivot_always": float((e.J_c - e.J_ref_always).sum()) / Lc,
                    "G_refine_only_if_ambiguous": float((e.J_c - e.J_ref_if_amb).sum()) / Lc,
                    "G_refine_if_ambiguous_noisy10": float((e.J_c - e.J_noisy_if_amb).sum()) / Lc})
    res = pd.DataFrame(out)
    res.to_csv(run / "range_ambiguity_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).T.to_string(), flush=True)


if __name__ == "__main__":
    main()
