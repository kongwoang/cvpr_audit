#!/usr/bin/env python
"""D2/D9 feasibility (exploratory): how accurate is an off-the-shelf verifier on the cheap decision's pivot?

For each frame whose cheap braking decision has a pivot (removal changes the action; else the object the rule reacts
to most), label the pivot real/phantom against the reference boxes (IoU >= 0.5, KITTI's 2D annotations), and ask each
cached high-fidelity detector whether it sees an object there (a detection with IoU >= 0.3 and conf >= t).  Reports
existence error rates on pivots (the population that matters), and on all cheap detections for contrast.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap.cache import DetCache
from rap.decision import KittiAdapter
from rap.mono import box_iou
from rap.paths import CACHE
from rap.risk import RiskConfig
from vtf import flips, runmeta
from vtf import headroom as HR

VERIFIERS = {"Y640": ("det", "full_640"), "Y960": ("det", "full_960"), "R640": ("rtdetr_kitti", "rt_full_640")}
THRS = (0.10, 0.20, 0.25, 0.35, 0.50)


def main():
    run = runmeta.new_run("verifier_accuracy_KITTI", {})
    cfg = RiskConfig()
    pp = P.PlannerParams()
    rng0 = np.random.default_rng(0)
    seqs = [p.stem for p in sorted((Path(CACHE) / "det" / "cheap_320").glob("*.npz"))]
    rows = []
    for s in seqs:
        geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s)
        c = DetCache(Path(CACHE) / "det" / "cheap_320" / f"{s}.npz")
        ver = {k: DetCache(Path(CACHE) / d / m / f"{s}.npz") for k, (d, m) in VERIFIERS.items()}
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            cheap = flips.Branch(c.det(i), c.geo(i), cfg.thr("cheap"), g, "mono", rng0)
            if not len(cheap):
                continue
            decide = lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp)   # noqa: E731
            S = HR.Obj(cheap.xyxy, cheap.cons)
            piv = HR.pivot_of(S, decide, v, pp)
            gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
            real_all = box_iou(cheap.xyxy, gt).max(1) >= 0.5 if len(gt) else np.zeros(len(cheap), bool)
            for k_, vc in ver.items():
                d = vc.det(vc.index[fr])
                bx, cf = d["xyxy"].astype(float), d["conf"].astype(float)
                iou = box_iou(cheap.xyxy, bx) if len(bx) else np.zeros((len(cheap), 0))
                for oi in range(len(cheap)):
                    best = np.where(iou[oi] >= 0.3, cf, 0.0).max() if iou.shape[1] else 0.0
                    rows.append({"seq": s, "frame": fr, "obj": oi, "pivot": oi == piv, "verifier": k_,
                                 "real": bool(real_all[oi]), "best_conf": float(best),
                                 "cheap_conf": float(cheap.conf[oi]), "box_h": float(cheap.xyxy[oi, 3] - cheap.xyxy[oi, 1])})
    d = pd.DataFrame(rows)
    d.to_pickle(run / "verifier_rows.pkl")
    out = []
    for (k_, piv), e in d.groupby(["verifier", "pivot"]):
        for t in THRS:
            says = e.best_conf >= t
            out.append({"verifier": k_, "population": "pivot" if piv else "non-pivot", "thr": t, "n": len(e),
                        "phantom_rate": float((~e.real).mean()),
                        "real_called_phantom": float((~says[e.real]).mean()),
                        "phantom_called_real": float(says[~e.real].mean()),
                        "error": float((says != e.real).mean())})
    res = pd.DataFrame(out)
    res.to_csv(run / "verifier_accuracy.csv", index=False)
    res.to_csv(runmeta.FINAL / "d2_verifier_accuracy.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
