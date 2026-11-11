#!/usr/bin/env python
"""Where does v3 beat a support-only crop policy? Per-frame split of v3's gain over a crop baseline (default: strict
temporal criticality) by whether a decision-relevant GT object (its own required deceleration reaches DECELERATE) is
NEW this keyframe (not visible, >= 10 px, in any earlier keyframe of the scene) — the case the unseen-object field
is for — versus all relevant objects seen before, versus no relevant object.
usage: field_value_analysis.py [results/baselines_eq__new_test_strict.pkl] [arm]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import NUSCENES_TRAINVAL

B = Path(os.environ["B"])


def main():
    pkl = sys.argv[1] if len(sys.argv) > 1 else str(B / "results" / "baselines_eq__new_test_strict.pkl")
    arm = sys.argv[2] if len(sys.argv) > 2 else "temporal_crit"
    d = pd.read_pickle(pkl)
    ad = make_adapter(NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval"))
    pp = P.PlannerParams()
    rows = []
    for s, gs in d.groupby("seq"):
        geom, speeds = ad.geometry(s), ad.speeds(s)
        seen = set()
        for fr in sorted(gs.frame):
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            rel = [j for j in range(len(g)) if P.discrete_action(P.required_decel(
                g["long_near"][j:j + 1], g["lat_min"][j:j + 1], g["lat_max"][j:j + 1], g["ttc"][j:j + 1], v, pp)[0], pp) > 0]
            rows.append({"seq": s, "frame": fr, "n_rel": len(rel), "new_rel": any(g["track_id"][j] not in seen for j in rel)})
            seen |= set(g["track_id"].tolist())
    m = d.merge(pd.DataFrame(rows), on=["seq", "frame"])
    L = m.J_c.sum()
    m["gain"] = (m[f"J_{arm}"] - m.J_v3) / L          # > 0: v3 better
    groups = {"relevant object NEW this frame": m[m.new_rel], "relevant objects all seen before": m[(m.n_rel > 0) & ~m.new_rel],
              "no relevant object": m[m.n_rel == 0]}
    print(f"v3 vs {arm}: total gain {m.gain.sum():+.4f} (share of all-cheap loss)")
    for k, sub in groups.items():
        print(f"  {k:34s} frames {len(sub):5d}  gain {sub.gain.sum():+.4f}  v3 better {int((sub.gain > 1e-12).sum()):4d}"
              f"  worse {int((sub.gain < -1e-12).sum()):4d}")


if __name__ == "__main__":
    main()
