#!/usr/bin/env python
"""field_value_analysis.py for Argoverse 2 val: v3 (confirmatory 2) vs each crop baseline of av2_baselines.py, split by
whether a decision-relevant GT object is NEW this frame (not visible >= 10 px in any earlier frame of the log)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from vtf import av2 as AV

B = Path(os.environ["B"])
ROOT = B / "datasets" / "av2" / "sensor" / "val"


def main():
    d = pd.read_pickle(B / "results" / "av2" / "baselines__val.pkl")
    ref = pd.read_pickle(B / "results" / "av2" / "eval__val_v3_confirm2.pkl")
    m = d.merge(ref[["log", "frame", "J_spec", "J_spec_ref"]], on=["log", "frame"])
    pp = P.PlannerParams()
    rows = []
    for log, gl in m.groupby("log"):
        L = AV.AV2Log(ROOT, log)
        geom, speeds = L.geometry(), L.speeds()
        seen = set()
        for fr in sorted(gl.frame):
            g = geom[geom["frame"] == fr] if len(geom) else geom
            g = g[(g["y2"] - g["y1"]) >= 10] if len(g) else g
            v = float(speeds[fr])
            rel = [j for j in range(len(g)) if P.discrete_action(P.required_decel(
                g["long_near"][j:j + 1], g["lat_min"][j:j + 1], g["lat_max"][j:j + 1], g["ttc"][j:j + 1], v, pp)[0], pp) > 0]
            rows.append({"log": log, "frame": fr, "n_rel": len(rel), "new_rel": any(g["track_id"][j] not in seen for j in rel)})
            seen |= set(np.asarray(g["track_id"]).tolist()) if len(g) else set()
    m = m.merge(pd.DataFrame(rows), on=["log", "frame"])
    Lc = m.J_c.sum()
    groups = {"relevant object NEW": m.new_rel, "relevant all seen before": (m.n_rel > 0) & ~m.new_rel, "no relevant object": m.n_rel == 0}
    for arm in ("temporal_crit", "uncertainty", "random"):
        gain = (m[f"J_{arm}"] - m.J_spec) / Lc
        print(f"v3 vs {arm}: total {gain.sum():+.4f}")
        for k, msk in groups.items():
            print(f"  {k:28s} frames {int(msk.sum()):5d}  gain {gain[msk].sum():+.4f}  better {int((gain[msk] > 1e-12).sum()):4d}"
                  f"  worse {int((gain[msk] < -1e-12).sum()):4d}")


if __name__ == "__main__":
    main()
