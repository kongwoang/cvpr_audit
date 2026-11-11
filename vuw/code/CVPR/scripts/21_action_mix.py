#!/usr/bin/env python
"""D33 (exploratory): how complementary are the compute actions? An oracle mix under one pixel budget.

Per frame, four actions with pixel costs (full-resolution pixels / image): none (0), pivot second look (crop area,
D9), DSF region crop (area, D3), frame escalation (1).  Their realised values come from the D9 and D3 runs (same
frames, same consumer, shared history).  For budget b (mean pixels per frame), the oracle picks at most one action per
frame to maximise total value — a multiple-choice knapsack solved by Lagrangian relaxation over lambda.  Reported next
to the best *single* action type under the same budget: the gap is what a unified scheduler could add.
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

from vtf import runmeta

DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"


def mck(values: np.ndarray, costs: np.ndarray, budget: float) -> float:
    """Max sum of chosen values, one action per row (column 0 = none, value 0, cost 0), mean cost <= budget."""
    n = len(values)
    best = 0.0
    for lam in np.r_[0.0, np.logspace(-4, 3, 400)]:
        score = values - lam * costs
        pick = score.argmax(1)
        cost = costs[np.arange(n), pick].mean()
        if cost <= budget + 1e-12:
            best = max(best, float(values[np.arange(n), pick].sum()))
    return best


def main():
    ds, geo = sys.argv[1], sys.argv[2]
    sl = pd.read_pickle(sorted(glob.glob(str(runmeta.RAW / f"*_secondlook_{ds}_{geo}" / "secondlook__*.pkl")))[-1])
    rg = pd.read_pickle(sorted(glob.glob(str(runmeta.RAW / f"*_region_{ds}_{geo}" / "region__*.pkl")))[-1])
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str})
    dv = dv[dv.track == ds].copy()
    dv["frame"] = dv.frame.astype(int)
    out = []
    for name in (["brake", "traj"] if ds == "KITTI" else ["brake"]):
        ref = dv[(dv.geometry == geo) & (dv.system == name)][["seq", "frame", "split", "V", "J_cheap"]]
        d = ref.merge(sl, on=["seq", "frame"]).merge(rg[["seq", "frame", f"{name}_J_r", f"{name}_area"]],
                                                   on=["seq", "frame"])
        for sp in ("all", "test"):
            e = d if sp == "all" else d[d.split == "test"]
            Lc = float(e.J_cheap.sum())
            Jc = e[f"{name}_J_c"].to_numpy()
            vals = np.stack([np.zeros(len(e)), Jc - e[f"{name}_J_sl1"], Jc - e[f"{name}_J_r"], e.V.to_numpy()], 1)
            costs = np.stack([np.zeros(len(e)), e[f"{name}_area_sl1"], e[f"{name}_area"], np.ones(len(e))], 1)
            for b in (0.05, 0.10, 0.20, 0.30):
                row = {"cell": f"{ds} {geo} {name}", "split": sp, "budget_pixels": b,
                       "mix_oracle": mck(vals, costs, b) / Lc}
                for a, nm in ((1, "second_look_only"), (2, "region_only"), (3, "frame_only")):
                    v2 = vals[:, [0, a]]
                    c2 = costs[:, [0, a]]
                    row[nm] = mck(v2, c2, b) / Lc
                out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(runmeta.FINAL / f"d33_action_mix__{ds}__{geo}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
