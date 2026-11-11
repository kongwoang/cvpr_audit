#!/usr/bin/env python
"""D62 intervals: paired unit bootstrap (DEEP's scorer) of the ensemble-sensitivity allocators vs DEEP's gate and random."""
import glob

import numpy as np
import pandas as pd

from rap.scoring import evaluate
from vtf import runmeta

DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"
RAW_RAP = runmeta.RAP_REPO / "results" / "raw"
dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
dv = dv[dv.track.isin(["KITTI", "nuScenes"]) & (dv.system == "brake")].copy()
dv["frame"] = dv.frame.astype(int)
gates = sorted(RAW_RAP.glob("*_budget_gate_scores_ego"))[-1]
out = []
for ds in ("KITTI", "nuScenes"):
    e = pd.read_pickle(sorted(glob.glob(str(runmeta.RAW / f"*_ensemble_sens_{ds}" / "ensemble.pkl")))[-1])
    for geo in ("mono", "oracle"):
        ref = dv[(dv.track == ds) & (dv.geometry == geo) & (dv.split == "test")]
        gz = np.load(gates / f"gates__{ds}__{geo}__brake__J.npz", allow_pickle=True)
        gk = pd.DataFrame({"seq": gz["key_seq"].astype(str), "frame": gz["key_frame"].astype(int), "gate_gbm": gz["gate_gbm"]})
        q = ref.merge(e, on=["seq", "frame"]).merge(gk, on=["seq", "frame"], how="left")
        tie = 1e-6 * q.v.to_numpy(float)
        scores = {"random": None, "ego speed": q.v.to_numpy(float), "DEEP gate GBM": q.gate_gbm.to_numpy(float)}
        for n in ("ens_all", "ens_range", "ens_miss", "ens_exist"):
            scores[n] = q[n].to_numpy(float) + tie
        ks, prize, point, draws, dropped = evaluate(q.V.to_numpy(float), q.unit.to_numpy(), scores, 1000,
                                                    np.random.default_rng(0))
        for n in scores:
            d_rand = draws[n][:, 1] - draws["random"][:, 1]
            d_gate = draws[n][:, 1] - draws["DEEP gate GBM"][:, 1]
            f = lambda x: np.nanpercentile(x, [2.5, 97.5])                                   # noqa: E731
            out.append({"cell": f"{ds} {geo} brake", "signal": n, "nDG20": float(point[n]["eta"][1]),
                        "minus_random_lo": f(d_rand)[0], "minus_random_hi": f(d_rand)[1],
                        "minus_gate_lo": f(d_gate)[0], "minus_gate_hi": f(d_gate)[1]})
res = pd.DataFrame(out)
res.to_csv(runmeta.FINAL / "d62_ensemble_bootstrap.csv", index=False)
pd.set_option("display.width", 250)
print(res.round(3).to_string())
