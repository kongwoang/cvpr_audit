#!/usr/bin/env python
"""K30 V6 (ii) table from results/navsim_exp/k30_probe_time/shard*.pkl (k30_probe_time.py): per variant (7x7, 5x5, 3x3,
plan corridor) planner calls, measured wall clock of the whole probing per frame (mean / median / p95, ms), time inside
PDM-Closed, share of probes that moved the plan; ratio of each variant's mean wall clock to 7x7 on the same tokens with a
paired log bootstrap (2,000, seed 0). K20's derived numbers (calls x mean frozen call) are joined for comparison.
results/navsim_exp/k30_probe_time/v6ii_probe_time.csv
"""
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ.get("B_REAL", os.environ["B"]))
D = B / "results" / "navsim_exp" / "k30_probe_time"
K20_DERIVED = {"7x7": 14908.38, "5x5": 7731.28, "3x3": 4155.02, "corridor": 4264.80}   # handoff/K20/pdm_variants.csv (frozen per-call)
K20_FAST_7x7 = 6192.13                                                                 # K20 fast implementation, measured


def main():
    rows = []
    for f in sorted(glob.glob(str(D / "shard*.pkl"))):
        rows += pickle.load(open(f, "rb"))
    d = pd.DataFrame(rows)
    w = d.pivot_table(index=["log", "token"], columns="variant", values="wall_s").reset_index()
    uq = np.array(sorted(w.log.unique()))
    inv = np.searchsorted(uq, w.log.to_numpy())
    rng = np.random.default_rng(0)
    bs = [rng.integers(0, len(uq), len(uq)) for _ in range(2000)]
    s7 = np.bincount(inv, w["7x7"].to_numpy(), len(uq))
    out = []
    for var in ("7x7", "5x5", "3x3", "corridor"):
        x = d[d.variant == var]
        r = {"variant": var, "tokens": len(x), "logs": x.log.nunique(), "calls_mean": x.calls.mean(), "calls_p95": x.calls.quantile(.95),
             "probe_calls_mean": x.n_probe.mean(), "support_calls_mean": x.n_support.mean(),
             "wall_ms_mean": 1e3 * x.wall_s.mean(), "wall_ms_median": 1e3 * x.wall_s.median(), "wall_ms_p95": 1e3 * x.wall_s.quantile(.95),
             "planner_ms_mean": 1e3 * x.planner_s.mean(), "ms_per_call_mean": 1e3 * (x.wall_s / x.calls).mean(),
             "moved_share_of_probes": x.n_moved.sum() / max(x.n_probe.sum() + x.n_support.sum(), 1),
             "k20_derived_ms": K20_DERIVED[var], "k20_fast_7x7_ms": K20_FAST_7x7}
        sv = np.bincount(inv, w[var].to_numpy(), len(uq))
        r["ratio_to_7x7"] = sv.sum() / s7.sum()
        v = [sv[b].sum() / s7[b].sum() for b in bs]
        r["ratio_ci_lo"], r["ratio_ci_hi"] = np.percentile(v, 2.5), np.percentile(v, 97.5)
        out.append(r)
    o = pd.DataFrame(out)
    o.to_csv(D / "v6ii_probe_time.csv", index=False)
    pd.set_option("display.width", 250)
    print(o.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
