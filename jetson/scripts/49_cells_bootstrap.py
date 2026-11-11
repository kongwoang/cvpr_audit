#!/usr/bin/env python
"""Paired sequence bootstrap for DEEP's four KITTI cells from a `40_e2e_v2.py --cells` run (latest by default)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
runs = sorted((ROOT / "results" / "raw").glob("*_e2e_v2_KITTI*"))
run = Path(sys.argv[1]) if len(sys.argv) > 1 else [r for r in runs if (r / "deep_cells.csv").exists()][-1]
d = pd.read_pickle(run / "e2e.pkl")
cells = {"brake mono": ("J_c_trt", {"full640": "J_full640_trt", "full960": "J_full960_trt", "spec": "J_spec_trt",
                                    "spec_ref": "J_spec_ref_trt"}),
         "brake oracle": ("Jbo_c", {k: f"Jbo_{k}" for k in ("full640", "full960", "spec", "spec_ref")}),
         "traj mono": ("Jtm_c", {k: f"Jtm_{k}" for k in ("full640", "full960", "spec", "spec_ref")}),
         "traj oracle": ("Jto_c", {k: f"Jto_{k}" for k in ("full640", "full960", "spec", "spec_ref")})}
uq = d.seq.unique(); idx = [np.flatnonzero(d.seq.to_numpy() == u) for u in uq]
b = np.random.default_rng(0)
boots = [np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
rows = []
for cell, (jc, arms) in cells.items():
    L = d[jc].to_numpy()
    V = {k: (d[jc] - d[c]).to_numpy() for k, c in arms.items()}
    r = {"cell": cell, **{k: v.sum() / L.sum() for k, v in V.items()}}
    for k in ("spec", "spec_ref"):
        dr = [(V[k][t].sum() - V["full960"][t].sum()) / L[t].sum() for t in boots]
        r[f"{k}-u960"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
    rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(run / "deep_cells_bootstrap.csv", index=False)
print(run.name)
print(res.round(3).to_string(index=False))
