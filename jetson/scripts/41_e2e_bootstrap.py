#!/usr/bin/env python
"""Paired unit bootstrap for the deployed v2 pipeline (scripts/40_e2e_v2.py): every pipeline minus uniform 640/960.

Units: KITTI sequences (DEEP's unit for KITTI).  Also reports the harmed share of affected frames per pipeline.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / "results" / "raw").glob("*_e2e_v2_KITTI"))[-1]
d = pd.read_pickle(run / "e2e.pkl")
L = d.J_c_trt.to_numpy()
G = {"uniform 640": d.J_c_trt - d.J_full640_trt, "uniform 960": d.J_c_trt - d.J_full960_trt,
     "speculative v2": d.J_c_trt - d.J_spec_trt, "speculative v2 + refiner": d.J_c_trt - d.J_spec_ref_trt}
G = {k: v.to_numpy() for k, v in G.items()}
units = d.seq.to_numpy()
uq = np.unique(units)
idx = [np.flatnonzero(units == u) for u in uq]
rng = np.random.default_rng(0)
draws = []
for _ in range(2000):
    t = np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))])
    if L[t].sum() > 0:
        draws.append({k: g[t].sum() / L[t].sum() for k, g in G.items()})
D = pd.DataFrame(draws)
rows = []
for k, g in G.items():
    aff = g != 0
    rows.append({"pipeline": k, "G": g.sum() / L.sum(), "affected": int(aff.sum()), "harmed_share": float((g < 0).sum() / max(aff.sum(), 1)),
                 "minus_u640": f"[{np.percentile(D[k] - D['uniform 640'], 2.5):+.3f}, {np.percentile(D[k] - D['uniform 640'], 97.5):+.3f}]",
                 "minus_u960": f"[{np.percentile(D[k] - D['uniform 960'], 2.5):+.3f}, {np.percentile(D[k] - D['uniform 960'], 97.5):+.3f}]"})
res = pd.DataFrame(rows)
res.to_csv(run / "bootstrap.csv", index=False)
print(f"run {run.name}; frames {len(d)}, units {len(uq)}, crops/frame {d.n_crops.mean():.2f}, rounds {d.rounds.mean():.2f}, "
      f"refined/frame {d.n_refined.mean():.2f}")
print(res.round(4).to_string(index=False))
