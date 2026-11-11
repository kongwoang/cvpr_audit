#!/usr/bin/env python
"""Where does v3 (spec) lose on NAVSIM? Per-token comparison of spec with the other branches (results/navsim_exp)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(os.environ["NAVSIM_EXP_ROOT"]) if "NAVSIM_EXP_ROOT" in os.environ else Path(os.environ["B"]) / "results/navsim_exp"
d = pd.read_csv(EXP / "pdms_states.csv")
print(len(d), "tokens")
for b in ("cheap", "full", "uncertainty", "fovea"):
    nw = (d.no_at_fault_collisions_spec < d[f"no_at_fault_collisions_{b}"]).sum()
    nb = (d.no_at_fault_collisions_spec > d[f"no_at_fault_collisions_{b}"]).sum()
    diff = d.score_spec - d[f"score_{b}"]
    print(f"vs {b:12s} NC worse {nw:4d} better {nb:4d} | PDMS worse {(diff < -1e-6).sum():4d} ({diff[diff < -1e-6].sum():7.1f}) "
          f"better {(diff > 1e-6).sum():4d} ({diff[diff > 1e-6].sum():7.1f})")
worse = d.score_spec < d.score_cheap - 1e-6
print("spec px: worse-vs-cheap tokens", round(d.info_pixels[worse].mean(), 3), "all", round(d.info_pixels.mean(), 3))
print(pd.crosstab(pd.cut(d.info_v, [-1, 0.5, 3, 8, 40]), worse))
print("rounds (worse):", d.info_rounds[worse].value_counts().to_dict(), " rounds (all):", d.info_rounds.value_counts().to_dict())
print("n_crops (worse) mean", d.info_n_crops[worse].mean(), "all", d.info_n_crops.mean())
