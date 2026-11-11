#!/usr/bin/env python
"""Reproduced vs published (shipped) bgt-ada VOC benchmark table: Spearman rho_s, peak mAP, AUC_rho."""
import os
from pathlib import Path

import pandas as pd

R = Path(os.environ["B"]) / "repro" / "bgt-ada" / "results"
pub = pd.read_csv(R / "voc" / "metrics" / "main_benchmark_table.csv")
new = pd.read_csv(R / "full_eval" / "voc" / "metrics" / "main_benchmark_table.csv")
cols = ["spearman_rho", "peak_map", "auc_0_5"]
m = pub[["estimator"] + cols].merge(new[["estimator"] + cols], on="estimator", suffixes=("_paper", "_repro"), how="left")
for c in cols:
    m[f"d_{c}"] = m[f"{c}_repro"] - m[f"{c}_paper"]
pd.set_option("display.width", 250)
print(m.to_string(index=False))
m.to_csv(Path(os.environ["B"]) / "results" / "repro_bgt_voc.csv", index=False)
