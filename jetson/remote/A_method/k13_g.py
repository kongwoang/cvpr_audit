#!/usr/bin/env python
"""K13(g) — AV2 dev check under the ego mask: v4-lite + ref v2 (reported), v3 + ref v1, v4 + ref v2 on AV2 TRAIN (700
logs, masked, av2_eval_v4 --split train --tag _confirm3 in the shadow tree) and, for the train winner, its AV2 VAL
(masked) result from the single masked av2_eval_v4 val run (eval__val_v4_final.pkl, all variants in one pass; K10).
Paired log bootstrap, 2,000. Post-hoc analysis, ego mask.  usage: python k13_g.py -> results/k13/g_dev_check.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
ARMS = {"v4-lite + ref v2 (reported)": "J_spec_ref_v4lite", "v3 + ref v1": "J_spec_ref_v3", "v4 + ref v2": "J_spec_ref_v4",
        "v4-lite (no refiner)": "J_spec_v4lite", "v4 (no refiner)": "J_spec_v4", "u960": "J_u960"}
PX = {"J_spec_ref_v4lite": "netpx_v4lite", "J_spec_ref_v3": "netpx_v3", "J_spec_ref_v4": "netpx_v4",
      "J_spec_v4lite": "netpx_v4lite", "J_spec_v4": "netpx_v4"}
CHEAP = 320 * 256 / (1550 * 2048)       # cheap canvas; area-based px as in K10 (cheap + crop area x scale^2)
out = []
for split, root, f in (("AV2 train (dev)", "ego mask", "eval__train_confirm3.pkl"), ("AV2 train (dev)", "unmasked", "eval__train_confirm3.pkl"),
                       ("AV2 val (test, post-hoc)", "ego mask", "eval__val_v4_final.pkl"), ("AV2 val (test, post-hoc)", "unmasked", "eval__val_v4_final.pkl")):
    d = pd.read_pickle((B / "k10_masked" if root == "ego mask" else B) / "results" / "av2" / f)
    L = d.J_c.to_numpy()
    uq = d.log.unique(); idx = [np.flatnonzero(d.log.to_numpy() == u) for u in uq]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    ref = L - d.J_spec_ref_v4lite.to_numpy()
    for nm, c in ARMS.items():
        V = L - d[c].to_numpy()
        r = {"split": split, "setting": root, "logs": len(uq), "frames": len(d), "method": nm, "G": V.sum() / L.sum(),
             "network_px_area": d[PX[c]].mean() + CHEAP if c in PX else (0.2323 if c == "J_u960" else np.nan)}
        if c != "J_spec_ref_v4lite":
            dr = [(V[b].sum() - ref[b].sum()) / L[b].sum() for b in boots]
            r["ci_minus_reported"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        out.append(r)
res = pd.DataFrame(out)
(B / "results" / "k13").mkdir(parents=True, exist_ok=True)
res.to_csv(B / "results" / "k13" / "g_dev_check.csv", index=False)
pd.set_option("display.width", 220)
print(res.round(4).to_string(index=False))
