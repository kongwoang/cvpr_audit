#!/usr/bin/env python
"""K13(d) — TTC-warning consumer on AV2 val, ours (vc 0, m 0) minus px-matched uncertainty crops at each multiple of
ours' detector px, unmasked vs ego mask (k9_tune curve pickles; paired log bootstrap, 2,000). Post-hoc analysis.
usage: python k13_k9pairs.py  ->  results/k13/k9_pairs.csv
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
out = []
for tag, root in (("unmasked", B), ("ego mask", B / "k10_masked")):
    rows = []
    for p in sorted((root / "results" / "k9_tune").glob("av2_test_curve__*.pkl")):
        rows += pickle.load(open(p, "rb"))
    d = pd.DataFrame(rows)
    Lc = d.J2_cheap.to_numpy()
    uq = d.unit.unique(); idx = [np.flatnonzero(d.unit.to_numpy() == u) for u in uq]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    Vo = Lc - d.J2_ours_vc0_m0.to_numpy()
    for n in sorted(c[3:] for c in d.columns if c.startswith("J2_unc_px_x")):
        V = Lc - d[f"J2_{n}"].to_numpy()
        dr = [(Vo[b].sum() - V[b].sum()) / Lc[b].sum() for b in boots]
        out.append({"setting": tag, "units": len(uq), "frames": len(d), "ours_G": Vo.sum() / Lc.sum(),
                    "ours_px": d.px_ours_vc0_m0.mean(), "control": n, "control_G": V.sum() / Lc.sum(),
                    "control_px": d[f"px_{n}"].mean(), "ours_minus_control": (Vo.sum() - V.sum()) / Lc.sum(),
                    "ci_lo": np.percentile(dr, 2.5), "ci_hi": np.percentile(dr, 97.5)})
res = pd.DataFrame(out)
(B / "results" / "k13").mkdir(parents=True, exist_ok=True)
res.to_csv(B / "results" / "k13" / "k9_pairs.csv", index=False)
print(res.round(4).to_string(index=False))
