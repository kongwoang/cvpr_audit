#!/usr/bin/env python
"""Combine pkl_cells chunks: decision value of every pipeline on DEEP's plan_ade / plan_fde cells (PKL planner vs
the real trajectory), mono and oracle submissions, old 85 scenes (DEEP's units; also DEEP's test split) and new test
scenes; paired scene bootstrap vs uniform 960 (2,000)."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
D = B / "results" / "pkl_cells"
split = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
split_of = dict(zip(split.seq, split["set"] + "_" + split.split))
rows = []
for dump, name in (("scenes__old_train+val+test_dump", "old85"), ("scenes__new_test_dump", "new_test")):
    for v in ("oracle", "mono"):
        fs = sorted(D.glob(f"{dump}_{v}_n*c*.csv"))
        if not fs:
            continue
        d = pd.concat([pd.read_csv(f) for f in fs])
        subsets = {name: d} if name == "new_test" else {name: d, "old_test": d[d.scene.map(split_of) == "old_test"]}
        pipes = [c[4:] for c in d.columns if c.startswith("ade_") and c not in ("ade_gt", "ade_cheap")]
        for sub, q in subsets.items():
            uq = q.scene.unique(); idx = [np.flatnonzero(q.scene.to_numpy() == u) for u in uq]
            b = np.random.default_rng(0)
            boots = [np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
            for m in ("ade", "fde"):
                L = q[f"{m}_cheap"].to_numpy()
                r = {"set": sub, "variant": v, "cell": f"plan_{m}", "samples": len(q), "scenes": len(uq)}
                V = {p: L - q[f"{m}_{p}"].to_numpy() for p in pipes}
                for p in pipes:
                    r[p] = V[p].sum() / L.sum()
                for p in pipes:
                    if p.startswith("spec"):
                        dr = [(V[p][t].sum() - V["u960"][t].sum()) / L[t].sum() for t in boots]
                        r[f"{p}-u960"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
                rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(D / "summary.csv", index=False)
pd.set_option("display.width", 250)
print(res.round(4).to_string(index=False))
