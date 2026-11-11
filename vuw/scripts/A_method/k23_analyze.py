#!/usr/bin/env python
"""K23 tables (development data only). From results/k23/<ds>_<tag>/*.pkl (k23_dev.py).
For every arm x refiner (none, frozen = v1 nuScenes / v2 AV2, v3a, v3b, v3c, v3c_veto<thr>) and the baselines (u960,
uncertainty crops at the base arm's crop area): G = sum(J_cheap - J) / sum(J_cheap), tensor px mean / p95, rounds,
and the paired unit bootstrap (2,000, seed 0) of (candidate - FROZEN), FROZEN = base arm with the frozen refiner.
Operating points: S = px <= 1.1 x frozen's dev px, L = <= 2 x.
usage: k23_analyze.py --tag alone|combos  ->  results/k23/tables/{components,sweep|combos}_{nusc,av2}.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
OUT = REAL / "results" / "k23" / "tables"


def load(ds, tag):
    rows = []
    for f in sorted(glob.glob(str(REAL / "results" / "k23" / f"{ds}_{tag}" / "*.pkl"))):
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows)


def table(d, ds):
    L = d.J_cheap.to_numpy(float)
    units = d.unit.to_numpy(); uq = pd.unique(units); idx = [np.flatnonzero(units == u) for u in uq]
    rng = np.random.default_rng(0)
    bs = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    V0 = L - d["J_base|frozen"].to_numpy(float)
    px0 = d.px_base.mean()
    rows = []
    for c in [c for c in d.columns if c.startswith("J_") and "|" in c]:
        arm, ref = c[2:].split("|")
        V = L - d[c].to_numpy(float)
        dd = [(V[b].sum() - V0[b].sum()) / L[b].sum() for b in bs]
        px = d[f"px_{arm}"]
        lo, hi = np.percentile(dd, 2.5), np.percentile(dd, 97.5)
        rows.append({"dataset": ds, "units": len(uq), "frames": len(d), "arm": arm, "refiner": ref, "G": V.sum() / L.sum(),
                     "minus_frozen": (V.sum() - V0.sum()) / L.sum(), "ci_lo": lo, "ci_hi": hi, "tensor_px": px.mean(),
                     "tensor_px_p95": px.quantile(.95), "px_ratio_to_frozen": px.mean() / px0,
                     "point": "S" if px.mean() <= 1.1 * px0 else ("L" if px.mean() <= 2 * px0 else "-"),
                     "rounds_mean": d[f"rounds_{arm}"].mean() if f"rounds_{arm}" in d else np.nan})
    return pd.DataFrame(rows).sort_values(["refiner", "arm"])


def halves(d, arm, ref):
    """G of one candidate and of FROZEN on two disjoint halves of the dev units (sorted units, alternate)"""
    uq = sorted(d.unit.unique())
    out = []
    for h in (0, 1):
        m = d.unit.isin(uq[h::2])
        L = d.J_cheap[m].sum()
        out.append({"half": h, "units": len(uq[h::2]), "G_candidate": (L - d.loc[m, f"J_{arm}|{ref}"].sum()) / L,
                    "G_frozen": (L - d.loc[m, "J_base|frozen"].sum()) / L})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="alone")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in ("nusc", "av2"):
        d = load(ds, args.tag)
        if not len(d):
            continue
        t = table(d, ds)
        t.to_csv(OUT / f"{'components' if args.tag == 'alone' else args.tag}_{ds}.csv", index=False)
        pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
        print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
