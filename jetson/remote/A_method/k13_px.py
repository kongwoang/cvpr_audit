#!/usr/bin/env python
"""K13(f) — tensor detector px on AV2 val WITH the ego mask vs without (post-hoc analysis, ego mask). From the instr_eval
base pickles (detector canvases incl. guard re-detections, 32-px rounding; px_* = tensor px / frame px): ours without
refiner (px_full), the area-based figure (cheap canvas + crop area x scale^2, the K10 convention) and the heuristic crop
arms, which instr_eval budgets with ours' per-frame crop area (strict) — under the mask, ours' masked crop area.
usage: python k13_px.py  ->  results/k13/px_masked.csv
"""
from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
SCALE = 960 / 2048                      # v4-lite crops at 960 / long side
ARMS = {"ours (no refiner)": "full", "uncertainty crops": "uncertainty", "temporal_crit crops": "temporal_crit",
        "random crops": "random", "u960": "u960", "native": "native"}


def load(root):
    rows = []
    for f in sorted(glob.glob(str(root / "results" / "instr" / "av2_test_base__*.pkl"))):
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows)


def main():
    out = []
    for tag, root in (("unmasked", B), ("ego mask", B / "k10_masked")):
        d = load(root)
        L = d.J_cheap.to_numpy()
        uq = d.unit.unique(); idx = [np.flatnonzero(d.unit.to_numpy() == u) for u in uq]
        rng = np.random.default_rng(0)
        boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
        Gf = L - d.J_full.to_numpy()
        for nm, k in ARMS.items():
            Gk = L - d[f"J_{k}"].to_numpy()
            r = {"setting": tag, "units": len(uq), "frames": len(d), "method": nm,
                 "tensor_px_mean": d[f"px_{k}"].mean(), "tensor_px_p95": d[f"px_{k}"].quantile(.95),
                 "G_no_refiner": Gk.sum() / L.sum()}
            if k == "full":
                r["area_px_mean"] = (d.px_cheap + d.crop_area_full * SCALE ** 2).mean()
                r["crops_per_frame"] = d.crops_full.mean()
            elif f"crops_{k}" in d:
                r["crops_per_frame"] = d[f"crops_{k}"].mean()
            if k != "full":
                dr = [(Gf[b].sum() - Gk[b].sum()) / L[b].sum() for b in boots]
                r["ci_ours_minus"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
            out.append(r)
    res = pd.DataFrame(out)
    (B / "results" / "k13").mkdir(parents=True, exist_ok=True)
    res.to_csv(B / "results" / "k13" / "px_masked.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
