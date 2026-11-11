#!/usr/bin/env python
"""K20(3) — cheaper approximations of the frozen PDM-Closed probing field, derived from it (no new probing): the frozen
field's sensitive cells restricted to a coarser grid or to the corridor of the current plan. The support set is kept.
  g5        x in {8, 12, 23, 40, 55} m, y in {-10.5, -3.5, 0, 3.5, 10.5} m (5 x 5 of the 7 x 7 grid)
  g3        x in {8, 23, 55} m, y in {-3.5, 0, 3.5} m
  corridor  cells whose lateral offset from the draft plan (k20_pdm --mode base: the 8 poses of PDM-Closed on the draft
            scene, path interpolated in x, held constant beyond the last pose) is <= 1.75 m + half the probe width
Each cell of a token is identified by projecting its probe box with the token's camera (as navsim_field) and matching
the stored image box. Writes results/navsim_exp/field_k20_<variant>/<log>.pkl in navsim_field's format plus
results/k20/variant_cells.csv (per token: probed cells, sensitive cells, image-footprint IoU with the frozen field's
sensitive boxes).
usage: k20_variants.py
"""
from __future__ import annotations

import importlib.util
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("kp", B / "scripts/A_navsim/k20_pdm.py")
KP = importlib.util.module_from_spec(_s); _s.loader.exec_module(KP)
NF, NP = KP.NF, KP.NP
G5X, G5Y = (8.0, 12.0, 23.0, 40.0, 55.0), (-10.5, -3.5, 0.0, 3.5, 10.5)
G3X, G3Y = (8.0, 23.0, 55.0), (-3.5, 0.0, 3.5)


def union_area(boxes, W=1920, H=1080, step=4):
    if not boxes:
        return np.zeros((H // step, W // step), bool)
    m = np.zeros((H // step, W // step), bool)
    for b in boxes:
        m[int(b[1]) // step:int(np.ceil(b[3])) // step + 1, int(b[0]) // step:int(np.ceil(b[2])) // step + 1] = True
    return m


def main():
    gz = float((KP.DETS / "ground_z.txt").read_text())
    priors = pickle.load(open(KP.X / "dets" / "size_priors.pkl", "rb"))
    L, Wd, Ht = priors["vehicle"]
    base = {}
    for f in sorted(KP.OUT.glob("base__*.pkl")):
        for r in pickle.load(open(f, "rb")):
            if "poses" in r:
                base[r["token"]] = r["poses"]
    rows = []
    outs = {v: KP.X / f"field_k20_{v}" for v in ("g5", "g3", "corridor")}
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=True)
    for f in sorted(KP.FIELD.glob("*.pkl")):
        log = f.stem
        fz = pickle.load(open(f, "rb"))
        dets = pickle.load(open(KP.DETS / f"{log}.pkl", "rb"))
        res = {v: {} for v in outs}
        for tok, rec in fz.items():
            if "err" in rec or tok not in dets:
                continue
            cam = NP.Cam(*dets[tok]["cam"])
            cells = []
            for x in NF.X_GRID:
                for y in NF.Y_GRID:
                    ib = cam.project(np.array([x, y, gz + Ht / 2, L, Wd, Ht, 0.0]))
                    if ib is None:
                        continue
                    sens = any(np.allclose(ib, b) for b in rec.get("field_boxes", []))
                    cells.append((x, y, ib, sens))
            p = base.get(tok)
            if p is not None:
                px, py = np.r_[0.0, p[:, 0]], np.r_[0.0, p[:, 1]]
                o = np.argsort(px)

                def lat_off(x, y):
                    return abs(y - np.interp(x, px[o], py[o]))
            sel = {"g5": lambda x, y: x in G5X and y in G5Y, "g3": lambda x, y: x in G3X and y in G3Y,
                   "corridor": (lambda x, y: lat_off(x, y) <= 1.75 + Wd / 2) if p is not None else None}
            fm = union_area([c[2] for c in cells if c[3]])
            row = {"log": log, "token": tok, "probed_7x7": len(cells), "sens_7x7": sum(c[3] for c in cells),
                   "support_calls_frozen": rec.get("n_calls", 0) - 1 - len(cells)}
            for v, fn in sel.items():
                if fn is None:
                    continue
                kept = [c for c in cells if fn(c[0], c[1])]
                boxes = [c[2] for c in kept if c[3]]
                res[v][tok] = {"field_boxes": boxes, "support_boxes": rec.get("support_boxes", []),
                               "n_calls": 1 + len(kept) + row["support_calls_frozen"]}
                vm = union_area(boxes)
                inter, uni = (vm & fm).sum(), (vm | fm).sum()
                row.update({f"probed_{v}": len(kept), f"sens_{v}": len(boxes), f"iou_{v}": inter / uni if uni else 1.0})
            rows.append(row)
        for v, d in outs.items():
            with open(d / f"{log}.pkl", "wb") as fh:
                pickle.dump(res[v], fh)
    pd.DataFrame(rows).to_csv(KP.OUT / "variant_cells.csv", index=False)
    print(pd.DataFrame(rows).describe().round(3).to_string())
    print("K20 VARIANTS DONE")


if __name__ == "__main__":
    main()
