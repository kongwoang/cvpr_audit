#!/usr/bin/env python
"""K18(A) tables + figures (post-hoc analysis; AV2 with the K10 ego mask): decision value against TENSOR px.
  heuristics      k18_heur: uncertainty / temporal_crit / random / traj_support / corridor with per-frame crop canvases
                  matched to {0.5, 1, 2, 4} x ours' crop tensor px (with and without refiner)
  routers         k18_routers (routers_{ds}.csv)
  ours            the frozen configuration (from the same k18_heur pass) + existing operating points (nuScenes: the K8
                  operating table; AV2: k17_variants v_close / probe arms when available)
  whole frame     u640 ... native (K14 per-frame tables; u1280 / u1536 from the K1 / K10 tables)
Paired unit bootstrap (2,000) of ours final minus each pipeline (every heuristic budget; routers at 1x). The budget is a
cap on the canvases of the heuristic's own crops; its guard re-detections come on top, so the REALISED tensor px
(px_ratio_to_ours) is reported and decides which row is the equal-px comparison.
usage: k18_analyze.py  ->  results/k18/pareto_{nusc,av2}.csv (+ heur_{ds}.csv)
"""
from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
S = B / "k10_masked"
OUT = B / "results" / "k18"
HEUR = ("uncertainty", "temporal_crit", "random", "traj_support", "corridor")


def load(pat):
    rows = []
    for f in sorted(glob.glob(str(pat))):
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows)


def main():
    for ds, root in (("nusc", B), ("av2", S), ("ahd", B)):
        d = load(root / "results" / "k18_heur" / f"{ds}__*.pkl")
        if not len(d):
            continue
        if "gt" in d:
            d = d[d["gt"].astype(bool)].reset_index(drop=True)
        L = d.J_cheap.to_numpy(float)
        units = d.unit.to_numpy(); uq = pd.unique(units); idx = [np.flatnonzero(units == u) for u in uq]
        rng = np.random.default_rng(0)
        bs = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
        Vo = L - d.J_full_ref.to_numpy(float)
        rows = [{"dataset": ds, "family": "ours", "method": "ours final (frozen)", "budget_x": 1.0, "G_no_refiner": (L - d.J_full).sum() / L.sum(),
                 "G_refiner": Vo.sum() / L.sum(), "tensor_px": d.px_full.mean(), "tensor_px_p95": d.px_full.quantile(.95),
                 "units": len(uq), "frames": len(d)}]
        for bx in (0.5, 1.0, 2.0, 4.0):
            for h in HEUR:
                nm = f"{h}_x{bx:g}"
                if f"J_{nm}" not in d:
                    continue
                Vn, Vr = L - d[f"J_{nm}"].to_numpy(float), L - d[f"J_{nm}_ref"].to_numpy(float)
                r = {"dataset": ds, "family": "crop heuristic (tensor-px matched per frame)", "method": h, "budget_x": bx,
                     "G_no_refiner": Vn.sum() / L.sum(), "G_refiner": Vr.sum() / L.sum(), "tensor_px": d[f"px_{nm}"].mean(),
                     "tensor_px_p95": d[f"px_{nm}"].quantile(.95), "px_ratio_to_ours": d[f"px_{nm}"].mean() / d.px_full.mean()}
                if True:                                    # every budget: the realised px ratio decides the comparison
                    dr = [(Vo[b].sum() - Vr[b].sum()) / L[b].sum() for b in bs]
                    dn = [(Vo[b].sum() - Vn[b].sum()) / L[b].sum() for b in bs]
                    r.update({"ci_lo_vs_refiner": np.percentile(dr, 2.5), "ci_hi_vs_refiner": np.percentile(dr, 97.5),
                              "ci_lo_vs_no_refiner": np.percentile(dn, 2.5), "ci_hi_vs_no_refiner": np.percentile(dn, 97.5)})
                rows.append(r)
        rt = pd.read_csv(OUT / f"routers_{ds}.csv") if (OUT / f"routers_{ds}.csv").exists() else pd.DataFrame({"method": pd.Series([], dtype=str)})
        for _, r in rt[rt.method.str.startswith(("oracle", "best"))].iterrows():
            rows.append({"dataset": ds, "family": "frame router", "method": r.method, "budget_x": r.budget_x, "G_no_refiner": r.G,
                         "tensor_px": r.tensor_px, "ci_lo_vs_no_refiner": r.get("ci_lo_ours_minus"), "ci_hi_vs_no_refiner": r.get("ci_hi_ours_minus"),
                         "note": "px within 2% of the budget" if r.get("px_within_2pct") else "budget not reachable within 2%"})
        k14 = load(root / "results" / "k14" / f"{ds}__*.pkl")
        k14 = k14[k14["gt"].astype(bool)]
        Lk = k14.J_cheap.to_numpy(float)
        for c in ("u960", "native"):
            rows.append({"dataset": ds, "family": "whole frame", "method": c, "G_no_refiner": (Lk - k14[f"J_{c}"]).sum() / Lk.sum(),
                         "G_refiner": (Lk - k14[f"J_{c}_ref"]).sum() / Lk.sum(), "tensor_px": k14[f"px_{c}"].mean()})
        rows.append({"dataset": ds, "family": "whole frame", "method": "cheap draft (u320)", "G_no_refiner": 0.0, "tensor_px": k14.px_cheap.mean()})
        res = pd.DataFrame(rows)
        res.to_csv(OUT / f"heur_{ds}.csv", index=False)
        pd.set_option("display.width", 250)
        print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
