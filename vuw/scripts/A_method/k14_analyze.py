#!/usr/bin/env python
"""K14 tables (post-hoc analysis). From results/k14/<ds>__<unit>.pkl (k14_controls.py; AV2 in the K10 shadow tree,
ego mask) plus the support-only / field-only rows of the instrumented runs (instr_eval: nuScenes new test, AV2 val
masked; k13_ahd_ablation: Argoverse-HD val), merged on (unit, frame).  G = sum(J_cheap - J_m) / sum(J_cheap); tensor
px = mean detector canvas px / frame px; paired unit bootstrap (scenes / logs / sequences), 2,000 resamples, seed 0, of
ours final (with refiner) minus the method WITH refiner (ci_lo, ci_hi) and minus the method WITHOUT refiner.
Argoverse-HD: frames with 3D labels only.
usage: k14_analyze.py  ->  results/k14_tables/refiner_controls_{nusc,av2,ahd}.csv
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
OUT = B / "results" / "k14_tables"
NAMES = {"full": "ours (planner-driven crops)", "uncertainty": "uncertainty crops", "temporal_crit": "temporal_crit crops",
         "random": "random crops", "traj_support": "traj_support (support + corridor tube)", "corridor": "ego-corridor tube",
         "u960": "whole frame u960", "native": "whole frame native", "support_only": "support only (no field)",
         "field_only": "field only (no support crops)"}


def load(pat):
    rows = []
    for f in sorted(glob.glob(str(pat))):
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = {"nusc": (B / "results" / "k14" / "nusc__*.pkl", B / "results" / "instr" / "nusc_test_base__*.pkl"),
           "av2": (S / "results" / "k14" / "av2__*.pkl", S / "results" / "instr" / "av2_test_base__*.pkl"),
           "ahd": (B / "results" / "k14" / "ahd__*.pkl", B / "results" / "k13_ahd_ablation" / "seq*.pkl")}
    for ds, (pk, pi) in src.items():
        d = load(pk)
        if not len(d):
            print(ds, "no K14 pickles yet"); continue
        d = d[d["gt"].astype(bool)].reset_index(drop=True)
        ins = load(pi)
        if "gt" in ins:
            ins = ins[ins["gt"].astype(bool)]
        cols = [c for c in ins.columns if any(c.startswith(p) for p in ("J_support_only", "J_field_only", "px_support_only",
                                                                         "px_field_only", "J_full_ref"))]
        d = d.merge(ins[["unit", "frame"] + cols].rename(columns={"J_full_ref": "J_full_ref_instr"}), on=["unit", "frame"], how="left")
        L = d.J_cheap.to_numpy(float)
        units = d.unit.to_numpy(); uq = pd.unique(units); idx = [np.flatnonzero(units == u) for u in uq]
        rng = np.random.default_rng(0)
        bs = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
        Vo = L - d.J_full_ref.to_numpy(float)
        parity = float(np.nanmax(np.abs(d.J_full_ref - d.J_full_ref_instr))) if d.J_full_ref_instr.notna().all() else np.nan
        rows = []
        for m in ("full", "uncertainty", "temporal_crit", "random", "traj_support", "corridor", "u960", "native",
                  "support_only", "field_only"):
            if f"J_{m}" not in d:
                continue
            Vn = L - d[f"J_{m}"].to_numpy(float)
            Vr = L - d[f"J_{m}_ref"].to_numpy(float)
            r = {"dataset": ds, "units": len(uq), "frames": len(d), "method": NAMES[m], "G_no_refiner": Vn.sum() / L.sum(),
                 "G_refiner": Vr.sum() / L.sum(), "tensor_px": d[f"px_{m}"].mean(), "tensor_px_p95": d[f"px_{m}"].quantile(.95)}
            if m != "full":
                dr = [(Vo[b].sum() - Vr[b].sum()) / L[b].sum() for b in bs]
                dn = [(Vo[b].sum() - Vn[b].sum()) / L[b].sum() for b in bs]
                r.update({"ours_final_minus_refiner": (Vo.sum() - Vr.sum()) / L.sum(), "ci_lo": np.percentile(dr, 2.5),
                          "ci_hi": np.percentile(dr, 97.5), "ours_final_minus_no_refiner": (Vo.sum() - Vn.sum()) / L.sum(),
                          "ci_lo_no_refiner": np.percentile(dn, 2.5), "ci_hi_no_refiner": np.percentile(dn, 97.5)})
            else:
                r["parity_J_full_ref_vs_instr_maxabs"] = parity
            rows.append(r)
        res = pd.DataFrame(rows)
        res.to_csv(OUT / f"refiner_controls_{ds}.csv", index=False)
        pd.set_option("display.width", 250)
        print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
