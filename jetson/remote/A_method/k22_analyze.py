#!/usr/bin/env python
"""K22 tables (post-hoc analysis).
  same_footing_noref.csv   ours WITHOUT refiner minus each heuristic WITHOUT refiner at budget b = 0.5 (realised px
                           ~ ours'), nuScenes new test, AV2 val (ego mask), Argoverse-HD val (3D-labelled frames);
                           unit bootstrap 2,000
  groupd_{av2,ahd}.csv     per group-d frame (k22_diag) + groupd_summary.csv (shares, median errors, F reproduces)
usage: k22_analyze.py
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
OUT = B / "results" / "k22"
HEUR = ("uncertainty", "temporal_crit", "corridor", "traj_support", "random")


def load(pat):
    rows = []
    for f in sorted(glob.glob(str(pat))):
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
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
        Vo = L - d.J_full.to_numpy(float)
        for h in HEUR:
            Vh = L - d[f"J_{h}_x0.5"].to_numpy(float)
            dd = [(Vo[b].sum() - Vh[b].sum()) / L[b].sum() for b in bs]
            rows.append({"dataset": ds, "heuristic": h, "units": len(uq), "frames": len(d), "G_ours_noref": Vo.sum() / L.sum(),
                         "G_heur_noref": Vh.sum() / L.sum(), "px_ratio_to_ours": d[f"px_{h}_x0.5"].mean() / d.px_full.mean(),
                         "ci_lo": np.percentile(dd, 2.5), "ci_hi": np.percentile(dd, 97.5)})
    pd.DataFrame(rows).to_csv(OUT / "same_footing_noref.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    summ = []
    for ds, root in (("av2", S), ("ahd", B)):
        d = load(root / "results" / "k22_diag" / f"{ds}__*.pkl")
        if not len(d):
            continue
        d.to_csv(OUT / f"groupd_{ds}.csv", index=False)
        dec = d[d.est_z.notna()] if "est_z" in d else d.iloc[:0]
        m = (dec[dec.get("gt_z", pd.Series(dtype=float)).notna()] if len(dec) else dec).copy()
        r = {"dataset": ds, "group_d_frames": len(d), "frames_ours_above_so": len(dec),
             "decisive_from_crop_and_new": float((dec.dec_from_crop & dec.dec_new_vs_so).mean()) if len(dec) else np.nan,
             "matched_annotated_object": len(m)}
        if len(m):
            for c in ("fix_range", "fix_lateral", "fix_ttc", "class_mismatch", "est_in_corridor", "gt_in_corridor"):
                m[c] = m[c].fillna(False).astype(bool)
            r.update({"median_range_error_m": float((m.est_z - m.gt_z).median()),
                      "median_abs_range_error_m": float((m.est_z - m.gt_z).abs().median()),
                      "median_range_ratio_est_over_gt": float((m.est_z / m.gt_z).median()),
                      "share_est_closer_than_gt": float((m.est_z < m.gt_z).mean()),
                      "share_lateral_in_ours_out_in_gt": float((m.est_in_corridor & ~m.gt_in_corridor).mean()),
                      "median_closing_speed_est_ms": float((m.est_z / m.est_ttc.clip(lower=0.05)).replace(np.inf, np.nan).median()),
                      "median_closing_speed_gt_ms": float((m.gt_z / m.gt_ttc.clip(lower=0.05)).replace(np.inf, np.nan).median()),
                      "share_class_mismatch": float(m.class_mismatch.mean()),
                      "share_fixed_by_range": float(m.fix_range.mean()), "share_fixed_by_lateral": float(m.fix_lateral.mean()),
                      "share_fixed_by_ttc": float(m.fix_ttc.mean()),
                      "share_fixed_by_none": float((~m.fix_range & ~m.fix_lateral & ~m.fix_ttc).mean())})
        if len(dec) and "F_has_object" in dec:
            r.update({"share_F_detects_same_object": float(dec.F_has_object.fillna(False).astype(bool).mean()),
                      "share_same_object_decisive_for_F": float(dec.F_object_decisive.fillna(False).astype(bool).mean())})
        summ.append(r)
    pd.DataFrame(summ).to_csv(OUT / "groupd_summary.csv", index=False)
    print(pd.DataFrame(summ).round(3).T.to_string())


if __name__ == "__main__":
    main()
