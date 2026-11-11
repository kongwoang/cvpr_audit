#!/usr/bin/env python
"""K30: every comparison row where the frozen v6-S is NOT significantly above (R6), from results/k30/tables (+ b2d,
navsim tables when present) -> results/k30/tables/not_above_all.csv (item, dataset, setting, row, metric, v6S_minus,
ci_lo, ci_hi, verdict) and not_above_counts.csv (per item: comparisons, v6-S above, n.s., row above).
Sign convention: v6S_minus > 0 means v6-S is better (for cost-like metrics the sign is flipped before this table).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ.get("B_REAL", os.environ["B"]))
T = B / "results" / "k30" / "tables"


def verdict(lo, hi):
    return "v6-S above" if lo > 0 else ("row above" if hi < 0 else "n.s.")


def rows():
    out = []

    def add(item, ds, setting, row, metric, pt, lo, hi, **kw):
        out.append({"item": item, "dataset": ds, "setting": setting, "row": row, "metric": metric, "v6S_minus": pt,
                    "ci_lo": lo, "ci_hi": hi, "verdict": verdict(lo, hi), **kw})
    f = T / "v1a_matched.csv"
    if f.exists():
        for _, r in pd.read_csv(f).dropna(subset=["ci_lo"]).iterrows():
            add("V1a", r.dataset, f"m={r.m:g}, refiner {r.refiner}", r.row, "G", r.v6S_minus, r.ci_lo, r.ci_hi,
                px_ratio_to_v6S=r.px_ratio_to_v6S)
    f = T / "v1b_axis.csv"
    if f.exists():
        d = pd.read_csv(f)
        for _, r in d[d.crop_long_side.notna() & (d.is_v6S.astype(str) != "True")].iterrows():
            add("V1b", r.dataset, "axis", f"crop side {int(r.crop_long_side)}, context {r.support_context:g}", "G",
                -r.minus_v6S, -r.ci_hi, -r.ci_lo)
    f = T / "v1c_ttc.csv"
    if f.exists():
        for _, r in pd.read_csv(f).dropna(subset=["ci_lo"]).iterrows():
            add("V1c", r.dataset, "TTC-warning consumer", r.row, "G2", r.v6S_minus, r.ci_lo, r.ci_hi)
    f = T / "v2_strata.csv"
    if f.exists():
        for _, r in pd.read_csv(f).dropna(subset=["ci_lo"]).iterrows():
            add("V2 strata", r.dataset, f"{r.stratum}={r.level}", r.row, "G", r.v6S_minus, r.ci_lo, r.ci_hi, frames=r.frames)
    f = T / "v2_tube.csv"
    if f.exists():
        for _, r in pd.read_csv(f).dropna(subset=["ci_lo"]).iterrows():
            add("V2 tube", r.dataset, "frames with the decisive object outside the tube", "hybrid tsp8", "G", r.v6S_minus, r.ci_lo,
                r.ci_hi, frames=r.frames)
    f = T / "v2b_idm.csv"
    if f.exists():
        for _, r in pd.read_csv(f).iterrows():
            if pd.notna(r.get("v6-S_ci_lo")):
                add("V2b", r.dataset, "IDMFollow consumer, vs frozen v6-S", r.row, "G_idm", r["v6-S_minus"], r["v6-S_ci_lo"], r["v6-S_ci_hi"])
            if pd.notna(r.get("ours-IDM_ci_lo")) and r.row.startswith("v6-S") is False:
                add("V2b", r.dataset, "IDMFollow consumer, vs ours driven by IDM", r.row, "G_idm", r["ours-IDM_minus"],
                    r["ours-IDM_ci_lo"], r["ours-IDM_ci_hi"], ours="ours-IDM")
    f = T / "v5a_loss_weights.csv"
    if f.exists():
        for _, r in pd.read_csv(f).dropna(subset=["ci_lo"]).iterrows():
            add("V5a", r.dataset, f"lam_brake x{r.lam_brake_x:g}, lam_coll x{r.lam_collision_x:g}, lam_jerk x{r.lam_jerk_x:g}",
                r.pipeline, "G", r.v6S_minus, r.ci_lo, r.ci_hi)
    f = T / "v5_prior.csv"
    if f.exists():
        for _, r in pd.read_csv(f).iterrows():
            add(f"{r.part} (prior work)", r.dataset, r.setting, r.prior, "G", r.v6S_minus_prior, r.ci_lo, r.ci_hi)
    f = T / "v5bc_consumer.csv"
    if f.exists():
        for _, r in pd.read_csv(f).iterrows():
            if pd.notna(r.get("v6S_minus")):
                add("V5b", r.dataset, r.setting, r.pipeline, "G", r.v6S_minus, r.ci_lo, r.ci_hi)
            elif pd.notna(r.get("minus_frozen_v6S")):
                add("V5c", r.dataset, r.setting, r.pipeline, "G", -r.minus_frozen_v6S, -r.ci_hi, -r.ci_lo)
    f = T / "v8c_kitti_cells.csv"
    if f.exists():
        d = pd.read_csv(f)
        for _, r in d.dropna(subset=["v6S_f0_minus_lo"]).iterrows():
            add("V8c KITTI cells", "kitti", f"{r.frames_set}; v6-S flag 0.0", r.pipeline, "mean of 4 cells", r.v6S_f0_minus_mean4,
                r.v6S_f0_minus_lo, r.v6S_f0_minus_hi)
    f = T / "v8c_kitti_ap_diff.csv"
    if f.exists():
        for _, r in pd.read_csv(f).iterrows():
            add("V8c KITTI AP", "kitti", f"{r['class']} {r.difficulty}", r.other, "AP R40", r.v6S_minus, r.ci_lo, r.ci_hi)
    f = T / "v8d_speed.csv"
    if f.exists():
        for _, r in pd.read_csv(f).dropna(subset=["ci_lo"]).iterrows():
            add("V8d", r.dataset, f"speed x{r.speed_scale:g}", r.pipeline, "G", r.v6S_minus, r.ci_lo, r.ci_hi)
    f = T / "v8eg_corner_coda.csv"
    if f.exists():
        d = pd.read_csv(f)
        for _, r in d[(d["item"] == "V8e") & d.ci_lo.notna()].iterrows():
            add("V8e", r.dataset, "corner-case frames", r.pipeline, "G", r.v6S_minus, r.ci_lo, r.ci_hi, frames=r.frames)
    f = B / "results" / "k30" / "b2d" / "b2d_intervals.csv"
    if f.exists():
        higher_better = {"DS", "RC", "penalty", "success", "efficiency"}
        for _, r in pd.read_csv(f).iterrows():
            m = r.metric
            if m.startswith("n_"):
                continue
            s = 1 if (m in higher_better or m.startswith("ab_")) else -1      # smoothness / hard brakes / infractions: lower is better
            pt, lo, hi = s * r.v6_minus, min(s * r.ci_lo, s * r.ci_hi), max(s * r.ci_lo, s * r.ci_hi)
            if lo == hi == 0:
                continue
            add("V4 Bench2Drive", "b2d", f"routes={r.routes}", r.branch, m, pt, lo, hi)
    return pd.DataFrame(out)


def main():
    d = rows()
    na = d[d.verdict != "v6-S above"]
    na.to_csv(T / "not_above_all.csv", index=False)
    c = d.groupby("item").verdict.value_counts().unstack(fill_value=0)
    c["comparisons"] = c.sum(1)
    c.to_csv(T / "not_above_counts.csv")
    pd.set_option("display.width", 200)
    print(c.to_string())
    print(f"{len(na)} not-above rows of {len(d)}")


if __name__ == "__main__":
    main()
