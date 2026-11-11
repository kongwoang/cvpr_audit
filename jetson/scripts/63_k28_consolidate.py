#!/usr/bin/env python
"""K28 consolidated REPORT, item (iii): every row, on every benchmark, where ours (v6) is not above a baseline.
Reads only the K28 handoff CSVs (no new evaluation) and writes one long table plus counts:
  docs/paper/handoff/K28/final/not_above_all.csv, not_above_counts.csv
Sources and what "not above" means in each:
  phase 2 (CONFIRMATORY)  not_outperformed_v6.csv: every (v6 configuration, row) pair whose interval is not above 0
                          (K24 verdicts: matches = lo > -0.02, below otherwise); in-class flag from class_matched_v6.csv.
  phase 3 S5 (post-hoc)   ctable_{y11s,ft0,ft1}.csv: rows v6-S does not outperform (with refiner).
  phase 3 nomask          class_matched_nomask_v6.csv: AV2 without the ego mask, each row vs its class-matched v6.
  phase 3 prior works     sota_intervals_v6.csv: v6-S / M / L vs AdaScale / FOVEA / SAHI / fine-tuned rows.
  phase 3 S3 (Jetson)     s3_dominance.csv: rows with latency <= ours (mean or p95) and G >= ours, for the frozen
                          v6-S / M / L and v5.1-S rows (variants are listed in s3_front.csv).
  phase 3 KITTI           kitti_v6S_cells.csv: rows whose mean of 4 cells is not significantly below v6-S.
  phase 3 sAP             sap_ahd_v6.csv: rows with a higher sAP than v6-S (no interval; point estimates).
  phase 3 K6 Bench2Drive  b2d_v6_summary*.csv: branches v6-S does not outperform in driving score (paired route bootstrap).
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

K = Path(__file__).resolve().parents[1] / "docs" / "paper" / "handoff" / "K28"
OUT = K / "final"
DS = {"nusc": "nuScenes new test", "av2": "AV2 val (ego mask)", "ahd": "Argoverse-HD val",
      "av2 (no ego mask)": "AV2 val (no ego mask)"}
OWN = {"ours", "ours (reference)", "ablation", "previous frozen", "previous frozen + components"}


def side(d):
    return "v6 higher" if d > 0 else ("row higher" if d < 0 else "equal")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    # phase 2
    no = pd.read_csv(K / "phase2" / "not_outperformed_v6.csv")
    cm = pd.read_csv(K / "phase2" / "class_matched_v6.csv")
    over = dict(zip(zip(cm.dataset, cm.baseline, cm.configuration), cm.over_budget))   # class-matched pairs -> over budget
    for _, r in no.iterrows():
        kind = "own configuration / ablation" if r.group in OWN else ("oracle (not deployable)" if "oracle" in r.baseline else "baseline")
        rows.append({"source": "phase 2 (confirmatory)", "benchmark": DS[r.dataset], "ours": r.configuration, "row": r.baseline,
                     "row_group": r.group, "row_kind": kind, "row_class": r.baseline_class, "row_cost_ratio": r.baseline_GFLOPs_ratio,
                     "class_matched": (r.dataset, r.baseline, r.configuration) in over,
                     "in_class": over.get((r.dataset, r.baseline, r.configuration)) is False or over.get((r.dataset, r.baseline, r.configuration)) == 0,
                     "metric": "G (with refiner)",
                     "ours_minus_row": r["diff"], "lo": r.lo, "hi": r.hi, "verdict": r.verdict, "point_estimate": side(r["diff"])})
    # phase 3 S5
    for tag in ("y11s", "ft0", "ft1"):
        t = pd.read_csv(K / "phase3" / f"ctable_{tag}.csv")
        t = t[(t.verdict_refiner != "outperforms") & (t.row != "v6S")]
        for _, r in t.iterrows():
            rows.append({"source": f"phase 3 S5 ({tag}, post-hoc)", "benchmark": DS[r.dataset], "ours": "v6-S", "row": r.row,
                         "row_group": "ours" if r.row == "v51S" else "control",
                         "row_kind": "own configuration / ablation" if r.row == "v51S" else "baseline",
                         "row_cost_ratio": None, "in_class": None, "metric": "G (with refiner)", "ours_minus_row": r.v6S_minus_refiner,
                         "lo": r.lo_refiner, "hi": r.hi_refiner, "verdict": r.verdict_refiner, "point_estimate": side(r.v6S_minus_refiner)})
    # phase 3 nomask, class-matched
    t = pd.read_csv(K / "phase3" / "class_matched_nomask_v6.csv")
    t = t[t.class_matched & (t.verdict != "outperforms")]
    for _, r in t.iterrows():
        rows.append({"source": "phase 3 AV2 without ego mask (post-hoc)", "benchmark": DS[r.dataset], "ours": r.configuration,
                     "row": r.row, "row_group": r.group, "row_kind": "baseline", "row_class": r["class"], "row_cost_ratio": r.GFLOPs_ratio,
                     "in_class": r["class"] != "over", "metric": "G (with refiner)", "ours_minus_row": r.v6_minus_row, "lo": r.lo, "hi": r.hi,
                     "verdict": r.verdict, "point_estimate": side(r.v6_minus_row)})
    # phase 3 prior works
    t = pd.read_csv(K / "phase3" / "sota_intervals_v6.csv")
    for _, r in t.iterrows():
        for c in ("v6S", "v6M", "v6L"):
            if r[f"{c}_verdict_refiner"] != "outperforms":
                rows.append({"source": "phase 3 prior works (post-hoc)", "benchmark": DS[r.dataset], "ours": c[:2] + "-" + c[2:],
                             "row": r.baseline, "row_group": "prior work", "row_kind": "baseline", "metric": "G (with refiner)",
                             "ours_minus_row": r[f"{c}_minus_refiner"], "lo": r[f"{c}_lo_refiner"], "hi": r[f"{c}_hi_refiner"],
                             "verdict": r[f"{c}_verdict_refiner"], "point_estimate": side(r[f"{c}_minus_refiner"])})
    # phase 3 S3: rows that are at least as fast and at least as good (any frozen ours row)
    t = pd.read_csv(K / "phase3" / "jetson" / "s3_dominance.csv")
    t = t[t.ours.isin(["v6-S", "v6-M", "v6-L"]) & (t["diff"] <= 0) & (t.row_faster_mean | t.row_faster_p95)]
    for _, r in t.iterrows():
        rows.append({"source": "phase 3 S3 Jetson (post-hoc)", "benchmark": {"nusc": "nuScenes, 20 local DEEP scenes (Jetson)",
                     "av2": "AV2, 8 local val logs (Jetson)"}[r.dataset], "ours": r.ours, "row": r.row, "row_group": r.row_group,
                     "row_kind": "own configuration / ablation" if r.row_group in ("ours", "previous ours") else "baseline",
                     "metric": f"G at <= latency (row faster: mean {r.row_faster_mean}, p95 {r.row_faster_p95}; "
                               f"ms ours {r.mean_ms_ours:.1f}/{r.p95_ms_ours:.1f}, row {r.mean_ms_row:.1f}/{r.p95_ms_row:.1f})",
                     "ours_minus_row": r["diff"], "lo": r.ci_lo, "hi": r.ci_hi,
                     "verdict": "outperforms" if r.ci_lo > 0 else ("matches" if r.ci_lo > -0.02 else "below"), "point_estimate": side(r["diff"])})
    # phase 3 KITTI (v6-S - row interval on the mean of 4 cells; the CSV holds the interval only)
    t = pd.read_csv(K / "phase3" / "jetson" / "kitti_v6S_cells.csv")
    v = t[t.pipeline == "ours"].mean4.iloc[0] if (t.pipeline == "ours").any() else None
    for _, r in t[t.pipeline != "ours"].iterrows():
        if not r.ours_minus_mean4_lo > 0:
            rows.append({"source": "phase 3 KITTI shared v6-S (post-hoc)", "benchmark": "KITTI test (6 sequences)", "ours": "v6-S",
                         "row": r.pipeline, "row_group": "ours variant" if r.pipeline.startswith("ours") else "whole frame",
                         "row_kind": "own configuration / ablation" if r.pipeline.startswith("ours") else "baseline", "metric": "mean of 4 cells",
                         "ours_minus_row": (v - r.mean4) if v is not None else None, "lo": r.ours_minus_mean4_lo, "hi": r.ours_minus_mean4_hi,
                         "verdict": "matches" if r.ours_minus_mean4_lo > -0.02 else "below",
                         "point_estimate": side(v - r.mean4) if v is not None else None})
    # phase 3 sAP
    t = pd.read_csv(K / "phase3" / "sap_ahd_v6.csv")
    s0 = t.sAP.iloc[0]
    for _, r in t.iloc[2:].iterrows():
        if r.sAP >= s0:
            rows.append({"source": "phase 3 streaming AP (post-hoc)", "benchmark": "Argoverse-HD val", "ours": "v6-S", "row": r.pipeline,
                         "row_group": r.source, "row_kind": "previous ours (K7)" if r.pipeline.startswith("ours") else "baseline",
                         "metric": "sAP (no interval)", "ours_minus_row": s0 - r.sAP,
                         "verdict": "below (point estimate)", "point_estimate": side(s0 - r.sAP)})
    # K6 Bench2Drive with v6-S
    f = sorted(glob.glob(str(K / "phase3" / "b2d_v6_summary*.csv")))
    f = [x for x in f if not x.endswith("_run1.csv")] or f
    t = pd.read_csv(f[0])
    for _, r in t[t.branch != "ours_v6"].iterrows():
        if not r.v6_minus_DS_ci_lo > 0:
            rows.append({"source": f"phase 3 K6 Bench2Drive (post-hoc; runs {r.runs_compared})", "benchmark": "Bench2Drive 219 routes",
                         "ours": "v6-S", "row": r.branch, "row_group": "perception branch",
                         "row_kind": "oracle (not deployable)" if r.branch == "privileged" else "baseline", "metric": "driving score",
                         "ours_minus_row": r.v6_minus_DS, "lo": r.v6_minus_DS_ci_lo, "hi": r.v6_minus_DS_ci_hi,
                         "verdict": "below (significant)" if r.v6_minus_DS_ci_hi < 0 else "n.s.", "point_estimate": side(r.v6_minus_DS)})
    a = pd.DataFrame(rows)
    a.to_csv(OUT / "not_above_all.csv", index=False)
    c = a.fillna({"point_estimate": "-"}).groupby(["source", "benchmark", "row_kind", "verdict", "point_estimate"]).size().rename("rows").reset_index()
    c.to_csv(OUT / "not_above_counts.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(a.groupby(["source", "row_kind"]).size().to_string())
    print(len(a), "rows")


if __name__ == "__main__":
    main()
