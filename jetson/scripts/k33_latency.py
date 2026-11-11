#!/usr/bin/env python
"""K33 G: Jetson latency of the selected rows from the timing runs' latency.csv (69_k33_jetson.py; tags k33selde /
k33selpub / k33selada / k33selrt, repetitions r1-r3; nusc_test20 -> nusc, av2 -> av2). Per dataset x row: mean over the
repetitions of the per-run mean and p95 (ms) and the spread of the per-run means. v6S = "ours@trt32" (v6-S on the device,
timed in every run; pooled over all runs). -> results/k33j_latency.csv (k33_report.py --latency)
"""
import glob
import re
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
rows = []
for d in sorted(glob.glob(str(R / "raw" / "*_k33j_timing_*_v6S_k33sel*_r[0-9]"))):
    m = re.search(r"k33j_timing_(nusc_test20|av2)_v6S_(k33sel[a-z]+)_r(\d)$", d)
    if not m or not Path(d, "latency.csv").exists():
        continue
    t = pd.read_csv(Path(d, "latency.csv"))
    for _, r in t.iterrows():
        p = str(r.pipeline)
        if p == "cheap" or p.endswith("_noref"):
            continue
        row = "v6S" if p == "ours@trt32" else p.rsplit("@", 1)[0]
        rows.append({"dataset": "nusc" if m[1] == "nusc_test20" else "av2", "row": row, "group": m[2], "rep": int(m[3]),
                     "frames": r.frames, "mean_ms": r.mean_ms, "p95_ms": r.p95_ms})
    rows += [{"dataset": "nusc" if m[1] == "nusc_test20" else "av2", "row": "cheap", "group": m[2], "rep": int(m[3]),
              "frames": r.frames, "mean_ms": r.mean_ms, "p95_ms": r.p95_ms} for _, r in t[t.pipeline == "cheap"].iterrows()]
D = pd.DataFrame(rows)
S = D.groupby(["dataset", "row"]).agg(mean_ms=("mean_ms", "mean"), p95_ms=("p95_ms", "mean"), mean_ms_sd=("mean_ms", "std"),
                                      runs=("rep", "size"), frames=("frames", "max")).reset_index()
S.to_csv(R / "k33j_latency.csv", index=False)
print(S.round(2).to_string(index=False))
