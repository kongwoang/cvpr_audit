#!/usr/bin/env python
"""K29 interim outputs (window C paused on 2026-10-15, Jetson resets under GPU load):
  maxn_partial/latency_rep1.csv   repetition 1 at MAXN (GPU 1377 MHz), every pipeline, reference vs the selected fast path:
                                  mean / median / p95 / p99 / max, speed-up of the mean and p95 (test frames, post-hoc)
  vin_reset.{pdf,png}             input voltage (INA3221 bus voltage of the GPU channel) and rail power, 1 s lines, of the
                                  run that ended in the 17:16 reset (logs/k29/vin_1015_171516.log)
usage: 66_k29_interim.py --sel trt32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "docs" / "paper" / "handoff" / "K29"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", required=True)
    ap.add_argument("--vin", default=str(ROOT / "logs" / "k29" / "vin_1015_171516.log"))
    args = ap.parse_args()
    r = pd.read_csv(K / "maxn_partial" / "latency_runs.csv")
    r = r[r.rep == 1]
    cols = ["mean_ms", "median_ms", "p95_ms", "p99_ms", "max_ms"]
    ref = r[r.path == "ref"].set_index(["dataset", "pipeline"])
    fast = r[r.path == args.sel].set_index(["dataset", "pipeline"])
    out = ref[["group", "frames", "units", "G"] + cols].copy()
    for c in cols:
        out[f"{c}_{args.sel}"] = fast[c]
    out["speedup_mean"] = out["mean_ms"] / out[f"mean_ms_{args.sel}"]
    out["speedup_p95"] = out["p95_ms"] / out[f"p95_ms_{args.sel}"]
    out = out.rename(columns={c: f"{c}_ref" for c in cols}).reset_index()
    out.to_csv(K / "maxn_partial" / "latency_rep1.csv", index=False)
    print(out[["dataset", "pipeline", "mean_ms_ref", f"mean_ms_{args.sel}", "p95_ms_ref", f"p95_ms_{args.sel}"]].round(1).to_string(index=False))

    v = pd.read_csv(args.vin, sep=" ")
    t = pd.to_datetime(v["time"], format="%H:%M:%S")
    s = (t - t.iloc[0]).dt.total_seconds()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(s, v.vmean_mV / 1000, c="tab:blue", lw=1.5, label="VDD_IN, 1 s mean")
    ax.fill_between(s, v.vmin_mV / 1000, v.vmax_mV / 1000, color="tab:blue", alpha=.2, lw=0, label="1 s min-max")
    ax.axhline(v.vmean_mV.iloc[:5].mean() / 1000 - 0.5, c="tab:red", ls="--", lw=1, label="idle - 0.5 V (power-bad trip, design guide)")
    ax.set_xlabel(f"seconds after {v['time'].iloc[0]} (2026-10-15); the board reset at {v['time'].iloc[-1]}+")
    ax.set_ylabel("input voltage (V)")
    a2 = ax.twinx()
    a2.plot(s, v.P_W, c="tab:orange", lw=1, label="rail power (GPU+CPU+SOC+CV+VDDRQ)")
    a2.set_ylabel("rail power (W)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center left")
    ax.grid(alpha=.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(K / f"vin_reset.{ext}", dpi=150)


if __name__ == "__main__":
    main()
