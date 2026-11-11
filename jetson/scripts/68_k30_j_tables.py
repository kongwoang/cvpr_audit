#!/usr/bin/env python
"""K30 J tables from the Jetson runs of scripts/67_k30_jetson.py (results/raw/*_k30j_timing_<dataset>_<config>_<tag>/):
  J1  latency-G points (mean and p95 over the 3 repetitions) of the heuristics + C1 + refiner at K28's S budget (tags
      j1s_r*, config v51S) and at V1's x1.0 budget (tags j1x_r*, config v6S), fast path trt32 -> J/j1_latency_G.csv
  J2  energy per frame: the module rails (INA3221 GPU + CPU + SOC + CV + VDDRQ; scripts/k29_vin_logger.py, 1 s means)
      integrated over each single-pipeline run (tags j2_*; the 7 ms draft bookkeeping included), between the first and the
      last timed frame; the hybrid's own energy = its run minus the run of its budget configuration alone -> J/j2_energy.csv
  J3  per-frame latency of the Argoverse-HD run (tag j3_r1) for the streaming-AP simulation on VUW -> J/j3_latency.csv
usage: 68_k30_j_tables.py <vin log>
"""
import glob
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper" / "handoff" / "K30" / "J"


DSN = ("nusc_test20", "nusc", "av2", "ahd")


def split_key(key):
    """'<dataset>_<config>_<tag>' with datasets that contain '_'"""
    for ds in DSN:
        if key.startswith(ds + "_"):
            cfg, tag = key[len(ds) + 1:].split("_", 1)
            return ds, cfg, tag
    raise ValueError(key)


def runs(pattern):
    out = {}
    for d in sorted(glob.glob(str(ROOT / "results" / "raw" / f"*_k30j_timing_{pattern}"))):
        if (Path(d) / "frames.pkl").exists():
            out[Path(d).name.split("_k30j_timing_")[1]] = Path(d)
    return out


def vin(path):
    """vin log lines 'HH:MM:SS vmin vmean vmax P Pmax ...' -> DataFrame with absolute datetimes (date from the file
    name's MMDD, day rollover when the clock goes back)"""
    md = Path(path).name.split("_")[1]
    day = datetime(2026, int(md[:2]), int(md[2:]))
    rows, prev = [], None
    for ln in open(path):
        p = ln.split()
        if len(p) < 6 or not p[1].isdigit():
            continue
        t = datetime.combine(day.date(), datetime.strptime(p[0], "%H:%M:%S").time())
        if prev is not None and t < prev - timedelta(hours=1):
            day += timedelta(days=1)
            t += timedelta(days=1)
        prev = t
        rows.append((t.timestamp(), float(p[2]) / 1e3, float(p[4])))
    return pd.DataFrame(rows, columns=["ts", "v", "P_W"])


def frames(d):
    f = pd.read_pickle(d / "frames.pkl")                          # 67 writes a DataFrame
    return f[~f["warm"].astype(bool)].reset_index(drop=True) if "warm" in f else f


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- J1
    rows = []
    for key, d in runs("*_j1*_r*").items():
        ds, cfg, tag = split_key(key)
        lat = pd.read_csv(d / "latency.csv")
        lat = lat.assign(dataset=ds, config=cfg, budget="S (K28)" if tag.startswith("j1s") else "x1.0 v6-S (V1)", rep=int(tag[-1]))
        rows.append(lat)
    if rows:
        j1 = pd.concat(rows)
        g = j1.groupby(["dataset", "budget", "pipeline"]).agg(reps=("rep", "nunique"), G=("G", "first"), mean_ms=("mean_ms", "mean"),
                                                                mean_ms_sd=("mean_ms", "std"), p95_ms=("p95_ms", "mean"),
                                                                p95_ms_sd=("p95_ms", "std"), px=("tensor_px_mean", "mean")).reset_index()
        g.to_csv(OUT / "j1_latency_G.csv", index=False)
        print(g.round(3).to_string(index=False))
    # ---- J2
    V = vin(sys.argv[1]) if len(sys.argv) > 1 else None
    rows = []
    for key, d in runs("*_j2_*").items():
        f = frames(d)
        if V is None or not len(f) or "t_wall" not in f:
            continue
        t0, t1 = float(f.t_wall.min()), float(f.t_wall.max())
        w = V[(V.ts >= t0) & (V.ts <= t1)]
        if len(w) < 5:
            continue
        E = float(w.P_W.sum())                                   # 1 s means -> J
        n = len(f) - 1
        ds, cfg, tag = split_key(key)
        rows.append({"dataset": ds, "config": cfg, "run": tag.replace("j2_", ""), "frames": n, "seconds": t1 - t0,
                     "mean_P_W": float(w.P_W.mean()), "max_P_W": float(w.P_W.max()), "min_VDD_IN_V": float(w.v.min()),
                     "energy_per_frame_J": E / n * (t1 - t0) / max(len(w), 1), "ms_per_frame": 1e3 * (t1 - t0) / n})
    if rows:
        e = pd.DataFrame(rows)
        for ds in e.dataset.unique():                            # the hybrid's own energy: run minus its budget configuration
            for hyb, base in (("hybS", "v51S_alone"), ("hybX", "v6S_alone")):
                a, b = e[(e.dataset == ds) & (e.run == hyb)], e[(e.dataset == ds) & (e.run == base)]
                if len(a) and len(b):
                    e.loc[a.index, "energy_own_J"] = a.energy_per_frame_J.values - b.energy_per_frame_J.values
        e.to_csv(OUT / "j2_energy.csv", index=False)
        print(e.round(4).to_string(index=False))
    # ---- J3
    for key, d in runs("ahd_*_j3_*").items():
        f = frames(d)
        cols = ["unit", "frame", "image_id", "gt"] + [c for c in f.columns if c.startswith("ms_")]
        f[cols].to_csv(OUT / "j3_latency.csv", index=False)
        print("J3 latency rows", len(f), [c for c in cols if c.startswith("ms_")])
    print("K30 J TABLES DONE")


if __name__ == "__main__":
    main()
