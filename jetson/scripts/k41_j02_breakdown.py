#!/usr/bin/env python
"""K41-6 J02 report: per-stage breakdown of v6-S / B2K-A / B1K-A (--breakdown repetition per test set, per-stage CUDA
syncs; profiler overhead kept out of the clean J04 timings). Per row: stage mean (ms) and share of the total, mean stage
times on the row's tail frames (latency >= its own p95 / p99) and on frames missing 50 / 66.7 ms; rest = total - stages
(merge, guard, lift, bookkeeping). usage: scripts/py scripts/k41_j02_breakdown.py -> docs/paper/handoff/K41/J04/"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/k41_jetson_20261030"
OUT = ROOT / "docs/paper/handoff/K41/J04"
ROWS = {"ours@trt32": "DESP v6-S", "ours_b2A@trt32": "DESP B2K-A", "ours_b1A@trt32": "DESP B1K-A"}


def main():
    st = json.loads((HERE / "logs/state.json").read_text()); out = []
    for r in st["runs"]:
        if r["tag"] != "bd":
            continue
        d = pd.read_pickle(Path(r["run"]) / "frames.pkl"); d = d[~d.warm]
        for key, lab in ROWS.items():
            stages = [c for c in d.columns if c.startswith("bd_") and c.endswith("_" + key)]
            crop = [c for c in stages if c.startswith("bd_crops_r")]
            S = pd.DataFrame({c[3:-len(key) - 1]: d[c].astype(float).fillna(0.0) for c in stages if c not in crop})
            if crop:
                S["crops_all_rounds"] = d[crop].astype(float).fillna(0.0).sum(1)
            tot = d["ms_" + key].astype(float); S["rest"] = tot - S.sum(1); S["total"] = tot
            sets = {"all": np.ones(len(d), bool), ">=p95": tot >= np.percentile(tot, 95), ">=p99": tot >= np.percentile(tot, 99),
                    ">50ms": tot > 50, ">66.7ms": tot > 66.7}
            for name, mk in sets.items():
                mk = np.asarray(mk)
                if not mk.any():
                    continue
                row = {"dataset": r["dataset"], "row": lab, "frames": int(mk.sum()), "subset": name}
                for c in S:
                    row[c + "_ms"] = float(S[c][mk].mean())
                rounds = d.get("rounds_" + key)
                if rounds is not None:
                    row["rounds_mean"] = float(rounds[mk].astype(float).mean())
                out.append(row)
    t = pd.DataFrame(out); OUT.mkdir(parents=True, exist_ok=True); t.to_csv(OUT / "k41_6_breakdown.csv", index=False)
    pd.set_option("display.width", 250); print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
