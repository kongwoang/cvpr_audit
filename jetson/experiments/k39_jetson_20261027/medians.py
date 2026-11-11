"""K39-1: dev-prefix round medians of v6-S / B2K-A / B1K-A (K38-1 medians.py with B1K-A added; --calibrate runs on
nusc_val / av2_dev, timed frames only) -> medians_k39.json, written BEFORE the test runs."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    out = {}
    for ds, run in (("nusc", sys.argv[1]), ("av2", sys.argv[2])):
        d = pd.read_pickle(Path(run) / "frames.pkl")
        d = d[~d.warm]
        out[ds] = {}
        for nm, kmax in (("ours", 3), ("ours_b2A", 2), ("ours_b1A", 1)):
            col = f"{nm}@trt32"
            dur = {k: [] for k in range(kmax)}
            for top, end in zip(d[f"rt_top_{col}"], d[f"rt_end_{col}"]):
                for k in range(min(len(end), kmax)):
                    dur[k].append(end[k] - top[k])
            out[ds][nm] = {"round": [float(np.median(dur[k])) if dur[k] else float(np.median(dur[0])) for k in range(kmax)],
                           "round_n": [len(dur[k]) for k in range(kmax)],
                           "round_p95": [float(np.percentile(dur[k], 95)) if dur[k] else None for k in range(kmax)],
                           "finish": float(np.median(d[f"fin_{col}"])), "frames": int(len(d))}
    Path(sys.argv[3]).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
