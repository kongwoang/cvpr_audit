"""J04: dev-prefix round medians of the iterated K41 rows (ana05hinfi, cf40q20i; --calibrate runs on nusc_val / av2_dev,
timed frames only, no stopping) -> medians41.json, written BEFORE the test runs (same rule as K38-1 / K39 medians.py)."""
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
        for nm in ("ana05hinfi", "cf40q20i"):
            col = f"{nm}@trt32"
            dur = {k: [] for k in range(3)}
            for top, end in zip(d[f"rt_top_{col}"], d[f"rt_end_{col}"]):
                for k in range(min(len(end), 3)):
                    dur[k].append(end[k] - top[k])
            out[ds][nm] = {"round": [float(np.median(dur[k])) if dur[k] else float(np.median(dur[0])) for k in range(3)],
                           "round_n": [len(dur[k]) for k in range(3)],
                           "round_p95": [float(np.percentile(dur[k], 95)) if dur[k] else None for k in range(3)],
                           "finish": float(np.median(d[f"fin_{col}"])), "frames": int(len(d))}
    Path(sys.argv[3]).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
