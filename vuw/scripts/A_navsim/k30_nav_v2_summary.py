#!/usr/bin/env python
"""K30 V8a navtest summary: per branch, the devkit's per-token EPDMS rows (run_pdm_score_one_stage.py CSVs under
<EXP>/k30_v2/navtest_<branch>/, latest file) -> mean EPDMS and sub-scores over valid tokens, and spec_pdm - branch on the
common valid tokens with a paired log bootstrap (2,000, seed 0; token -> log from dets_final). Writes
<EXP>/k30_v2/navtest_summary.csv."""
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(os.environ.get("NAVSIM_EXP_ROOT", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR/results/navsim_exp"))
BR = {"reference": "reference", "cheap": "cheap", "full960": "full960", "native": "native", "spec_pdm": "spec_pdm",
      "hybrid_pdm (fixed tube)": "hybrid_pdm_fixed2"}
SUB = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance",
       "ego_progress", "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]


def main():
    tok2log = {}
    for f in glob.glob(str(EXP / "dets_final" / "*.pkl")):
        for t in pickle.load(open(f, "rb")):
            tok2log[t] = Path(f).stem
    d = {}
    for lab, dn in BR.items():
        fs = sorted(glob.glob(str(EXP / "k30_v2" / f"navtest_{dn}" / "*.csv")))
        if fs:
            x = pd.read_csv(fs[-1])
            d[lab] = x[(x.token != "average_all_frames") & x.valid.astype(bool)].set_index("token")
    out = []
    ref = d.get("spec_pdm")
    for lab, x in d.items():
        r = {"branch": lab, "tokens_valid": len(x), "EPDMS": x.score.mean(), **{c: x[c].mean() for c in SUB if c in x}}
        if ref is not None and lab != "spec_pdm":
            com = sorted(set(ref.index) & set(x.index))
            diff = ref.loc[com, "score"].to_numpy() - x.loc[com, "score"].to_numpy()
            logs = np.array([tok2log.get(t, "?") for t in com])
            uq, inv = np.unique(logs, return_inverse=True)
            s, n = np.bincount(inv, diff, len(uq)), np.bincount(inv, None, len(uq))
            rng = np.random.default_rng(0)
            bs = [rng.integers(0, len(uq), len(uq)) for _ in range(2000)]
            v = [s[b].sum() / n[b].sum() for b in bs]
            r.update({"common_tokens": len(com), "spec_pdm_minus": diff.mean(), "ci_lo": np.percentile(v, 2.5), "ci_hi": np.percentile(v, 97.5)})
        out.append(r)
    o = pd.DataFrame(out)
    o.to_csv(EXP / "k30_v2" / "navtest_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(o.round(4).to_string(index=False))
    print("K30 NAVTEST SUMMARY DONE")


if __name__ == "__main__":
    main()
