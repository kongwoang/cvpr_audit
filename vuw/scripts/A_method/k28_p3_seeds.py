#!/usr/bin/env python
"""K28 PHASE 3 (post-hoc) — S6 on the test units: G of v6-S / M / L with refiner seeds 0 / 1 / 2 (k28_p3.py --part ours
--seeds --tag seeds), mean / std (ddof 1, as phase-1 seeds_r2.csv), parity of seed 0 with the phase-2 pass, and the S6
check against the smallest S1 margin claimed in phase 2 (tables_v6/class_matched_v6.csv: in-class, deployable, outperforms;
at S, where S1 is met, and over every budget). Argoverse-HD: frames with 3D labels only.
Outputs: results/k28/p3/tables/seeds_test_v6.csv, s6_test_v6.csv
"""
from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
R = REAL / "results" / "k28"
V6 = {"v6-S": ("v6S", "v3a"), "v6-M": ("v6M", "own-yolov8m"), "v6-L": ("v6L", "own-yolo11m")}
OWN = {"ours", "ours v6", "ablation", "previous frozen", "previous frozen + components"}


def load(d):
    rows = []
    for f in sorted(glob.glob(str(d / "*.pkl"))):
        rows += pickle.load(open(f, "rb"))
    x = pd.DataFrame(rows)
    return x[x["gt"].astype(bool)].reset_index(drop=True) if "gt" in x else x


def main():
    rows = []
    for ds in ("nusc", "av2", "ahd"):
        s = load(R / "p3" / f"{ds}_test_ours_seeds")
        p2 = load(R / f"{ds}_test_v6ours")
        m = s.merge(p2, on=["unit", "frame"], suffixes=("", "@p2"))
        L = s.J_cheap.to_numpy(float)
        for lab, (k, rf) in V6.items():
            g = [float((L - s[f"J_{k}|{c}"].to_numpy(float)).sum() / L.sum()) for c in (rf, f"{rf}_s1", f"{rf}_s2")]
            par = float(np.isclose(m[f"J_{k}|{rf}"].astype(float), m[f"J_{k}|{rf}@p2"].astype(float)).mean())
            rows.append({"dataset": ds, "configuration": lab, "units": s.unit.nunique(), "frames": len(s), "G_seed0": g[0], "G_seed1": g[1],
                         "G_seed2": g[2], "G_mean": float(np.mean(g)), "G_std": float(np.std(g, ddof=1)),
                         "seed0_identical_to_phase2": par, "frames_matched_phase2": len(m)})
    t = pd.DataFrame(rows)
    out = R / "p3" / "tables"
    t.to_csv(out / "seeds_test_v6.csv", index=False)
    print(t.round(4).to_string(index=False))
    c = pd.read_csv(R / "tables_v6" / "class_matched_v6.csv")
    c = c[(~c.group.isin(OWN)) & (~c.baseline.str.contains("oracle")) & (c["class"] != "over") & (c.verdict == "outperforms")]
    ms = float(c[c["class"] == "S"]["diff"].min()); ma = float(c["diff"].min())
    sd = float(t.G_std.max())
    s6 = pd.DataFrame([{"max_seed_std_test": sd, "smallest_S1_margin_at_S": ms, "half_S": ms / 2, "met_at_S": sd < ms / 2,
                        "smallest_outperform_margin_any_budget": ma, "half_any": ma / 2, "met_any": sd < ma / 2}])
    s6.to_csv(out / "s6_test_v6.csv", index=False)
    print(s6.round(4).to_string(index=False))
    print("K28 SEEDS TABLES DONE")


if __name__ == "__main__":
    main()
