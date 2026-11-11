#!/usr/bin/env python
"""K39-2 report (post-hoc; handoff/K39/PREREG.md): the 8 S heuristics + C1 + v3a under the STRICT post-guard per-frame
px cap (budget = v6-S's realised crop px of the frame) and under the nominal m = 1 cap (K30 V1a rule), three test
splits (results/k39/vuw/<ds>, GT frames). Per row: G, realised px / v6-S px (mean, max), skipped crop / guard calls;
paired unit bootstrap (2,000, seed 0; sorted units, default_rng(0) integers) of v6-S - row.
Primary rows (PREREG): tsp8 everywhere; nuScenes unc4 / tcr3 / cor0; AV2 and Argoverse-HD unc2 / tcr0 / cor8.
usage: k39_2_report.py <out_dir>"""
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
HEUR8 = ("unc2", "unc4", "tcr0", "tcr3", "cor0", "cor8", "tsp2", "tsp8")
FAM = {"unc": "uncertainty", "tcr": "temporal criticality", "cor": "ego corridor", "tsp": "support + ego-corridor tube"}
PRIMARY = {"nusc": {"tsp8", "unc4", "tcr3", "cor0"}, "av2": {"tsp8", "unc2", "tcr0", "cor8"}, "ahd": {"tsp8", "unc2", "tcr0", "cor8"}}


def recs(d):
    rr = []
    for f in sorted(glob.glob(str(d / "*.pkl"))):
        rr += pickle.load(open(f, "rb"))
    x = pd.DataFrame(rr)
    x["unit"] = x.unit.astype(str)
    return x


def main():
    od = Path(sys.argv[1])
    od.mkdir(parents=True, exist_ok=True)
    rows, diffs = [], []
    for ds in ("nusc", "av2", "ahd"):
        d = recs(B / "results/k39/vuw" / ds)
        d = d[d["gt"].astype(bool)].reset_index(drop=True)
        units = np.array(sorted(d.unit.unique()))
        idx = np.random.default_rng(0).integers(0, len(units), (2000, len(units)))
        g = lambda col: d.groupby("unit")[col].sum().reindex(units).to_numpy()   # noqa: E731
        c, v = g("J_cheap"), g("J_v6S|v3a")
        Gv = (c.sum() - v.sum()) / c.sum()
        rows.append({"dataset": ds, "row": "v6-S", "cap": "-", "frames": len(d), "units": len(units), "G": Gv,
                     "px_mean": d.px_v6S.mean(), "px_ratio_mean": 1.0, "px_ratio_max": 1.0})
        for cfg in HEUR8:
            for cap, nm in (("strict", f"{cfg}@strict+C1"), ("nominal m=1", f"{cfg}@x1.00+C1")):
                j = g(f"J_{nm}|v3a")
                ratio = (d[f"px_{nm}"] / d.px_v6S).to_numpy()
                pt = (j.sum() - v.sum()) / c.sum()
                dist = (j[idx].sum(1) - v[idx].sum(1)) / c[idx].sum(1)
                lo, hi = np.percentile(dist, [2.5, 97.5])
                r = {"dataset": ds, "row": cfg, "family": FAM[cfg[:3]], "primary": cfg in PRIMARY[ds], "cap": cap,
                     "frames": len(d), "units": len(units), "G": (c.sum() - j.sum()) / c.sum(), "px_mean": d[f"px_{nm}"].mean(),
                     "px_ratio_mean": float(ratio.mean()), "px_ratio_max": float(ratio.max()),
                     "frames_over_cap": int((ratio > 1 + 1e-9).sum())}
                if cap == "strict":
                    r["skipped_crop_calls_per_frame"] = d[f"skip_crop_{nm}"].mean()
                    r["skipped_guard_calls_per_frame"] = d[f"skip_guard_{nm}"].mean()
                    r["frames_with_a_skip"] = float(((d[f"skip_crop_{nm}"] + d[f"skip_guard_{nm}"]) > 0).mean())
                rows.append(r)
                diffs.append({"dataset": ds, "row": cfg, "family": FAM[cfg[:3]], "primary": cfg in PRIMARY[ds], "cap": cap,
                              "v6S_minus_row": pt, "ci_lo": lo, "ci_hi": hi, "excludes_0": bool(lo > 0 or hi < 0)})
    R, D = pd.DataFrame(rows), pd.DataFrame(diffs)
    R.to_csv(od / "k39_2_rows.csv", index=False)
    D.to_csv(od / "k39_2_diffs.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string(index=False))
    print(D.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
