#!/usr/bin/env python
"""K42-4(a) (post-hoc, CPU; handoff/K42/PREREG.md): decisive-object coverage of K34-2 with the region restricted to the
field + support crops (no guard crops). K30 match stores crops_v6S = crops + grown with ncrops_v6S = len(crops), so the
field + support crops are crops_v6S[:ncrops_v6S]. Population, centre rule and bootstrap = K34 (k34_common.bootstrap:
units of the whole dataset, default_rng(0).integers((2000, n)), ratio of per-unit sums).
Checks: len(crops_v6S) = ncrops + nguard; all-crop coverage recomputed = K34 coverage_frames.covered on every frame;
the K34 all-crop interval is reproduced exactly.
usage: k42_4_coverage.py <out_dir>"""
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from vtf import speculative as SP

B = Path(os.environ.get("B_REAL", os.environ["B"]))
K34 = B / "results" / "k34"


def recs(d, cols):
    rr = []
    for f in sorted(glob.glob(str(d / "*.pkl"))):
        rr += [{c: r.get(c) for c in cols} for r in pickle.load(open(f, "rb"))]
    x = pd.DataFrame(rr); x["unit"] = x.unit.astype(str)
    return x


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    rows, ints, checks, frames = [], [], {}, []
    for ds in ("nusc", "av2", "ahd"):
        q = pd.read_csv(K34 / "coverage" / ds / "coverage_frames.csv", dtype={"unit": str})
        old = pd.read_csv(K34 / "coverage" / ds / "coverage_intervals.csv")
        osum = pd.read_csv(K34 / "coverage" / ds / "coverage_summary.csv")
        m = recs(B / "results/k30/match" / ds, ["unit", "frame", "crops_v6S", "ncrops_v6S", "nguard_v6S"]).set_index(["unit", "frame"])
        gt = pd.read_pickle(B / "results/k30/gt" / f"{ds}.pkl"); gt["unit"] = gt.unit.astype(str); gt = gt.set_index(["unit", "frame"])
        allc, fs, nfs, ng, badlen = [], [], [], [], 0
        for u, f in zip(q.unit, q.frame):
            r = m.loc[(u, f)]; cr = list(r.crops_v6S); k = int(r.ncrops_v6S)
            badlen += int(len(cr) != k + int(r.nguard_v6S))
            box = np.array([gt.loc[(u, f)].dec_box], float)
            allc.append(bool(SP.centres_in(box, cr)[0])); fs.append(bool(SP.centres_in(box, cr[:k])[0])); nfs.append(k); ng.append(len(cr) - k)
        q["covered_all_recomputed"] = allc; q["covered_field_support"] = fs; q["n_field_support_crops"] = nfs; q["n_guard_crops"] = ng
        checks[ds] = {"frames": len(q), "len(crops) != ncrops + nguard": badlen,
                      "all-crop coverage != K34 covered": int((q.covered_all_recomputed != q.covered.astype(bool)).sum()),
                      "field+support covered but all-crop not (impossible)": int((q.covered_field_support & ~q.covered_all_recomputed).sum())}
        nall = int(osum.loc[(osum.stratification == "all"), "all_units"].iloc[0])
        units_all = np.array(sorted(set(recs(B / "results/k28/p3" / f"{ds}_test_instr", ["unit", "frame", "gt"]).query("gt == True").unit)))
        checks[ds]["units (instr GT) vs K34 all_units"] = f"{len(units_all)} vs {nall}"
        draws = np.random.default_rng(0).integers(0, len(units_all), (2000, len(units_all)))
        C = np.stack([np.bincount(x, minlength=len(units_all)) for x in draws]).astype(float)
        n = q.groupby("unit").size().reindex(units_all, fill_value=0).to_numpy(float)
        est = {}
        for lab, col in (("all crops (K34)", "covered_all_recomputed"), ("field + support crops", "covered_field_support")):
            num = q.groupby("unit")[col].sum().reindex(units_all, fill_value=0).to_numpy(float)
            dist = (C @ num) / (C @ n)
            est[lab] = (num, dist)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            ints.append({"dataset": ds, "region": lab, "required_frames": len(q), "covered": int(num.sum()), "coverage": num.sum() / n.sum(), "ci_lo": lo, "ci_hi": hi})
        o = old[(old.stratification == "all") & (old.metric == "coverage")].iloc[0]
        r = ints[-2]
        checks[ds]["K34 all-crop interval reproduced (max abs diff)"] = float(max(abs(r["coverage"] - o.estimate), abs(r["ci_lo"] - o.ci_lo), abs(r["ci_hi"] - o.ci_hi)))
        dd = est["all crops (K34)"][1] - est["field + support crops"][1]
        num_d = est["all crops (K34)"][0] - est["field + support crops"][0]
        lo, hi = np.percentile(dd, [2.5, 97.5])
        ints.append({"dataset": ds, "region": "difference (all - field + support) = covered only by guard crops", "required_frames": len(q),
                     "covered": int(num_d.sum()), "coverage": num_d.sum() / n.sum(), "ci_lo": lo, "ci_hi": hi})
        frames.append(q[["dataset", "unit", "frame", "covered", "covered_field_support", "n_field_support_crops", "n_guard_crops", "class", "range_m", "act_gt"]])
    pd.DataFrame(ints).to_csv(od / "k42_4_coverage.csv", index=False)
    pd.concat(frames).to_csv(od / "k42_4_coverage_frames.csv", index=False)
    (od / "k42_4_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    print(json.dumps(checks, indent=1)); print(pd.DataFrame(ints).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
