#!/usr/bin/env python
"""K39-4 (post-hoc; handoff/K39/PREREG.md): exactness cross-tabulation and single-intervention attribution.
  crosstab  K28 p3 test instr (GT frames), required action = act_gt; per dataset and refiner setting (none / v3a):
            (X = pi(F)) x (X = required) x (pi(F) = required) for X = DESP v6-S and X = native + C1, pi(F) = u960 + C1
  single    results/k39/vuw/<ds> (deployed procedure WITHOUT refiner): D0 = frames with a_v6S|none != a_wf960+C1|none;
            per intervention X alone: resolved (a_X = a_F) on D0, share of D0, newly disagreeing frames outside D0;
            frames resolved by no single intervention; the cumulative chain shares (stored ch_*) for reference
usage: k39_4_report.py <out_dir> [--no-single]"""
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
NATIVE = {"nusc": "wf1600", "av2": "wf2048", "ahd": "wf1920"}
SINGLE = {"sa": "a exact locality", "sb": "b F's geometry (history)", "sc1": "c1 closing probe 10 m/s",
          "sc2": "c2 closing probe 20 m/s", "sd": "d extended grid", "se": "e three probe templates", "sf": "f round cap 10"}
CHAIN = {"ch_a": "a", "ch_b": "a+b", "ch_c1": "a..c1", "ch_c2": "a..c2", "ch_d": "a..d", "ch_e": "a..e", "ch_f": "a..f"}


def recs(d):
    rr = []
    for f in sorted(glob.glob(str(d / "*.pkl"))):
        rr += pickle.load(open(f, "rb"))
    x = pd.DataFrame(rr)
    x["unit"] = x.unit.astype(str)
    return x


def crosstab(ds, d):
    out = []
    req = d.act_gt.to_numpy(int)
    for ref in ("none", "v3a"):
        F = d[f"a_wf960+C1|{ref}"].to_numpy(int)
        for lab, col in (("DESP v6-S", f"a_v6S|{ref}"), ("native + C1", f"a_{NATIVE[ds]}+C1|{ref}")):
            X = d[col].to_numpy(int)
            for xf in (True, False):
                for xr in (True, False):
                    for fr in (True, False):
                        m = ((X == F) == xf) & ((X == req) == xr) & ((F == req) == fr)
                        out.append({"dataset": ds, "refiner": ref, "pipeline": lab, "X_eq_piF": xf, "X_eq_required": xr,
                                    "piF_eq_required": fr, "frames": int(m.sum()), "share": float(m.mean()), "total": len(d)})
    return out


def single(ds):
    d = recs(B / "results/k39/vuw" / ds)
    d = d[d["gt"].astype(bool)].reset_index(drop=True)
    F, O = d["a_wf960+C1|none"].to_numpy(int), d["a_v6S|none"].to_numpy(int)
    D0 = O != F
    rows, anyres = [], np.zeros(len(d), bool)
    for k, lab in SINGLE.items():
        X = d[f"a_{k}|none"].to_numpy(int)
        res = D0 & (X == F)
        anyres |= res
        rows.append({"dataset": ds, "intervention": lab, "mode": "alone", "frames": len(d), "disagree_D0": int(D0.sum()),
                     "agreement_before": float((~D0).mean()), "agreement_after": float((X == F).mean()),
                     "resolved": int(res.sum()), "share_of_D0": float(res.sum() / max(D0.sum(), 1)),
                     "new_disagreements": int((~D0 & (X != F)).sum())})
    rows.append({"dataset": ds, "intervention": "resolved by no single intervention", "mode": "alone", "frames": len(d),
                 "disagree_D0": int(D0.sum()), "resolved": int((D0 & ~anyres).sum()),
                 "share_of_D0": float((D0 & ~anyres).sum() / max(D0.sum(), 1))})
    k28 = recs(B / "results/k28/p3" / f"{ds}_test_instr")
    k28 = k28[k28["gt"].astype(bool)]
    m = d[["unit", "frame"]].merge(k28, on=["unit", "frame"], validate="1:1")
    Fk, Ok = m["a_wf960+C1|none"].to_numpy(int), m["a_v6S|none"].to_numpy(int)
    D0k = Ok != Fk
    prev = D0k.copy()
    for k, lab in CHAIN.items():
        X = m[f"a_{k}"].to_numpy(int)
        still = X != Fk
        rows.append({"dataset": ds, "intervention": lab, "mode": "cumulative (stored chain)", "frames": len(m),
                     "disagree_D0": int(D0k.sum()), "agreement_after": float((X == Fk).mean()),
                     "resolved": int((D0k & ~still).sum()), "share_of_D0": float((D0k & ~still).sum() / max(D0k.sum(), 1)),
                     "step_share": float((prev & ~still).sum() / max(D0k.sum(), 1))})
        prev = D0k & still
    return rows


def main():
    od = Path(sys.argv[1])
    od.mkdir(parents=True, exist_ok=True)
    ct = []
    for ds in ("nusc", "av2", "ahd"):
        d = recs(B / "results/k28/p3" / f"{ds}_test_instr")
        ct += crosstab(ds, d[d["gt"].astype(bool)].reset_index(drop=True))
    ct = pd.DataFrame(ct)
    ct.to_csv(od / "k39_4_crosstab.csv", index=False)
    pd.set_option("display.width", 250)
    print(ct.to_string(index=False))
    if "--no-single" not in sys.argv:
        sg = pd.DataFrame([r for ds in ("nusc", "av2", "ahd") for r in single(ds)])
        sg.to_csv(od / "k39_4_single.csv", index=False)
        print(sg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
