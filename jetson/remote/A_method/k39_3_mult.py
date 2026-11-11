#!/usr/bin/env python
"""K39-3 (post-hoc, CPU; handoff/K39/PREREG.md): multiplicity sensitivity of the paired-bootstrap comparisons.
  family a      Tab. 1 @cap, 12 methods x 3 datasets (k33_report.table1's v6-S - row@cap, its own resamples:
                sorted units, default_rng(0) integers, 2,000); a_all = the same with the two reference rows (14 x 3)
  family b      ablation field only / support only / pivots (refiner v3a) x 3 datasets (k28_p3_p0.abl's v6-S - row,
                k28_phase0.Boot resamples)
p = 2 min(P*(D <= 0), P*(D >= 0)) (capped at 1) and the +1-corrected 2 min(#{D* <= 0} + 1, #{D* >= 0} + 1) / 2,001;
Holm-Bonferroni and Benjamini-Hochberg within each family at 0.05. Checks: point and CI reproduce table1_v7.csv /
ablation_v6.csv.
usage: k39_3_mult.py <table1_v7.csv> <ablation_v6.csv> <out_dir>
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])


def mod(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


T1 = mod("k33r", B / "scripts/A_method/k33_report.py")
AB = mod("k28p3p0", B / "scripts/A_method/k28_p3_p0.py")
PRIMARY_A = ["ada+1t1280", "sahi640o1s10", "fovi1280b64a2", "fovs1280b64a1", "bar1280_0.5", "dvr1280_0.5", "ufp20s512",
             "ch128B_csc", "ana05hinfo", "ana05hinfi", "cf40q20o", "cf40q20i"]
ALL_A = PRIMARY_A + ["ch128B_reinforce", "ana25hinfo"]
ABL = {"support only (no field)": "so6S", "field only (no support)": "fo6S", "pivots instead of support": "pv6S"}


def pvals(dd):
    dd = np.asarray(dd, float)
    n = len(dd)
    le, ge = int((dd <= 0).sum()), int((dd >= 0).sum())
    return min(1.0, 2 * min(le, ge) / n), min(1.0, 2 * min(le + 1, ge + 1) / (n + 1))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p, kind="stable")
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p, kind="stable")
    m = len(p)
    adj = np.empty(m)
    run = 1.0
    for k in range(m - 1, -1, -1):
        i = o[k]
        run = min(run, m * p[i] / (k + 1))
        adj[i] = min(1.0, run)
    return adj


def family_a(t1):
    out, chk = [], {}
    for ds in ("nusc", "av2", "ahd"):
        d = T1.frames(ds)
        units = np.array(sorted(d.unit.unique()))
        idx = T1.boot_idx(units)
        Jv = T1.cols(d, "v6S")[0]
        for row in ALL_A:
            Jk = T1.cols(d, row, cap=True)[0]
            g = d.assign(_j=Jk, _v=Jv).groupby("unit")[["J_cheap", "_j", "_v"]].sum().reindex(units)
            c, j, v = (g[x].to_numpy() for x in ("J_cheap", "_j", "_v"))
            pt = (j.sum() - v.sum()) / c.sum()
            dd = (j[idx].sum(1) - v[idx].sum(1)) / c[idx].sum(1)
            lo, hi = np.percentile(dd, [2.5, 97.5])
            ref = t1[(t1.dataset == ds) & (t1.row == row)].iloc[0]
            chk[f"a {ds} {row}"] = float(max(abs(pt - ref.delta_cap), abs(lo - ref.delta_cap_lo), abs(hi - ref.delta_cap_hi)))
            p, p1 = pvals(dd)
            out.append({"family": "a", "method": row, "label": ref.family, "dataset": ds, "delta": pt, "ci_lo": lo, "ci_hi": hi,
                        "p_raw": p, "p_raw_plus1": p1, "primary": row in PRIMARY_A})
    return out, chk


def family_b(ab):
    out, chk = [], {}
    for ds in ("nusc", "av2", "ahd"):
        d = AB.k24v6(ds, p3=True)
        L = d.J_cheap.to_numpy(float)
        bt = AB.Boot(d.unit.to_numpy())
        Jp = d["J_v6S|v3a"].to_numpy(float)
        for lab, k in ABL.items():
            Jm = d[f"J_{k}|v3a"].to_numpy(float)
            dd = np.array([(Jm[b] - Jp[b]).sum() / L[b].sum() for b in bt.bs])
            pt = (Jm - Jp).sum() / L.sum()
            lo, hi = np.percentile(dd, [2.5, 97.5])
            ref = ab[(ab.dataset == ds) & (ab.row == lab) & (ab.refiner == "v3a")].iloc[0]
            chk[f"b {ds} {k}"] = float(max(abs(pt - ref.v6S_minus), abs(lo - ref.lo), abs(hi - ref.hi)))
            p, p1 = pvals(dd)
            out.append({"family": "b", "method": k, "label": lab, "dataset": ds, "delta": pt, "ci_lo": lo, "ci_hi": hi,
                        "p_raw": p, "p_raw_plus1": p1, "primary": True})
    return out, chk


def main():
    t1, ab, od = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2]), Path(sys.argv[3])
    od.mkdir(parents=True, exist_ok=True)
    ra, ca = family_a(t1)
    rb, cb = family_b(ab)
    df = pd.DataFrame(ra + rb)
    fams = {"a (12 x 3, primary)": (df.family == "a") & df.primary, "a_all (14 x 3)": df.family == "a", "b (3 x 3)": df.family == "b"}
    res = []
    for fn, m in fams.items():
        x = df[m].copy()
        x["family"] = fn
        for sfx, col in (("", "p_raw"), ("_plus1", "p_raw_plus1")):
            x[f"p_holm{sfx}"] = holm(x[col].to_numpy())
            x[f"p_bh{sfx}"] = bh(x[col].to_numpy())
        x["sig_ci"] = (x.ci_lo > 0) | (x.ci_hi < 0)
        res.append(x)
    out = pd.concat(res, ignore_index=True)
    out.to_csv(od / "k39_3_pvalues.csv", index=False)
    summ = []
    for fn, x in out.groupby("family", sort=False):
        summ.append({"family": fn, "comparisons": len(x), "ci_excludes_0": int(x.sig_ci.sum()),
                     "p_raw_lt_05": int((x.p_raw < 0.05).sum()), "holm_lt_05": int((x.p_holm < 0.05).sum()),
                     "bh_lt_05": int((x.p_bh < 0.05).sum()), "holm_lt_05_plus1": int((x.p_holm_plus1 < 0.05).sum()),
                     "bh_lt_05_plus1": int((x.p_bh_plus1 < 0.05).sum()),
                     "positive_sig_holm": int(((x.p_holm < 0.05) & (x.delta > 0)).sum()),
                     "negative_sig_holm": int(((x.p_holm < 0.05) & (x.delta < 0)).sum())})
    pd.DataFrame(summ).to_csv(od / "k39_3_summary.csv", index=False)
    chk = {**ca, **cb}
    (od / "k39_3_checks.json").write_text(json.dumps({"max_abs_diff_vs_published": max(chk.values()), "per_cell": chk}, indent=1))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summ).to_string(index=False))
    print(out[["family", "method", "dataset", "delta", "ci_lo", "ci_hi", "p_raw", "p_holm", "p_bh"]].round(4).to_string(index=False))
    print("max |diff| vs published:", max(chk.values()))


if __name__ == "__main__":
    main()
