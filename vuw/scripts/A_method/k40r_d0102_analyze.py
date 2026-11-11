#!/usr/bin/env python
"""K40R analysis of the authorized detector jobs D01 (K40-3, mean-budget frame-level policies with own history) and
D02 (K40-6, action-independent speed tubes tau1 / tau2 under the nominal, K39-literal strict and observed-strict caps).
Rules: K40R/PREREG.md (K40-3, K40-6). Evaluated frames = gt frames (36,005); paired unit bootstrap of the K40 release
(analysis/common.Bootstrap: sorted units, default_rng(0), 2,000 draws, percentile 95%), Delta = DESP - row.
Frozen parity: J_v6S|v3a of every job frame must equal the released records (the paper's tab:main G).
Repair rate (K39-6): agreement of the no-refiner action with pi(F) = u960 + C1 (no refiner) on the frames where the
cheap draft differs from pi(F). Natural-miss recovery: decision-relevant objects not found by the cheap draft and
found by the row (found_* per relevant object).
K40-3 family: the historical 12 x 3 tab:main capped comparisons with AdaScale / BAR / Router-G / Chanakya replaced by
their D01 mean-budget rows (the other 8 methods keep their per-frame-cap records); raw CI count and Holm on the
bootstrap p-values p = 2 min(P*(D<=0), P*(D>=0)) (K39-3).
usage: k40r_d0102_analyze.py <out_dir>   (on VUW, after source env.sh)"""
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path("/vol/dl-nguyenb5-solar/users/hoangpc/CVPR")
REL = B / "tmp/desp_k40"
sys.path.insert(0, str(REL))
from analysis.common import Bootstrap, records  # noqa: E402  (read-only use of the K40 release helpers)
import analysis.tab_main as M  # noqa: E402

NAME = {"nusc": "nuscenes", "av2": "av2", "ahd": "argoverse_hd"}
RECS = REL / "artifacts/records"


def load(job, ds):
    rows = []
    for f in sorted(glob.glob(str(B / "results/k40r" / job / ds / "*.pkl"))):
        rows += pickle.load(open(f, "rb"))
    d = pd.DataFrame(rows)
    d["unit"] = d.unit.astype(str)
    return d[d["gt"].astype(bool)].sort_values(["unit", "frame"]).reset_index(drop=True)


def G(d, col):
    return float(1 - d[col].astype(float).sum() / d.J_cheap.astype(float).sum())


def draws(bt, L, Ja, Jb):
    l, a, b = bt.sums(L), bt.sums(Ja), bt.sums(Jb)
    return (b[bt.idx].sum(1) - a[bt.idx].sum(1)) / l[bt.idx].sum(1)


def holm(p, alpha=0.05):
    p = np.sort(np.asarray(p))
    k = 0
    while k < len(p) and p[k] <= alpha / (len(p) - k):
        k += 1
    return k


def recovery(d, row):
    hit = n = 0
    for fc, fr in zip(d["found_cheap"], d[f"found_{row}"]):
        if isinstance(fc, (list, np.ndarray)) and len(fc):
            fc, fr = np.asarray(fc, bool), np.asarray(fr, bool)
            n += int((~fc).sum()); hit += int((fr & ~fc).sum())
    return hit / n if n else np.nan, n


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    checks, d01, d02, fam = {}, [], [], []
    for ds, nm in NAME.items():
        R = records(nm, RECS)
        for job, rows in (("mean_history", ["AdaScale", "BAR", "Router-G", "Chanakya"]),
                          ("tubes", [f"speed_{t}@{c}" for t in ("tau1", "tau2") for c in ("nominal", "k39_strict", "strict_observed")])):
            d = load(job, ds)
            m = d[["unit", "frame", "J_v6S|v3a", "J_cheap"]].merge(R[["unit", "frame", "J_v6S|v3a", "J_cheap", "a_wf960+C1|none"]],
                                                                    on=["unit", "frame"], how="left", suffixes=("", "@rel"), validate="1:1")
            checks[f"{job}/{ds}"] = {"frames": len(d), "released_frames": len(R),
                                     "max_abs_dJ_v6S": float(np.nanmax(np.abs(m["J_v6S|v3a"] - m["J_v6S|v3a@rel"]))),
                                     "max_abs_dJ_cheap": float(np.nanmax(np.abs(m["J_cheap"] - m["J_cheap@rel"]))),
                                     "G_v6S": G(d, "J_v6S|v3a")}
            assert len(d) == len(R) and checks[f"{job}/{ds}"]["max_abs_dJ_v6S"] < 1e-9, checks[f"{job}/{ds}"]
            bt = Bootstrap(d.unit)
            L, Jd = d.J_cheap.to_numpy(float), d["J_v6S|v3a"].to_numpy(float)
            F = m["a_wf960+C1|none"].to_numpy(int)
            rep = d.a_cheap.to_numpy(int) != F
            e6 = d["a_v6S|none"].to_numpy(int) == F
            rec6 = recovery(d, "v6S")
            for row in rows:
                for ref in ("v3a", "none"):   # control under the same refiner (fix 2026-11-02; was v6-S|v3a for both)
                    x = draws(bt, L, d[f"J_v6S|{ref}"].to_numpy(float), d[f"J_{row}|{ref}"].to_numpy(float))
                    eq = d[f"a_{row}|none"].to_numpy(int) == F
                    sd = bt.share_delta(e6, eq, rep)
                    r = {"dataset": ds, "row": row, "refiner": ref, "units": len(bt.units), "frames": len(d),
                         "G": G(d, f"J_{row}|{ref}"), "G_v6S": G(d, f"J_v6S|{ref}"),
                         "px_mean": float(d[f"px_{row}"].mean()), "px_v6S": float(d.px_v6S.mean()),
                         "px_ratio": float(d[f"px_{row}"].mean() / d.px_v6S.mean()),
                         "delta": G(d, f"J_v6S|{ref}") - G(d, f"J_{row}|{ref}"),
                         "ci_lo": float(np.percentile(x, 2.5)), "ci_hi": float(np.percentile(x, 97.5)),
                         "p_boot": float(2 * min((x <= 0).mean(), (x >= 0).mean())),
                         "repair_row": float(eq[rep].mean()), "repair_v6S": float(e6[rep].mean()),
                         "repair_delta": sd[0], "repair_ci_lo": sd[1], "repair_ci_hi": sd[2], "repair_frames": int(rep.sum())}
                    if f"found_{row}" in d:
                        rr, n = recovery(d, row)
                        r.update({"natural_recovery_row": rr, "natural_recovery_v6S": rec6[0], "natural_misses": n})
                    (d01 if job == "mean_history" else d02).append(r)
            if job == "mean_history":
                sub = {"AdaScale": "J_ada+1t1280@cap|v3a", "BAR": "J_bar1280_0.5@cap", "Router-G": "J_dvr1280_0.5@cap", "Chanakya": "J_ch128B_csc@cap"}
                btR = Bootstrap(R.unit)
                LR, JdR = R.J_cheap.to_numpy(float), R["J_v6S|v3a"].to_numpy(float)
                for lab, j, p, jc in M.rows(nm)[3:15]:
                    key = next((k for k, v in sub.items() if v == jc), None)
                    for fam_name in ("historical_per_frame_cap", "mean_budget_substituted"):
                        if fam_name == "mean_budget_substituted" and key:
                            x = draws(bt, L, Jd, d[f"J_{key}|v3a"].to_numpy(float)); src = f"D01 {key}"
                        else:
                            x = draws(btR, LR, JdR, R[jc].to_numpy(float)); src = jc
                        fam.append({"family": fam_name, "dataset": ds, "row": lab, "source": src,
                                    "ci_lo": float(np.percentile(x, 2.5)), "ci_hi": float(np.percentile(x, 97.5)),
                                    "p_boot": float(2 * min((x <= 0).mean(), (x >= 0).mean()))})
    f = pd.DataFrame(fam)
    summ = []
    for name, g in f.groupby("family"):
        summ.append({"family": name, "comparisons": len(g), "desp_sig_better_ci": int((g.ci_lo > 0).sum()),
                     "row_sig_better_ci": int((g.ci_hi < 0).sum()), "holm_rejections": holm(g.p_boot)})
    pd.DataFrame(d01).to_csv(od / "k40_3_mean_budget.csv", index=False)
    pd.DataFrame(d02).to_csv(od / "k40_6_speed_tubes.csv", index=False)
    f.to_csv(od / "k40_3_family.csv", index=False)
    pd.DataFrame(summ).to_csv(od / "k40_3_family_summary.csv", index=False)
    (od / "k40r_d0102_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    pd.set_option("display.width", 250)
    for nm_, t in (("K40-3", pd.DataFrame(d01)), ("K40-6", pd.DataFrame(d02))):
        t = t[t.refiner == "v3a"]
        print(nm_); print(t[["dataset", "row", "G", "G_v6S", "px_ratio", "delta", "ci_lo", "ci_hi", "repair_row", "repair_v6S", "repair_ci_lo", "repair_ci_hi"]
                           + [c for c in ("natural_recovery_row", "natural_recovery_v6S") if c in t]].round(3).to_string(index=False))
    print(pd.DataFrame(summ).to_string(index=False)); print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
