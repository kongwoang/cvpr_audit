#!/usr/bin/env python
"""K30 V5 for the best published method (post-hoc, CPU): nuScenes SAHI scale 1.0, AV2 (ego mask) FOVEA KDE_I 960, re-run
by k30_sahi_nusc.py / k30_fovea_av2.py with the action, the lifted geometry and a_gt stored per frame. Prior works keep their
official settings (no refiner, no C1), so their geometry does not depend on the consumer and the actions are recomputed:
  V5a  loss weights (full 3 x 3 x 3 grid of k30_stored.py): prior and v6-S / cheap re-scored from the stored actions
       (v6-S: K28 phase 3 instr records, a_gt from the K30 GT pass, previous draft action)
  V5b  consumer parameters (k30_eval.py part cons settings): prior action = rap.planner on its geometry with the setting's
       PlannerParams; required action, previous draft action and the v6-S / cheap costs from the cons records of that setting
  V5c  probe assumptions change only v6-S: the prior's G is its base G (reported for completeness)
Paired unit bootstrap 2,000, seed 0. results/k30/tables/v5_prior.csv
"""
from __future__ import annotations

import glob
import os
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P

B = Path(os.environ.get("B_REAL", os.environ["B"]))
R30 = B / "results" / "k30"
PRIOR = {"nusc": ("nusc_sahi_s10.pkl", "seq", "sahi_s10", "SAHI 640-px slices (scale 1.0)"),
         "av2": ("av2_fovea_kdei960.pkl", "log", "fovea_kdei960", "FOVEA KDE_I 960")}
CONS = {"tau0.8": dict(thr_decel=0.8, thr_hard=2.8), "tau1.2": dict(thr_decel=1.2, thr_hard=4.2),
        "tr0.3": dict(reaction_t=0.3), "tr0.6": dict(reaction_t=0.6)}


def boot(units, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    uq, inv = np.unique(np.asarray(units), return_inverse=True)
    return uq, inv, [rng.integers(0, len(uq), len(uq)) for _ in range(n)]


def gdiff(inv, nu, bs, L, Ja, Jb):
    l, a, b = (np.bincount(inv, np.asarray(x, float), nu) for x in (L, Ja, Jb))
    v = [(b[s].sum() - a[s].sum()) / l[s].sum() for s in bs]
    return float((b.sum() - a.sum()) / l.sum()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    pp0, cp0 = P.PlannerParams(), P.CostParams()
    out = []
    for ds, (fn, ucol, key, lab) in PRIOR.items():
        f = R30 / "prior" / fn
        if not f.exists():
            print("missing", f); continue
        pr = pd.read_pickle(f).rename(columns={ucol: "unit"})
        # --- V5a: v6-S / cheap from the K28 instr records
        rows = []
        for g in sorted(glob.glob(str(B / "results" / "k28" / "p3" / f"{ds}_test_instr" / "*.pkl"))):
            rows += pickle.load(open(g, "rb"))
        d = pd.DataFrame(rows)
        d["prev_cheap"] = d.groupby("unit")["a_cheap"].shift(1)
        gt = pd.read_pickle(R30 / "gt" / f"{ds}.pkl")[["unit", "frame", "a_gt"]]
        d = d.merge(gt, on=["unit", "frame"]).merge(pr[["unit", "frame", f"a_{key}", f"J_{key}", "J_c", "a_gt", "prev_cheap", f"geo_{key}"]],
                                                     on=["unit", "frame"], suffixes=("", "_prior"))
        d = d[d["gt"].astype(bool)].reset_index(drop=True)
        chk = {"J_c_equal": float(np.isclose(d.J_c, d.J_cheap).mean()), "a_gt_equal": float(np.isclose(d.a_gt, d.a_gt_prior).mean())}
        uq, inv, bs = boot(d.unit.to_numpy())
        for b in (0.5, 1.0, 2.0):
            for c in (0.5, 1.0, 2.0):
                for j in (0.0, 1.0, 2.0):
                    cp = replace(cp0, lam_brake=cp0.lam_brake * b, lam_collision=cp0.lam_collision * c, lam_jerk=cp0.lam_jerk * j)
                    pv = [None if pd.isna(x) else int(x) for x in d.prev_cheap]
                    Jc = np.array([P.decision_cost(int(a), g_, p_, pp0, cp)["J"] for a, g_, p_ in zip(d.a_cheap, d.a_gt, pv)])
                    J6 = np.array([P.decision_cost(int(a), g_, p_, pp0, cp)["J"] for a, g_, p_ in zip(d["a_v6S|v3a"], d.a_gt, pv)])
                    Jp = np.array([P.decision_cost(int(a), g_, p_, pp0, cp)["J"] for a, g_, p_ in zip(d[f"a_{key}"], d.a_gt, pv)])
                    pt, lo, hi = gdiff(inv, len(uq), bs, Jc, J6, Jp)
                    out.append({"dataset": ds, "part": "V5a", "setting": f"lam_brake x{b} / lam_collision x{c} / lam_jerk x{j}",
                                "prior": lab, "G_prior": float((Jc.sum() - Jp.sum()) / Jc.sum()), "G_v6S": float((Jc.sum() - J6.sum()) / Jc.sum()),
                                "v6S_minus_prior": pt, "ci_lo": lo, "ci_hi": hi, "frames": len(d), **chk})
        # --- V5b: consumer parameters from the cons records
        cf = sorted(glob.glob(str(R30 / "cons" / ds / "*.pkl")))
        if not cf:
            print(ds, "cons records missing"); continue
        rows = []
        for g in cf:
            rows += pickle.load(open(g, "rb"))
        c_ = pd.DataFrame(rows)
        c_ = c_[c_["gt"].astype(bool)].merge(pr[["unit", "frame", f"geo_{key}", "v"]], on=["unit", "frame"], suffixes=("", "_prior"))
        uq, inv, bs = boot(c_.unit.to_numpy())
        for sn, kw in [("base", {})] + list(CONS.items()):
            pp = replace(pp0, **kw)
            sfx = "" if sn == "base" else f"@{sn}"
            prev = c_.groupby("unit")[f"a_cheap{sfx}"].shift(1)
            a_p = [P.discrete_action(P.required_decel(*geo, v, pp)[0], pp) for geo, v in zip(c_[f"geo_{key}"], c_.v)]
            Jp = np.array([P.decision_cost(a, g_, None if pd.isna(p_) else int(p_), pp, cp0)["J"] for a, g_, p_ in zip(a_p, c_[f"a_gt{sfx}"], prev)])
            Jc, J6 = c_[f"J_cheap{sfx}"].to_numpy(float), c_[f"J_v6S|v3a{sfx}"].to_numpy(float)
            pt, lo, hi = gdiff(inv, len(uq), bs, Jc, J6, Jp)
            out.append({"dataset": ds, "part": "V5b", "setting": sn, "prior": lab, "G_prior": float((Jc.sum() - Jp.sum()) / Jc.sum()),
                        "G_v6S": float((Jc.sum() - J6.sum()) / Jc.sum()), "v6S_minus_prior": pt, "ci_lo": lo, "ci_hi": hi, "frames": len(c_)})
    od = R30 / "tables"
    od.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(od / "v5_prior.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(out).round(4).to_string(index=False))
    print("K30 V5 PRIOR DONE")


if __name__ == "__main__":
    main()
