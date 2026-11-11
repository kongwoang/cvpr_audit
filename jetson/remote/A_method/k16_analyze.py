#!/usr/bin/env python
"""K16 tables (post-hoc analysis; ours WITHOUT refiner, F = u960, AV2 with the K10 ego mask) from results/k16 (k16_attr).
  attribution_<ds>.csv  counterfactual chain: agreement with pi(F) after each cumulative intervention (unit bootstrap
                        2,000), share of the ORIGINAL disagreeing frames fixed (cumulative and by this step, net), the
                        loss-weighted version (sum |L_ours - L_F| over the original disagreeing frames, same split)
  residual_<ds>.csv     frames still disagreeing after the last step, with F's deciding object (range, lateral, class,
                        inside ours' verified region R), ours' / F's action, rounds
  rounds.csv            rounds 0/1/2/3 of ours and the share stopping at the cap without having converged
  coverage_<ds>.csv     F's objects whose demand reaches tau_{a+1} for ours' action a: share with box centre in R by
                        range bin, lateral bin and class; objects at 70-80 m and how often they decide pi(F)
  direction_<ds>.csv    frames where ours brakes less / more than F: decision loss of ours, F and cheap
usage: k16_analyze.py
"""
from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
OUT = B / "results" / "k16_tables"


def load(root, ds):
    rows = []
    for f in sorted(glob.glob(str(root / "results" / "k16" / f"{ds}__*.pkl"))):
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rounds_rows = []
    for ds, root in (("nusc", B), ("av2", B / "k10_masked")):
        d = load(root, ds)
        if not len(d):
            continue
        steps = ["ours", "a", "b"] + (["c"] if ds == "nusc" else ["c1", "c2"]) + ["d", "e", "f"]
        names = {"ours": "ours (real crops)", "a": "a. exact locality (F's objects in the crops)", "b": "b. + F's ttc history",
                 "c": "c. + closing probe 20 m/s", "c1": "c1. + closing probe 10 m/s", "c2": "c2. + closing probe 20 m/s",
                 "d": "d. + extended grid (8-80 m, +-9 m)", "e": "e. + probe templates (ped / car / truck)", "f": "f. + round cap 10"}
        units = d.unit.to_numpy(); uq = pd.unique(units); idx = [np.flatnonzero(units == u) for u in uq]
        rng = np.random.default_rng(0)
        bs = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
        aF = d.a_F.to_numpy()
        dis0 = d.a_ours.to_numpy() != aF
        w = np.abs(d.J_ours - d.J_F).to_numpy()
        rows, prev_fixed = [], np.zeros(len(d), bool)
        for s in steps:
            col = "a_ours" if s == "ours" else f"a_{s}"
            ag = d[col].to_numpy() == aF
            fixed = dis0 & ag
            r = {"dataset": ds, "units": len(uq), "frames": len(d), "step": names[s], "agreement": ag.mean(),
                 "ci_lo": np.percentile([ag[b].mean() for b in bs], 2.5), "ci_hi": np.percentile([ag[b].mean() for b in bs], 97.5),
                 "orig_disagreeing": int(dis0.sum()), "fixed_cum": int(fixed.sum()), "fixed_cum_share": fixed.sum() / dis0.sum(),
                 "fixed_this_step_share": (fixed.sum() - prev_fixed.sum()) / dis0.sum(),
                 "newly_broken_frames": int((~dis0 & ~ag).sum()),
                 "loss_fixed_cum_share": w[fixed].sum() / w[dis0].sum(),
                 "loss_fixed_this_step_share": (w[fixed].sum() - w[prev_fixed].sum()) / w[dis0].sum()}
            rows.append(r)
            prev_fixed = fixed
        last = "a_" + steps[-1]
        res_m = dis0 & (d[last].to_numpy() != aF)
        rows.append({"dataset": ds, "step": "g. residual (still disagreeing after f)", "orig_disagreeing": int(dis0.sum()),
                     "fixed_this_step_share": res_m.sum() / dis0.sum(), "loss_fixed_this_step_share": w[res_m].sum() / w[dis0].sum()})
        pd.DataFrame(rows).to_csv(OUT / f"attribution_{ds}.csv", index=False)
        # residual frames with F's deciding object
        rr = []
        for i in np.flatnonzero(res_m):
            r = d.iloc[i]
            drv = [c for c in r["cov"] if c["is_F_driver"]]
            dv = drv[0] if drv else {}
            rr.append({"unit": r.unit, "frame": r.frame, "v": round(r.v, 2), "a_ours": r.a_ours, "a_F": r.a_F, "a_after_f": r[last],
                       "act_gt": r.act_gt, "rounds": r.rounds, "converged": r.converged, "F_driver_in_cov": bool(drv),
                       "F_driver_range_m": dv.get("z"), "F_driver_lateral_m": dv.get("y"), "F_driver_cls": dv.get("cls"),
                       "F_driver_in_R": dv.get("inR"), "n_cov_objects": len(r["cov"]), "J_ours": r.J_ours, "J_F": r.J_F})
        pd.DataFrame(rr).to_csv(OUT / f"residual_{ds}.csv", index=False)
        # rounds
        rc = d.rounds.value_counts(normalize=True)
        rounds_rows.append({"dataset": ds, "frames": len(d), **{f"rounds_{k}": float(rc.get(k, 0)) for k in range(4)},
                            "stop_at_cap_not_converged": float(((d.rounds == 3) & ~d.converged.astype(bool)).mean()),
                            "stop_at_cap": float((d.rounds == 3).mean())})
        # coverage
        cv = pd.DataFrame([dict(c, unit=r.unit, frame=r.frame, a_ours=r.a_ours, a_F=r.a_F) for _, r in d.iterrows() for c in r["cov"]])
        crow = []
        if len(cv):
            cv["range_bin"] = pd.cut(cv.z, [-np.inf, 12.5, 70, 80, np.inf], labels=["<12.5 m", "12.5-70 m", "70-80 m", ">80 m"], right=False)
            cv["lat_bin"] = np.where(cv.y.abs() <= 6, "|y| <= 6 m", "|y| > 6 m")
            for key in ("range_bin", "lat_bin", "cls"):
                for k, g in cv.groupby(key, observed=True):
                    crow.append({"dataset": ds, "split": key, "bin": str(k), "objects": len(g), "share_in_R": g.inR.mean(),
                                 "F_deciding_objects": int(g.is_F_driver.sum()),
                                 "F_deciding_and_F_brakes_more": int((g.is_F_driver & (g.a_F > g.a_ours)).sum())})
            crow.append({"dataset": ds, "split": "all", "bin": "all", "objects": len(cv), "share_in_R": cv.inR.mean(),
                         "F_deciding_objects": int(cv.is_F_driver.sum()), "F_deciding_and_F_brakes_more": int((cv.is_F_driver & (cv.a_F > cv.a_ours)).sum())})
            crow.append({"dataset": ds, "split": "note", "bin": "classes are the detector's coarse classes (vehicle / person / cyclist); "
                         "car vs truck / bus is not separable", "objects": np.nan})
        pd.DataFrame(crow).to_csv(OUT / f"coverage_{ds}.csv", index=False)
        # direction
        drows = []
        for cls, m in (("ours brakes less than F", d.a_ours < d.a_F), ("ours brakes more than F", d.a_ours > d.a_F)):
            drows.append({"dataset": ds, "class": cls, "frames": int(m.sum()), "loss_ours": d.loc[m, "J_ours"].sum(),
                          "loss_F": d.loc[m, "J_F"].sum(), "loss_cheap": d.loc[m, "J_cheap"].sum()})
        pd.DataFrame(drows).to_csv(OUT / f"direction_{ds}.csv", index=False)
        pd.set_option("display.width", 250)
        print(pd.DataFrame(rows).round(4).to_string(index=False)); print(pd.DataFrame(crow).round(3).to_string(index=False))
        print(pd.DataFrame(drows).round(2).to_string(index=False))
    pd.DataFrame(rounds_rows).to_csv(OUT / "rounds.csv", index=False)
    print(pd.DataFrame(rounds_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
