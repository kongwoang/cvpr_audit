#!/usr/bin/env python
"""K30 analyses on STORED per-frame records (no re-run; post-hoc). Sources: K28 phase 3 instrumented run
results/k28/p3/<ds>_test_instr (k28_p3.py, setting base; AV2 with the ego mask) and the K30 ground-truth pass
results/k30/gt/<ds>.pkl (k30_gt.py: continuous a_gt, decisive object, yaw rate). Writes results/k30/tables/:
  v5a_loss_weights.csv   V5a: every pipeline re-scored with lam_brake x {0.5, 1, 2}, lam_collision x {0.5, 1, 2},
                         lam_jerk x {0, 1, 2} (full grid; stored actions, a_gt from the GT pass, prev action = the draft's,
                         exactly as the stored costs): G, v6-S - pipeline with paired unit bootstrap (2,000, seed 0), rank
  v5a_check.csv          the base setting reproduces every stored cost (max |dJ|)
  v3_taxonomy.csv        V3: frames where pi(X) != pi(F), F = u960 + C1 (no refiner): direction, who equals the annotated
                         required action, the safety-critical subset (required >= DECELERATE and X lower than F), shares of
                         frames and of X's decision loss; X = v6-S (deployed, refiner v3a), v6-S without refiner, cheap,
                         native + C1 (+ v3a / without)
  v3_top_frames.csv      the 20 most costly safety-critical frames of v6-S (deployed) per dataset
usage: k30_stored.py
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
OUT = B / "results" / "k30" / "tables"
NWS = {"nusc": 1600, "av2": 2048, "ahd": 1920}
DSL = {"nusc": "nuScenes new test", "av2": "AV2 val (ego mask)", "ahd": "Argoverse-HD val"}


def pipes(ds):
    n = NWS[ds]
    return {"cheap": "a_cheap", "v6-S": "a_v6S|v3a", "u960 + C1 + v3a": "a_wf960+C1|v3a", "native + C1 + v3a": f"a_wf{n}+C1|v3a",
            "hybrid tsp8@S + C1 + v3a": "a_tsp8@S+C1|v3a", "temporal-crit tcr0@S + C1 + v3a": "a_tcr0@S+C1|v3a",
            "temporal-crit tcr3@S + C1 + v3a": "a_tcr3@S+C1|v3a", "uncertainty unc2@S + C1 + v3a": "a_unc2@S+C1|v3a",
            "uncertainty unc4@S + C1 + v3a": "a_unc4@S+C1|v3a", "ego corridor cor0@S + C1 + v3a": "a_cor0@S+C1|v3a",
            "ego corridor cor8@S + C1 + v3a": "a_cor8@S+C1|v3a"}


def load(ds):
    rows = []
    for f in sorted(glob.glob(str(B / "results" / "k28" / "p3" / f"{ds}_test_instr" / "*.pkl"))):
        rows += pickle.load(open(f, "rb"))
    d = pd.DataFrame(rows)
    d["prev_cheap"] = d.groupby("unit")["a_cheap"].shift(1)        # the stored costs' jerk term (k28_p3: prev = draft action)
    gt = pd.read_pickle(B / "results" / "k30" / "gt" / f"{ds}.pkl")
    d = d.merge(gt[["unit", "frame", "a_gt", "dec_range", "dec_lat", "dec_class", "dec_cls", "yaw_rate"]], on=["unit", "frame"], how="left")
    d = d[d["gt"].astype(bool)].reset_index(drop=True)
    return d


def cost_vec(a, a_gt, prev, pp, cp):
    out = np.empty(len(a))
    for i, (x, g, p) in enumerate(zip(a, a_gt, prev)):
        out[i] = P.decision_cost(int(x), float(g), None if pd.isna(p) else int(p), pp, cp)["J"]
    return out


def unit_boot(units, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    uq, inv = np.unique(units, return_inverse=True)
    return uq, inv, [rng.integers(0, len(uq), len(uq)) for _ in range(n)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pp, cp0 = P.PlannerParams(), P.CostParams()
    grid = [(b, c, j) for b in (0.5, 1.0, 2.0) for c in (0.5, 1.0, 2.0) for j in (0.0, 1.0, 2.0)]
    v5, chk, tax, top = [], [], [], []
    for ds in ("nusc", "av2", "ahd"):
        d = load(ds)
        Pm = pipes(ds)
        uq, inv, bs = unit_boot(d.unit.to_numpy())
        # --- V5a
        stored = {lab: d["J" + c[1:]].to_numpy(float) for lab, c in Pm.items()}
        for (b, c, j) in grid:
            cp = replace(cp0, lam_brake=cp0.lam_brake * b, lam_collision=cp0.lam_collision * c, lam_jerk=cp0.lam_jerk * j)
            J = {lab: cost_vec(d[col].to_numpy(int), d.a_gt.to_numpy(float), d.prev_cheap.to_numpy(), pp, cp) for lab, col in Pm.items()}
            if (b, c, j) == (1.0, 1.0, 1.0):
                for lab in Pm:
                    chk.append({"dataset": ds, "pipeline": lab, "frames": len(d), "max_abs_dJ": float(np.max(np.abs(J[lab] - stored[lab])))})
            us = {lab: np.bincount(inv, J[lab], len(uq)) for lab in J}
            L = us["cheap"]
            Gs = {lab: float((L.sum() - us[lab].sum()) / L.sum()) for lab in J}
            order = sorted([k for k in Gs if k != "cheap"], key=lambda k: -Gs[k])
            for lab in J:
                if lab == "cheap":
                    continue
                r = {"dataset": ds, "lam_brake_x": b, "lam_collision_x": c, "lam_jerk_x": j, "pipeline": lab, "G": Gs[lab],
                     "rank_of_pipeline": order.index(lab) + 1, "pipelines_ranked": len(order)}
                if lab != "v6-S":
                    dd = [((us[lab][s].sum() - us["v6-S"][s].sum()) / L[s].sum()) for s in bs]
                    r.update({"v6S_minus": Gs["v6-S"] - Gs[lab], "ci_lo": float(np.percentile(dd, 2.5)), "ci_hi": float(np.percentile(dd, 97.5))})
                v5.append(r)
        # --- V3
        nat = NWS[ds]
        aF = d["a_wf960+C1|none"].to_numpy(int)
        g = d.act_gt.to_numpy(int)
        X = {"v6-S (deployed, + v3a)": ("a_v6S|v3a", "J_v6S|v3a"), "v6-S without refiner": ("a_v6S|none", "J_v6S|none"),
             "cheap": ("a_cheap", "J_cheap"), "native + C1 + v3a": (f"a_wf{nat}+C1|v3a", f"J_wf{nat}+C1|v3a"),
             "native + C1 (no refiner)": (f"a_wf{nat}+C1|none", f"J_wf{nat}+C1|none")}
        JF = d["J_wf960+C1|none"].to_numpy(float)
        for lab, (ac, jc) in X.items():
            a = d[ac].to_numpy(int); J = d[jc].to_numpy(float)
            dis = a != aF
            sc = (g >= P.DECELERATE) & (a < aF)
            tax.append({"dataset": ds, "X": lab, "frames": len(d), "disagree_share": float(dis.mean()),
                        "X_higher_share": float((a > aF).mean()), "X_lower_share": float((a < aF).mean()),
                        "of_disagreements_X_equals_required": float(((a == g) & dis).sum() / max(dis.sum(), 1)),
                        "of_disagreements_F_equals_required": float(((aF == g) & dis).sum() / max(dis.sum(), 1)),
                        "of_disagreements_neither": float(((a != g) & (aF != g) & dis).sum() / max(dis.sum(), 1)),
                        "safety_critical_frames": int(sc.sum()), "safety_critical_share": float(sc.mean()),
                        "share_of_X_loss_on_disagreements": float(J[dis].sum() / J.sum()),
                        "share_of_X_loss_on_safety_critical": float(J[sc].sum() / J.sum()),
                        "X_minus_F_loss_on_disagreements_per_1k_frames": float((J[dis].sum() - JF[dis].sum()) / len(d) * 1000),
                        "X_minus_F_loss_on_safety_critical_per_1k_frames": float((J[sc].sum() - JF[sc].sum()) / len(d) * 1000),
                        "X_below_required_share": float((a < g).mean()), "F_below_required_share": float((aF < g).mean())})
        a6 = d["a_v6S|v3a"].to_numpy(int)
        sc6 = (g >= P.DECELERATE) & (a6 < aF)
        t = d[sc6].copy()
        t["J_v6S"] = t["J_v6S|v3a"]; t["J_F"] = t["J_wf960+C1|none"]
        t = t.sort_values("J_v6S", ascending=False).head(20)
        for rank, (_, r) in enumerate(t.iterrows(), 1):
            top.append({"dataset": ds, "rank": rank, "unit": r.unit, "frame": int(r.frame), "v": r.v, "act_gt": int(r.act_gt),
                        "a_gt": r.a_gt, "a_v6S": int(r["a_v6S|v3a"]), "a_F": int(r["a_wf960+C1|none"]), "J_v6S": r.J_v6S,
                        "J_F": r.J_F, "dec_range": r.dec_range, "dec_lat": r.dec_lat, "dec_class": r.dec_class,
                        "found_v6S_relevant": r.get("found_v6S"), "a_v6S_no_refiner": int(r["a_v6S|none"])})
        print(ds, "done", len(d), flush=True)
    pd.DataFrame(v5).to_csv(OUT / "v5a_loss_weights.csv", index=False)
    pd.DataFrame(chk).to_csv(OUT / "v5a_check.csv", index=False)
    pd.DataFrame(tax).to_csv(OUT / "v3_taxonomy.csv", index=False)
    pd.DataFrame(top).to_csv(OUT / "v3_top_frames.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(pd.DataFrame(chk).to_string(index=False))
    print(pd.DataFrame(tax).round(3).to_string(index=False))
    print("K30 STORED DONE")


if __name__ == "__main__":
    main()
