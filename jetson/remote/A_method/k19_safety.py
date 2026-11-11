#!/usr/bin/env python
"""K19 — safety-relevant quantities from the existing per-frame results (post-hoc analysis; no new inference).
Frames: the K14 frames (nuScenes new test; AV2 val with the K10 ego mask; Argoverse-HD val frames with 3D labels).
Per frame and pipeline the action is taken from the per-frame tables; the loss terms are recomputed with the consumer's
cost (rap.planner.decision_cost: a_gt from the annotated scene, previous action = the draft's, as every table):
insufficient braking lam_risk * shortfall^2, excessive braking lam_brake * excess^2, command changes lam_jerk * |jerk|,
collisions lam_collision * 1[shortfall >= collision_a]. Their sum equals the tables' J (checked).
Sources: K14 (cheap, u960, native, ours +- refiner, uncertainty, temporal_crit, a_gt), instr_eval / k13_ahd_ablation
(u640 [nuScenes, AV2], support only, field only, both with refiner as in the ablation), k10_hood (u640 / u1280 on
Argoverse-HD, u1280 on nuScenes), the K10 masked native table (AV2 u1280: action recovered from its cost, unique
given a_gt and the previous action), routers (k18_routers at 1x ours' tensor px: best Budget-Adaptive Routing and the
oracle; action = u960's on routed frames, the draft's otherwise; not available on Argoverse-HD), k19_gt (corridor
objects, minimum ttc / range in the corridor).
Intervals: paired unit bootstrap, 2,000 resamples, seed 0, of ours minus each pipeline.
usage: k19_safety.py  ->  results/k19/safety_{nusc,av2,ahd}.csv (+ _scenes, _collision)
"""
from __future__ import annotations

import glob
import importlib.util
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from rap import planner as P

B = Path(os.environ["B"])
S = B / "k10_masked"
OUT = B / "results" / "k19"
pp, cp = P.PlannerParams(), P.CostParams()
EVENTS = {"missed hard brake": (2, 0), "delayed hard brake": (2, 1), "missed deceleration": (1, 0),
          "unnecessary deceleration": (0, 1), "unnecessary hard brake": (0, 2)}


def load(pat):
    rows = []
    for f in sorted(glob.glob(str(pat))):
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows)


def terms(a, a_gt, prev):
    r = P.decision_cost(int(a), float(a_gt), None if prev is None or (isinstance(prev, float) and np.isnan(prev)) else int(prev), pp, cp)
    return (cp.lam_risk * r["shortfall"] ** 2, cp.lam_brake * r["excess"] ** 2, cp.lam_jerk * r["jerk"],
            cp.lam_collision * r["collision"], r["J"])


def invert(J, a_gt, prev):
    c = [a for a in range(3) if abs(terms(a, a_gt, prev)[4] - J) < 1e-6]
    return (c[0], len(c)) if c else (-1, 0)


def router_actions(ds, d):
    """best Budget-Adaptive Routing (picked on val) and the oracle at 1x ours' tensor px, merged onto d."""
    _s = importlib.util.spec_from_file_location("kr", B / "scripts" / "A_method" / "k18_routers.py")
    KR = importlib.util.module_from_spec(_s); _s.loader.exec_module(KR)
    from src.offloader import Offloader
    D, te, _, Sc, pc, ps = KR.load(ds)
    target = KR.TPX[ds]
    Vt = (te.J_c - te.J_u960).to_numpy(); Vv = (D["val"].J_c - D["val"].J_u960).to_numpy(); Lv = D["val"].J_c.to_numpy()

    def px_of(kind, m):
        return pc + m.mean() * ps if kind == "cond" else (1 - m.mean()) * pc + m.mean() * ps

    def solve(name, split):
        kind = name.split("_")[0]
        lo, hi, best = 0.0, 1.0, None
        for _ in range(40):
            r = (lo + hi) / 2
            m = Offloader("offload_binary").decide_calibrated(Sc[name][split], r, calibration_predictions=Sc[name]["train"]).mask
            px = px_of(kind, m)
            if best is None or abs(px - target) < abs(best[1] - target):
                best = (m, px)
            if abs(px - target) <= 0.02 * target:
                break
            lo, hi = (r, hi) if px < target else (lo, r)
        return best
    gv = {n: float((Vv * solve(n, "val")[0]).sum() / Lv.sum()) for n in ("skip_offbin", "cond_moric") if n in Sc}
    best = max(gv, key=gv.get)
    m_best, px_best = solve(best, "test")
    k = int(round((target - pc) / ps * len(te)))
    orc = np.zeros(len(te), bool); orc[np.argsort(-Vt, kind="stable")[:k]] = True
    r = pd.DataFrame({"unit": te.seq.to_numpy(), "frame": te.frame.to_numpy(), "route_best": m_best, "route_oracle": orc})
    d = d.merge(r, on=["unit", "frame"], how="left")
    for nm in ("best", "oracle"):
        d[f"a_router_{nm}"] = np.where(d[f"route_{nm}"].astype(bool), d.a_u960, d.a_cheap)
    return d, f"best Budget-Adaptive Routing ({best}, picked on val; px {px_best:.4f})", f"oracle router (px {pc + k / len(te) * ps:.4f})"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in ("nusc", "av2", "ahd"):
        root = S if ds == "av2" else B
        d = load(root / "results" / "k14" / f"{ds}__*.pkl")
        d = d[d["gt"].astype(bool)].reset_index(drop=True)
        gt = pd.read_pickle(B / "results" / "k19" / f"gt_{ds}.pkl")
        d = d.merge(gt[["unit", "frame", "n_corridor", "min_ttc_corridor", "min_range_corridor"]], on=["unit", "frame"], how="left")
        names = {"cheap": "cheap draft", "u960": "u960", "native": "native", "full": "ours without refiner",
                 "full_ref": "ours", "uncertainty": "uncertainty crops", "temporal_crit": "temporal_crit crops"}
        if ds in ("nusc", "av2"):
            ins = load(root / "results" / "instr" / f"{ds}_test_base__*.pkl")
            d = d.merge(ins[["unit", "frame", "a_u640", "a_support_only_ref", "a_field_only_ref"]], on=["unit", "frame"], how="left")
        else:
            ab = load(B / "results" / "k13_ahd_ablation" / "seq*.pkl")
            hd = load(B / "results" / "k10_hood" / "ahd__*.pkl")
            d = d.merge(ab[["unit", "frame", "a_support_only_ref", "a_field_only_ref"]], on=["unit", "frame"], how="left")
            d = d.merge(hd[["unit", "frame", "a_u640", "a_u1280"]], on=["unit", "frame"], how="left")
        names.update({"u640": "u640", "support_only_ref": "support only (+ refiner)", "field_only_ref": "field only (+ refiner)"})
        note = ""
        if ds == "nusc":
            hd = load(B / "results" / "k10_hood" / "nusc__*.pkl")
            d = d.merge(hd[["unit", "frame", "a_u1280"]], on=["unit", "frame"], how="left")
        elif ds == "av2":
            nat = pd.read_pickle(S / "results" / "native" / "av2.pkl")[["unit", "frame", "J_u1280"]]
            d = d.merge(nat, on=["unit", "frame"], how="left")
            inv = [invert(J, ag, pv) for J, ag, pv in zip(d.J_u1280, d.a_gt, d.prev_act)]
            d["a_u1280"] = [a for a, _ in inv]
            note = f"AV2 u1280 action recovered from its cost: unique on {sum(n == 1 for _, n in inv)} / {len(inv)} frames"
        names["u1280"] = "u1280"
        if ds != "ahd":
            d, nb, no = router_actions(ds, d)
            names.update({"router_best": nb, "router_oracle": no})
        order = ["cheap", "u640", "u960", "u1280", "native", "router_best", "router_oracle", "uncertainty", "temporal_crit",
                 "support_only_ref", "field_only_ref", "full", "full_ref"]
        order = [k for k in order if f"a_{k}" in d]
        # ---- per-frame loss terms
        T = {}
        for k in order:
            t = np.array([terms(a, ag, pv) for a, ag, pv in zip(d[f"a_{k}"], d.a_gt, d.prev_act)])
            T[k] = t
        chk = float(np.abs(T["full_ref"][:, 4] - d.J_full_ref).max())
        Lc = T["cheap"][:, 4].sum()
        units = d.unit.to_numpy(); uq = pd.unique(units); idx = [np.flatnonzero(units == u) for u in uq]
        rng = np.random.default_rng(0)
        bs = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
        ag = d.act_gt.to_numpy()
        multi = (d.n_corridor >= 2).to_numpy()
        rows = []
        for k in order:
            a = d[f"a_{k}"].to_numpy()
            r = {"dataset": ds, "units": len(uq), "frames": len(d), "pipeline": names[k],
                 "G": (Lc - T[k][:, 4].sum()) / Lc, "loss_total": T[k][:, 4].sum()}
            for j, tn in enumerate(("insufficient_braking", "excessive_braking", "command_changes", "collisions")):
                r[f"{tn}_sum"] = T[k][:, j].sum(); r[f"{tn}_share_of_cheap"] = T[k][:, j].sum() / Lc
            r["collision_frames"] = int((T[k][:, 3] > 0).sum())
            r["collision_units"] = int(d.loc[T[k][:, 3] > 0, "unit"].nunique())
            for en, (req, tk) in EVENTS.items():
                ev = ((ag == req) & (a == tk)).astype(float)
                r[f"{en} /1000"] = 1000 * ev.mean()
                if k != "full_ref":
                    eo = ((ag == req) & (d.a_full_ref.to_numpy() == tk)).astype(float)
                    dd = [1000 * (eo[b].mean() - ev[b].mean()) for b in bs]
                    r[f"{en} ours-minus ci"] = f"[{np.percentile(dd, 2.5):+.2f}, {np.percentile(dd, 97.5):+.2f}]"
            # multi-obstacle frames
            Lm = T["cheap"][multi, 4].sum()
            r["multi_frames"] = int(multi.sum())
            r["G_multi"] = (Lm - T[k][multi, 4].sum()) / Lm if Lm > 0 else np.nan
            for en, (req, tk) in EVENTS.items():
                r[f"multi {en} /1000"] = 1000 * ((ag[multi] == req) & (a[multi] == tk)).mean() if multi.any() else np.nan
            rows.append(r)
        res = pd.DataFrame(rows)
        res["note"] = note
        res["check_maxabs_J_full_ref"] = chk
        res.to_csv(OUT / f"safety_{ds}.csv", index=False)
        # ---- collision term detail
        col = []
        for k in order:
            m = T[k][:, 3] > 0
            col.append({"dataset": ds, "pipeline": names[k], "collision_frames": int(m.sum()), "collision_units": int(d.loc[m, "unit"].nunique()),
                        "median_gt_min_ttc_corridor_s": float(np.median(d.loc[m, "min_ttc_corridor"])) if m.any() else np.nan,
                        "median_gt_min_range_corridor_m": float(np.median(d.loc[m, "min_range_corridor"])) if m.any() else np.nan,
                        "median_shortfall_ms2": float(np.median(np.sqrt(T[k][m, 0] / cp.lam_risk))) if m.any() else np.nan})
        pd.DataFrame(col).to_csv(OUT / f"safety_{ds}_collision.csv", index=False)
        # ---- per-scene sign test
        sc = []
        Ju = {k: pd.Series(T[k][:, 4]).groupby(units).sum() for k in order}
        for ref in ("cheap", "u960", "native"):
            dlt = Ju["full_ref"] - Ju[ref]
            lo, eq, hi = int((dlt < -1e-9).sum()), int((dlt.abs() <= 1e-9).sum()), int((dlt > 1e-9).sum())
            p = binomtest(lo, lo + hi).pvalue if lo + hi else np.nan
            sc.append({"dataset": ds, "against": names[ref], "units": len(dlt), "ours_lower": lo, "equal": eq, "ours_higher": hi,
                       "sign_test_p": p})
        pd.DataFrame(sc).to_csv(OUT / f"safety_{ds}_scenes.csv", index=False)
        pd.set_option("display.width", 250)
        print(res[["pipeline", "G", "insufficient_braking_share_of_cheap", "excessive_braking_share_of_cheap",
                   "command_changes_share_of_cheap", "collisions_share_of_cheap", "collision_frames"]].round(4).to_string(index=False))
        print(pd.DataFrame(sc).to_string(index=False), "\n", note, "check", chk)


if __name__ == "__main__":
    main()
