#!/usr/bin/env python
"""K30 tables from the k30_eval.py runs (post-hoc) + the GT pass (k30_gt.py). Paired unit bootstrap 2,000, seed 0 (units =
scenes / logs / sequences); frames without labels (Argoverse-HD) are dropped after the histories; a dataset enters a table
only when all its units are present (or --partial). Writes results/k30/tables/:
  v1a_matched.csv     G (with refiner v3a, and without) and px of v6-S and of the 8 S heuristics + C1 at m x v6-S's px,
                      m in {0.75, 1, 1.25, 1.5}; v6-S - row with interval; px ratio row / v6-S
  v1b_axis.csv        v6-S over crop long side x support context (post-hoc axis): G, px, minus v6-S
  v1c_ttc.csv         TTC-warning consumer: G2 of v6-S, hybrid with braking support and with the TTC consumer's support, at
                      K28's S budget and at the v6-S budget; intervals
  v2_strata.csv       per stratum (ego-speed tercile, |yaw rate| tercile, required action, decisive-object range bin) for v6-S,
                      hybrid tsp8@x1.00, ego corridor cor8@x1.00 / cor0@x1.00: frames, G, px, decisive-object recall,
                      unnecessary / missed per 1,000 frames, v6-S - hybrid with interval
  v2_tube.csv         share of v6-S's field footprint outside the hybrid's tube; G of both on frames whose decisive object is
                      outside the tube
  v2b_idm.csv         V2b: G under IDMFollow of every pipeline, intervals of ours-IDM / v6-S minus each
  v5bc_consumer.csv   V5b / V5c: G per setting and pipeline, v6-S minus each, rank of v6-S
  v3_causes.csv       cause of the 20 most costly safety-critical v6-S frames (from v3_top_frames.csv and part match)
  v8d_speed.csv       V8d (k30_eval2.py part speed, nuScenes): per ego-speed scale s in {1, 0.5, 1.5}: G and px of cheap's
                      comparators (u960 / native + C1 + v3a, v5.1-S, hybrid tsp8@S + C1 + v3a, v6-S), v6-S minus each with
                      interval; v6-S's field (sensitive cells, farthest sensitive range, field crop area) and the hybrid's
                      crop area; safety-critical frame counts
usage: k30_analyze.py [--partial] [--only v1a,v1b,...]
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P

B = Path(os.environ.get("B_REAL", os.environ["B"]))
R30 = B / "results" / "k30"
OUT = R30 / "tables"
NU = {"nusc": 228, "av2": 150, "ahd": 24}
NWS = {"nusc": 1600, "av2": 2048, "ahd": 1920}
HEUR8 = ("unc2", "unc4", "tcr0", "tcr3", "cor0", "cor8", "tsp2", "tsp8")
FAM = {"unc": "uncertainty", "tcr": "temporal-crit", "cor": "ego corridor", "tsp": "hybrid (support + tube)"}
MULTS = (0.75, 1.0, 1.25, 1.5)


def load(part, ds, partial):
    fs = sorted(glob.glob(str(R30 / part / ds / "*.pkl")))
    if not fs or (len(fs) < NU[ds] and not partial):
        return None, len(fs)
    rows = []
    for f in fs:
        rows += pickle.load(open(f, "rb"))
    d = pd.DataFrame(rows)
    d = d[d["gt"].astype(bool)].reset_index(drop=True)
    return d, len(fs)


class Boot:
    def __init__(self, units, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        self.uq, self.inv = np.unique(np.asarray(units), return_inverse=True)
        self.bs = [rng.integers(0, len(self.uq), len(self.uq)) for _ in range(n)]

    def sums(self, x):
        return np.bincount(self.inv, np.asarray(x, float), len(self.uq))

    def diff(self, L, Ja, Jb):
        """G(a) - G(b) = (sum Jb - sum Ja) / sum L, point and 95% interval"""
        l, a, b = self.sums(L), self.sums(Ja), self.sums(Jb)
        v = [(b[s].sum() - a[s].sum()) / l[s].sum() for s in self.bs]
        return float((b.sum() - a.sum()) / l.sum()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def G(L, J):
    L, J = np.asarray(L, float), np.asarray(J, float)
    return float((L.sum() - J.sum()) / L.sum()) if L.sum() > 0 else np.nan


def v1a(ds, d, bt):
    out = []
    L = d.J_cheap.to_numpy(float)
    px6 = d.px_v6S.to_numpy(float)
    for ref in ("v3a", "none"):
        J6 = d[f"J_v6S|{ref}"].to_numpy(float)
        out.append({"dataset": ds, "refiner": ref, "row": "v6-S", "family": "ours", "m": 1.0, "G": G(L, J6), "px_mean": px6.mean(),
                    "px_ratio_to_v6S": 1.0})
        for m in MULTS:
            for cfg in HEUR8:
                k = f"{cfg}@x{m:.2f}+C1"
                if f"J_{k}|{ref}" not in d:
                    continue
                Jx = d[f"J_{k}|{ref}"].to_numpy(float)
                pt, lo, hi = bt.diff(L, J6, Jx)
                out.append({"dataset": ds, "refiner": ref, "row": f"{cfg} + C1" + (" + v3a" if ref == "v3a" else ""), "family": FAM[cfg[:3]],
                            "m": m, "G": G(L, Jx), "px_mean": d[f"px_{k}"].mean(), "px_ratio_to_v6S": d[f"px_{k}"].mean() / px6.mean(),
                            "v6S_minus": pt, "ci_lo": lo, "ci_hi": hi,
                            "verdict": "v6-S above" if lo > 0 else ("row above" if hi < 0 else "n.s.")})
    return out


def v1b(ds, d, bt, d_match):
    out = []
    L = d.J_cheap.to_numpy(float)
    base = "v6S_s960_c2"
    for k in sorted({c.split("|")[0][2:] for c in d.columns if c.startswith("J_v6S_s")}):
        side, ctx = k.split("_")[1][1:], k.split("_")[2][1:]
        J = d[f"J_{k}|v3a"].to_numpy(float)
        pt, lo, hi = bt.diff(L, J, d[f"J_{base}|v3a"].to_numpy(float))
        out.append({"dataset": ds, "crop_long_side": int(side), "support_context": float(ctx), "G": G(L, J), "px_mean": d[f"px_{k}"].mean(),
                    "minus_v6S": pt, "ci_lo": lo, "ci_hi": hi, "is_v6S": k == base})
    if d_match is not None:                          # the hybrid's G-px points from part match, for the plane plot
        L2 = d_match.J_cheap.to_numpy(float)
        for m in MULTS:
            k = f"tsp8@x{m:.2f}+C1"
            out.append({"dataset": ds, "crop_long_side": np.nan, "support_context": np.nan, "G": G(L2, d_match[f"J_{k}|v3a"]),
                        "px_mean": d_match[f"px_{k}"].mean(), "row": f"hybrid tsp8 + C1 + v3a @ {m} x v6-S px"})
    return out


def v1c(ds, d, bt):
    out = []
    L2 = d.J2_cheap.to_numpy(float)
    J6 = d["J2_v6S|v3a"].to_numpy(float)
    out.append({"dataset": ds, "row": "v6-S (braking-driven, deployed)", "G2": G(L2, J6), "px_mean": d.px_v6S.mean()})
    for k, lab in (("tsp8@S+C1", "hybrid, braking support, K28 S budget (= K28 consumer2 row)"),
                   ("tsp8c2@S+C1", "hybrid, TTC consumer's support, K28 S budget"),
                   ("tsp8@x1.00+C1", "hybrid, braking support, v6-S budget"),
                   ("tsp8c2@x1.00+C1", "hybrid, TTC consumer's support, v6-S budget")):
        if f"J2_{k}|v3a" not in d:
            continue
        J = d[f"J2_{k}|v3a"].to_numpy(float)
        pt, lo, hi = bt.diff(L2, J6, J)
        out.append({"dataset": ds, "row": lab, "G2": G(L2, J), "px_mean": d[f"px_{k}"].mean(), "v6S_minus": pt, "ci_lo": lo, "ci_hi": hi})
    return out


def area(c):
    return max(c[2] - c[0], 0.0) * max(c[3] - c[1], 0.0)


def inter(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def v2(ds, d, gt, bt):
    x = d.merge(gt[["unit", "frame", "yaw_rate", "dec_range", "dec_box", "a_gt"]], on=["unit", "frame"], how="left")
    rows_ = {"v6-S": "v6S", "hybrid tsp8 @ v6-S px": "tsp8@x1.00+C1", "ego corridor cor8 @ v6-S px": "cor8@x1.00+C1",
             "ego corridor cor0 @ v6-S px": "cor0@x1.00+C1"}
    x["speed_tercile"] = pd.qcut(x.v, 3, labels=["low", "mid", "high"])
    x["yaw_tercile"] = pd.qcut(x.yaw_rate.abs(), 3, labels=["low", "mid", "high"])
    x["required_action"] = x.act_gt.map({0: "KEEP", 1: "DECELERATE", 2: "HARD_BRAKE"})
    x["range_bin"] = pd.cut(x.dec_range, [-np.inf, 20, 40, 60, np.inf], labels=["<20", "20-40", "40-60", ">60"]).astype(object).fillna("no decisive object")
    out = []
    for sv in ("speed_tercile", "yaw_tercile", "required_action", "range_bin"):
        for lev, s in x.groupby(sv, observed=True):
            if not len(s):
                continue
            bt_s = Boot(s.unit.to_numpy())
            L = s.J_cheap.to_numpy(float)
            r0 = {"dataset": ds, "stratum": sv, "level": str(lev), "frames": len(s)}
            for lab, k in rows_.items():
                a = s[f"a_{k}|v3a"].to_numpy(int); g = s.act_gt.to_numpy(int)
                dec = s.dec >= 0
                r = dict(r0, row=lab, G=G(L, s[f"J_{k}|v3a"]), px_mean=s[f"px_{k}"].mean(),
                         decisive_recall=float(s.loc[dec, f"dec_found_{k}"].astype(float).mean()) if dec.any() and f"dec_found_{k}" in s else np.nan,
                         unnecessary_per_1k=float((a > g).mean() * 1000), missed_per_1k=float((a < g).mean() * 1000))
                if k != "v6S":
                    pt, lo, hi = bt_s.diff(L, s["J_v6S|v3a"].to_numpy(float), s[f"J_{k}|v3a"].to_numpy(float))
                    r.update({"v6S_minus": pt, "ci_lo": lo, "ci_hi": hi})
                out.append(r)
    # field footprint outside the hybrid's tube; decisive object outside the tube
    shares, outside = [], np.zeros(len(x), bool)
    for i, r in x.iterrows():
        tb = r.get("tube_tsp8@x1.00+C1")
        fl = r.get("field_v6S") or []
        if fl and tb is not None:
            fa = sum(area(c) for c in fl)
            shares.append(1.0 - sum(inter(c, tb) for c in fl) / fa if fa > 0 else np.nan)
        if r.dec >= 0 and r.dec_box is not None:
            b = np.asarray(r.dec_box, float); cxy = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            outside[i] = tb is None or not (tb[0] <= cxy[0] <= tb[2] and tb[1] <= cxy[1] <= tb[3])
    tube = [{"dataset": ds, "measure": "share of v6-S field footprint (area) outside the hybrid's tube", "frames": len(shares),
             "value": float(np.nanmean(shares)) if shares else np.nan}]
    s = x[outside]
    if len(s):
        bt_s = Boot(s.unit.to_numpy())
        L = s.J_cheap.to_numpy(float)
        pt, lo, hi = bt_s.diff(L, s["J_v6S|v3a"].to_numpy(float), s["J_tsp8@x1.00+C1|v3a"].to_numpy(float))
        tube.append({"dataset": ds, "measure": "frames whose decisive object lies outside the hybrid's tube", "frames": len(s),
                     "G_v6S": G(L, s["J_v6S|v3a"]), "G_hybrid": G(L, s["J_tsp8@x1.00+C1|v3a"]), "v6S_minus": pt, "ci_lo": lo, "ci_hi": hi,
                     "decisive_recall_v6S": float(s["dec_found_v6S"].astype(float).mean()),
                     "decisive_recall_hybrid": float(s["dec_found_tsp8@x1.00+C1"].astype(float).mean())})
    return out, tube


def v2b(ds, d, bt):
    out = []
    L = d.J3_cheap.to_numpy(float)
    rows_ = {"cheap": None, "u960 + C1 + v3a": "wf960+C1", f"native + C1 + v3a": f"wf{NWS[ds]}+C1", "ours driven by IDMFollow": "oursIDM",
             "v6-S (frozen, braking-driven)": "v6S", "hybrid (IDM support + tube) @ ours-IDM px": "tsp8@IDM+C1",
             "uncertainty unc2 @ ours-IDM px": "unc2@IDM+C1"}
    for lab, k in rows_.items():
        if k is None:
            continue
        J = d[f"J3_{k}|v3a"].to_numpy(float)
        r = {"dataset": ds, "row": lab, "G_idm": G(L, J), "px_mean": d[f"px_{k.split('+')[0]}" if k.startswith("wf") else f"px_{k}"].mean()}
        for ref, rk in (("oursIDM", "ours-IDM"), ("v6S", "v6-S")):
            if k != ref:
                pt, lo, hi = bt.diff(L, d[f"J3_{ref}|v3a"].to_numpy(float), J)
                r.update({f"{rk}_minus": pt, f"{rk}_ci_lo": lo, f"{rk}_ci_hi": hi})
        out.append(r)
    return out


def v5bc(ds, d, bt):
    out = []
    sets = ["base", "tau0.8", "tau1.2", "tr0.3", "tr0.6"]
    pipes_ = {"v6-S": "v6S", "v5.1-S": "v51S", "u960 + C1 + v3a": "wf960+C1", "native + C1 + v3a": f"wf{NWS[ds]}+C1",
              "hybrid tsp8@S": "tsp8@S+C1", "temporal-crit tcr0@S": "tcr0@S+C1", "temporal-crit tcr3@S": "tcr3@S+C1",
              "uncertainty unc2@S": "unc2@S+C1", "uncertainty unc4@S": "unc4@S+C1"}
    for sn in sets:
        sfx = "" if sn == "base" else f"@{sn}"
        L = d[f"J_cheap{sfx}"].to_numpy(float)
        Gs = {lab: G(L, d[f"J_{k}|v3a{sfx}"]) for lab, k in pipes_.items() if f"J_{k}|v3a{sfx}" in d}
        order = sorted(Gs, key=lambda k: -Gs[k])
        for lab, g_ in Gs.items():
            r = {"dataset": ds, "setting": sn, "kind": "consumer", "pipeline": lab, "G": g_, "rank": order.index(lab) + 1, "ranked": len(order)}
            if lab != "v6-S":
                pt, lo, hi = bt.diff(L, d[f"J_v6S|v3a{sfx}"].to_numpy(float), d[f"J_{pipes_[lab]}|v3a{sfx}"].to_numpy(float))
                r.update({"v6S_minus": pt, "ci_lo": lo, "ci_hi": hi})
            out.append(r)
    L = d.J_cheap.to_numpy(float)
    for pn in ("vc5", "vc10", "ped", "truck", "union"):
        k = f"v6S@{pn}"
        if f"J_{k}|v3a" not in d:
            continue
        pt, lo, hi = bt.diff(L, d[f"J_{k}|v3a"].to_numpy(float), d["J_v6S|v3a"].to_numpy(float))
        out.append({"dataset": ds, "setting": pn, "kind": "probe (v6-S only)", "pipeline": f"v6-S with probe {pn}", "G": G(L, d[f"J_{k}|v3a"]),
                    "px_mean": d[f"px_{k}"].mean(), "minus_frozen_v6S": pt, "ci_lo": lo, "ci_hi": hi})
    return out


def v3c(ds, d, top):
    out = []
    pp = P.PlannerParams()
    t = top[top.dataset == ds]
    x = d.set_index(["unit", "frame"])
    for _, r in t.iterrows():
        k = (r.unit, int(r.frame))
        if k not in x.index:
            continue
        q = x.loc[k]
        found, inc = bool(q.get("dec_found_v6S", False)), bool(q.get("dec_in_crop_v6S", False))
        cause = None
        if not found:
            cause = "decisive object inside a crop but missed" if inc else "decisive object outside every crop (not found)"
        else:
            z, lmin, lmax, ttc = q["dec_z_v6S"], q["dec_lmin_v6S"], q["dec_lmax_v6S"], q["dec_ttc_v6S"]
            gtz, gttc = r.dec_range, np.nan
            a_h = P.discrete_action(P.required_decel([z], [lmin], [lmax], [ttc], r.v, pp)[0], pp)
            a_r = P.discrete_action(P.required_decel([gtz], [lmin], [lmax], [ttc], r.v, pp)[0], pp)
            if a_r > a_h:
                cause = "range error (decisive object found; its estimated range keeps the demand low)"
            else:
                cause = "ttc / lateral error (found, range not the reason)"
        out.append({"dataset": ds, "rank": int(r["rank"]), "unit": r.unit, "frame": int(r.frame), "J_v6S": r.J_v6S, "act_gt": int(r.act_gt),
                    "a_v6S": int(r.a_v6S), "a_F": int(r.a_F), "dec_range_gt": r.dec_range, "dec_class": r.dec_class, "found": found,
                    "inside_crop": inc, "dec_z_est": q.get("dec_z_v6S"), "dec_ttc_est": q.get("dec_ttc_v6S"), "cause": cause,
                    "panel_id": f"{ds}/{r.unit}/{int(r.frame)}"})
    return out


def v8d(ds, d, bt):
    out = []
    pipes_ = {"v6-S": "v6S", "v5.1-S": "v51S", "u960 + C1 + v3a": "wf960+C1", "native + C1 + v3a": f"wf{NWS[ds]}+C1",
              "hybrid tsp8@S + C1 + v3a": "tsp8@S+C1"}
    for sn, s_ in (("base", 1.0), ("s0.5", 0.5), ("s1.5", 1.5)):
        sfx = "" if sn == "base" else f"@{sn}"
        L = d[f"J_cheap{sfx}"].to_numpy(float)
        J6 = d[f"J_v6S|v3a{sfx}"].to_numpy(float)
        act = d[f"act_gt{sfx}"].to_numpy(int)
        com = {"dataset": ds, "speed_scale": s_, "frames": len(d), "units": d.unit.nunique(), "safety_critical_frames": int((act > 0).sum()),
               "hard_brake_frames": int((act >= 2).sum()), "L_cheap_sum": float(L.sum())}
        for lab, k in pipes_.items():
            pxc = f"px_{k}{sfx}" if f"px_{k}{sfx}" in d else (f"px_{k.split('+')[0]}" if k.startswith("wf") else f"px_{k}")
            r = dict(com, pipeline=lab, G=G(L, d[f"J_{k}|v3a{sfx}"]), G_norefiner=G(L, d[f"J_{k}|none{sfx}"]), px_mean=d[pxc].mean())
            if lab != "v6-S":
                pt, lo, hi = bt.diff(L, J6, d[f"J_{k}|v3a{sfx}"].to_numpy(float))
                r.update({"v6S_minus": pt, "ci_lo": lo, "ci_hi": hi, "verdict": "v6-S above" if lo > 0 else ("row above" if hi < 0 else "n.s.")})
            out.append(r)
        fc = d[f"field_cells{sfx}"].to_numpy(float)
        nz = fc > 0
        out.append(dict(com, pipeline="(v6-S field and hybrid crops)", field_nonempty_share=float(nz.mean()), field_cells_mean=float(fc.mean()),
                        field_rmax_mean_nonempty=float(d[f"field_rmax{sfx}"].to_numpy(float)[nz].mean()) if nz.any() else np.nan,
                        field_area_v6S_mean=float(d[f"field_area_v6S{sfx}"].mean()), crop_area_tsp8_mean=float(d[f"crop_area_tsp8{sfx}"].mean()),
                        ncrops_v6S_mean=float(d[f"ncrops_v6S{sfx}"].mean())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--src", default="", help="suffix of the part folders (smoke)")
    args = ap.parse_args()
    only = {x for x in args.only.split(",") if x}
    OUT.mkdir(parents=True, exist_ok=True)
    acc = {k: [] for k in ("v1a", "v1b", "v1c", "v2", "v2t", "v2b", "v5bc", "v3c", "v8d")}
    top = pd.read_csv(OUT / "v3_top_frames.csv") if (OUT / "v3_top_frames.csv").exists() else None
    sfx = f"_{args.src}" if args.src else ""
    for ds in ("nusc", "av2", "ahd"):
        gt = pd.read_pickle(R30 / "gt" / f"{ds}.pkl")
        dm, n = load("match" + sfx, ds, args.partial)
        print(ds, "match units", n, flush=True)
        if dm is not None:
            bt = Boot(dm.unit.to_numpy())
            if not only or "v1a" in only:
                acc["v1a"] += v1a(ds, dm, bt)
            if not only or "v1c" in only:
                acc["v1c"] += v1c(ds, dm, bt)
            if not only or "v2" in only:
                a, b = v2(ds, dm, gt, bt); acc["v2"] += a; acc["v2t"] += b
            if top is not None and (not only or "v3" in only):
                acc["v3c"] += v3c(ds, dm, top)
        da, n = load("axis" + sfx, ds, args.partial)
        print(ds, "axis units", n, flush=True)
        if da is not None and (not only or "v1b" in only):
            acc["v1b"] += v1b(ds, da, Boot(da.unit.to_numpy()), dm)
        di, n = load("idm" + sfx, ds, args.partial)
        print(ds, "idm units", n, flush=True)
        if di is not None and (not only or "v2b" in only):
            acc["v2b"] += v2b(ds, di, Boot(di.unit.to_numpy()))
        dc, n = load("cons" + sfx, ds, args.partial)
        print(ds, "cons units", n, flush=True)
        if dc is not None and (not only or "v5" in only):
            acc["v5bc"] += v5bc(ds, dc, Boot(dc.unit.to_numpy()))
        if ds == "nusc" and (not only or "v8d" in only):
            ds_, n = load("speed" + sfx, ds, args.partial)
            print(ds, "speed units", n, flush=True)
            if ds_ is not None:
                acc["v8d"] += v8d(ds, ds_, Boot(ds_.unit.to_numpy()))
    names = {"v1a": "v1a_matched", "v1b": "v1b_axis", "v1c": "v1c_ttc", "v2": "v2_strata", "v2t": "v2_tube", "v2b": "v2b_idm",
             "v5bc": "v5bc_consumer", "v3c": "v3_causes", "v8d": "v8d_speed"}
    for k, rows in acc.items():
        if rows:
            pd.DataFrame(rows).to_csv(OUT / f"{names[k]}{sfx}.csv", index=False)
            print(k, len(rows), "rows", flush=True)
    print("K30 ANALYZE DONE", flush=True)


if __name__ == "__main__":
    main()
