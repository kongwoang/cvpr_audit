#!/usr/bin/env python
"""K27 tables (development data only). From results/k27/<ds>_<tag>/*.pkl (k27_dev.py). Unit bootstrap 2,000, seed 0.
  temporal_<ds>.csv  every arm (v5 geometry; nuScenes also v_close 10) x refiner (none / frozen / v3a): G, (arm - v5)
                     at the same refiner and against v5 with v3a, tensor px (incl. retrospective px), operating point,
                     field-raise frames per 1,000 (v3a) with median estimated / annotated ttc and the share the
                     annotated object does not require, G on the K17(3) regime, unnecessary / missed per 1,000
  controls_<ds>.csv  u960 (base / T2_k / T4), uncertainty / ego-corridor at v5's crop tensor px (base / T1 / T2_k / T4):
                     the same columns; (policy - the control's base) and (control - v5 | v3a)
  diag_<ds>.csv      section A diagnosis (0) on v5 (base): action-raising crop objects by pairing source, box heights
                     at the detector scale, relative change, ttc, u960's box of the object; split by "only ours brakes"
                     section B DIAGNOSTIC changed reference (gate on every pipeline incl. the draft): G vs the gated draft
  selection.csv      the §2e rule at S / L for every arm (v3a), both datasets; halves.csv for the chosen arm
usage: k27_analyze.py [--tag main]
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
KS = ("1", "1.5", "2")
ALONE = ["base", "T1"] + [f"T2_{k}" for k in KS] + ["T3", "T4"]
COMBO = [f"T3+T2_{k}" for k in KS] + [f"T4+T2_{k}" for k in KS] + ["T3+T4"] + [f"T3+T4+T2_{k}" for k in KS]
ARMS = ALONE + COMBO
CTRL = ["base", "T1"] + [f"T2_{k}" for k in KS] + ["T4"]
REFS = ("none", "frozen", "v3a")


def load(ds, tag):
    rows = []
    for f in sorted(glob.glob(str(REAL / "results" / "k27" / f"{ds}_{tag}" / "*.pkl"))):
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows)


class Boot:
    def __init__(self, d):
        self.L = d.J_cheap.to_numpy(float)
        u = d.unit.to_numpy(); uq = pd.unique(u); idx = [np.flatnonzero(u == x) for x in uq]
        rng = np.random.default_rng(0)
        self.bs = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
        self.n = len(uq)

    def diff(self, Ja, Jb, L=None):
        L = self.L if L is None else L
        Va, Vb = L - Ja, L - Jb
        dd = [(Va[b].sum() - Vb[b].sum()) / L[b].sum() for b in self.bs]
        return (Va.sum() - Vb.sum()) / L.sum(), np.percentile(dd, 2.5), np.percentile(dd, 97.5)


def events(d, a):
    a = d[a].to_numpy(int); g = d.act_gt.to_numpy(int)
    return 1000 * (a > g).mean(), 1000 * (a < g).mean()


def row(d, bt, pre, ref, base_col, v5_col, px0):
    J = d[f"J_{pre}|{ref}"].to_numpy(float)
    L = bt.L
    reg = ((d.act_gt >= 1) & (d.a_cheap < d.act_gt)).to_numpy()
    m, lo, hi = bt.diff(J, d[base_col].to_numpy(float))
    m5, lo5, hi5 = bt.diff(J, d[v5_col].to_numpy(float))
    un, mi = events(d, f"a_{pre}|{ref}")
    px = d[f"px_{pre}"] if f"px_{pre}" in d else pd.Series(np.nan, index=d.index)
    r = {"pipeline": pre, "refiner": ref, "units": bt.n, "frames": len(d), "G": (L - J).sum() / L.sum(),
         "minus_base": m, "ci_lo": lo, "ci_hi": hi, "minus_v5_v3a": m5, "ci5_lo": lo5, "ci5_hi": hi5,
         "tensor_px": px.mean(), "tensor_px_p95": px.quantile(.95), "px_ratio_to_v5": px.mean() / px0,
         "point": "S" if px.mean() <= 1.1 * px0 else ("L" if px.mean() <= 2 * px0 else "-"),
         "regime_frames": int(reg.sum()), "G_regime": (L[reg] - J[reg]).sum() / L[reg].sum() if reg.any() else np.nan,
         "unnecessary_per_1k": un, "missed_per_1k": mi}
    for c, nm in ((f"pxretro_{pre}", "retro_px"), (f"nretro_{pre}", "retro_crops_per_frame"), (f"rounds_{pre}", "rounds_mean")):
        if c in d:
            r[nm] = d[c].mean()
    return r


def field_stats(d, arm):
    fr = d[f"fr_{arm}"].astype(bool) if f"fr_{arm}" in d else pd.Series(False, index=d.index)
    out = {"field_raise_per_1k": 1000 * fr.mean()}
    if fr.any():
        e = d.loc[fr, f"fr_est_ttc_{arm}"].astype(float)
        out["fr_median_est_ttc"] = e.median()
        out["fr_share_est_ttc_finite"] = np.isfinite(e).mean()
        if f"fr_gt_ttc_{arm}" in d:
            gt = d.loc[fr, f"fr_gt_ttc_{arm}"].astype(float)
            out["fr_median_gt_ttc"] = gt.median()
        out["fr_share_not_required"] = 1 - d.loc[fr, f"fr_gt_req_{arm}"].astype(float).fillna(0).mean()
        out["fr_median_est_z"] = d.loc[fr, f"fr_est_z_{arm}"].astype(float).median()
    return out


def temporal(d, ds):
    bt = Boot(d)
    px0 = d["px_base"].mean()
    rows = []
    for tagp in ("", "vc10:"):
        if tagp and ds != "nusc":
            continue
        for arm in ARMS:
            pre = f"{tagp}{arm}"
            for ref in REFS:
                r = row(d, bt, pre, ref, f"J_{tagp}base|{ref}" if tagp else f"J_base|{ref}", "J_base|v3a", px0)
                r.update({"dataset": ds, "geometry": "v_close 10" if tagp else "v5 (v_close 0)", "arm": arm})
                if not tagp and ref == "v3a":
                    r.update(field_stats(d, arm))
                rows.append(r)
    return pd.DataFrame(rows), bt


def controls(d, ds, bt):
    px0 = d["px_base"].mean()
    rows = []
    for fam, pols in (("u960", ["base"] + CTRL[2:]), ("unc", CTRL), ("cor", CTRL)):
        for pol in pols:
            pre = "u960" if (fam == "u960" and pol == "base") else f"{fam}:{pol}"
            base_pre = "u960" if fam == "u960" else f"{fam}:base"
            for ref in REFS:
                r = row(d, bt, pre, ref, f"J_{base_pre}|{ref}", "J_base|v3a", px0)
                r.update({"dataset": ds, "control": {"u960": "u960", "unc": "uncertainty", "cor": "ego corridor"}[fam], "policy": pol})
                rows.append(r)
    return pd.DataFrame(rows)


def diag(d, ds, bt):
    rows = []
    if "dg" in d:
        g = d[d.dg.fillna(False).astype(bool)].copy()
        g["only_ours"] = g.dg_a > g.dg_a960
        g["dh_det"] = g.dg_h - g.dg_h_prev
        for split, sub in (("all", g), ("only ours brakes", g[g.only_ours]), ("u960 also brakes", g[~g.only_ours])):
            for pair, s2 in [("any", sub)] + [(p, sub[sub.dg_pair == p]) for p in ("draft", "crop", "guard", "none")]:
                if not len(s2):
                    continue
                r = {"section": "A diagnosis (0), v5 base, v3a", "dataset": ds, "frames_split": split, "pairing": pair,
                     "frames": len(s2), "per_1k": 1000 * len(s2) / len(d), "share_of_split": len(s2) / max(len(sub), 1),
                     "obj_from_guard": (s2.dg_src_obj == "guard").mean(), "share_ttc_driven": s2.dg_ttc_driven.astype(float).mean(),
                     "share_field_raise": s2.dg_field.astype(float).mean(),
                     "median_h_det_px": s2.dg_h.median(), "median_h_prev_det_px": s2.dg_h_prev.median(),
                     "median_dh_det_px": s2.dh_det.median(), "median_rel_change": s2.dg_rel.median(),
                     "median_est_ttc": s2.dg_ttc.median(), "median_est_z": s2.dg_z.median()}
                if "dg_gt_ttc" in s2:
                    r.update({"share_gt_matched": s2.dg_gt_ttc.notna().mean(), "median_gt_ttc": s2.dg_gt_ttc.median(),
                              "share_gt_not_required": 1 - s2.dg_gt_req.astype(float).fillna(0).mean()})
                if "dg9_has" in s2:
                    h9 = s2.dg9_has.fillna(False).astype(bool)
                    r.update({"u960_has_object": h9.mean(), "u960_median_h_det_px": s2.loc[h9, "dg9_h"].median(),
                              "u960_median_rel_change": s2.loc[h9, "dg9_rel"].median(),
                              "u960_median_dh_det_px": (s2.loc[h9, "dg9_h"] - s2.loc[h9, "dg9_h_prev"]).median(),
                              "u960_median_ttc": s2.loc[h9, "dg9_ttc"].median(),
                              "u960_share_ttc_finite": np.isfinite(s2.loc[h9, "dg9_ttc"].astype(float)).mean()})
                rows.append(r)
    L = bt.L
    for k in ("ungated",) + KS:
        Lk = L if k == "ungated" else d[f"J_g{k}:cheap"].to_numpy(float)
        for nm in ("u640", "u960", "native", "ours"):
            for ref in ("none", "v3a"):
                col = (f"J_{'base' if nm == 'ours' else nm}|{ref}" if k == "ungated" else f"J_g{k}:{nm}|{ref}")
                J = d[col].to_numpy(float)
                pxc = (f"px_{'base' if nm == 'ours' else nm}" if k == "ungated" else (f"px_g{k}:ours" if nm == "ours" else f"px_{nm}"))
                rows.append({"section": "B DIAGNOSTIC changed reference (gate on every pipeline incl. the draft)", "dataset": ds,
                             "gate_k_px": k, "pipeline": nm, "refiner": ref, "G": (Lk - J).sum() / Lk.sum(),
                             "J_cheap_mean": Lk.mean(), "tensor_px": d[pxc].mean() if pxc in d else np.nan})
    out = pd.DataFrame(rows)
    out["draft_ttc_parity_share"] = d.draft_ttc_parity.mean() if "draft_ttc_parity" in d else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    out = REAL / "results" / "k27" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 40)
    T, D = {}, {}
    for ds in ("nusc", "av2"):
        d = load(ds, args.tag)
        if not len(d):
            continue
        D[ds] = d
        t, bt = temporal(d, ds)
        T[ds] = t
        t.to_csv(out / f"temporal_{ds}.csv", index=False)
        controls(d, ds, bt).to_csv(out / f"controls_{ds}.csv", index=False)
        diag(d, ds, bt).to_csv(out / f"diag_{ds}.csv", index=False)
        print(t[t.refiner == "v3a"][["geometry", "arm", "G", "minus_base", "ci_lo", "ci_hi", "px_ratio_to_v5", "point",
                                     "field_raise_per_1k", "G_regime", "unnecessary_per_1k", "missed_per_1k"]].round(4).to_string(index=False))
    if len(T) == 2:
        sel = []
        for arm in ARMS[1:]:
            r = {"arm": arm}
            for ds in T:
                x = T[ds][(T[ds].arm == arm) & (T[ds].refiner == "v3a") & (T[ds].geometry != "v_close 10")].iloc[0]
                r.update({f"{ds}_G": x.G, f"{ds}_minus_v5": x.minus_base, f"{ds}_ci_lo": x.ci_lo, f"{ds}_ci_hi": x.ci_hi,
                          f"{ds}_point": x.point})
            r["mean_G"] = (r["nusc_G"] + r["av2_G"]) / 2
            r["alone"] = arm in ALONE
            for pt in ("S", "L"):
                ok_pt = all(r[f"{ds}_point"] in (("S",) if pt == "S" else ("S", "L")) for ds in T)
                passes = ((r["nusc_ci_lo"] > 0 and r["av2_minus_v5"] >= 0) or (r["av2_ci_lo"] > 0 and r["nusc_minus_v5"] >= 0))
                r[f"passes_{pt}"] = bool(ok_pt and passes)
            sel.append(r)
        sel = pd.DataFrame(sel)
        passed_alone = {a for a in ALONE if sel.loc[sel.arm == a, "passes_L"].any()}
        # a combination is considered only if every component passed alone (§2g)
        sel["eligible"] = [a in ALONE or all(tok in passed_alone for tok in a.split("+")) for a in sel.arm]
        sel.to_csv(out / "selection.csv", index=False)
        print(sel.round(4).to_string(index=False))
        cand = sel[sel.eligible & sel.passes_L]
        if len(cand):
            best = cand.sort_values("mean_G", ascending=False).iloc[0].arm
            hv = []
            for ds, d in D.items():
                uq = sorted(d.unit.unique())
                for h in (0, 1):
                    m = d.unit.isin(uq[h::2])
                    L = d.J_cheap[m].sum()
                    hv.append({"dataset": ds, "chosen": best, "half": h, "units": len(uq[h::2]),
                               "G_chosen": (L - d.loc[m, f"J_{best}|v3a"].sum()) / L, "G_v5": (L - d.loc[m, "J_base|v3a"].sum()) / L})
            pd.DataFrame(hv).to_csv(out / "halves.csv", index=False)
            print("CHOSEN", best); print(pd.DataFrame(hv).round(4).to_string(index=False))
        else:
            print("NO ARM PASSES: v5 stays")


if __name__ == "__main__":
    main()
