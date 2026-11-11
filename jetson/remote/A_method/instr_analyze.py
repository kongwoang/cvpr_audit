#!/usr/bin/env python
"""Tables for K2 / K3 / K4 / K9 from results/instr/<dataset>_<split>_<setting>__<unit>.pkl (instr_eval.py).
usage: instr_analyze.py <dataset> <split>   -> results/instr_tables/<dataset>_<split>/*.csv
Bootstrap: units (scenes / logs), 2,000 resamples, seed 0, 95% intervals."""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
IN = B / "results" / "instr"
NB = 2000
VARIANTS = ("full", "field_only", "support_only", "pivots", "one_round", "no_guard", "draft_hist", "no_guard_draft_hist")
HEUR = ("uncertainty", "temporal_crit", "random")


def load(ds, split, setting):
    rows = []
    for p in sorted(IN.glob(f"{ds}_{split}_{setting}__*.pkl")):
        rows += pickle.load(open(p, "rb"))
    return pd.DataFrame(rows)


def boots(units, seed=0):
    uq = pd.unique(units); idx = [np.flatnonzero(units == u) for u in uq]
    rng = np.random.default_rng(seed)
    return [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(NB)]


def ci(stat, bs):
    v = [stat(b) for b in bs]
    return round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)


def main():
    ds, split = sys.argv[1], sys.argv[2]
    out = B / "results" / "instr_tables" / f"{ds}_{split}"
    out.mkdir(parents=True, exist_ok=True)
    d = load(ds, split, "base")
    n = len(d)
    bs = boots(d.unit.to_numpy())
    L = d.J_cheap.to_numpy(float)
    label = f"{ds} {split} ({d.unit.nunique()} units, {n} frames)"
    # ---------------- K2: agreement with pi(F)
    rows = []
    def agree(col, ref):
        x, y = d[col].to_numpy(), d[ref].to_numpy()
        return (x == y).mean(), ci(lambda b: (x[b] == y[b]).mean(), bs)
    meths = [("ours", "full", "no refiner", "a_full"), ("ours", "full", "with refiner", "a_full_ref")] + \
        [(h, "strict budget = ours", "-", f"a_{h}") for h in HEUR] + [("u640", "whole frame", "-", "a_u640"), ("cheap", "draft", "-", "a_cheap")]
    for ref_name, ref in (("u960", "a_u960"), ("native", "a_native")):
        for m, var, refv, col in meths:
            if col in d:
                a, (lo, hi) = agree(col, ref)
                rows.append({"dataset": label, "against": ref_name, "method": m, "variant": f"{var}, {refv}", "n_frames": n,
                             "agreement": round(a, 4), "ci_lo": lo, "ci_hi": hi})
    if "a_loc" in d:
        sub = d[d.a_loc.notna()]
        for col, nm in (("a_full", "real crops (ours, no refiner)"), ("a_loc", "exact locality (F's objects inside the crops)"),
                        ("a_loc_Fhist", "exact locality + F's ttc history")):
            rows.append({"dataset": label + f" — diagnostic subset {sub.unit.nunique()} units / {len(sub)} frames", "against": "u960",
                         "method": "ours", "variant": nm, "n_frames": len(sub), "agreement": round((sub[col] == sub.a_u960).mean(), 4)})
    pd.DataFrame(rows).to_csv(out / "agreement.csv", index=False)
    for col, nm in (("a_full", "noref"), ("a_full_ref", "ref")):
        for ref in ("u960", "native"):
            cm = pd.crosstab(d[col].rename(f"ours_{nm}"), d[f"a_{ref}"].rename(f"pi_F_{ref}")).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
            cm.index = ["KEEP", "DECELERATE", "HARD_BRAKE"]; cm.columns = ["KEEP", "DECELERATE", "HARD_BRAKE"]
            cm.to_csv(out / f"confusion_ours_{nm}_vs_{ref}.csv")
    rows = []
    for col, nm in (("a_full", "ours no refiner"), ("a_full_ref", "ours with refiner")):
        jc = "J_full" if col == "a_full" else "J_full_ref"
        for ref in ("u960", "native"):
            for cls, msk in (("ours brakes less", d[col] < d[f"a_{ref}"]), ("ours brakes more", d[col] > d[f"a_{ref}"])):
                rows.append({"dataset": label, "method": nm, "against": ref, "class": cls, "frames": int(msk.sum()),
                             "sum_loss_ours": round(float(d.loc[msk, jc].sum()), 2), "sum_loss_F": round(float(d.loc[msk, f"J_{ref}"].sum()), 2),
                             "sum_loss_cheap": round(float(d.loc[msk, "J_cheap"].sum()), 2)})
    pd.DataFrame(rows).to_csv(out / "disagreement_split.csv", index=False)
    # ---------------- K3: ablations
    rows = []
    Vfull = L - d.J_full.to_numpy(float); Vfull_r = L - d.J_full_ref.to_numpy(float)
    for vn in VARIANTS:
        if f"J_{vn}" not in d:
            continue
        V = L - d[f"J_{vn}"].to_numpy(float); Vr = L - d[f"J_{vn}_ref"].to_numpy(float)
        aff = np.abs(Vr) > 1e-9
        r = {"dataset": label, "variant": vn, "G_no_refiner": round(V.sum() / L.sum(), 3), "G_with_refiner": round(Vr.sum() / L.sum(), 3),
             "network_px_mean": round(float(d[f"px_{vn}"].mean()), 4), "crops_per_frame": round(float(d[f"crops_{vn}"].mean()), 3),
             "rounds_mean": round(float(d[f"rounds_{vn}"].mean()), 3), "guard_redet_per_frame": round(float(d[f"guard_redet_{vn}"].mean()), 3),
             "agreement_u960_no_ref": round(float((d[f"a_{vn}"] == d.a_u960).mean()), 4),
             "agreement_u960_ref": round(float((d[f"a_{vn}_ref"] == d.a_u960).mean()), 4),
             "harmed_share_ref": round(float((Vr < -1e-9).sum() / max(aff.sum(), 1)), 3)}
        if vn != "full":
            r["ci_full_minus_variant_no_ref"] = ci(lambda b: (Vfull[b].sum() - V[b].sum()) / L[b].sum(), bs)
            r["ci_full_minus_variant_ref"] = ci(lambda b: (Vfull_r[b].sum() - Vr[b].sum()) / L[b].sum(), bs)
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "ablation.csv", index=False)
    # cost stats (K1 G gap)
    cs = {"dataset": label, "crops_mean": d.crops_full.mean(), "crops_p95": d.crops_full.quantile(0.95), "share_zero_crops": (d.crops_full == 0).mean(),
          "guard_redet_mean": d.guard_redet_full.mean(), "guard_redet_p95": d.guard_redet_full.quantile(0.95),
          "network_px_mean": d.px_full.mean(), "network_px_p95": d.px_full.quantile(0.95)}
    for k in range(4):
        cs[f"share_rounds_{k}"] = (d.rounds_full == k).mean()
    pd.DataFrame([cs]).round(4).to_csv(out / "cost_stats.csv", index=False)
    # ---------------- K4 A: natural misses
    POL = ("full", "field_only", "support_only") + HEUR + ("u960",)
    rows = []
    has = d["found_cheap"].notna() if "found_cheap" in d else pd.Series(False, index=d.index)
    nat_frames, objs = [], {p: [0, 0] for p in POL}
    for i, r in d[has].iterrows():
        fc, fu = np.asarray(r.found_cheap), np.asarray(r.found_u960)
        miss = (~fc) & fu
        if miss.any():
            nat_frames.append(i)
            for p in POL:
                fp = np.asarray(r.get(f"found_{p}", np.zeros(len(fc), bool)))
                objs[p][0] += int(miss.sum()); objs[p][1] += int((fp & miss).sum())
    sub = d.loc[nat_frames]
    for p in POL:
        Ls = sub.J_cheap.sum()
        rows.append({"dataset": label, "policy": p, "frames": len(sub), "missed_objects": objs[p][0],
                     "recovery_rate": round(objs[p][1] / max(objs[p][0], 1), 3),
                     "agreement_u960": round(float((sub[f"a_{p}"] == sub.a_u960).mean()), 3) if len(sub) else np.nan,
                     "G_on_these_frames": round(float((sub.J_cheap - sub[f"J_{p}"]).sum() / Ls), 3) if Ls > 0 else np.nan})
    pd.DataFrame(rows).to_csv(out / "missed_natural.csv", index=False)
    # ---------------- K4 B: controlled deletion
    rows = []
    for setting in ("del_all", "del_half"):
        e = load(ds, split, setting)
        if not len(e):
            continue
        e = e[e.deleted > 0]
        tert = np.quantile(e.v, [1 / 3, 2 / 3]) if len(e) else [0, 0]
        e = e.assign(speed_tercile=np.digitize(e.v, tert))
        POLB = ("cheap", "full", "field_only", "support_only") + HEUR + ("u960",)
        def strata():
            yield "all", lambda r_, k: True
            for lo, hi, nm in ((0, 30, "<30 m"), (30, 50, "30-50 m"), (50, 1e9, ">50 m")):
                yield f"range {nm}", (lambda r_, k, lo=lo, hi=hi: lo <= r_.deleted_gt_range[k] < hi)
            for t_ in range(3):
                yield f"speed tercile {t_ + 1}", (lambda r_, k, t_=t_: r_.speed_tercile == t_)
        for sname, pred in strata():
            for p in POLB:
                n_del = rec_ = 0
                frames = []
                for i, r in e.iterrows():
                    dm = np.asarray(r.deleted_mask)
                    fp = np.asarray(r.get(f"found_{p}", np.zeros(len(dm), bool))) if isinstance(r.get(f"found_{p}"), (list, np.ndarray)) else np.zeros(len(dm), bool)
                    ks = [k for k in range(len(dm)) if dm[k] and pred(r, list(np.flatnonzero(dm)).index(k))]
                    if ks:
                        frames.append(i); n_del += len(ks); rec_ += int(fp[ks].sum())
                es = e.loc[frames]
                Lu = es.J_cheap_unmod.sum()
                rows.append({"dataset": label, "setting": setting, "stratum": sname, "policy": p, "frames": len(es), "deleted_objects": n_del,
                             "recovery_rate": round(rec_ / max(n_del, 1), 3) if p != "cheap" else 0.0,
                             "agreement_u960": round(float((es[f"a_{p}"] == es.a_u960).mean()), 3) if len(es) else np.nan,
                             "loss_vs_unmodified_cheap": round(float((es[f"J_{p}"].sum() - Lu) / Lu), 3) if Lu > 0 else np.nan,
                             "network_px_mean": round(float(es[f"px_{p}"].mean()), 4) if f"px_{p}" in es and len(es) else np.nan})
    pd.DataFrame(rows).to_csv(out / "missed_controlled.csv", index=False)
    # ---------------- K9: second consumer
    rows = []
    L2 = d.J2_cheap.to_numpy(float)
    bs2 = bs
    V2o = L2 - d.J2_ours_c2.to_numpy(float)
    for p, jc, ac, pxc in (("cheap", "J2_cheap", "a2_cheap", "px_cheap"), ("u960", "J2_u960", "a2_u960", "px_u960"),
                           ("ours (TTC field + support)", "J2_ours_c2", "a2_ours_c2", "px_ours_c2"), ("uncertainty crops (same px)", "J2_unc_c2", "a2_unc_c2", "px_unc_c2")):
        V2 = L2 - d[jc].to_numpy(float)
        r = {"dataset": label, "consumer": f"TTC warning (warn <= 4 s, brake <= 2 s)", "method": p,
             "G": round(V2.sum() / L2.sum(), 3) if L2.sum() > 0 else np.nan, "network_px_mean": round(float(d[pxc].mean()), 4),
             "agreement_u960": round(float((d[ac] == d.a2_u960).mean()), 4)}
        if not p.startswith("ours"):
            r["ci_ours_minus"] = ci(lambda b: (V2o[b].sum() - V2[b].sum()) / L2[b].sum(), bs2)
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "consumer2.csv", index=False)
    print(f"tables -> {out}")


if __name__ == "__main__":
    main()
