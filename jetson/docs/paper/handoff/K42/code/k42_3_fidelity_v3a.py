#!/usr/bin/env python
"""K42-3 (post-hoc, CPU; handoff/K42/PREREG.md): k39_6_fidelity.py (md5 261a63085a8006ad50d30f25a628b62e) with every
"|none" replaced by "|v3a" and nothing else in the computation: pi(F) = a_wf960+C1|v3a and every row with refiner v3a.
Outputs renamed k42_3_*. Router rows: a_wf<s>|v3a when routed, else a_cheap; gate = stored refined J reproduced.
usage: k42_3_fidelity_v3a.py <out_dir>"""
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
NAT = {"nusc": 1600, "av2": 2048, "ahd": 1920}
HEUR = {"nusc": ("tsp8", "unc4", "tcr3", "cor0"), "av2": ("tsp8", "unc2", "tcr0", "cor8"), "ahd": ("tsp8", "unc2", "tcr0", "cor8")}
FAMN = {"tsp": "hybrid tsp8 (support + ego-corridor tube)", "unc": "uncertainty", "tcr": "temporal criticality", "cor": "ego corridor"}
PUB = {"ada+1t1280": "AdaScale", "sahi640o1s10": "SAHI", "fovi1280b64a2": "FOVEA KDE_I", "fovs1280b64a1": "FOVEA KDE_S", "ufp20s512": "UFP"}
DE = {"ana05hinfo": "AnA one pass", "ana05hinfi": "AnA iterated", "cf40q20o": "counterfactual one pass", "cf40q20i": "counterfactual iterated"}
RC = {"bar1280_0.5": "BAR router", "dvr1280_0.5": "decision-value router (router-G)", "ch128B_csc": "Chanakya"}
P = pd.DataFrame  # noqa


def recs(d):
    rr = []
    for f in sorted(glob.glob(str(d / "*.pkl"))):
        rr += pickle.load(open(f, "rb"))
    x = pd.DataFrame(rr)
    x["unit"] = x.unit.astype(str)
    return x


def load(ds):
    key = ["unit", "frame"]
    base = recs(B / "results/k28/p3" / f"{ds}_test_instr")
    base = base.sort_values(key).reset_index(drop=True)
    F = base["a_wf960+C1|v3a"].to_numpy(int)
    prevF = base.groupby("unit")["a_wf960+C1|v3a"].shift(1)
    base["_trans"] = (prevF.notna() & (prevF != base["a_wf960+C1|v3a"])).to_numpy()
    base = base[base["gt"].astype(bool)].reset_index(drop=True)
    T = {"unit": base.unit, "frame": base.frame, "F": base["a_wf960+C1|v3a"].to_numpy(int), "act_gt": base.act_gt.to_numpy(int),
         "cheap": base.a_cheap.to_numpy(int), "trans": base._trans.to_numpy(bool)}
    rows, checks = {}, {}

    def add(lab, fam, a, px, passes=None, batches=None):
        rows[lab] = (fam, np.asarray(a, int), np.asarray(px, float),
                     None if passes is None else np.asarray(passes, float), None if batches is None else np.asarray(batches, float))
    add("cheap", "reference", base.a_cheap, base.px_cheap, np.ones(len(base)), np.zeros(len(base)))
    for k, lab in (("v6S", "DESP v6-S"), ("so6S", "support only"), ("fo6S", "field only"), ("pv6S", "pivots instead of support")):
        pas = 1 + base.get(f"ncrops_{k}", np.nan) + base.get(f"nguard_{k}", np.nan)
        add(lab, "ours / ablation", base[f"a_{k}|v3a"], base[f"px_{k}"], pas, base.get(f"rounds_{k}"))
    for c in HEUR[ds]:
        nm = f"{c}@S+C1"
        add(f"{FAMN[c[:3]]} [{c}] @S (Tab. 1 form)", "heuristic", base[f"a_{nm}|v3a"], base[f"px_{nm}"],
            1 + base.get(f"ncrops_{nm}", np.nan) + base.get(f"nguard_{nm}", np.nan), np.ones(len(base)))

    def merged(d, cols):
        d = d[d["gt"].astype(bool)] if "gt" in d else d
        m = base[key].merge(d[key + cols], on=key, how="left", validate="1:1")
        assert m[cols].notna().all().all(), (ds, cols[:3], int(m[cols].isna().any(axis=1).sum()))
        return m
    w = recs(B / "results/k28" / f"{ds}_test_v6basewf-yolo11m")
    sides = [s for s in (448, 512, 640, 960, 1280, 1536, NAT[ds]) if f"a_wf{s}+C1|v3a" in w]
    m = merged(w, [f"a_wf{s}+C1|v3a" for s in sides] + [f"px_wf{s}" for s in sides])
    checks["basewf wf960+C1 vs F (differing frames)"] = int((m["a_wf960+C1|v3a"].to_numpy(int) != T["F"]).sum())
    for s in sides:
        add(f"whole frame u{'native' if s == NAT[ds] else s} + C1" + (" (= F)" if s == 960 else ""), "whole frame",
            m[f"a_wf{s}+C1|v3a"], m[f"px_wf{s}"], np.ones(len(m)), np.zeros(len(m)))
    checks["whole frame 800"] = "no stored full-split record (reported missing)"
    de = recs(B / "results/k33" / f"{ds}_de_test")
    pub = recs(B / "results/k33" / f"{ds}_pub_test")
    for src, table in ((de, DE), (pub, PUB)):
        for k, lab in table.items():
            for cap in ("", "@cap"):
                m = merged(src, [f"a_{k}{cap}|v3a", f"px_{k}{cap}"])
                add(f"{lab}{' @cap' if cap else ''}", "published / object-level" if src is pub else "object-level (AnA / counterfactual)",
                    m[f"a_{k}{cap}|v3a"], m[f"px_{k}{cap}"])
    m = merged(pub, ["a_wf960|v3a"])
    checks["K33 pub wf960 (C1, no refiner) vs F (differing frames)"] = int((m["a_wf960|v3a"].to_numpy(int) != T["F"]).sum())
    rc = pd.read_pickle(B / "results/k33/rc" / f"test_{ds}.pkl").assign(unit=lambda x: x.unit.astype(str))
    pcols = [f"{x}_wf{s}|{r}" for s in (640, 768, 960, 1280) for x in ("a", "J") for r in ("none", "v3a")]
    pm = merged(pub, pcols + ["J_cheap", "a_cheap"])
    for k, lab in RC.items():
        for cap in ("", "@cap"):
            nm = k + cap
            r = base[key].merge(rc[key + [f"size_{nm}", f"J_{nm}", f"px_{nm}"]], on=key, how="left", validate="1:1")
            sz = r[f"size_{nm}"].fillna(0).astype(int).to_numpy()
            J = pm.J_cheap.to_numpy(float).copy(); a = pm.a_cheap.to_numpy(int).copy()
            for s in (640, 768, 960, 1280):
                q = sz == s
                J[q] = pm[f"J_wf{s}|v3a"].to_numpy(float)[q]
                a[q] = pm[f"a_wf{s}|v3a"].to_numpy(int)[q]
            bad = int((~np.isclose(J, r[f"J_{nm}"].to_numpy(float), rtol=0, atol=1e-9)).sum())
            checks[f"rc {nm}: stored refined J reproduced from routing sizes (differing frames)"] = bad
            if bad == 0:
                add(f"{lab}{' @cap' if cap else ''}", "published (router)", a, r[f"px_{nm}"], 1 + (sz > 0), np.zeros(len(r)))
    k30 = recs(B / "results/k30/match" / ds)
    k39 = recs(B / "results/k39/vuw" / ds)
    for c in HEUR[ds]:
        m = merged(k30, [f"a_{c}@x1.00+C1|v3a", f"px_{c}@x1.00+C1"])
        add(f"{FAMN[c[:3]]} [{c}] nominal cap x1.00", "heuristic", m[f"a_{c}@x1.00+C1|v3a"], m[f"px_{c}@x1.00+C1"], None, np.ones(len(m)))
        m = merged(k39, [f"a_{c}@strict+C1|v3a", f"px_{c}@strict+C1"])
        add(f"{FAMN[c[:3]]} [{c}] strict cap", "heuristic", m[f"a_{c}@strict+C1|v3a"], m[f"px_{c}@strict+C1"], None, np.ones(len(m)))
    return T, rows, checks


def stats(T, a):
    F, g, ch, tr = T["F"], T["act_gt"], T["cheap"], T["trans"]
    eq = a == F
    masks = {"A": np.ones(len(F), bool), "A|cheap!=F": ch != F, "A|F transition": tr, "A|braking required": g >= 1, "A|F>=DECEL": F >= 1}
    return eq, masks


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    out, diffs, allchk = [], [], {}
    for ds in ("nusc", "av2", "ahd"):
        T, rows, checks = load(ds)
        allchk[ds] = checks
        units = np.array(sorted(pd.unique(T["unit"])))
        ui = np.searchsorted(units, T["unit"].to_numpy())
        idx = np.random.default_rng(0).integers(0, len(units), (2000, len(units)))
        C = np.stack([np.bincount(x, minlength=len(units)) for x in idx]).astype(float)
        eq6, masks = stats(T, rows["DESP v6-S"][1])

        def per_unit(v, m):
            return np.bincount(ui[m], weights=v[m].astype(float), minlength=len(units)), np.bincount(ui[m], minlength=len(units)).astype(float)
        for lab, (fam, a, px, pas, bat) in rows.items():
            eq, _ = stats(T, a)
            r = {"dataset": ds, "row": lab, "family": fam, "frames": len(a)}
            for k, m in masks.items():
                r[k] = float(eq[m].mean()) if m.any() else np.nan
                r[f"n_{k}"] = int(m.sum())
                if lab != "DESP v6-S":
                    n1, d1 = per_unit(eq6, m); n2, _ = per_unit(eq, m)
                    dist = (C @ n1 - C @ n2) / np.maximum(C @ d1, 1e-12)
                    lo, hi = np.percentile(dist, [2.5, 97.5])
                    diffs.append({"dataset": ds, "row": lab, "metric": k, "v6S_minus_row": float(eq6[m].mean() - eq[m].mean()),
                                  "ci_lo": lo, "ci_hi": hi, "excludes_0": bool(lo > 0 or hi < 0)})
            ne = ~eq
            r["n_disagree"] = int(ne.sum())
            r["disagree: row = required"] = float((a[ne] == T["act_gt"][ne]).mean()) if ne.any() else np.nan
            r["disagree: pi(F) = required"] = float((T["F"][ne] == T["act_gt"][ne]).mean()) if ne.any() else np.nan
            r["disagree: neither"] = float(((a[ne] != T["act_gt"][ne]) & (T["F"][ne] != T["act_gt"][ne])).mean()) if ne.any() else np.nan
            r["KEEP share of the row"] = float((a == 0).mean())
            r["px_mean"] = float(np.nanmean(px))
            r["detector_passes_mean"] = float(np.nanmean(pas)) if pas is not None and np.isfinite(pas).any() else np.nan
            r["sequential_crop_batches_mean"] = float(np.nanmean(bat)) if bat is not None and np.isfinite(np.asarray(bat, float)).any() else np.nan
            out.append(r)
        out.append({"dataset": ds, "row": "always KEEP (reference)", "family": "reference", "frames": len(T["F"]),
                    **{k: float((T["F"][m] == 0).mean()) for k, m in masks.items()}, "px_mean": 0.0})
    R, D = pd.DataFrame(out), pd.DataFrame(diffs)
    R.to_csv(od / "k42_3_fidelity.csv", index=False)
    D.to_csv(od / "k42_3_diffs.csv", index=False)
    (od / "k42_3_checks.json").write_text(json.dumps(allchk, indent=1) + "\n")
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60)
    print(json.dumps(allchk, indent=1))
    print(R[["dataset", "row", "A", "A|cheap!=F", "A|F transition", "A|braking required", "A|F>=DECEL", "disagree: row = required",
             "disagree: pi(F) = required", "px_mean", "detector_passes_mean"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
