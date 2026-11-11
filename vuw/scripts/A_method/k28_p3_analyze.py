#!/usr/bin/env python
"""K28 PHASE 3 tables (post-hoc, reporting only; v6 frozen in method_freeze_draft §2n). Inputs: k28_p3.py runs
(results/k28/p3/), the phase-2 pass (results/k28/<ds>_test_v6{ours,abl,basewf-*,baseheur*}) and K24 (k24_analyze.merged).
Argoverse-HD: frames with 3D labels only. Unit bootstrap 2,000, seed 0 (k24_analyze.Boot); verdict as K24.
Writes results/k28/p3/tables/:
  parity_p3.csv           instr rows = phase-2 rows (v5.1-S, v6-S, whole frame 640 / 960 / native +- C1, so6S / fo6S, S heuristics + C1)
  agreement_v6.csv        K2: share of frames with the action of F (F = u960 + C1, and u960 plain), per pipeline, without /
                          with refiner, unit-bootstrap interval; ours minus the best heuristic
  attribution_v6.csv      K16: agreement along the counterfactual chain and the share of ours' disagreement removed by each step
  attribution_detail_v6.csv  rounds of v6-S, cap without convergence, residual frames (who matches the annotation), coverage
  missed_natural_v6.csv   K4 A: frames where the draft misses a decision-relevant annotated object that u960 finds
  missed_controlled_v6.csv  K4 B (del_all): recovery of the deleted objects by stratum, loss vs the unmodified draft
  regime_v6.csv           K17: G where braking is required and the draft holds no support object (and the K24 regime), incl. v6-M / L
  consumer2_v6.csv        K9: TTC-warning consumer, G2 per pipeline; ours (TTC loop) minus each
  ctable_<tag>.csv        y11s (nusc / av2 / ahd), ft0 / ft1 (nusc), nomask (av2): every row, G, px, v6-S minus it, verdict
  failure_v6.csv          frames where v6-S (+ v3a) has a higher cost than u960 + C1 (+ v3a), by category; failure_frames_v6.csv the
                          5 largest per dataset and category (input of the capture / panels run)
  roadmap_v6.csv          additive roadmap on the test units: cheap -> previous frozen -> v5 -> + v3a -> + T4 (v5.1-S) -> C1 (v6-S) -> v6-M -> v6-L
usage: k28_p3_analyze.py [--only parity,agreement,...]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
_a = importlib.util.spec_from_file_location("k24a", B / "scripts" / "A_method" / "k24_analyze.py")
K24A = importlib.util.module_from_spec(_a); _a.loader.exec_module(K24A)
P3 = REAL / "results" / "k28" / "p3"
OUT = P3 / "tables"
NWS = {"nusc": 1600, "av2": 2048, "ahd": 1920}
CFG_S = ("unc2@S", "unc4@S", "tcr0@S", "tcr3@S", "cor0@S", "cor8@S", "tsp2@S", "tsp8@S", "rnd@S")
CHAIN = (("a", "exact locality (F's objects in ours' crops)"), ("b", "+ F's ttc (C1 on F's history)"),
         ("c1", "+ closing probe 10 m/s"), ("c2", "+ closing probe 20 m/s"), ("d", "+ extended grid"),
         ("e", "+ probe templates"), ("f", "+ round cap 10"))


def load_dir(d, gt_only=True):
    fs = sorted(glob.glob(str(d / "*.pkl")))
    rows = []
    for f in fs:
        rows += pickle.load(open(f, "rb"))
    x = pd.DataFrame(rows)
    if len(x) and gt_only and "gt" in x:
        x = x[x["gt"].astype(bool)].reset_index(drop=True)
    return x, len(fs)


def G(L, J):
    return float((L - J).sum() / L.sum()) if L.sum() > 0 else np.nan


def agree_ci(bt, eq):
    eq = np.asarray(eq, float)
    v = np.array([eq[b].mean() for b in bt.bs])
    return float(eq.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def pipelines(d, ds):
    """instr pipelines: label -> (key, refs available)"""
    nat = NWS[ds]
    P = {"draft (cheap)": ("cheap", None)}
    for L in (640, 960, nat):
        nm = "native" if L == nat else f"u{L}"
        P[f"whole frame {nm}"] = (f"wf{L}", ("none", "v3a"))
        P[f"whole frame {nm} + C1"] = (f"wf{L}+C1", ("none", "v3a"))
    P.update({"v5.1-S": ("v51S", ("none", "v3a")), "v6-S": ("v6S", ("none", "v3a")),
              "v6-S support only": ("so6S", ("none", "v3a")), "v6-S field only": ("fo6S", ("none", "v3a"))})
    for c in CFG_S:
        P[f"heuristic {c} + C1"] = (f"{c}+C1", ("none", "v3a"))
    return {k: v for k, v in P.items() if (f"a_{v[0]}" in d or f"a_{v[0]}|none" in d)}


def pxcol(d, k):
    c = f"px_{k}" if f"px_{k}" in d else f"px_{re.sub(r'[+](T4|C1)$', '', k)}"
    return d[c] if c in d else None


def col(d, kind, key, ref):
    return d[f"{kind}_{key}"] if ref is None else d[f"{kind}_{key}|{ref}"]


# ------------------------------------------------------------------------------------------------------------ parity
def parity(ds, d):
    out = []
    srcs = {"ours": REAL / "results" / "k28" / f"{ds}_test_v6ours", "abl": REAL / "results" / "k28" / f"{ds}_test_v6abl",
            "basewf": REAL / "results" / "k28" / f"{ds}_test_v6basewf-yolo11m", "heur": REAL / "results" / "k28" / f"{ds}_test_v6baseheur"}
    nat = NWS[ds]
    keys = {"ours": ["v51S", "v6S"], "abl": ["so6S", "fo6S"], "basewf": ["wf640", "wf640+C1", "wf960", "wf960+C1", f"wf{nat}", f"wf{nat}+C1"],
            "heur": [f"{c}+C1" for c in CFG_S]}
    for s, p in srcs.items():
        r, n = load_dir(p)
        if not n:
            continue
        m = d.merge(r, on=["unit", "frame"], suffixes=("", "@p2"))
        for k in keys[s]:
            for c in (f"J_{k}|none", f"J_{k}|v3a", f"a_{k}|none"):
                if c in m and f"{c}@p2" in m:
                    out.append({"dataset": ds, "column": c, "source": p.name, "frames": len(m),
                                "identical": float((m[c].to_numpy(float) == m[f"{c}@p2"].to_numpy(float)).mean())})
    return out


# ------------------------------------------------------------------------------------------------------------ K2 / K16
def agreement(ds, d, bt):
    rows = []
    P = pipelines(d, ds)
    for fn, fk in (("u960 + C1", "wf960+C1"), ("u960", "wf960")):
        aF = d[f"a_{fk}|none"].to_numpy(int)
        best_h = None
        for lab, (k, refs) in P.items():
            for r in (refs or (None,)):
                a = col(d, "a", k, r).to_numpy(int)
                m, lo, hi = agree_ci(bt, a == aF)
                px = pxcol(d, k)
                rows.append({"dataset": ds, "F": fn, "pipeline": lab, "refiner": r or "-", "agreement": m, "lo": lo, "hi": hi,
                             "px_mean": float(px.mean()) if px is not None else np.nan})
                if lab.startswith("heuristic") and r == "none" and (best_h is None or m > best_h[1]):
                    best_h = (lab, m, a)
        a6 = d["a_v6S|none"].to_numpy(int)
        if best_h is not None:
            eq6, eqh = (a6 == aF).astype(float), (best_h[2] == aF).astype(float)
            v = np.array([eq6[b].mean() - eqh[b].mean() for b in bt.bs])
            rows.append({"dataset": ds, "F": fn, "pipeline": f"v6-S minus best heuristic ({best_h[0]})", "refiner": "none",
                         "agreement": float(eq6.mean() - eqh.mean()), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))})
    return rows


def attribution(ds, d, bt):
    rows, det = [], []
    if "a_ch_a" not in d:
        return rows, det
    aF = d["a_wf960+C1|none"].to_numpy(int)
    a0 = d["a_v6S|none"].to_numpy(int)
    dis0 = float((a0 != aF).mean())
    prev, prev_lab = a0, "ours (v6-S, no refiner)"
    m, lo, hi = agree_ci(bt, a0 == aF)
    rows.append({"dataset": ds, "step": "ours", "description": prev_lab, "agreement": m, "lo": lo, "hi": hi,
                 "share_of_disagreement_removed": np.nan})
    for s, desc in CHAIN:
        a = d[f"a_ch_{s}"].to_numpy(int)
        m, lo, hi = agree_ci(bt, a == aF)
        rows.append({"dataset": ds, "step": s, "description": desc, "agreement": m, "lo": lo, "hi": hi,
                     "share_of_disagreement_removed": ((prev != aF).mean() - (a != aF).mean()) / dis0 if dis0 > 0 else np.nan})
        prev = a
    rows.append({"dataset": ds, "step": "residual", "description": "disagreement left after f", "agreement": np.nan, "lo": np.nan,
                 "hi": np.nan, "share_of_disagreement_removed": (prev != aF).mean() / dis0 if dis0 > 0 else np.nan})
    res = prev != aF
    g = d.act_gt.to_numpy(int)
    r = d["rounds_v6S"].to_numpy(int)
    det.append({"dataset": ds, "frames": len(d), "disagreement_ours": dis0, "residual_frames": int(res.sum()),
                "residual_F_matches_annotation": int((aF[res] == g[res]).sum()), "residual_ours_matches_annotation": int((a0[res] == g[res]).sum()),
                **{f"rounds_{k}": float((r == k).mean()) for k in range(4)},
                "cap_without_convergence": float((~d["conv_v6S"].astype(bool)).mean())})
    cov = [c for cs in d["cov"] for c in (cs or [])]
    if cov:
        cv = pd.DataFrame(cov)
        det[-1].update({"F_raising_objects": len(cv), "coverage_inR": float(cv.inR.mean()),
                        "coverage_inR_70_80m": float(cv[(cv.z >= 70) & (cv.z <= 80)].inR.mean()) if ((cv.z >= 70) & (cv.z <= 80)).any() else np.nan,
                        "coverage_inR_F_driver": float(cv[cv.is_F_driver].inR.mean()) if cv.is_F_driver.any() else np.nan})
    return rows, det


# ------------------------------------------------------------------------------------------------------------ K4
def missed_natural(ds, d, bt):
    rows = []
    has = d["found_cheap"].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0) if "found_cheap" in d else pd.Series(False, index=d.index)
    P = pipelines(d, ds)
    nat_idx, objs = [], {lab: [0, 0] for lab in P}
    for i in np.flatnonzero(has.to_numpy()):
        r = d.iloc[i]
        fc, fu = np.asarray(r.found_cheap, bool), np.asarray(r["found_wf960"], bool)
        miss = (~fc) & fu
        if miss.any():
            nat_idx.append(i)
            for lab, (k, _) in P.items():
                fp = np.asarray(r.get(f"found_{k}"), bool) if isinstance(r.get(f"found_{k}"), (list, np.ndarray)) else np.zeros(len(fc), bool)
                objs[lab][0] += int(miss.sum()); objs[lab][1] += int((fp & miss).sum())
    sub = d.iloc[nat_idx]
    L = sub.J_cheap.to_numpy(float)
    for lab, (k, refs) in P.items():
        for rf in (refs or (None,)):
            J = col(sub, "J", k, rf).to_numpy(float)
            rows.append({"dataset": ds, "pipeline": lab, "refiner": rf or "-", "frames": len(sub), "missed_objects": objs[lab][0],
                         "recovery_rate": objs[lab][1] / max(objs[lab][0], 1),
                         "agreement_u960_C1": float((col(sub, "a", k, rf) == sub["a_wf960+C1|none"]).mean()) if len(sub) else np.nan,
                         "G_on_these_frames": G(L, J)})
    return rows


def missed_controlled(ds, e):
    rows = []
    e = e[e.deleted > 0].reset_index(drop=True)
    if not len(e):
        return rows
    tert = np.quantile(e.v, [1 / 3, 2 / 3])
    e = e.assign(speed_tercile=np.digitize(e.v, tert))
    P = pipelines(e, ds_of(e))

    def strata():
        yield "all", lambda r_, k: True
        for lo, hi, nm in ((0, 30, "<30 m"), (30, 50, "30-50 m"), (50, 1e9, ">50 m")):
            yield f"range {nm}", (lambda r_, k, lo=lo, hi=hi: lo <= r_.deleted_gt_range[k] < hi)
        for t_ in range(3):
            yield f"speed tercile {t_ + 1}", (lambda r_, k, t_=t_: r_.speed_tercile == t_)
    for sname, pred in strata():
        for lab, (k, refs) in P.items():
            n_del = rec_ = 0
            fr = []
            for i, r in e.iterrows():
                dm = np.asarray(r.deleted_mask, bool)
                fp = np.asarray(r.get(f"found_{k}"), bool) if isinstance(r.get(f"found_{k}"), (list, np.ndarray)) else np.zeros(len(dm), bool)
                pos = list(np.flatnonzero(dm))
                ks = [j for j in pos if pred(r, pos.index(j))]
                if ks:
                    fr.append(i); n_del += len(ks); rec_ += int(fp[ks].sum())
            es = e.loc[fr]
            Lu = es.J_cheap_unmod.sum()
            for rf in (refs or (None,)):
                rows.append({"dataset": e.attrs.get("ds", ""), "stratum": sname, "pipeline": lab, "refiner": rf or "-", "frames": len(es),
                             "deleted_objects": n_del, "recovery_rate": rec_ / max(n_del, 1) if k != "cheap" else 0.0,
                             "agreement_u960_C1": float((col(es, "a", k, rf) == es["a_wf960+C1|none"]).mean()) if len(es) else np.nan,
                             "loss_vs_unmodified_draft": float((col(es, "J", k, rf).sum() - Lu) / Lu) if Lu > 0 else np.nan,
                             "tensor_px_mean": float(pxcol(es, k).mean()) if pxcol(es, k) is not None and len(es) else np.nan})
    return rows


def ds_of(e):
    return e.attrs.get("ds", "nusc")


# ------------------------------------------------------------------------------------------------------------ K17 / K9
def regime(ds, d, bt):
    rows = []
    reg_ns = ((d.act_gt >= 1) & d.nosup_cheap.astype(bool)).to_numpy()
    reg_k24 = ((d.act_gt >= 1) & (d.a_cheap < d.act_gt)).to_numpy()
    L = d.J_cheap.to_numpy(float)
    P = dict(pipelines(d, ds))
    ours, n = load_dir(REAL / "results" / "k28" / f"{ds}_test_v6ours")
    extra = {}
    if n:
        m = d[["unit", "frame"]].merge(ours, on=["unit", "frame"], how="left")
        for lab, k, rf in (("v6-M", "v6M", "own-yolov8m"), ("v6-L", "v6L", "own-yolo11m")):
            if f"J_{k}|{rf}" in m:
                extra[lab] = {"none": m[f"J_{k}|none"].to_numpy(float), "ref": m[f"J_{k}|{rf}"].to_numpy(float)}
    for rn, reg in (("no support in the draft, braking required", reg_ns), ("draft below the required action (K24)", reg_k24)):
        for lab, (k, refs) in P.items():
            for rf in (refs or (None,)):
                J = col(d, "J", k, rf).to_numpy(float)
                rows.append({"dataset": ds, "regime": rn, "frames": int(reg.sum()), "share_of_frames": float(reg.mean()),
                             "pipeline": lab, "refiner": rf or "-", "G_regime": G(L[reg], J[reg]), "G_outside": G(L[~reg], J[~reg])})
        for lab, e in extra.items():
            for rf, J in e.items():
                rows.append({"dataset": ds, "regime": rn, "frames": int(reg.sum()), "share_of_frames": float(reg.mean()),
                             "pipeline": lab, "refiner": rf, "G_regime": G(L[reg], J[reg]), "G_outside": G(L[~reg], J[~reg])})
    return rows


def consumer2(ds, d, bt):
    rows = []
    if "J2_c2|none" not in d:
        return rows
    L2 = d.J2_cheap.to_numpy(float)
    P = dict(pipelines(d, ds))
    P["ours, TTC-warning loop (field + support of the consumer) + C1"] = ("c2", ("none", "v3a"))
    P["uncertainty crops unc2 + C1 at the TTC loop's tensor px"] = ("unc2@c2+C1", ("none", "v3a"))
    bt2 = K24A.Boot(d.assign(J_cheap=d.J2_cheap))
    for lab, (k, refs) in P.items():
        for rf in (refs or (None,)):
            J2 = d[f"J2_{k}"].to_numpy(float) if rf is None else d[f"J2_{k}|{rf}"].to_numpy(float)
            row = {"dataset": ds, "consumer": "TTC warning (warn <= 4 s, brake <= 2 s)", "pipeline": lab, "refiner": rf or "-",
                   "G2": G(L2, J2), "tensor_px_mean": float(pxcol(d, k).mean()) if pxcol(d, k) is not None else np.nan}
            for ok in ("none", "v3a"):
                Jo = d[f"J2_c2|{ok}"].to_numpy(float)
                mdiff, lo, hi = bt2.diff(J2, Jo)
                row.update({f"c2{ok}_minus": mdiff, f"c2{ok}_minus_lo": lo, f"c2{ok}_minus_hi": hi})
            rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------------------ failures
def failures(ds, d):
    """frames where v6-S + v3a costs more than u960 + C1 + v3a; categories from the actions and the matched objects"""
    a6, aF, g = d["a_v6S|v3a"].to_numpy(int), d["a_wf960+C1|v3a"].to_numpy(int), d.act_gt.to_numpy(int)
    dJ = d["J_v6S|v3a"].to_numpy(float) - d["J_wf960+C1|v3a"].to_numpy(float)
    fr = []
    for i in np.flatnonzero(dJ > 0):
        r = d.iloc[i]
        fo = np.asarray(r.get("found_v6S"), bool) if isinstance(r.get("found_v6S"), (list, np.ndarray)) else np.zeros(0, bool)
        fF = np.asarray(r.get("found_wf960+C1"), bool) if isinstance(r.get("found_wf960+C1"), (list, np.ndarray)) else np.zeros(0, bool)
        if a6[i] > g[i]:
            cat = "unnecessary action (ours above the required action)"
        elif len(fo) and (fF & ~fo).any():
            cat = "missed: a relevant object u960 finds is not in ours' scene"
        elif a6[i] < g[i]:
            cat = "under-reaction with the relevant objects in ours' scene (range / ttc)"
        else:
            cat = "other (same action class as required, switching cost)"
        fr.append({"dataset": ds, "unit": r.unit, "frame": int(r.frame), "category": cat, "act_gt": int(g[i]), "a_v6S": int(a6[i]),
                   "a_u960_C1": int(aF[i]), "a_cheap": int(r.a_cheap), "dJ": float(dJ[i]), "v": float(r.v),
                   "n_relevant": int(r.n_relevant), "rounds": int(r.rounds_v6S), "px_v6S": float(r.px_v6S)})
    f = pd.DataFrame(fr)
    if not len(f):
        return [], []
    L = d.J_cheap.sum()
    summ = [{"dataset": ds, "category": c, "frames": len(x), "share_of_frames": len(x) / len(d), "loss_excess_share_of_cheap_loss": x.dJ.sum() / L}
            for c, x in f.groupby("category")]
    top = f.sort_values("dJ", ascending=False, kind="stable").groupby("category").head(5)
    return summ, top.to_dict("records")


# ------------------------------------------------------------------------------------------------------------ ctables
def ctable(tag, ds, parts):
    """every row of the k28_p3 ours / basewf / baseheur runs of one tag: G, px, (v6-S - row) at the same refiner"""
    d = None
    for p in parts:
        x, n = load_dir(P3 / f"{ds}_test_{p}_{tag}")
        if not n:
            return [], f"{ds} {p}_{tag}: no units"
        if n != K24A.NUNITS[ds]:
            return [], f"{ds} {p}_{tag}: {n} of {K24A.NUNITS[ds]} units"
        keep = ["unit", "frame", "J_cheap"] + [c for c in x.columns if c != "J_cheap" and c.startswith(("J_", "a_", "px_")) and (d is None or c not in d)]
        d = x[keep] if d is None else d.merge(x[keep], on=["unit", "frame"], suffixes=("", "_r"))
        if "J_cheap_r" in d:
            assert np.allclose(d.J_cheap, d.J_cheap_r); d = d.drop(columns="J_cheap_r")
    bt = K24A.Boot(d)
    L = d.J_cheap.to_numpy(float)
    rows = []
    keys = sorted({re.sub(r"\|.*", "", c[2:]) for c in d.columns if c.startswith("J_") and "|" in c})
    for k in keys:
        rf = next(c.split("|")[1] for c in d.columns if c.startswith(f"J_{k}|") and not c.endswith("|none"))
        Jn, Jr = d[f"J_{k}|none"].to_numpy(float), d[f"J_{k}|{rf}"].to_numpy(float)
        pxc = f"px_{k}" if f"px_{k}" in d else f"px_{re.sub(r'[+](T4|C1)$', '', k)}"
        row = {"tag": tag, "dataset": ds, "units": bt.n, "frames": len(d), "row": k, "refiner": rf, "G_no_refiner": G(L, Jn),
               "G_refiner": G(L, Jr), "tensor_px_mean": float(d[pxc].mean()) if pxc in d else np.nan}
        for set_, Jm, Jo in (("no_refiner", Jn, d["J_v6S|none"].to_numpy(float)), ("refiner", Jr, d["J_v6S|v3a"].to_numpy(float))):
            mdiff, lo, hi = bt.diff(Jm, Jo)
            row.update({f"v6S_minus_{set_}": mdiff, f"lo_{set_}": lo, f"hi_{set_}": hi, f"verdict_{set_}": K24A.verdict(lo)})
        rows.append(row)
    return rows, None


# ------------------------------------------------------------------------------------------------------------ roadmap
def roadmap(ds):
    d, why = K24A.merged(ds, "main", "supp")
    if d is None:
        return [], why
    o, n = load_dir(REAL / "results" / "k28" / f"{ds}_test_v6ours")
    if n != K24A.NUNITS[ds]:
        return [], f"{ds} ours {n} units"
    keep = ["unit", "frame"] + [c for c in o.columns if c.startswith(("J_v6", "px_v6"))]
    d = d.merge(o[keep], on=["unit", "frame"], how="left")
    bt = K24A.Boot(d)
    L = d.J_cheap.to_numpy(float)
    if ds == "nusc":
        steps = [("draft (cheap)", "J_cheap", "px_cheap"), ("v3 (previous frozen loop, no refiner)", "J_v3|none", "px_v3"),
                 ("+ refiner v1 (previous frozen, as reported)", "J_v3|frozen", "px_v3"),
                 ("v5 loop (v_close 0, crops 960; no refiner)", "J_v5|none", "px_v5")]
    else:
        steps = [("draft (cheap)", "J_cheap", "px_cheap"), ("v4-lite = v5 loop (no refiner)", "J_v5|none", "px_v5"),
                 ("+ refiner v2 (previous frozen, as reported)", "J_v5|frozen", "px_v5")]
    steps += [("v5 + refiner v3a", "J_v5|v3a", "px_v5"), ("+ T4 robust looming (v5.1-S)", "J_v51S|v3a", "px_v51S"),
              ("C1 learned temporal cue instead of T4 (v6-S)", "J_v6S|v3a", "px_v6S"),
              ("YOLOv8m crops 768 + own refiner (v6-M)", "J_v6M|own-yolov8m", "px_v6M"),
              ("YOLO11m crops 960 + own refiner (v6-L)", "J_v6L|own-yolo11m", "px_v6L")]
    rows, prev = [], None
    for lab, jc, pc in steps:
        if jc not in d:
            rows.append({"dataset": ds, "step": lab, "column": jc, "missing": True}); continue
        J = d[jc].to_numpy(float)
        row = {"dataset": ds, "step": lab, "column": jc, "G": G(L, J), "tensor_px_mean": float(d[pc].mean()) if pc in d else np.nan}
        if prev is not None:
            mdiff, lo, hi = bt.diff(prev, J)                       # (step - previous step) = (Jprev - J) / L
            row.update({"delta_vs_previous": mdiff, "lo": lo, "hi": hi})
        rows.append(row)
        prev = J
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(x for x in args.only.split(",") if x)
    OUT.mkdir(parents=True, exist_ok=True)
    acc = {k: [] for k in ("parity", "agreement", "attribution", "attribution_detail", "missed_natural", "missed_controlled",
                           "regime", "consumer2", "failure", "failure_frames", "roadmap")}
    notes = []
    for ds in ("nusc", "av2", "ahd"):
        d, n = load_dir(P3 / f"{ds}_test_instr")
        if n == K24A.NUNITS[ds] and (not only or only & {"parity", "agreement", "attribution", "missed_natural", "regime", "consumer2"}):
            d = d.reset_index(drop=True)
            bt = K24A.Boot(d)
            acc["parity"] += parity(ds, d)
            acc["agreement"] += agreement(ds, d, bt)
            a, b = attribution(ds, d, bt); acc["attribution"] += a; acc["attribution_detail"] += b
            acc["missed_natural"] += missed_natural(ds, d, bt)
            acc["regime"] += regime(ds, d, bt)
            acc["consumer2"] += consumer2(ds, d, bt)
            a, b = failures(ds, d); acc["failure"] += a; acc["failure_frames"] += b
        else:
            notes.append(f"{ds} instr: {n} units")
        e, n = load_dir(P3 / f"{ds}_test_instr-del_all")
        if n == K24A.NUNITS[ds]:
            e.attrs["ds"] = ds
            acc["missed_controlled"] += missed_controlled(ds, e)
        else:
            notes.append(f"{ds} instr-del_all: {n} units")
        if not only or "roadmap" in only:
            r, why = roadmap(ds)
            acc["roadmap"] += r
            if why:
                notes.append(f"roadmap {why}")
    for k, v in acc.items():
        if v:
            pd.DataFrame(v).to_csv(OUT / f"{k}_v6.csv", index=False)
    for tag, dss, parts in (("y11s", ("nusc", "av2", "ahd"), ("ours", "basewf", "baseheur")), ("ft0", ("nusc",), ("ours", "basewf", "baseheur")),
                            ("ft1", ("nusc",), ("ours", "basewf", "baseheur")), ("nomask", ("av2",), ("ours", "basewf", "baseheur"))):
        rows = []
        for ds in dss:
            r, why = ctable(tag, ds, parts)
            rows += r
            if why:
                notes.append(why)
        if rows:
            pd.DataFrame(rows).to_csv(OUT / f"ctable_{tag}.csv", index=False)
    print("notes:", notes)
    print("K28 P3 TABLES DONE", OUT)


if __name__ == "__main__":
    main()
