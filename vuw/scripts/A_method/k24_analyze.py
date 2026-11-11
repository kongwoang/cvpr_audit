#!/usr/bin/env python
"""K24 tables (CONFIRMATORY, method_freeze_draft §2i, revised spec "baselines for v5.1"). From results/k24/<ds>_main
(k24_test.py) + results/k24/<ds>_supp (k24_supp.py; merged on unit / frame, J_cheap checked) + the K18 router tables /
scores (nuScenes, AV2 masked; Argoverse-HD has no trained router: oracle only). Argoverse-HD: frames with 3D labels only.
Unit bootstrap 2,000, seed 0, by scene / log / sequence. "+ components" = refiner v3a AND the temporal rule of v5.1-S
(T4 with the T2_1 gate). Refuses to run (prints nothing about results) until both passes are complete.
  table_main_<ds>_v51.csv   every row of (1)-(5) and ours: G without / with refiner, tensor px mean / p95, for P in
                            (v5.1-S, v5.1-L, v5): (P - row) at the same refiner setting, interval, verdict (outperforms:
                            lo > 0; matches: lo > -0.02; else below); G on the K17(3) regime and unnecessary / missed
                            per 1,000 frames (with refiner)
  ablation2x2_<ds>_v51.csv  (5): support only / field only (+ components) and the 2 x 2 {refiner v3a} x {T4}: G, px,
                            (v5.1-S - cell), K17(3) G, unnecessary / missed per 1,000
  pareto_<ds>_v51.csv       tensor px vs G (both settings)
  headline_v51.csv          v5.1-S vs the previous frozen configuration as reported, per dataset
  not_outperformed_v51.csv  every (v5.1-S | v5.1-L, baseline row) whose interval with refiner is not above 0
usage: k24_analyze.py [--tag main] [--supp supp]
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
HEUR = ("uncertainty", "temporal_crit", "random", "traj_support", "corridor")
WF = ("u640", "u960", "u1280", "u1536", "native")
PRIM = {"v5.1-S": "v51S", "v5.1-L": "v51L", "v5": "v5"}
EST = ("skip_offbin", "cond_moric", "skip_V", "cond_V")
NUNITS = {"nusc": 228, "av2": 150, "ahd": 24}


def load(ds, tag):
    rows = []
    fs = sorted(glob.glob(str(REAL / "results" / "k24" / f"{ds}_{tag}" / "*.pkl")))
    for f in fs:
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows), len(fs)


def merged(ds, tag, supp):
    d, n1 = load(ds, tag)
    s, n2 = load(ds, supp)
    if n1 != NUNITS[ds] or n2 != NUNITS[ds]:
        return None, f"{ds}: main {n1} / supp {n2} of {NUNITS[ds]} units"
    keep = ["unit", "frame", "J_cheap"] + [c for c in s.columns if c.startswith(("J_", "a_", "px_", "rounds_", "ncrops_"))
                                          and c not in d.columns]
    m = d.merge(s[keep], on=["unit", "frame"], how="inner", suffixes=("", "_supp"))
    assert len(m) == len(d) == len(s), (ds, len(m), len(d), len(s))
    assert np.allclose(m.J_cheap.to_numpy(float), m.J_cheap_supp.to_numpy(float)), ds
    if ds == "ahd":
        m = m[m["gt"].astype(bool)].reset_index(drop=True)
    return m, None


def methods(d, ds):
    """name -> dict(Jn, Jr, An, Ar, px, group, section)"""
    m = {}

    def add(name, k_none, k_ref, px, group, section):
        m[name] = {"Jn": d[f"J_{k_none}"], "Jr": d[f"J_{k_ref}"], "An": d[f"a_{k_none}"], "Ar": d[f"a_{k_ref}"],
                   "px": d[px], "group": group, "section": section}
    for lab, k in PRIM.items():
        add(lab, f"{k}|none", f"{k}|v3a", f"px_{k}", "ours", "ours" if lab != "v5" else "ours (reference)")
    for nm in WF:
        add(f"whole frame {nm}", f"{nm}|none", f"{nm}|v3a", f"px_{nm}", "whole frame", "(1)")
        add(f"whole frame {nm} + components", f"{nm}+T4|none", f"{nm}+T4|v3a", f"px_{nm}", "whole frame + components", "(1)")
    for ref, sec in (("S", "(2)"), ("L", "(2) extra: budget matched to v5.1-L")):
        for bx in ("0.5", "1", "2"):
            for hn in HEUR:
                k = f"{hn}_{ref}x{bx}"
                add(f"{hn} @ {bx}x v5.1-{ref} px", f"{k}|none", f"{k}|v3a", f"px_{k}", "heuristic", sec)
                add(f"{hn} @ {bx}x v5.1-{ref} px + components", f"{k}+T4|none", f"{k}+T4|v3a", f"px_{k}",
                    "heuristic + components", sec)
    if ds == "nusc":
        add("previous frozen v3 + refiner v1 (as reported)", "v3|none", "v3|frozen", "px_v3", "previous frozen", "(4)")
        add("previous frozen v3 + components", "v3+T4|none", "v3+T4|v3a", "px_v3+T4", "previous frozen + components", "(4)")
        add("previous frozen v3 + refiner v3a", "v3|none", "v3|v3a", "px_v3", "previous frozen", "(4) extra")
    else:
        add("previous frozen v4-lite + refiner v2 (as reported)", "v5|none", "v5|frozen", "px_v5", "previous frozen", "(4)")
        add("previous frozen v4-lite + components (= v5.1-S by construction)", "v51S|none", "v51S|v3a", "px_v51S",
            "previous frozen + components", "(4)")
    add("support only + components", "so51S|none", "so51S|v3a", "px_so51S", "ablation", "(5)")
    add("field only + components", "fo51S|none", "fo51S|v3a", "px_fo51S", "ablation", "(5)")
    return m


def routers(d, ds):
    """K18 protocol re-matched to v5.1-S (and, extra, v5.1-L) mean tensor px (within 2%); escalated frames run u960
    (plain) or u960 + components; oracle on every dataset"""
    out = {}
    pc = d.px_cheap.mean()
    ps = d.px_u960.mean()
    Jc, Ac = d.J_cheap.to_numpy(float), d.a_cheap.to_numpy(int)
    esc = {"": {r: (d[f"J_u960|{r}"].to_numpy(float), d[f"a_u960|{r}"].to_numpy(int)) for r in ("none", "v3a")},
           " + components": {r: (d[f"J_u960+T4|{r}"].to_numpy(float), d[f"a_u960+T4|{r}"].to_numpy(int)) for r in ("none", "v3a")}}

    def put(lab, msk, px, sec, extra=None):
        for suf, E in esc.items():
            o = {"Jn": np.where(msk, E["none"][0], Jc), "Jr": np.where(msk, E["v3a"][0], Jc),
                 "An": np.where(msk, E["none"][1], Ac), "Ar": np.where(msk, E["v3a"][1], Ac),
                 "px": np.full(len(d), px), "group": "router" + suf, "section": sec}
            o.update(extra or {})
            out[lab + suf] = o
    for ref in ("S", "L"):
        sec = "(3)" if ref == "S" else "(3) extra: matched to v5.1-L"
        target = d[f"px_{PRIM['v5.1-' + ref]}"].mean()
        k = int(round((target - pc) / ps * len(d)))
        # oracle: top-k by true V of plain u960 with refiner (the router's escalation target)
        V = Jc - esc[""]["v3a"][0]
        orc = np.zeros(len(d), bool)
        if 0 <= k <= len(d):
            orc[np.argsort(-V, kind="stable")[:k]] = True
        put(f"oracle frame router (true V, top-k) @ v5.1-{ref} px", orc, pc + orc.mean() * ps, sec,
            {"within_2pct": bool(abs(pc + orc.mean() * ps - target) <= 0.02 * target)})
    if ds == "ahd":
        return out
    sys.path.insert(0, str(B / "third_party" / "bgt-ada"))
    from src.offloader import Offloader
    if ds == "nusc":
        O = B / "results" / "router"
        D = {sp: pd.read_pickle(O / f"frames__{sp}.pkl") for sp in ("val", "test")}
    else:
        O = B / "k10_masked" / "results" / "av2_sota"
        D = {sp: pd.read_pickle(O / f"frames__{sp}.pkl").rename(columns={"log": "seq"}) for sp in ("val", "test")}
    S_ = {n: pickle.load(open(O / f"scores__{n}.pkl", "rb")) for n in EST if (O / f"scores__{n}.pkl").exists()}
    te = D["test"].reset_index(drop=True)
    key = pd.DataFrame({"seq": d.unit.values, "frame": d.frame.values, "i": np.arange(len(d))})
    mm = te[["seq", "frame", "J_c"]].reset_index().merge(key, on=["seq", "frame"], how="inner")
    assert len(mm) == len(d) == len(te), (len(mm), len(d), len(te))
    jc_agree = float(np.mean(np.isclose(mm.J_c.to_numpy(float), Jc[mm.i.to_numpy()])))
    perm = mm.sort_values("i")["index"].to_numpy()         # router row for each of our frames
    Vv = (D["val"].J_c - D["val"].J_u960).to_numpy(); Lv = D["val"].J_c.to_numpy()

    def px_of(kind, msk):
        return pc + msk.mean() * ps if kind == "cond" else (1 - msk.mean()) * pc + msk.mean() * ps

    def route(name, split, r):
        return Offloader("offload_binary").decide_calibrated(S_[name][split], r, calibration_predictions=S_[name]["train"]).mask

    def solve(name, split, target):
        kind = name.split("_")[0]
        lo, hi, best = 0.0, 1.0, None
        for _ in range(40):
            r = (lo + hi) / 2
            msk = np.asarray(route(name, split, r), bool)
            px = px_of(kind, msk)
            if best is None or abs(px - target) < abs(best[1] - target):
                best = (msk, px, r)
            if abs(px - target) <= 0.02 * target:
                break
            lo, hi = (r, hi) if px < target else (lo, r)
        return best
    for ref in ("S", "L"):
        sec = "(3)" if ref == "S" else "(3) extra: matched to v5.1-L"
        target = d[f"px_{PRIM['v5.1-' + ref]}"].mean()
        picks = {}
        for name in S_:
            mv, _, _ = solve(name, "val", target)
            picks[name] = float((Vv * mv).sum() / Lv.sum())
        for grp, cand in (("best Budget-Adaptive Routing (picked on val)", ("skip_offbin", "cond_moric")),
                          ("best decision-value router (picked on val)", ("skip_V", "cond_V"))):
            c = [x for x in cand if x in picks]
            if not c:
                continue
            nm = max(c, key=lambda x: picks[x])
            msk, px, _ = solve(nm, "test", target)
            put(f"{grp}: {nm} @ v5.1-{ref} px", msk[perm], px, sec,
                {"within_2pct": bool(abs(px - target) <= 0.02 * target), "jc_agree": jc_agree})
    return out


class Boot:
    def __init__(self, d):
        self.L = d.J_cheap.to_numpy(float)
        u = d.unit.to_numpy(); uq = pd.unique(u); idx = [np.flatnonzero(u == x) for x in uq]
        rng = np.random.default_rng(0)
        self.bs = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
        self.n = len(uq)

    def diff(self, Jm, Jp):
        """(P - method) in G = (sum Jm - sum Jp) / sum L, point and 95% interval"""
        L = self.L
        dd = np.array([(Jm[b] - Jp[b]).sum() / L[b].sum() for b in self.bs])
        return (Jm - Jp).sum() / L.sum(), np.percentile(dd, 2.5), np.percentile(dd, 97.5)


def verdict(lo):
    return "outperforms" if lo > 0 else ("matches" if lo > -0.02 else "below")


def extras(d, J, A):
    L = d.J_cheap.to_numpy(float)
    g = d.act_gt.to_numpy(int)
    reg = ((d.act_gt >= 1) & (d.a_cheap < d.act_gt)).to_numpy()
    A = np.asarray(A, int)
    return {"regime_frames": int(reg.sum()),
            "G_K17_3_regime": (L[reg] - J[reg]).sum() / L[reg].sum() if reg.any() else np.nan,
            "unnecessary_per_1k": 1000 * (A > g).mean(), "missed_per_1k": 1000 * (A < g).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--supp", default="supp")
    args = ap.parse_args()
    data = {}
    for ds in ("nusc", "av2", "ahd"):
        d, why = merged(ds, args.tag, args.supp)
        if d is None:
            print(f"NOT READY: {why}; no table written")
            return
        data[ds] = d
    out = REAL / "results" / "k24" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    no, par, head = [], [], []
    for ds, d in data.items():
        L = d.J_cheap.to_numpy(float)
        bt = Boot(d)
        M = methods(d, ds)
        R = routers(d, ds)
        M.update(R)
        rows = []
        for lab, e in M.items():
            Jn, Jr = np.asarray(e["Jn"], float), np.asarray(e["Jr"], float)
            px = np.asarray(e["px"], float)
            row = {"dataset": ds, "units": bt.n, "frames": len(d), "section": e["section"], "method": lab,
                   "group": e["group"], "G_no_refiner": (L - Jn).sum() / L.sum(), "G_refiner": (L - Jr).sum() / L.sum(),
                   "tensor_px_mean": float(px.mean()), "tensor_px_p95": float(np.quantile(px, .95))}
            if lab in R:
                row["router_px_within_2pct"] = e.get("within_2pct", np.nan)
                row["router_J_cheap_agreement"] = e.get("jc_agree", np.nan)
            row.update(extras(d, Jr, e["Ar"]))
            for P_, k in PRIM.items():
                tag = P_.replace(".", "").replace("-", "")
                for set_, Jm, Jp in (("no_refiner", Jn, d[f"J_{k}|none"].to_numpy(float)),
                                     ("refiner", Jr, d[f"J_{k}|v3a"].to_numpy(float))):
                    mdiff, lo, hi = bt.diff(Jm, Jp)
                    row[f"{tag}_minus_{set_}"], row[f"{tag}_minus_{set_}_lo"], row[f"{tag}_minus_{set_}_hi"] = mdiff, lo, hi
                    row[f"{tag}_verdict_{set_}"] = verdict(lo)
                    if set_ == "refiner" and P_ != "v5" and e["group"] != "ours" and lo <= 0:
                        no.append({"dataset": ds, "configuration": P_, "section": e["section"], "baseline": lab,
                                   "diff": mdiff, "lo": lo, "hi": hi, "verdict": verdict(lo),
                                   "baseline_px": float(px.mean()), "config_px": d[f"px_{k}"].mean()})
            if e["section"] == "(4)" and "as reported" in lab:
                for set_ in ("no_refiner", "refiner"):
                    gk = "G_no_refiner" if set_ == "no_refiner" else "G_refiner"
                    head.append({"dataset": ds, "baseline": lab, "setting": set_, "G_baseline": row[gk],
                                 "G_v51S": row[gk] + row[f"v51S_minus_{set_}"],
                                 "diff": row[f"v51S_minus_{set_}"], "lo": row[f"v51S_minus_{set_}_lo"],
                                 "hi": row[f"v51S_minus_{set_}_hi"], "verdict": row[f"v51S_verdict_{set_}"]})
            rows.append(row)
            par.append({"dataset": ds, "section": e["section"], "method": lab, "group": e["group"],
                        "tensor_px": float(px.mean()), "G_no_refiner": row["G_no_refiner"], "G_refiner": row["G_refiner"]})
        t = pd.DataFrame(rows)
        t.to_csv(out / f"table_main_{ds}_v51.csv", index=False)
        # (5) ablation 2 x 2 + support / field only
        Lc = d.J_cheap.to_numpy(float)
        Jp = d["J_v51S|v3a"].to_numpy(float)
        ab = []
        for lab, jk in (("v5.1-S (refiner v3a, T4)", "v51S|v3a"), ("v5.1-S without refiner (T4 only)", "v51S|none"),
                        ("v5.1-S without T4 (refiner v3a only; = v5)", "v5|v3a"), ("v5.1-S without refiner and without T4", "v5|none"),
                        ("support only + components", "so51S|v3a"), ("field only + components", "fo51S|v3a"),
                        ("support only + T4, no refiner", "so51S|none"), ("field only + T4, no refiner", "fo51S|none")):
            J = d[f"J_{jk}"].to_numpy(float)
            pk = "px_" + jk.split("|")[0]
            mdiff, lo, hi = bt.diff(J, Jp)
            r = {"dataset": ds, "cell": lab, "refiner": "v3a" if jk.endswith("v3a") else "none",
                 "T4": jk.split("|")[0] != "v5", "G": (Lc - J).sum() / Lc.sum(), "tensor_px_mean": d[pk].mean(),
                 "tensor_px_p95": d[pk].quantile(.95), "v51S_minus": mdiff, "lo": lo, "hi": hi, "verdict": verdict(lo)}
            r.update(extras(d, J, d[f"a_{jk}"]))
            ab.append(r)
        pd.DataFrame(ab).to_csv(out / f"ablation2x2_{ds}_v51.csv", index=False)
        print(f"{ds}: {bt.n} units, {len(d)} frames, {len(rows)} rows")
    P_ = pd.DataFrame(par)
    for ds in P_.dataset.unique():
        P_[P_.dataset == ds].to_csv(out / f"pareto_{ds}_v51.csv", index=False)
    pd.DataFrame(no).to_csv(out / "not_outperformed_v51.csv", index=False)
    pd.DataFrame(head).to_csv(out / "headline_v51.csv", index=False)
    print("tables written to", out)


if __name__ == "__main__":
    main()
