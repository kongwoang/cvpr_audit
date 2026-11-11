#!/usr/bin/env python
"""K28 phase-1 tables (DEVELOPMENT data only; method_freeze_draft §2j rule). Inputs results/k28/<ds>_<part>_<tag>
(k28_dev.py ours / base / ver-<det>, k28_tdev.py temporal); all merged on unit / frame (J_cheap checked). Unit bootstrap
2,000, seed 0, by scene / log; halves = units sorted, alternate.
  candidates_<tag>.csv    every candidate of ours (A, B, C) per dataset: G without / with refiner (B: own refiner, also v3a),
                          (candidate - v5.1-S | v3a) and interval, tensor px mean / p95, px and GFLOPs ratio to v5.1-S,
                          K17(3) G, unnecessary / missed per 1,000, the same G / difference on the two halves
  selection_<tag>.csv     the §2j rule per candidate: passes, budget class (S / M / L / over), chosen per budget
  baselines_<tag>.csv     family E per configuration and budget: G plain / + components (none / v3a), px; the dev-best
                          configuration per family and budget (mean of the two dev G with components) and the K24 form
  wholeframe_<tag>.csv    whole frame per long side, plain / + components; dev-best under each cap and unconstrained
  bwholeframe_<tag>.csv   B's counterpart: whole frame with the B detector (own refiner) per long side, GFLOPs ratio
usage: k28_analyze.py [--tag r1]
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
OUT = REAL / "results" / "k28" / "tables"
A_NAMES = ["vc5", "vc10", "va0.25", "va0.5", "va1", "T3", "ctx2.5", "ctx3", "s1280", "N2", "N3"]
A_DESC = {"vc5": "v_close 5", "vc10": "v_close 10", "va0.25": "v_close 0.25 x speed", "va0.5": "v_close 0.5 x speed",
          "va1": "v_close 1 x speed", "T3": "retrospective re-detection (= v5.1-L)", "ctx2.5": "support context 2.5",
          "ctx3": "support context 3", "s1280": "crop long side 1280", "N2": "bounded loop N = 2", "N3": "bounded loop N = 3"}
CAPS = {"S": 1.0, "M": 1.5, "L": 2.5}
TOL = 1.02
GF = {"yolov8s": 28.6, "yolov8m": 78.9, "yolov8l": 165.2, "yolo11m": 68.0}     # GFLOPs at 640 x 640 (overwritten below)
NATIVE = {"nusc": 1600 * 900, "av2": 2048 * 1550}
FAM = {"unc": "uncertainty", "tcr": "temporal_crit", "cor": "ego-corridor", "tsp": "traj_support", "rnd": "random"}
K24_FORM = {"unc": "unc2", "tcr": "tcr3", "cor": "cor8", "tsp": "tsp8", "rnd": "rnd"}   # TUBE index 8 = near 10, far 70, fill (k24_test.tube_px)


def load(ds, part, tag):
    rows = []
    for f in sorted(glob.glob(str(REAL / "results" / "k28" / f"{ds}_{part}_{tag}" / "*.pkl"))):
        rows += pickle.load(open(f, "rb"))
    d = pd.DataFrame(rows)
    if part.startswith("ver-"):                         # make the B columns detector-specific
        det = part[4:]
        d.columns = [re.sub(r"_(ver|vwf\d+|v51S_v8s)(?=[|+]|$)", lambda m: f"_{m.group(1)}-{det}", c) for c in d.columns]
    return d


def merged(ds, tag):
    parts = {}
    for p in ["ours", "base", "temporal"] + [os.path.basename(x)[len(ds) + 1:-len(tag) - 1]
                                             for x in glob.glob(str(REAL / "results" / "k28" / f"{ds}_ver-*_{tag}"))]:
        d = load(ds, p, tag)
        if len(d):
            parts[p] = d
    base = parts.pop("ours")
    for p, d in parts.items():
        keep = ["unit", "frame", "J_cheap"] + [c for c in d.columns if c not in base.columns]
        n0 = len(base)
        base = base.merge(d[keep], on=["unit", "frame"], how="left", suffixes=("", f"_{p}"))
        assert len(base) == n0
        ok = base[f"J_cheap_{p}"].notna()
        assert np.allclose(base.loc[ok, "J_cheap"], base.loc[ok, f"J_cheap_{p}"]), (ds, p)
        base[f"has_{p}"] = ok
    return base, sorted(parts)


class Boot:
    def __init__(self, d):
        self.L = d.J_cheap.to_numpy(float)
        u = d.unit.to_numpy(); uq = sorted(pd.unique(u)); idx = [np.flatnonzero(u == x) for x in uq]
        rng = np.random.default_rng(0)
        self.bs = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
        self.half = [d.unit.isin(uq[h::2]).to_numpy() for h in (0, 1)]
        self.n = len(uq)

    def diff(self, Jc, Jr, m=None):
        """(candidate - reference) in G, interval; on the frames m if given (point only)"""
        L = self.L
        if m is not None:
            return (Jr[m] - Jc[m]).sum() / L[m].sum()
        dd = np.array([(Jr[b] - Jc[b]).sum() / L[b].sum() for b in self.bs])
        return (Jr - Jc).sum() / L.sum(), np.percentile(dd, 2.5), np.percentile(dd, 97.5)


def g_of(L, J, m=None):
    if m is None:
        return (L - J).sum() / L.sum()
    return (L[m] - J[m]).sum() / L[m].sum()


def stats(d, bt, key, ref_key="v51S|v3a"):
    L = bt.L
    J = d[f"J_{key}"].to_numpy(float)
    A = d[f"a_{key}"].to_numpy(int)
    g = d.act_gt.to_numpy(int)
    reg = ((d.act_gt >= 1) & (d.a_cheap < d.act_gt)).to_numpy()
    Jr = d[f"J_{ref_key}"].to_numpy(float)
    m, lo, hi = bt.diff(J, Jr)
    r = {"G": g_of(L, J), "minus_v51S": m, "ci_lo": lo, "ci_hi": hi, "G_K17_3_regime": g_of(L, J, reg),
         "unnecessary_per_1k": 1000 * (A > g).mean(), "missed_per_1k": 1000 * (A < g).mean()}
    for h in (0, 1):
        r[f"G_half{h}"] = g_of(L, J, bt.half[h]); r[f"minus_v51S_half{h}"] = bt.diff(J, Jr, bt.half[h])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r1")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gfp = OUT.parent / "phase0" / "gflops_640.csv"
    if gfp.exists():
        GF.update(pd.read_csv(gfp).iloc[0].to_dict())
    cand, base_rows, wf_rows, bwf_rows = [], [], [], []
    D = {}
    for ds in ("nusc", "av2"):
        d, parts = merged(ds, args.tag)
        D[ds] = d
        bt = Boot(d)
        nat = NATIVE[ds]
        px0 = d.px_v51S.mean()
        gf0 = (d.px_cheap * GF["yolov8s"] + (d.px_v51S - d.px_cheap) * GF["yolov8s"]).mean() * nat / 640 ** 2
        # ---- ours: reference, A, B, C
        items = [("v5.1-S (reference)", "ref", "v51S", "v51S", "yolov8s")]
        items += [(A_DESC[a], "A", a, a, "yolov8s") for a in A_NAMES if f"J_{a}|v3a" in d]
        for p in parts:
            if p.startswith("ver-"):
                det = p[4:]
                items.append((f"verifier {det} (own refiner)", "B", f"ver-{det}", f"ver-{det}", det))
        if "temporal" in parts:
            for c, desc in (("C1", "learned temporal cue, replacing T4"), ("C2", "learned temporal cue, mean with T4")):
                items.append((desc, "C", c, c, "yolov8s"))
        for desc, fam, key, pxk, det in items:
            sub = d
            if fam == "B":
                sub = d[d[f"has_ver-{det}"]].reset_index(drop=True)
            if fam == "C":
                sub = d[d["has_temporal"]].reset_index(drop=True)
            b2 = Boot(sub) if (fam in ("B", "C")) else bt
            kref = f"{key}|own" if fam == "B" else f"{key}|v3a"
            r = {"dataset": ds, "family": fam, "candidate": desc, "key": key, "units": b2.n, "frames": len(sub),
                 "G_no_refiner": g_of(b2.L, sub[f"J_{pxk}|none"].to_numpy(float))}
            r.update(stats(sub, b2, kref))
            if fam == "B":
                r["G_v3a_refiner"] = g_of(b2.L, sub[f"J_{key}|v3a"].to_numpy(float))
            px = sub[f"px_{pxk}"]
            r["tensor_px_mean"], r["tensor_px_p95"] = px.mean(), px.quantile(.95)
            r["px_ratio"] = px.mean() / px0
            crop = px - sub.px_cheap
            r["GFLOPs_mean"] = float(((sub.px_cheap * GF["yolov8s"] + crop * GF[det]) * nat / 640 ** 2).mean())
            r["GFLOPs_ratio"] = r["GFLOPs_mean"] / gf0
            for c in ("rounds", "ncrops", "nguard", "ncalls"):
                if f"{c}_{pxk}" in sub:
                    r[f"{c}_mean"] = sub[f"{c}_{pxk}"].mean()
            cand.append(r)
        # ---- B counterpart: whole frame with the same detector (own refiner), plain / + components
        for p in parts:
            if not p.startswith("ver-"):
                continue
            det = p[4:]
            sub = d[d[f"has_{p}"]].reset_index(drop=True)
            b2 = Boot(sub)
            for L_ in sorted({int(m.group(1)) for c in sub.columns for m in [re.match(rf"px_vwf(\d+)-{det}$", c)] if m}):
                for comp in ("", "+T4"):
                    k = f"vwf{L_}-{det}{comp}"
                    px = sub[f"px_vwf{L_}-{det}"]
                    r = {"dataset": ds, "detector": det, "long_side": L_, "components": comp == "+T4", "units": b2.n,
                         "G_no_refiner": g_of(b2.L, sub[f"J_{k}|none"].to_numpy(float)), "tensor_px_mean": px.mean(),
                         "px_ratio": px.mean() / px0, "GFLOPs_mean": float((px * GF[det] * nat / 640 ** 2).mean())}
                    r["GFLOPs_ratio"] = r["GFLOPs_mean"] / gf0
                    r.update(stats(sub, b2, f"{k}|own"))
                    bwf_rows.append(r)
        # ---- E: heuristics per configuration and budget
        if "has_base" in d:
            sb = d[d.has_base].reset_index(drop=True)
            b3 = Boot(sb)
            for cap in CAPS:
                for fam in FAM:
                    nconf = {"unc": 12, "tcr": 12, "cor": 12, "tsp": 12, "rnd": 1}[fam]
                    for i in range(nconf):
                        vn = f"{fam}{i}" if fam != "rnd" else "rnd"
                        for comp in ("", "+T4"):
                            k = f"{vn}@{cap}{comp}"
                            if f"J_{k}|v3a" not in sb:
                                continue
                            r = {"dataset": ds, "family": FAM[fam], "config": vn, "K24_form": vn == K24_FORM[fam], "budget": cap,
                                 "components": comp == "+T4", "G_no_refiner": g_of(b3.L, sb[f"J_{k}|none"].to_numpy(float)),
                                 "tensor_px_mean": sb[f"px_{k}"].mean(), "px_ratio": sb[f"px_{k}"].mean() / sb.px_v51S.mean()}
                            r.update(stats(sb, b3, f"{k}|v3a"))
                            base_rows.append(r)
            for L_ in sorted({int(c[5:]) for c in sb.columns if c.startswith("px_wf")}):
                for comp in ("", "+T4"):
                    k = f"wf{L_}{comp}"
                    r = {"dataset": ds, "long_side": L_, "components": comp == "+T4",
                         "G_no_refiner": g_of(b3.L, sb[f"J_{k}|none"].to_numpy(float)), "tensor_px_mean": sb[f"px_wf{L_}"].mean(),
                         "px_ratio": sb[f"px_wf{L_}"].mean() / sb.px_v51S.mean()}
                    r.update(stats(sb, b3, f"{k}|v3a"))
                    wf_rows.append(r)
    C = pd.DataFrame(cand)
    C.to_csv(OUT / f"candidates_{args.tag}.csv", index=False)
    if bwf_rows:
        pd.DataFrame(bwf_rows).to_csv(OUT / f"bwholeframe_{args.tag}.csv", index=False)
    # ---- selection (§2j)
    sel = []
    for key, g in C[C.family != "ref"].groupby("key", sort=False):
        if set(g.dataset) != {"nusc", "av2"}:
            continue
        n, a = g[g.dataset == "nusc"].iloc[0], g[g.dataset == "av2"].iloc[0]
        passes = bool((n.ci_lo > 0 and a.minus_v51S >= 0) or (a.ci_lo > 0 and n.minus_v51S >= 0))
        ratio = max(n.GFLOPs_ratio, a.GFLOPs_ratio) if n.family == "B" else max(n.px_ratio, a.px_ratio)
        cls = next((c for c, v in CAPS.items() if ratio <= v * TOL), "over")
        sel.append({"key": key, "family": n.family, "candidate": n.candidate, "nusc_minus": n.minus_v51S, "nusc_lo": n.ci_lo,
                    "nusc_hi": n.ci_hi, "av2_minus": a.minus_v51S, "av2_lo": a.ci_lo, "av2_hi": a.ci_hi, "passes": passes,
                    "budget_ratio": ratio, "budget_class": cls, "mean_G": (n.G + a.G) / 2,
                    "halves_nonneg": bool(min(n.minus_v51S_half0, n.minus_v51S_half1, a.minus_v51S_half0, a.minus_v51S_half1) >= 0)})
    S = pd.DataFrame(sel)
    chosen = {}
    for cap in CAPS:
        ok = S[S.passes & S.budget_class.isin(list(CAPS)[:list(CAPS).index(cap) + 1])] if len(S) else S
        chosen[cap] = ok.sort_values("mean_G", ascending=False).iloc[0].key if len(ok) else None
    if len(S):
        for cap in CAPS:
            S[f"chosen_{cap}"] = S.key == chosen[cap]
    S.to_csv(OUT / f"selection_{args.tag}.csv", index=False)
    # ---- E: dev-best per family and budget (with components, mean of the two dev G)
    E = pd.DataFrame(base_rows)
    if len(E):
        m = E[E.components].pivot_table(index=["family", "config", "budget"], columns="dataset", values="G").reset_index()
        m["mean_G"] = m[["nusc", "av2"]].mean(1)
        best = m.loc[m.groupby(["family", "budget"]).mean_G.idxmax()][["family", "budget", "config", "mean_G"]]
        E = E.merge(best.rename(columns={"config": "best_config", "mean_G": "best_mean_G"}), on=["family", "budget"], how="left")
        E["dev_best"] = E.config == E.best_config
        E.to_csv(OUT / f"baselines_{args.tag}.csv", index=False)
    W = pd.DataFrame(wf_rows)
    if len(W):
        m = W[W.components].pivot_table(index="long_side", columns="dataset", values=["G", "px_ratio"]).reset_index()
        m.columns = ["long_side"] + [f"{a}_{b}" for a, b in m.columns[1:]]
        m["mean_G"] = m[["G_nusc", "G_av2"]].mean(1)
        m["max_px_ratio"] = m[["px_ratio_nusc", "px_ratio_av2"]].max(1)
        pick = {cap: (m[m.max_px_ratio <= v * TOL].sort_values("mean_G", ascending=False).long_side.iloc[0]
                      if (m.max_px_ratio <= v * TOL).any() else None) for cap, v in CAPS.items()}
        pick["any"] = m.sort_values("mean_G", ascending=False).long_side.iloc[0]
        for k, v in pick.items():
            W[f"dev_best_{k}"] = W.long_side == v
        W.to_csv(OUT / f"wholeframe_{args.tag}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(S.round(4).to_string(index=False) if len(S) else "no candidates")
    print("chosen", chosen)


if __name__ == "__main__":
    main()
