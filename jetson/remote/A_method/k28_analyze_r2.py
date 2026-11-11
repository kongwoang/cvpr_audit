#!/usr/bin/env python
"""K28 round-2 tables (DEVELOPMENT data only; method_freeze_draft §2k rule). Inputs: round 1 (k28_analyze.merged, tag r1)
and round 2 (results/k28/<ds>_r2ours_r2, _r2basewf_r2, _r2baseheur[-<det>]_r2), merged on unit / frame (J_cheap checked).
Parity checks: r2 v51S|v3a = r1 v51S|v3a; r2 C1|v3a = r1 C1|v3a; r2 B11|own = r1 ver-yolo11m|own (fraction of identical
frames). Unit bootstrap 2,000, seed 0 (k28_analyze.Boot).
  candidates_r2.csv   R2-1..4 (+ the reference rows) as candidates_r1.csv
  selection_r2.csv    the §2j / §2k rule over the passing candidates of rounds 1 and 2; chosen per S / M / L
  seeds_r2.csv        family D: dev G per refiner seed 0 / 1 / 2, mean / std
  basewf_r2.csv       whole frame + C1 (YOLOv8s at the E long sides; YOLO11m at the B long sides); dev-best per cap with + C1
  baseheur_r2.csv     heuristic configurations + C1 (and the YOLO11m rows + C1 / + T4)
  parity_r2.csv       the parity checks
usage: k28_analyze_r2.py
"""
from __future__ import annotations

import glob
import importlib.util
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
_a = importlib.util.spec_from_file_location("k28a", Path(__file__).resolve().parent / "k28_analyze.py")
KA = importlib.util.module_from_spec(_a); _a.loader.exec_module(KA)
OUT = REAL / "results" / "k28" / "tables"
R2 = {"B11C1": "R2-1 YOLO11m + C1", "B11C2": "R2-2 YOLO11m + C2", "B11C1s768": "R2-3 YOLO11m + C1, crops 768",
      "B11C1s640": "R2-4 YOLO11m + C1, crops 640"}
DET = "yolo11m"


def load(ds, part):
    rows = []
    for f in sorted(glob.glob(str(REAL / "results" / "k28" / f"{ds}_{part}_r2" / "*.pkl"))):
        rows += pickle.load(open(f, "rb"))
    return pd.DataFrame(rows)


def attach(base, d, sfx):
    if not len(d):
        return base, None
    keep = ["unit", "frame", "J_cheap"] + [c for c in d.columns if c not in ("unit", "frame", "J_cheap")]
    d = d[keep].rename(columns={c: f"{c}@{sfx}" for c in keep[3:]})
    n0 = len(base)
    base = base.merge(d, on=["unit", "frame"], how="left", suffixes=("", f"_{sfx}"))
    assert len(base) == n0
    ok = base[f"J_cheap_{sfx}"].notna()
    assert np.allclose(base.loc[ok, "J_cheap"], base.loc[ok, f"J_cheap_{sfx}"]), sfx
    base[f"has_{sfx}"] = ok
    return base, ok


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gfp = OUT.parent / "phase0" / "gflops_640.csv"
    if gfp.exists():
        KA.GF.update(pd.read_csv(gfp).iloc[0].to_dict())
    GF = KA.GF
    cand, seeds, par, wf_rows, heur_rows = [], [], [], [], []
    for ds in ("nusc", "av2"):
        d, _ = KA.merged(ds, "r1")
        nat = KA.NATIVE[ds]
        px0 = d.px_v51S.mean()
        gf0 = (d.px_v51S * GF["yolov8s"]).mean() * nat / 640 ** 2
        d, ok_o = attach(d, load(ds, "r2ours"), "o")
        d, ok_w = attach(d, load(ds, "r2basewf"), "w")
        hparts = sorted(os.path.basename(x)[len(ds) + 1:-3] for x in glob.glob(str(REAL / "results" / "k28" / f"{ds}_r2baseheur*_r2")))
        for hp in hparts:
            d, _ = attach(d, load(ds, hp), hp)
        # ---- parity
        if ok_o is not None:
            s = d[ok_o]
            for a, b in (("J_v51S|v3a@o", "J_v51S|v3a"), ("J_C1|v3a@o", "J_C1|v3a"), ("J_B11|own@o", f"J_ver-{DET}|own"),
                         ("px_v51S@o", "px_v51S"), ("px_B11@o", f"px_ver-{DET}")):
                if a in s and b in s:
                    m = s[b].notna()
                    par.append({"dataset": ds, "r2": a, "r1": b, "frames": int(m.sum()),
                                "identical": float(np.isclose(s.loc[m, a], s.loc[m, b]).mean())})
        if ok_w is not None and "J_wf960+T4|v3a@w" in d:
            s = d[ok_w & d.get("has_base", False)]
            if len(s):
                par.append({"dataset": ds, "r2": "J_wf960+T4|v3a@w", "r1": "J_wf960+T4|v3a", "frames": len(s),
                            "identical": float(np.isclose(s["J_wf960+T4|v3a@w"], s["J_wf960+T4|v3a"]).mean())})
            s = d[ok_w & d.has_temporal]
            par.append({"dataset": ds, "r2": "J_wf960+C1|v3a@w", "r1": "J_u960+C1|v3a", "frames": len(s),
                        "identical": float(np.isclose(s["J_wf960+C1|v3a@w"], s["J_u960+C1|v3a"]).mean())})
        # ---- round-2 candidates (+ reference rows), with their seed-0 refiner
        if ok_o is not None:
            sub = d[ok_o].reset_index(drop=True)
            bt = KA.Boot(sub)
            items = [("v5.1-S (reference)", "ref", "v51S", "v3a", "yolov8s"), ("C1 (round-1 S / M)", "ref", "C1", "v3a", "yolov8s"),
                     ("YOLO11m (round-1 L)", "ref", "B11", "own", DET)] + [(R2[k], "R2", k, "own", DET) for k in R2]
            for desc, fam, key, rf, det in items:
                J = sub[f"J_{key}|{rf}@o"].to_numpy(float)
                r = {"dataset": ds, "family": fam, "candidate": desc, "key": key, "units": bt.n, "frames": len(sub),
                     "G_no_refiner": KA.g_of(bt.L, sub[f"J_{key}|none@o"].to_numpy(float))}
                tmp = sub.assign(**{f"J_{key}|{rf}": J, f"a_{key}|{rf}": sub[f"a_{key}|{rf}@o"], "J_v51S|v3a": sub["J_v51S|v3a@o"]})
                r.update(KA.stats(tmp, bt, f"{key}|{rf}"))
                px = sub[f"px_{key}@o"]
                r["tensor_px_mean"], r["tensor_px_p95"], r["px_ratio"] = px.mean(), px.quantile(.95), px.mean() / px0
                r["GFLOPs_mean"] = float(((sub.px_cheap * GF["yolov8s"] + (px - sub.px_cheap) * GF[det]) * nat / 640 ** 2).mean())
                r["GFLOPs_ratio"] = r["GFLOPs_mean"] / gf0
                for c in ("rounds", "ncrops", "nguard", "ncalls"):
                    if f"{c}_{key}@o" in sub:
                        r[f"{c}_mean"] = sub[f"{c}_{key}@o"].mean()
                cand.append(r)
                # family D
                sd = [c for c in (f"{rf}", f"{rf}_s1", f"{rf}_s2") if f"J_{key}|{c}@o" in sub]
                if len(sd) == 3:
                    gs = [KA.g_of(bt.L, sub[f"J_{key}|{c}@o"].to_numpy(float)) for c in sd]
                    seeds.append({"dataset": ds, "candidate": desc, "key": key, "G_seed0": gs[0], "G_seed1": gs[1], "G_seed2": gs[2],
                                  "G_mean": float(np.mean(gs)), "G_std": float(np.std(gs, ddof=1))})
        # ---- whole frame + C1 (R4)
        if ok_w is not None:
            sub = d[ok_w].reset_index(drop=True)
            bt = KA.Boot(sub)
            for pre, det, rf in (("wf", "yolov8s", "v3a"), ("vwf", DET, "own")):
                for L_ in sorted({int(m.group(1)) for c in sub.columns for m in [re.match(rf"px_{pre}(\d+)@w$", c)] if m}):
                    k = f"{pre}{L_}+C1"
                    tmp = sub.assign(**{f"J_{k}|{rf}": sub[f"J_{k}|{rf}@w"], f"a_{k}|{rf}": sub[f"a_{k}|{rf}@w"],
                                        "J_v51S|v3a": sub["J_v51S|v3a@w"]})
                    px = sub[f"px_{pre}{L_}@w"]
                    r = {"dataset": ds, "detector": det, "long_side": L_, "components": "+C1", "units": bt.n,
                         "G_no_refiner": KA.g_of(bt.L, sub[f"J_{k}|none@w"].to_numpy(float)), "tensor_px_mean": px.mean(),
                         "px_ratio": px.mean() / px0, "GFLOPs_ratio": float((px * GF[det]).mean() * nat / 640 ** 2) / gf0}
                    r.update(KA.stats(tmp, bt, f"{k}|{rf}"))
                    wf_rows.append(r)
        # ---- heuristics + C1 (R4)
        for hp in hparts:
            sub = d[d[f"has_{hp}"]].reset_index(drop=True)
            bt = KA.Boot(sub)
            for c in [c for c in sub.columns if c.startswith("px_") and c.endswith(f"@{hp}")]:
                k = c[3:-len(hp) - 1]
                for rf in ("v3a", "own"):
                    if f"J_{k}|{rf}@{hp}" not in sub:
                        continue
                    tmp = sub.assign(**{f"J_{k}|{rf}": sub[f"J_{k}|{rf}@{hp}"], f"a_{k}|{rf}": sub[f"a_{k}|{rf}@{hp}"],
                                        "J_v51S|v3a": sub[f"J_v51S|v3a@{hp}"]})
                    r = {"dataset": ds, "part": hp, "row": k, "refiner": rf, "units": bt.n,
                         "G_no_refiner": KA.g_of(bt.L, sub[f"J_{k}|none@{hp}"].to_numpy(float)), "tensor_px_mean": sub[c].mean(),
                         "px_ratio": sub[c].mean() / px0}
                    r.update(KA.stats(tmp, bt, f"{k}|{rf}"))
                    heur_rows.append(r)
    C = pd.DataFrame(cand)
    C.to_csv(OUT / "candidates_r2.csv", index=False)
    pd.DataFrame(par).to_csv(OUT / "parity_r2.csv", index=False)
    pd.DataFrame(seeds).to_csv(OUT / "seeds_r2.csv", index=False)
    # ---- selection over rounds 1 and 2
    S1 = pd.read_csv(OUT / "selection_r1.csv")
    sel = []
    for key, g in C[C.family == "R2"].groupby("key", sort=False):
        n, a = g[g.dataset == "nusc"].iloc[0], g[g.dataset == "av2"].iloc[0]
        passes = bool((n.ci_lo > 0 and a.minus_v51S >= 0) or (a.ci_lo > 0 and n.minus_v51S >= 0))
        ratio = max(n.GFLOPs_ratio, a.GFLOPs_ratio)
        cls = next((c for c, v in KA.CAPS.items() if ratio <= v * KA.TOL), "over")
        sel.append({"round": 2, "key": key, "family": "R2", "candidate": n.candidate, "nusc_minus": n.minus_v51S, "nusc_lo": n.ci_lo,
                    "nusc_hi": n.ci_hi, "av2_minus": a.minus_v51S, "av2_lo": a.ci_lo, "av2_hi": a.ci_hi, "passes": passes,
                    "budget_ratio": ratio, "budget_class": cls, "mean_G": (n.G + a.G) / 2,
                    "halves_nonneg": bool(min(n.minus_v51S_half0, n.minus_v51S_half1, a.minus_v51S_half0, a.minus_v51S_half1) >= 0)})
    S = pd.concat([S1.drop(columns=[c for c in S1.columns if c.startswith("chosen_")]).assign(round=1), pd.DataFrame(sel)],
                  ignore_index=True)
    caps = list(KA.CAPS)
    chosen = {}
    for cap in caps:
        ok = S[S.passes & S.budget_class.isin(caps[:caps.index(cap) + 1])]
        chosen[cap] = ok.sort_values("mean_G", ascending=False).iloc[0].key if len(ok) else None
        S[f"chosen_{cap}"] = S.key == chosen[cap]
    S.to_csv(OUT / "selection_r2.csv", index=False)
    W = pd.DataFrame(wf_rows)
    if len(W):
        m = W[W.detector == "yolov8s"].pivot_table(index="long_side", columns="dataset", values=["G", "px_ratio"]).reset_index()
        m.columns = ["long_side"] + [f"{a}_{b}" for a, b in m.columns[1:]]
        m["mean_G"] = m[["G_nusc", "G_av2"]].mean(axis=1)
        m["max_px_ratio"] = m[["px_ratio_nusc", "px_ratio_av2"]].max(axis=1)
        for cap, v in list(KA.CAPS.items()) + [("any", np.inf)]:
            okm = m[m.max_px_ratio <= v * KA.TOL]
            best = okm.sort_values("mean_G", ascending=False).long_side.iloc[0] if len(okm) else None
            W[f"dev_best_{cap}"] = (W.detector == "yolov8s") & (W.long_side == best)
        W.to_csv(OUT / "basewf_r2.csv", index=False)
    pd.DataFrame(heur_rows).to_csv(OUT / "baseheur_r2.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(pd.DataFrame(par).round(4).to_string(index=False))
    print(S.round(4).to_string(index=False))
    print("chosen", chosen)


if __name__ == "__main__":
    main()
