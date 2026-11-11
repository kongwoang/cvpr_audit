#!/usr/bin/env python
"""K28 round-3 tables (DEVELOPMENT data only; method_freeze_draft §2l rule = §2j / §2k). Inputs: round 1 (k28_analyze.merged,
tag r1), round 3 (results/k28/<ds>_r3ours-yolov8m_r3, _r3ours-yolov8l_r3) merged on unit / frame (J_cheap checked), and
selection_r2.csv (rounds 1 and 2). Unit bootstrap 2,000, seed 0 (k28_analyze.Boot).
  parity_r3.csv       r3 v51S|v3a = r1 v51S|v3a, px; r3 B8m|own = r1 ver-yolov8m|own, px (fraction of identical frames)
  candidates_r3.csv   R3-1..5 (+ the B8m reference row) as candidates_r2.csv
  selection_r3.csv    the rule over the passing candidates of rounds 1-3; chosen per S / M / L
usage: k28_analyze_r3.py
"""
from __future__ import annotations

import glob
import importlib.util
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
_a = importlib.util.spec_from_file_location("k28a", Path(__file__).resolve().parent / "k28_analyze.py")
KA = importlib.util.module_from_spec(_a); _a.loader.exec_module(KA)
OUT = REAL / "results" / "k28" / "tables"
PARTS = {"m": "yolov8m", "l": "yolov8l"}
R3 = {"B8mC1": ("R3-1 YOLOv8m + C1", "m"), "B8mC1s768": ("R3-2 YOLOv8m + C1, crops 768", "m"),
      "B8mC1s640": ("R3-3 YOLOv8m + C1, crops 640", "m"), "B8lC1s768": ("R3-4 YOLOv8l + C1, crops 768", "l"),
      "B8lC1s640": ("R3-5 YOLOv8l + C1, crops 640", "l")}


def load(ds, det):
    rows = []
    for f in sorted(glob.glob(str(REAL / "results" / "k28" / f"{ds}_r3ours-{det}_r3" / "*.pkl"))):
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
    gfp = OUT.parent / "phase0" / "gflops_640.csv"
    if gfp.exists():
        KA.GF.update(pd.read_csv(gfp).iloc[0].to_dict())
    GF = KA.GF
    cand, par = [], []
    for ds in ("nusc", "av2"):
        d, _ = KA.merged(ds, "r1")
        nat = KA.NATIVE[ds]
        px0 = d.px_v51S.mean()
        gf0 = (d.px_v51S * GF["yolov8s"]).mean() * nat / 640 ** 2
        oks = {}
        for sfx, det in PARTS.items():
            d, oks[sfx] = attach(d, load(ds, det), sfx)
        # ---- parity
        for sfx, ok in oks.items():
            if ok is None:
                continue
            s = d[ok]
            pairs = [(f"J_v51S|v3a@{sfx}", "J_v51S|v3a"), (f"px_v51S@{sfx}", "px_v51S")]
            if sfx == "m":
                pairs += [("J_B8m|own@m", "J_ver-yolov8m|own"), ("J_B8m|none@m", "J_ver-yolov8m|none"), ("px_B8m@m", "px_ver-yolov8m")]
            for a, b in pairs:
                if a in s and b in s:
                    m = s[b].notna()
                    par.append({"dataset": ds, "r3": a, "r1": b, "frames": int(m.sum()),
                                "identical": float(np.isclose(s.loc[m, a], s.loc[m, b]).mean())})
        # ---- round-3 candidates (+ the B8m reference row), own seed-0 refiner
        items = [("YOLOv8m (round-1 B1, reference)", "ref", "B8m", "m")] + [(v[0], "R3", k, v[1]) for k, v in R3.items()]
        for desc, fam, key, sfx in items:
            if oks.get(sfx) is None:
                continue
            sub = d[oks[sfx]].reset_index(drop=True)
            bt = KA.Boot(sub)
            det = PARTS[sfx]
            r = {"dataset": ds, "family": fam, "candidate": desc, "key": key, "units": bt.n, "frames": len(sub),
                 "G_no_refiner": KA.g_of(bt.L, sub[f"J_{key}|none@{sfx}"].to_numpy(float))}
            tmp = sub.assign(**{f"J_{key}|own": sub[f"J_{key}|own@{sfx}"], f"a_{key}|own": sub[f"a_{key}|own@{sfx}"],
                                "J_v51S|v3a": sub[f"J_v51S|v3a@{sfx}"]})
            r.update(KA.stats(tmp, bt, f"{key}|own"))
            px = sub[f"px_{key}@{sfx}"]
            r["tensor_px_mean"], r["tensor_px_p95"], r["px_ratio"] = px.mean(), px.quantile(.95), px.mean() / px0
            r["GFLOPs_mean"] = float(((sub.px_cheap * GF["yolov8s"] + (px - sub.px_cheap) * GF[det]) * nat / 640 ** 2).mean())
            r["GFLOPs_ratio"] = r["GFLOPs_mean"] / gf0
            for c in ("rounds", "ncrops", "nguard", "ncalls"):
                if f"{c}_{key}@{sfx}" in sub:
                    r[f"{c}_mean"] = sub[f"{c}_{key}@{sfx}"].mean()
            cand.append(r)
    C = pd.DataFrame(cand)
    C.to_csv(OUT / "candidates_r3.csv", index=False)
    pd.DataFrame(par).to_csv(OUT / "parity_r3.csv", index=False)
    # ---- selection over rounds 1-3
    S2 = pd.read_csv(OUT / "selection_r2.csv")
    sel = []
    for key, g in C[C.family == "R3"].groupby("key", sort=False):
        if set(g.dataset) != {"nusc", "av2"}:
            continue
        n, a = g[g.dataset == "nusc"].iloc[0], g[g.dataset == "av2"].iloc[0]
        passes = bool((n.ci_lo > 0 and a.minus_v51S >= 0) or (a.ci_lo > 0 and n.minus_v51S >= 0))
        ratio = max(n.GFLOPs_ratio, a.GFLOPs_ratio)
        cls = next((c for c, v in KA.CAPS.items() if ratio <= v * KA.TOL), "over")
        sel.append({"round": 3, "key": key, "family": "R3", "candidate": n.candidate, "nusc_minus": n.minus_v51S, "nusc_lo": n.ci_lo,
                    "nusc_hi": n.ci_hi, "av2_minus": a.minus_v51S, "av2_lo": a.ci_lo, "av2_hi": a.ci_hi, "passes": passes,
                    "budget_ratio": ratio, "budget_class": cls, "mean_G": (n.G + a.G) / 2,
                    "halves_nonneg": bool(min(n.minus_v51S_half0, n.minus_v51S_half1, a.minus_v51S_half0, a.minus_v51S_half1) >= 0)})
    S = pd.concat([S2.drop(columns=[c for c in S2.columns if c.startswith("chosen_")]), pd.DataFrame(sel)], ignore_index=True)
    caps = list(KA.CAPS)
    chosen = {}
    for cap in caps:
        ok = S[S.passes & S.budget_class.isin(caps[:caps.index(cap) + 1])]
        chosen[cap] = ok.sort_values("mean_G", ascending=False).iloc[0].key if len(ok) else None
        S[f"chosen_{cap}"] = S.key == chosen[cap]
    S.to_csv(OUT / "selection_r3.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(pd.DataFrame(par).round(4).to_string(index=False))
    print(C.round(4).to_string(index=False))
    print(S.round(4).to_string(index=False))
    print("chosen", chosen)


if __name__ == "__main__":
    main()
