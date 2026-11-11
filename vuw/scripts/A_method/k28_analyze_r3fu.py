#!/usr/bin/env python
"""K28 end of phase 1 (method_freeze_draft §2m): tables of the round-3 follow-ups (DEVELOPMENT data only; not selection
inputs). Inputs: round 1 (k28_analyze.merged, tag r1), round 3 (results/k28/<ds>_r3ours-yolov8m_r3) and the follow-up
parts results/k28/<ds>_r3{oursd,basewf3,baseheur}-yolov8m_fu, merged on unit / frame (J_cheap checked).
Unit bootstrap 2,000, seed 0 (k28_analyze.Boot).
  parity_fu.csv     v51S|v3a, px = r1; fu B8mC1s768|none / own, px = r3; fu vwf<L> / + T4 (none / own), px = r1 vwf<L>-yolov8m
  seeds_fu.csv      family D for R3-2 (B8mC1s768): dev G with the own refiner of seeds 0 / 1 / 2
  basewf_fu.csv     YOLOv8m whole frame at the B long sides: plain / + T4 / + C1 (own refiner), vs v5.1-S [CI], GFLOPs class
  baseheur_fu.csv   YOLOv8m heuristics at the S crop budget, + T4 / + C1 (own refiner), vs v5.1-S [CI], GFLOPs class
usage: k28_analyze_r3fu.py
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
DET = "yolov8m"
PARTS = {"m": f"r3ours-{DET}_r3", "d": f"r3oursd-{DET}_fu", "w": f"r3basewf3-{DET}_fu", "h": f"r3baseheur-{DET}_fu"}


def load(ds, part):
    rows = []
    for f in sorted(glob.glob(str(REAL / "results" / "k28" / f"{ds}_{part}" / "*.pkl"))):
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


def cls_of(ratio):
    return next((c for c, v in KA.CAPS.items() if ratio <= v * KA.TOL), "over")


def main():
    gfp = OUT.parent / "phase0" / "gflops_640.csv"
    if gfp.exists():
        KA.GF.update(pd.read_csv(gfp).iloc[0].to_dict())
    GF = KA.GF
    par, seeds, wf_rows, heur_rows = [], [], [], []
    for ds in ("nusc", "av2"):
        d, _ = KA.merged(ds, "r1")
        nat = KA.NATIVE[ds]
        px0 = d.px_v51S.mean()
        gf0 = (d.px_v51S * GF["yolov8s"]).mean() * nat / 640 ** 2
        oks = {}
        for sfx, part in PARTS.items():
            d, oks[sfx] = attach(d, load(ds, part), sfx)
        # ---- parity
        for sfx, ok in oks.items():
            if ok is None:
                continue
            s = d[ok]
            pairs = [(f"J_v51S|v3a@{sfx}", "J_v51S|v3a"), (f"px_v51S@{sfx}", "px_v51S")]
            if sfx == "d":
                pairs += [(f"J_B8mC1s768|{rf}@d", f"J_B8mC1s768|{rf}@m") for rf in ("none", "own")] + [("px_B8mC1s768@d", "px_B8mC1s768@m")]
            if sfx == "w":
                for L_ in sorted({int(m.group(1)) for c in s.columns for m in [re.match(r"px_vwf(\d+)@w$", c)] if m}):
                    pairs += [(f"J_vwf{L_}{c}|{rf}@w", f"J_vwf{L_}-{DET}{c}|{rf}") for c in ("", "+T4") for rf in ("none", "own")]
                    pairs += [(f"px_vwf{L_}@w", f"px_vwf{L_}-{DET}")]
            for a, b in pairs:
                if a in s and b in s:
                    m = s[b].notna() & s[a].notna()
                    par.append({"dataset": ds, "fu": a, "reference": b, "frames": int(m.sum()),
                                "identical": float(np.isclose(s.loc[m, a], s.loc[m, b]).mean())})
        # ---- family D for R3-2
        if oks["d"] is not None:
            sub = d[oks["d"]].reset_index(drop=True)
            bt = KA.Boot(sub)
            gs = [KA.g_of(bt.L, sub[f"J_B8mC1s768|{rf}@d"].to_numpy(float)) for rf in ("own", "own_s1", "own_s2")]
            seeds.append({"dataset": ds, "candidate": "R3-2 YOLOv8m + C1, crops 768", "key": "B8mC1s768", "units": bt.n,
                          "G_seed0": gs[0], "G_seed1": gs[1], "G_seed2": gs[2], "G_mean": float(np.mean(gs)),
                          "G_std": float(np.std(gs, ddof=1))})
        # ---- YOLOv8m whole frame (R4)
        if oks["w"] is not None:
            sub = d[oks["w"]].reset_index(drop=True)
            bt = KA.Boot(sub)
            for L_ in sorted({int(m.group(1)) for c in sub.columns for m in [re.match(r"px_vwf(\d+)@w$", c)] if m}):
                px = sub[f"px_vwf{L_}@w"]
                gfr = float((px * GF[DET]).mean() * nat / 640 ** 2) / gf0
                for comp in ("", "+T4", "+C1"):
                    k = f"vwf{L_}{comp}"
                    tmp = sub.assign(**{f"J_{k}|own": sub[f"J_{k}|own@w"], f"a_{k}|own": sub[f"a_{k}|own@w"],
                                        "J_v51S|v3a": sub["J_v51S|v3a@w"]})
                    r = {"dataset": ds, "detector": DET, "long_side": L_, "components": comp or "plain", "units": bt.n,
                         "G_no_refiner": KA.g_of(bt.L, sub[f"J_{k}|none@w"].to_numpy(float)), "tensor_px_mean": px.mean(),
                         "px_ratio": px.mean() / px0, "GFLOPs_ratio": gfr, "class": cls_of(gfr)}
                    r.update(KA.stats(tmp, bt, f"{k}|own"))
                    wf_rows.append(r)
        # ---- YOLOv8m heuristics at the S crop budget (R4)
        if oks["h"] is not None:
            sub = d[oks["h"]].reset_index(drop=True)
            bt = KA.Boot(sub)
            for c in [c for c in sub.columns if c.startswith("px_") and c.endswith("@h") and c != "px_v51S@h" and c != "px_cheap@h"]:
                k = c[3:-2]
                if f"J_{k}|own@h" not in sub:
                    continue
                px = sub[c]
                gfr = float(((sub.px_cheap * GF["yolov8s"] + (px - sub.px_cheap) * GF[DET]) * nat / 640 ** 2).mean()) / gf0
                tmp = sub.assign(**{f"J_{k}|own": sub[f"J_{k}|own@h"], f"a_{k}|own": sub[f"a_{k}|own@h"],
                                    "J_v51S|v3a": sub["J_v51S|v3a@h"]})
                r = {"dataset": ds, "row": k, "refiner": "own", "units": bt.n,
                     "G_no_refiner": KA.g_of(bt.L, sub[f"J_{k}|none@h"].to_numpy(float)), "tensor_px_mean": px.mean(),
                     "px_ratio": px.mean() / px0, "GFLOPs_ratio": gfr, "class": cls_of(gfr)}
                r.update(KA.stats(tmp, bt, f"{k}|own"))
                heur_rows.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(par).to_csv(OUT / "parity_fu.csv", index=False)
    pd.DataFrame(seeds).to_csv(OUT / "seeds_fu.csv", index=False)
    pd.DataFrame(wf_rows).to_csv(OUT / "basewf_fu.csv", index=False)
    pd.DataFrame(heur_rows).to_csv(OUT / "baseheur_fu.csv", index=False)
    print(pd.DataFrame(par).to_string(index=False))
    print(pd.DataFrame(seeds).to_string(index=False))
    print("K28 FU TABLES", len(wf_rows), "whole-frame rows,", len(heur_rows), "heuristic rows", flush=True)


if __name__ == "__main__":
    main()
