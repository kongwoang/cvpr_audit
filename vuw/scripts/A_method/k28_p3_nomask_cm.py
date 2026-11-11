#!/usr/bin/env python
"""K28 PHASE 3 (post-hoc, reporting only): AV2 val WITHOUT the ego mask, class-matched as in phase 2 (k28_analyze_test.py):
GFLOPs per frame (whole frame / YOLOv8s rows = px x GF; v6-M / L = draft YOLOv8s + crops with their detector), ratio vs
v5.1-S of the same run, class S / M / L / over (k28_analyze CAPS, TOL), each row against the v6 configuration of its class
(over -> v6-L), every v6 configuration with its own refiner, rows with v3a. Unit bootstrap 2,000, seed 0; verdict as K24.
Inputs: results/k28/p3/av2_test_{ours,basewf,baseheur}_nomask. Output: results/k28/p3/tables/class_matched_nomask_v6.csv
(one row per (row, v6 configuration)), headline_nomask_v6.csv (v6 vs v5.1-S).
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

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
_a = importlib.util.spec_from_file_location("k24a", B / "scripts" / "A_method" / "k24_analyze.py")
K24A = importlib.util.module_from_spec(_a); _a.loader.exec_module(K24A)
_b = importlib.util.spec_from_file_location("k28a", B / "scripts" / "A_method" / "k28_analyze.py")
KA = importlib.util.module_from_spec(_b); _b.loader.exec_module(KA)
P3 = REAL / "results" / "k28" / "p3"
NAT = 2048 * 1550
V6 = {"v6-S": ("v6S", "v3a", "yolov8s"), "v6-M": ("v6M", "own-yolov8m", "yolov8m"), "v6-L": ("v6L", "own-yolo11m", "yolo11m")}


def load(part):
    rows, fs = [], sorted(glob.glob(str(P3 / f"av2_test_{part}_nomask" / "*.pkl")))
    for f in fs:
        rows += pickle.load(open(f, "rb"))
    x = pd.DataFrame(rows)
    if len(x) and "gt" in x:
        x = x[x["gt"].astype(bool)].reset_index(drop=True)
    return x, len(fs)


def main():
    GF = dict(KA.GF)
    gfp = REAL / "results" / "k28" / "phase0" / "gflops_640.csv"
    if gfp.exists():
        GF.update(pd.read_csv(gfp).iloc[0].to_dict())
    d = None
    for part in ("ours", "basewf", "baseheur"):
        x, n = load(part)
        assert n == K24A.NUNITS["av2"], (part, n)
        keep = ["unit", "frame", "J_cheap"] + [c for c in x.columns if c != "J_cheap" and c.startswith(("J_", "px_")) and (d is None or c not in d)]
        d = x[keep] if d is None else d.merge(x[keep], on=["unit", "frame"], suffixes=("", "_r"))
        if "J_cheap_r" in d:
            assert np.allclose(d.J_cheap, d.J_cheap_r); d = d.drop(columns="J_cheap_r")
    bt = K24A.Boot(d)
    L = d.J_cheap.to_numpy(float)
    pc = d.px_cheap.to_numpy(float)
    gf0 = float((d.px_v51S * GF["yolov8s"]).mean())

    def ratio(px, det):
        px = np.asarray(px, float)
        g = px * GF["yolov8s"] if det == "yolov8s" else pc * GF["yolov8s"] + (px - pc) * GF[det]
        return float(g.mean()) / gf0

    def G(J):
        return float((L - J).sum() / L.sum())

    ours = {lab: (d[f"J_{k}|{rf}"].to_numpy(float), d[f"J_{k}|none"].to_numpy(float), ratio(d[f"px_{k}"], det), float(d[f"px_{k}"].mean()))
            for lab, (k, rf, det) in V6.items()}
    ours["v5.1-S"] = (d["J_v51S|v3a"].to_numpy(float), d["J_v51S|none"].to_numpy(float), 1.0, float(d.px_v51S.mean()))
    head = []
    for lab in V6:
        Jr, Jn, r_, px = ours[lab]
        m, lo, hi = bt.diff(ours["v5.1-S"][0], Jr)
        head.append({"dataset": "av2 (no ego mask)", "configuration": lab, "G_refiner": G(Jr), "G_no_refiner": G(Jn), "GFLOPs_ratio": r_,
                     "tensor_px_mean": px, "G_v51S": G(ours["v5.1-S"][0]), "minus_v51S": m, "lo": lo, "hi": hi, "verdict": K24A.verdict(lo)})
    rows = []
    keys = sorted({re.sub(r"\|.*", "", c[2:]) for c in d.columns if c.startswith("J_") and "|v3a" in c} - {"v6S", "v51S"})
    for k in keys:
        pxc = f"px_{k}" if f"px_{k}" in d else f"px_{re.sub(r'[+](T4|C1)$', '', k)}"
        r_ = ratio(d[pxc], "yolov8s")
        cls = next((c_ for c_, v in KA.CAPS.items() if r_ <= v * KA.TOL), "over")
        Jr = d[f"J_{k}|v3a"].to_numpy(float)
        for lab in V6:
            m, lo, hi = bt.diff(Jr, ours[lab][0])
            rows.append({"dataset": "av2 (no ego mask)", "row": k, "group": ("whole frame" if k.startswith("wf") else "heuristic")
                         + (" + C1" if k.endswith("+C1") else " + T4" if k.endswith("+T4") else " plain"),
                         "G_refiner": G(Jr), "tensor_px_mean": float(d[pxc].mean()), "GFLOPs_ratio": r_, "class": cls,
                         "configuration": lab, "class_matched": lab == {"S": "v6-S", "M": "v6-M"}.get(cls, "v6-L"),
                         "G_v6": G(ours[lab][0]), "v6_minus_row": m, "lo": lo, "hi": hi, "verdict": K24A.verdict(lo)})
    out = P3 / "tables"
    pd.DataFrame(head).to_csv(out / "headline_nomask_v6.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "class_matched_nomask_v6.csv", index=False)
    print("units", bt.n, "frames", len(d), "rows", len(rows), "K28 P3 NOMASK CM DONE")


if __name__ == "__main__":
    main()
