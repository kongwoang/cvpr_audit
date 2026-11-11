#!/usr/bin/env python
"""K28 PHASE 2 tables (CONFIRMATORY, method_freeze_draft §2n; K24 protocol, suffix _v6). Inputs: the K24 test pass
(k24_analyze.merged: results/k24/<ds>_main + _supp; its rows and routers unchanged) and the K28 test pass
(results/k28/<ds>_test_v6{ours,basewf-<det>,baseheur,baseheur-<det>}; k28_test.py), merged on unit / frame (J_cheap checked;
the whole-frame rows of the detector <det> are renamed vwf<L>-<det>).
Argoverse-HD: frames with 3D labels only. Unit bootstrap 2,000, seed 0, by scene / log / sequence (k24_analyze.Boot).
Refuses to write tables until every part has all units.
  table_main_<ds>_v6.csv   every row: G without / with refiner, tensor px mean / p95, GFLOPs ratio and class vs v5.1-S
                           (test), for P in (v6-S, v6-M, v6-L, v5.1-S): (P - row) at the same refiner setting, interval,
                           verdict (outperforms: lo > 0; matches: lo > -0.02; else below); K17(3) regime G, unnecessary /
                           missed per 1,000 (with refiner)
  headline_v6.csv          v6-S / v6-M / v6-L vs v5.1-S (K24's frozen configuration), per dataset
  class_matched_v6.csv     every baseline row vs the v6 configuration of its class (S / M / L by GFLOPs ratio; over-budget
                           rows vs v6-L, flagged)
  not_outperformed_v6.csv  every (v6 configuration, baseline row) whose interval with refiner is not above 0
  parity_v6.csv            K28 test v5.1-S = K24 v5.1-S; K28 whole frame 960 (plain / + T4) = K24 u960; K28 K24-form
                           heuristics vs K24's K18-protocol rows (informational)
usage: k28_analyze_test.py [--parts ours,basewf-yolo11m,basewf-yolov8m,baseheur,baseheur-yolo11m,baseheur-yolov8m]
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
_b = importlib.util.spec_from_file_location("k28a", B / "scripts" / "A_method" / "k28_analyze.py")
KA = importlib.util.module_from_spec(_b); _b.loader.exec_module(KA)
NATIVE = {"nusc": 1600 * 900, "av2": 2048 * 1550, "ahd": 1920 * 1200}
V6 = {"v6-S": ("v6S", "v3a", "yolov8s"), "v6-M": ("v6M", "own-yolov8m", "yolov8m"), "v6-L": ("v6L", "own-yolo11m", "yolo11m")}
K24FORM = {"unc2": "uncertainty", "tcr3": "temporal_crit", "cor8": "corridor", "tsp8": "traj_support", "rnd": "random"}


def load(ds, part):
    rows, fs = [], sorted(glob.glob(str(REAL / "results" / "k28" / f"{ds}_test_v6{part}" / "*.pkl")))
    for f in fs:
        rows += pickle.load(open(f, "rb"))
    d = pd.DataFrame(rows)
    if part.startswith("basewf-"):
        d.columns = [re.sub(r"_vwf(\d+)(?=[|+]|$)", rf"_vwf\1-{part[7:]}", c) for c in d.columns]
    return d, len(fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ours,basewf-yolo11m,basewf-yolov8m,baseheur,baseheur-yolo11m,baseheur-yolov8m")
    args = ap.parse_args()
    gfp = REAL / "results" / "k28" / "phase0" / "gflops_640.csv"
    GF = dict(KA.GF)
    if gfp.exists():
        GF.update(pd.read_csv(gfp).iloc[0].to_dict())
    data = {}
    for ds in ("nusc", "av2", "ahd"):
        d, why = K24A.merged(ds, "main", "supp")
        if d is None:
            print(f"NOT READY: {why}; no table written"); return
        for part in args.parts.split(","):
            p, n = load(ds, part)
            if n != K24A.NUNITS[ds]:
                print(f"NOT READY: {ds} {part} {n} of {K24A.NUNITS[ds]} units; no table written"); return
            keep = ["unit", "frame", "J_cheap"] + [c for c in p.columns if c != "J_cheap" and c.startswith(("J_", "a_", "px_", "cpx_", "rounds_", "ncrops_"))]
            p = p[keep].rename(columns={c: f"{c}@k28" for c in keep[3:] if c in d.columns})   # e.g. v5.1-S of the K28 pass
            keep = list(p.columns)
            n0 = len(d)
            d = d.merge(p[keep], on=["unit", "frame"], how="left", suffixes=("", f"_{part}"))
            assert len(d) == n0 and d[f"J_cheap_{part}"].notna().all(), (ds, part)
            assert np.allclose(d.J_cheap.to_numpy(float), d[f"J_cheap_{part}"].to_numpy(float)), (ds, part)
        data[ds] = d
    out = REAL / "results" / "k28" / "tables_v6"
    out.mkdir(parents=True, exist_ok=True)
    head, cm, no, par = [], [], [], []
    for ds, d in data.items():
        L = d.J_cheap.to_numpy(float)
        bt = K24A.Boot(d)
        nat = NATIVE[ds]
        gf0 = float((d.px_v51S * GF["yolov8s"]).mean()) * nat / 640 ** 2

        def gfl(px, det, wf=False):
            """GFLOPs per frame: whole frame = px x GF; crop pipelines = draft (YOLOv8s) + crops with their detector"""
            px = np.asarray(px, float)
            g = px * GF[det] if (wf or det == "yolov8s") else d.px_cheap * GF["yolov8s"] + (px - d.px_cheap) * GF[det]
            return float(np.mean(g)) * nat / 640 ** 2

        # ---- parity
        for a, b in (("J_v51S|v3a@k28", "J_v51S|v3a"), ("J_v51S|none@k28", "J_v51S|none"), ("px_v51S@k28", "px_v51S"),
                     ("J_wf960|v3a", "J_u960|v3a"), ("J_wf960+T4|v3a", "J_u960+T4|v3a")) + tuple(
                (f"J_{k}@S{s}|v3a", f"J_{v}_Sx1{s}|v3a") for k, v in K24FORM.items() for s in ("", "+T4")):
            if a in d and b in d:
                par.append({"dataset": ds, "k28": a, "k24": b, "frames": len(d), "identical": float(np.isclose(d[a], d[b]).mean()),
                            "informational": a.startswith(("J_unc", "J_tcr", "J_cor", "J_tsp", "J_rnd"))})
        # ---- methods: K24 rows + routers + K28 rows
        M = K24A.methods(d, ds)
        M.update(K24A.routers(d, ds))
        for lab, (k, rf, det) in V6.items():
            M[lab] = {"Jn": d[f"J_{k}|none"], "Jr": d[f"J_{k}|{rf}"], "An": d[f"a_{k}|none"], "Ar": d[f"a_{k}|{rf}"],
                      "px": d[f"px_{k}"], "group": "ours v6", "section": "v6", "det": det}
        for c in sorted(d.columns):
            m = re.match(r"px_(v?wf)(\d+)(-[a-z0-9]+)?$", c)
            if m:
                pre, Ls = m.group(1), m.group(2)
                det, rf = ("yolov8s", "v3a") if pre == "wf" else (m.group(3)[1:], "own")
                for suf, grp in (("", "whole frame"), ("+T4", "whole frame + T4"), ("+C1", "whole frame + C1")):
                    k = f"{pre}{Ls}{m.group(3) or ''}{suf}"
                    M[f"K28 whole frame {det} {Ls}{' ' + suf if suf else ''}"] = {
                        "Jn": d[f"J_{k}|none"], "Jr": d[f"J_{k}|{rf}"], "An": d[f"a_{k}|none"], "Ar": d[f"a_{k}|{rf}"],
                        "px": d[c], "group": f"K28 {grp}", "section": "K28 E / R4", "det": det, "wf": True}
            m = re.match(r"px_((unc|tcr|cor|tsp|rnd)\d*@[SML])(-[a-z0-9]+)?(\+T4|\+C1)?$", c)
            if m:
                k = c[3:]
                det, rf = ("yolov8s", "v3a") if not m.group(3) else (m.group(3)[1:], "own")
                M[f"K28 heuristic {k}"] = {"Jn": d[f"J_{k}|none"], "Jr": d[f"J_{k}|{rf}"], "An": d[f"a_{k}|none"],
                                           "Ar": d[f"a_{k}|{rf}"], "px": d[c], "group": "K28 heuristic" + (m.group(4) or " plain"),
                                           "section": "K28 E / R4", "det": det}
        rows = []
        for lab, e in M.items():
            Jn, Jr = np.asarray(e["Jn"], float), np.asarray(e["Jr"], float)
            px = np.asarray(e["px"], float)
            det = e.get("det", "yolov8s")
            ratio = gfl(px, det, e.get("wf", False)) / gf0
            cls = next((c_ for c_, v in KA.CAPS.items() if ratio <= v * KA.TOL), "over")
            row = {"dataset": ds, "units": bt.n, "frames": len(d), "section": e["section"], "method": lab, "group": e["group"],
                   "crop_detector": det, "G_no_refiner": (L - Jn).sum() / L.sum(), "G_refiner": (L - Jr).sum() / L.sum(),
                   "tensor_px_mean": float(px.mean()), "tensor_px_p95": float(np.quantile(px, .95)),
                   "GFLOPs_ratio": ratio, "class": cls}
            row.update(K24A.extras(d, Jr, e["Ar"]))
            for P_ in list(V6) + ["v5.1-S"]:
                Pn, Pr = (M[P_]["Jn"], M[P_]["Jr"]) if P_ in V6 else (d["J_v51S|none"], d["J_v51S|v3a"])
                for set_, Jm, Jp in (("no_refiner", Jn, np.asarray(Pn, float)), ("refiner", Jr, np.asarray(Pr, float))):
                    mdiff, lo, hi = bt.diff(Jm, Jp)
                    tg = P_.replace(".", "").replace("-", "")
                    row[f"{tg}_minus_{set_}"], row[f"{tg}_minus_{set_}_lo"], row[f"{tg}_minus_{set_}_hi"] = mdiff, lo, hi
                    row[f"{tg}_verdict_{set_}"] = K24A.verdict(lo)
                    if set_ == "refiner" and P_ in V6 and e["group"] not in ("ours v6",) and lo <= 0:
                        no.append({"dataset": ds, "configuration": P_, "section": e["section"], "baseline": lab, "group": e["group"],
                                   "diff": mdiff, "lo": lo, "hi": hi, "verdict": K24A.verdict(lo), "baseline_class": cls,
                                   "baseline_GFLOPs_ratio": ratio})
            if e["group"] not in ("ours", "ours v6"):
                P_ = {"S": "v6-S", "M": "v6-M", "L": "v6-L"}.get(cls, "v6-L")
                tg = P_.replace("-", "")
                cm.append({"dataset": ds, "baseline": lab, "group": e["group"], "class": cls, "GFLOPs_ratio": ratio,
                           "G_baseline": row["G_refiner"], "configuration": P_, "diff": row[f"{tg}_minus_refiner"],
                           "lo": row[f"{tg}_minus_refiner_lo"], "hi": row[f"{tg}_minus_refiner_hi"],
                           "verdict": row[f"{tg}_verdict_refiner"], "over_budget": cls == "over"})
            if lab in V6:
                head.append({"dataset": ds, "configuration": lab, "G_no_refiner": row["G_no_refiner"], "G_refiner": row["G_refiner"],
                             "GFLOPs_ratio": ratio, "class": cls, "minus_v51S": -row["v51S_minus_refiner"],
                             "lo": -row["v51S_minus_refiner_hi"], "hi": -row["v51S_minus_refiner_lo"],
                             "verdict": K24A.verdict(-row["v51S_minus_refiner_hi"])})
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / f"table_main_{ds}_v6.csv", index=False)
        print(f"{ds}: {bt.n} units, {len(d)} frames, {len(rows)} rows")
    pd.DataFrame(head).to_csv(out / "headline_v6.csv", index=False)
    pd.DataFrame(cm).to_csv(out / "class_matched_v6.csv", index=False)
    pd.DataFrame(no).to_csv(out / "not_outperformed_v6.csv", index=False)
    pd.DataFrame(par).to_csv(out / "parity_v6.csv", index=False)
    print("tables written to", out)


if __name__ == "__main__":
    main()
