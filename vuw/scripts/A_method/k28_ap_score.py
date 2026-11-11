#!/usr/bin/env python
"""K28 PHASE 3 (reporting only): whole-image detection metrics of v6 against tensor px / GFLOPs, from k28_ap.py dumps.
  Argoverse-HD val (24 sequences, every frame): COCO AP / AP50 / AP_s / AP_m / AP_l on the three road-user groups of K7
      (argoverse_hd_eval.py's ground truth and grouping; outputs with conf >= 0.1, the K7 definition for crop pipelines).
  nuScenes new test (every 4th frame of each scene, as det_metrics.py): AP50 (truncated at conf 0.25), recall all / small /
      decision-relevant, FP per frame, relevant precision; DEEP's projected boxes (det_metrics.py's functions).
  px = mean tensor px / native px; GFLOPs per frame = draft (YOLOv8s) + crops with their detector, or whole frame x GF; ratio
  vs v5.1-S of the same dump. Parity: J / px of v5.1-S, v6-S / M / L and whole frame vs the phase-2 pass (k28_test.py).
outputs: results/k28/tables_v6/ap_ahd_v6.csv, detm_nusc_v6.csv, parity_ap_v6.csv
usage: k28_ap_score.py [--datasets ahd,nusc] [--step 4]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", str(B)))
R = REAL / "results" / "k28"
OUT = R / "tables_v6"
V6DET = {"v51S": "yolov8s", "v6S": "yolov8s", "v6M": "yolov8m", "v6L": "yolo11m"}
SPLIT, TAG = "test", ""                                                         # --split dev --tag smoke: smoke checks only


def load(ds, part):
    rows, dets, ids = [], {}, {}
    for f in sorted(glob.glob(str(R / f"{ds}_{SPLIT}_v6{part}{TAG}" / "*.pkl"))):
        z = pickle.load(open(f, "rb"))
        if isinstance(z, dict):
            u = Path(f).stem
            rows += z["rows"]
            dets.update({(u,) + k: v for k, v in z["dets"].items()})
            ids[u] = z.get("image_ids")
        else:
            rows += z
    return pd.DataFrame(rows), dets, ids


def det_of(m):
    if m in V6DET:
        return V6DET[m], False
    if m.startswith("vwf"):
        return m.split("-", 1)[1], True
    return "yolov8s", True                                                     # wf<L>, cheap


def px_gflops(df, methods, GF, nat):
    gf0 = float((df.px_v51S * GF["yolov8s"]).mean()) * nat / 640 ** 2
    out = {}
    for m in methods:
        px = df.px_cheap if m == "cheap" else df[f"px_{m}"]
        det, wf = det_of(m)
        g = px * GF[det] if (wf or det == "yolov8s") else df.px_cheap * GF["yolov8s"] + (px - df.px_cheap) * GF[det]
        out[m] = (float(px.mean()), float(g.mean()) * nat / 640 ** 2 / gf0)
    return out


def parity(ds, df):
    rows = []
    for part, pairs in (("ours", [("J_v51S|v3a", "J_v51S|v3a"), ("px_v51S", "px_v51S"), ("J_v6S|v3a", "J_v6S|v3a"),
                                  ("J_v6M|own-yolov8m", "J_v6M|own-yolov8m"), ("J_v6L|own-yolo11m", "J_v6L|own-yolo11m"),
                                  ("px_v6S", "px_v6S"), ("px_v6M", "px_v6M"), ("px_v6L", "px_v6L")]),
                        ("basewf-yolo11m", [(f"J_wf{L}|none", f"J_wf{L}|none") for L in (448, 512, 640, 960, 1280, 1536, 1600, 1920)]
                         + [(f"J_vwf{L}-yolo11m|none", f"J_vwf{L}|none") for L in (448, 512, 640, 768, 896, 960, 1280, 1536, 1600, 1920, 2048)]),
                        ("basewf-yolov8m", [(f"J_vwf{L}-yolov8m|none", f"J_vwf{L}|none") for L in (448, 512, 640, 768, 896, 960, 1280, 1536, 1600, 1920, 2048)])):
        p, _, _ = load(ds, part)
        if not len(p):
            continue
        m = df.merge(p, on=["unit", "frame"], suffixes=("", "@p2"))
        for a, b in pairs:
            b_ = b if b not in df.columns else f"{b}@p2"
            if a in m and b_ in m:
                rows.append({"dataset": ds, "apdump": a, "phase2": f"{part}:{b}", "frames": len(m),
                             "identical": float(np.isclose(m[a].astype(float), m[b_].astype(float)).mean())})
    return rows


def ahd(GF):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HDm = runpy.run_path(str(B / "scripts" / "A_method" / "argoverse_hd_eval.py"), run_name="ahd_lib")
    df, dets, ids = load("ahd", "apdump")
    d = json.loads((HDm["HD"] / "val.json").read_text())
    gt = {"images": d["images"], "categories": [{"id": i, "name": n} for i, n in enumerate(("vehicle", "person", "two-wheeler"))],
          "annotations": [{**a, "category_id": HDm["GROUP"][a["category_id"]]} for a in d["annotations"] if a["category_id"] in HDm["GROUP"]]}
    tmp = OUT / "ahd_gt_groups.json"
    tmp.write_text(json.dumps(gt))
    cg = COCO(str(tmp))
    methods = sorted({k[2] for k in dets})
    pg = px_gflops(df, methods, GF, 1920 * 1200)
    res = []
    for m in methods:
        out, img = [], set()
        for (u, fr, mm), o in dets.items():
            if mm != m or o["ap"] is None:
                continue
            iid = int(ids[u][fr]); img.add(iid)
            for b, c, k in zip(o["ap"]["xyxy"], o["ap"]["conf"], o["ap"]["coarse"]):
                out.append({"image_id": iid, "category_id": HDm["CGROUP"][str(k)], "score": float(c),
                            "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
        E = COCOeval(cg, cg.loadRes(out), "bbox")
        E.params.imgIds = sorted(img)
        E.evaluate(); E.accumulate(); E.summarize()
        det, wf = det_of(m)
        res.append({"dataset": "ahd", "method": m, "detector": det, "whole_frame": wf and m != "cheap", "frames": len(img),
                    "AP": E.stats[0], "AP50": E.stats[1], "AP_s": E.stats[3], "AP_m": E.stats[4], "AP_l": E.stats[5],
                    "tensor_px": pg[m][0], "GFLOPs_ratio": pg[m][1]})
    return pd.DataFrame(res), parity("ahd", df)


def nusc(GF, step):
    sys.path.insert(0, str(B / "scripts" / "A_method"))
    from det_metrics import CLASSES, THR, ap_allpoint, keep, match
    from rap import planner as P
    from rap.nusc import NuScenesDB, make_adapter
    from rap.paths import NUSCENES_TRAINVAL
    df, dets, _ = load("nusc", "apdump")
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    pp = P.PlannerParams()
    methods = sorted({k[2] for k in dets})
    pg = px_gflops(df, methods, GF, 1600 * 900)
    acc = {m: {"conf": {c: [] for c in CLASSES}, "tp": {c: [] for c in CLASSES}, "hit_all": [], "hit_small": [], "hit_rel": [],
               "fp": 0, "rel_det": 0, "rel_tp": 0} for m in methods}
    ngt = {c: 0 for c in CLASSES}
    nfr = 0
    for s in sorted({k[0] for k in dets}):
        geom, speeds = ad.geometry(s), ad.speeds(s)
        frames = sorted({k[1] for k in dets if k[0] == s})
        for i, fr in enumerate(frames):
            if i % step:
                continue
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            gx = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
            gc = np.asarray(g["coarse"]).astype(str) if len(g) else np.zeros(0, "<U8")
            small = (gx[:, 3] - gx[:, 1]) < 32
            rel = np.array([P.discrete_action(P.required_decel(g["long_near"][j:j + 1], g["lat_min"][j:j + 1],
                                                               g["lat_max"][j:j + 1], g["ttc"][j:j + 1], v, pp)[0], pp) > 0
                            for j in range(len(g))], bool)
            for c in CLASSES:
                ngt[c] += int((gc == c).sum())
            nfr += 1
            for m in methods:
                raw = dets[(s, fr, m)]["sc"]
                d = keep(raw)
                tp, hit = match(d, gx, gc)
                A = acc[m]
                for c in CLASSES:
                    kk = d["coarse"] == c
                    A["conf"][c].append(d["conf"][kk]); A["tp"][c].append(tp[kk])
                A["hit_all"].append(hit); A["hit_small"].append(hit[small]); A["hit_rel"].append(hit[rel])
                A["fp"] += int((~tp).sum())
                kk = np.asarray(raw["conf"], float) >= THR
                mono = [np.asarray(a, float)[kk] for a in raw["mono"]]
                own = np.array([P.discrete_action(P.required_decel(*(a[j:j + 1] for a in mono), v, pp)[0], pp) > 0
                                for j in range(len(d["conf"]))], bool)
                A["rel_det"] += int(own.sum()); A["rel_tp"] += int((own & tp).sum())
    rows = []
    for m in methods:
        A = acc[m]
        aps = [ap_allpoint(np.concatenate(A["conf"][c]), np.concatenate(A["tp"][c]), ngt[c]) for c in CLASSES]
        cat = lambda k: np.concatenate(A[k]) if A[k] else np.zeros(0)                    # noqa: E731
        det, wf = det_of(m)
        rows.append({"dataset": "nusc", "method": m, "detector": det, "whole_frame": wf and m != "cheap", "frames": nfr,
                     "AP50": np.nanmean(aps), **{f"AP50_{c}": a for c, a in zip(CLASSES, aps)},
                     "R_all": cat("hit_all").mean(), "R_small": cat("hit_small").mean(), "R_rel": cat("hit_rel").mean(),
                     "n_rel": len(cat("hit_rel")), "FP_per_frame": A["fp"] / nfr, "P_rel": A["rel_tp"] / max(A["rel_det"], 1),
                     "rel_FP_per_1k_frames": 1000 * (A["rel_det"] - A["rel_tp"]) / nfr,
                     "tensor_px": pg[m][0], "GFLOPs_ratio": pg[m][1]})
    return pd.DataFrame(rows), parity("nusc", df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ahd,nusc")
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    global SPLIT, TAG, OUT
    SPLIT, TAG = args.split, (f"_{args.tag}" if args.tag else "")
    if args.split != "test":
        OUT = REAL / "tmp" / "k28_apsmoke"
    _b = runpy.run_path(str(B / "scripts" / "A_method" / "k28_analyze.py"), run_name="k28a")
    GF = dict(_b["GF"])
    gfp = REAL / "results" / "k28" / "phase0" / "gflops_640.csv"
    if gfp.exists():
        GF.update(pd.read_csv(gfp).iloc[0].to_dict())
    OUT.mkdir(parents=True, exist_ok=True)
    par = []
    for ds in args.datasets.split(","):
        t, p = ahd(GF) if ds == "ahd" else nusc(GF, args.step)
        t.to_csv(OUT / ("ap_ahd_v6.csv" if ds == "ahd" else "detm_nusc_v6.csv"), index=False)
        par += p
        pd.set_option("display.width", 220)
        print(t.round(4).to_string(index=False))
    pd.DataFrame(par).to_csv(OUT / "parity_ap_v6.csv", index=False)
    print(pd.DataFrame(par).to_string(index=False))
    print("K28 AP SCORE DONE")


if __name__ == "__main__":
    main()
