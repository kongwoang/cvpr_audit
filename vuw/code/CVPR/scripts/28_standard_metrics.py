#!/usr/bin/env python
"""How does DSF-guided re-perception score on standard perception metrics? (KITTI mono, all frames, proxy crops)

Recall and precision at IoU >= 0.5 against the reference boxes (height >= 10 px) for: cheap (320), full (640), and the
DSF-region fusion (cheap outside the crop, full inside), over (a) every object, (b) objects in the braking corridor
within 80 m (reference geometry), (c) objects whose box centre lies in the DSF crop.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap.cache import DetCache
from rap.decision import KittiAdapter
from rap.mono import box_iou
from rap.paths import CACHE
from rap.risk import RiskConfig
from vtf import runmeta

cfg = RiskConfig()
rg = pd.read_pickle(sorted(glob.glob(str(runmeta.RAW / "*_region_KITTI_mono" / "region__*.pkl")))[0])
crops = {(r.seq, int(r.frame)): r.brake_crop for r in rg.itertuples()}
half_w = P.PlannerParams().corridor_half_w
acc = {k: {"tp": 0, "gt": 0, "det": 0} for k in
       [f"{m}|{pop}" for m in ("cheap", "full", "dsf_fused") for pop in ("all", "corridor", "in_crop")]}


def centre_in(b, crop):
    if crop is None or not len(b):
        return np.zeros(len(b), bool)
    u, v = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    return (u >= crop[0]) & (u <= crop[2]) & (v >= crop[1]) & (v <= crop[3])


for s in sorted(set(rg.seq)):
    geom = KittiAdapter.geometry(s)
    c = DetCache(Path(CACHE) / "det" / "cheap_320" / f"{s}.npz")
    f = DetCache(Path(CACHE) / "det" / "full_640" / f"{s}.npz")
    for i, fr in enumerate(c.frames):
        fr = int(fr)
        g = geom[geom["frame"] == fr]
        g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
        gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
        corr = (P.in_corridor(g["lat_min"], g["lat_max"], half_w) & (g["long_near"] <= 80)) if len(g) else np.zeros(0, bool)
        crop = crops[(s, fr)]
        dc, df = c.det(i), f.det(f.index[fr])
        bc = dc["xyxy"][dc["conf"] >= 0.25].astype(float)
        bf = df["xyxy"][df["conf"] >= 0.25].astype(float)
        fused = np.r_[bc[~centre_in(bc, crop)], bf[centre_in(bf, crop)]]
        gin = centre_in(gt, crop)
        for m, b in (("cheap", bc), ("full", bf), ("dsf_fused", fused)):
            iou = box_iou(gt, b) if len(gt) and len(b) else np.zeros((len(gt), len(b)))
            hit = iou.max(1) >= 0.5 if len(b) else np.zeros(len(gt), bool)
            dtp = iou.max(0) >= 0.5 if len(gt) else np.zeros(len(b), bool)
            for pop, gm, dm in (("all", np.ones(len(gt), bool), np.ones(len(b), bool)),
                                ("corridor", corr, np.ones(len(b), bool)),
                                ("in_crop", gin, centre_in(b, crop))):
                a = acc[f"{m}|{pop}"]
                a["tp"] += int(hit[gm].sum()); a["gt"] += int(gm.sum())
                if pop != "corridor":
                    a["det"] += int(dm.sum()); a.setdefault("dtp", 0); a["dtp"] += int(dtp[dm].sum())
rows = []
for k, a in acc.items():
    m, pop = k.split("|")
    rows.append({"mode": m, "population": pop, "recall": a["tp"] / max(a["gt"], 1), "n_gt": a["gt"],
                 "precision": (a.get("dtp", np.nan) / a["det"]) if a["det"] else np.nan})
res = pd.DataFrame(rows).pivot(index="population", columns="mode", values=["recall", "precision"])
res.to_csv(runmeta.FINAL / "d3_standard_metrics.csv")
print(res.round(3).to_string())
