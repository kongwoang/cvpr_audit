#!/usr/bin/env python
"""Refiner data v2: every TRAIN and VAL scene of nuScenes (old 85 + new 765; no test scene of either set), same
recipe as scripts/34_refiner_dataset.py: full-mode (384x640) detections matched to a projected GT box (IoU >= 0.3),
64x64 crop, 12 box features, target log(z_true / z_lift). Writes data/refiner_data__nuScenesAll.npz."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rap.cache import DetCache
from rap.mono import box_iou
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL
import sys
B = Path(os.environ["B"])
sys.path.insert(0, str(B))
from nusc_refiner_crop import crop64                                            # noqa: E402

CLS = {"vehicle": 0, "person": 1, "cyclist": 2}


def main():
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    sp = sp[sp.split.isin(["train", "val"])]
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    crops, feats, target, split, seq = [], [], [], [], []
    for s, spl in zip(sp.seq, sp.split):
        geom = ad.geometry(s)
        f = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_full_640" / f"{s}.npz")
        toks = db.samples(by[s])
        for i, fr in enumerate(f.frames):
            fr = int(fr)
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            d, geo = f.det(i), f.geo(i)
            k = d["conf"] >= 0.25
            if not len(g) or not k.any():
                continue
            bx = d["xyxy"][k].astype(float)
            iou = box_iou(bx, np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float))
            hit = np.flatnonzero(iou.max(1) >= 0.3)
            if not len(hit):
                continue
            img = cv2.imread(str(db.image_path(toks[fr])))
            H, W = img.shape[:2]
            z, zg, zh = (np.asarray(geo[x][k], float) for x in ("z", "z_ground", "z_height"))
            conf, coarse = d["conf"][k], d["coarse"][k]
            for o in hit:
                zt = float(g["long_near"][int(np.argmax(iou[o]))])
                if zt <= 0.5 or z[o] <= 0.5:
                    continue
                x1, y1, x2, y2 = bx[o]
                crops.append(crop64(img, bx[o]))
                feats.append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(conf[o]),
                              float(CLS[str(coarse[o])]), z[o], zg[o], zh[o], float(np.log(max(zg[o], .5) / max(zh[o], .5)))])
                target.append(float(np.log(zt / z[o])))
                split.append(spl)
                seq.append(s)
        print(f"  {s} ({spl}): {len(target)} boxes", flush=True)
    np.savez_compressed(B / "data" / "refiner_data__nuScenesAll.npz", crops=np.asarray(crops, np.uint8),
                        feats=np.asarray(feats, np.float32), target=np.asarray(target, np.float32),
                        split=np.asarray(split), seq=np.asarray(seq))
    t = pd.DataFrame({"split": split, "t": np.abs(target)})
    print(t.groupby("split").t.agg(["count", "mean"]).to_string())


if __name__ == "__main__":
    main()
