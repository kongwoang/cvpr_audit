#!/usr/bin/env python
"""B1 step 1: dataset for an image-level range refiner.

Every full-mode (640) detection matched to a reference box (IoU >= 0.5 KITTI, >= 0.3 nuScenes, whose projected boxes
are loose) on ALL units, with the split recorded so that training uses train units only: a 64x64 crop of the original
image around the box (1.5x context), 12 box features (as in 20_learned_refiner.py), the monocular lift range and the
reference range.  Target: log(reference range / lifted range).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rap.cache import DetCache
from rap.decision import KittiAdapter
from rap.mono import box_iou
from rap.paths import CACHE, KITTI_IMAGES, NUSCENES_TRAINVAL
from vtf import runmeta

CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
S, CTXF = 64, 1.5
DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"


def crop64(img, box):
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * CTXF, (y2 - y1) * CTXF
    side = max(w, h, 8.0)
    X1, Y1, X2, Y2 = int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)
    H, W = img.shape[:2]
    pad = cv2.copyMakeBorder(img, max(-Y1, 0), max(Y2 - H, 0), max(-X1, 0), max(X2 - W, 0), cv2.BORDER_CONSTANT, value=(114, 114, 114))
    oy, ox = max(-Y1, 0), max(-X1, 0)
    patch = pad[Y1 + oy:Y2 + oy, X1 + ox:X2 + ox]
    return cv2.resize(patch, (S, S), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["KITTI", "nuScenes"], required=True)
    args = ap.parse_args()
    run = runmeta.new_run(f"refiner_data_{args.dataset}", vars(args))
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str})
    dv = dv[(dv.track == args.dataset) & (dv.system == "brake") & (dv.geometry == "mono")]
    split = dict(zip(dv.seq, dv.split))
    if args.dataset == "KITTI":
        det_dir, fm, adapter, iou_thr = Path(CACHE) / "det", "full_640", KittiAdapter, 0.5
        img_of = lambda s, fr: Path(KITTI_IMAGES) / s / f"{fr:06d}.png"                   # noqa: E731
    else:
        from rap.nusc import NuScenesDB, make_adapter
        db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
        det_dir, fm, adapter, iou_thr = Path(CACHE) / "nusc_det_tv", "ns_full_640", make_adapter(db), 0.3
        by_name = {db.scene_name(t): t for t in db.scenes}
        img_of = lambda s, fr: db.image_path(db.samples(by_name[s])[fr])                    # noqa: E731
    crops, feats, meta = [], [], []
    for s in sorted(split):
        geom = adapter.geometry(s)
        f = DetCache(det_dir / fm / f"{s}.npz")
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
            hit = np.flatnonzero(iou.max(1) >= iou_thr)
            if not len(hit):
                continue
            img = cv2.imread(str(img_of(s, fr)))
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
                meta.append((s, fr, split[s], float(np.log(zt / z[o])), zt, z[o]))
        print(f"  {s}: {len(meta)} boxes so far", flush=True)
    m = pd.DataFrame(meta, columns=["seq", "frame", "split", "target", "z_ref", "z_lift"])
    np.savez_compressed(run / f"refiner_data__{args.dataset}.npz", crops=np.asarray(crops, np.uint8),
                        feats=np.asarray(feats, np.float32), target=m.target.to_numpy(np.float32),
                        split=m.split.to_numpy(), seq=m.seq.to_numpy(), frame=m.frame.to_numpy())
    print(m.groupby("split").target.agg(["count", lambda x: float(np.mean(np.abs(x)))]).to_string(), flush=True)


if __name__ == "__main__":
    main()
