#!/usr/bin/env python
"""D75: decision-focused source-free adaptation to night, nuScenes CAM_FRONT.

Adaptation data: the NIGHT scenes of the new train + val splits, WITHOUT labels. Pseudo-labels: the full mode (COCO
YOLOv8s at 384x640, cached ns_full_640 detections, conf >= 0.25) in COCO ids. Arms (fine-tune the cheap model at 320):
  plain     every adaptation image once
  decision  images where the full mode sees an object in the braking corridor within 50 m that the cheap mode missed
            (IoU < 0.3 with every cheap box) are repeated 3x — adaptation effort goes where the decision is at stake
Validation: the night val scenes with the same pseudo-labels (no ground truth used anywhere in training).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap.cache import DetCache
from rap.mono import box_iou
from rap.nusc import NuScenesDB
from rap.paths import CACHE, NUSCENES_TRAINVAL
import json

B = Path(os.environ["B"])
OUT = B / "data" / "d75"
COCO = {"vehicle": 2, "person": 0, "cyclist": 1}
W, H = 1600, 900


def line(c, b):
    x1, y1, x2, y2 = np.clip(b, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1])
    return f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"


def main():
    sc = json.load(open(Path(NUSCENES_TRAINVAL) / "v1.0-trainval" / "scene.json"))
    night = {s["name"] for s in sc if "night" in s["description"].lower()}
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    sp = sp[(sp["set"] == "new") & sp.seq.isin(night)]
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    by = {db.scene_name(t): t for t in db.scenes}
    pp = P.PlannerParams()
    stats = []
    for arm in ("plain", "decision"):
        for sub in ("images/train", "labels/train", "images/val", "labels/val"):
            (OUT / arm / sub).mkdir(parents=True, exist_ok=True)
    for s, split in zip(sp.seq, sp.split):
        if split == "test":
            continue
        sub = "train" if split == "train" else "val"
        cc = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
        cf = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_full_640" / f"{s}.npz")
        toks = db.samples(by[s])
        for i, fr in enumerate(cf.frames):
            fr = int(fr)
            df, gf = cf.det(i), cf.geo(i)
            kf = df["conf"] >= 0.25
            bf, cof = df["xyxy"][kf].astype(float), df["coarse"][kf]
            dc = cc.det(cc.index[fr])
            bc = dc["xyxy"][dc["conf"] >= 0.25].astype(float)
            z, lo, hi = (np.asarray(gf[k][kf], float) for k in ("z", "lat_min", "lat_max"))
            rel = P.in_corridor(lo, hi, pp.corridor_half_w) & (z <= 50)
            missed = (box_iou(bf, bc).max(1) < 0.3) if len(bc) else np.ones(len(bf), bool)
            key = bool((rel & missed).any())
            text = "\n".join(line(COCO[str(c)], b) for c, b in zip(cof, bf))
            src = db.image_path(toks[fr])
            name = f"{s}_{fr:03d}"
            for arm in ("plain", "decision"):
                reps = 3 if (arm == "decision" and key and sub == "train") else 1
                for r in range(reps):
                    nm = f"{name}_r{r}"
                    dst = OUT / arm / f"images/{sub}" / f"{nm}.jpg"
                    if not dst.exists():
                        dst.symlink_to(src)
                    (OUT / arm / f"labels/{sub}" / f"{nm}.txt").write_text(text)
            stats.append({"seq": s, "frame": fr, "split": sub, "key": key})
    names = {i: f"c{i}" for i in range(80)}
    names.update({0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"})
    for arm in ("plain", "decision"):
        (OUT / arm / "data.yaml").write_text(f"path: {OUT / arm}\ntrain: images/train\nval: images/val\nnames:\n" +
                                            "".join(f"  {i}: {n}\n" for i, n in names.items()))
    st = pd.DataFrame(stats)
    st.to_csv(OUT / "stats.csv", index=False)
    print(st.groupby("split").agg(frames=("key", "size"), key_frames=("key", "sum")).to_string())


if __name__ == "__main__":
    main()
