#!/usr/bin/env python
"""D68: decision-aware copy-paste augmentation — build three YOLO training sets from nuScenes CAM_FRONT keyframes.

Train units: DEEP split 'train' (old + new scenes); validation: 'val' of the new scenes (plain images).
Labels: DEEP's projected ground-truth boxes (height >= 10 px), in COCO ids so the fine-tuned model is a drop-in for
DEEP's detector (vehicle -> 2 car, person -> 0, cyclist -> 1 bicycle).
Arms (identical except where pasted objects go; same number of pastes per image):
  plain   originals only
  rand    each image, with prob. 0.5, gets 1-2 objects pasted at ground cells drawn uniformly from the field grid
  dsf     the same count, drawn from the cells where an inserted object would RAISE the braking action given the
          ground-truth scene and ego speed (vtf.dsf.insertion_grid, v_close 0); images with no sensitive cell fall back
          to the uniform draw (so both arms paste exactly the same number of objects)
Pasted objects come from a bank of unoccluded, untruncated train objects (height >= 40 px), rescaled by range ratio.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rap import planner as P
from rap.cache import Camera
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import NUSCENES_TRAINVAL
from vtf import dsf
from vtf import speculative as SP

B = Path(os.environ["B"])
OUT = B / "data" / "d68"
COCO = {"vehicle": 2, "person": 0, "cyclist": 1}
W, H = 1600, 900


def yolo_line(c, b):
    x1, y1, x2, y2 = np.clip(b, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1])
    return f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"


def paste(img, labels, bank, cells, objs, cam):
    P2 = cam.calib.P2
    fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
    for (r, y), oi in zip(cells, objs):
        o = bank[oi]
        s = o["z"] / r
        h, w = int(round(o["crop"].shape[0] * s)), int(round(o["crop"].shape[1] * s))
        if h < 8 or w < 8 or h > H // 2:
            continue
        vb, uc = cy + fy * cam.cam_h / r, cx - fx * y / r
        x1, y1 = int(round(uc - w / 2)), int(round(vb - h))
        if x1 < 0 or y1 < 0 or x1 + w > W or y1 + h > H:
            continue
        patch = cv2.resize(o["crop"], (w, h), interpolation=cv2.INTER_AREA)
        m = np.ones((h, w), np.float32)
        k = max(1, min(h, w) // 10)                                    # feathered edges
        m = cv2.blur(np.pad(m[k:-k, k:-k], k) if min(h, w) > 2 * k else m, (2 * k + 1, 2 * k + 1))[..., None]
        roi = img[y1:y1 + h, x1:x1 + w].astype(np.float32)
        img[y1:y1 + h, x1:x1 + w] = (m * patch + (1 - m) * roi).astype(np.uint8)
        nb = np.array([x1, y1, x1 + w, y1 + h], float)
        keep = []
        for c, b in labels:                                            # drop labels now >= 60% covered
            ix = max(0, min(b[2], nb[2]) - max(b[0], nb[0])) * max(0, min(b[3], nb[3]) - max(b[1], nb[1]))
            keep.append(ix / max((b[2] - b[0]) * (b[3] - b[1]), 1e-6) < 0.6)
        labels = [lb for lb, k_ in zip(labels, keep) if k_] + [(COCO[o["coarse"]], nb)]
    return img, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--out", default="d68")
    args = ap.parse_args()
    global OUT
    OUT = B / "data" / args.out
    rng = np.random.default_rng(0)
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    train = sorted(sp[sp.split == "train"].seq)[: args.max_scenes or None]
    val = sorted(sp[(sp["set"] == "new") & (sp.split == "val")].seq)[: (args.max_scenes // 4) or None]
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    pp = P.PlannerParams()
    # object bank (train scenes, every 4th keyframe)
    bank = []
    for s in train:
        g = ad.geometry(s)
        toks = db.samples(by[s])
        ok = g[(g["truncated"] == 0) & (g["occluded"] == 0) & ((g["y2"] - g["y1"]) >= 40) & (g["x1"] > 1) &
               (g["x2"] < W - 2) & (g["long_near"] > 3)]
        for fr in sorted(set(ok["frame"]))[::4]:
            img = cv2.imread(str(db.image_path(toks[fr])))
            for o in ok[ok["frame"] == fr]:
                x1, y1, x2, y2 = (int(round(float(o[k]))) for k in ("x1", "y1", "x2", "y2"))
                bank.append({"crop": img[y1:y2, x1:x2].copy(), "z": float(o["long_near"]), "coarse": str(o["coarse"])})
    print(f"bank {len(bank)} objects", flush=True)
    grid = [(r, y) for r in SP.R_GRID for y in SP.Y_GRID]
    counts = {a: 0 for a in ("rand", "dsf")}
    for arm in ("plain", "rand", "dsf"):
        for sub in ("images/train", "labels/train", "images/val", "labels/val"):
            (OUT / arm / sub).mkdir(parents=True, exist_ok=True)
    stats = []
    for s in train + val:
        split = "train" if s in train else "val"
        g = ad.geometry(s)
        g = g[(g["y2"] - g["y1"]) >= 10]
        speeds = ad.speeds(s)
        cam = Camera("nuScenes", s)
        toks = db.samples(by[s])
        for fr, tk in enumerate(toks):
            gf = g[g["frame"] == fr]
            labels = [(COCO[str(o["coarse"])], np.array([o["x1"], o["y1"], o["x2"], o["y2"]], float)) for o in gf]
            src = db.image_path(tk)
            name = f"{s}_{fr:03d}"
            if split == "val":
                for arm in ("plain", "rand", "dsf"):
                    dst = OUT / arm / "images/val" / f"{name}.jpg"
                    if not dst.exists():
                        dst.symlink_to(src)
                    (OUT / arm / "labels/val" / f"{name}.txt").write_text("\n".join(yolo_line(c, b) for c, b in labels))
                continue
            dst = OUT / "plain" / "images/train" / f"{name}.jpg"
            if not dst.exists():
                dst.symlink_to(src)
            (OUT / "plain" / "labels/train" / f"{name}.txt").write_text("\n".join(yolo_line(c, b) for c, b in labels))
            n = int(rng.integers(1, 3)) if rng.random() < 0.5 else 0
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            gt = tuple(np.asarray(gf[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc"))
            sens = dsf.insertion_grid(gt, v, SP.R_GRID, SP.Y_GRID, SP.OBJ_W, pp, 0.0)
            scells = [(r, y) for i, r in enumerate(SP.R_GRID) for j, y in enumerate(SP.Y_GRID) if sens[i, j]]
            cells = {"rand": [grid[i] for i in rng.integers(0, len(grid), n)],
                     "dsf": [scells[i] for i in rng.integers(0, len(scells), n)] if scells
                     else [grid[i] for i in rng.integers(0, len(grid), n)]}
            objs = rng.integers(0, len(bank), n)                   # the same objects for both arms
            stats.append({"seq": s, "frame": fr, "n": n, "n_sensitive_cells": len(scells)})
            for arm in ("rand", "dsf"):
                dimg = OUT / arm / "images/train" / f"{name}.jpg"
                if n == 0:
                    if not dimg.exists():
                        dimg.symlink_to(src)
                    lab = labels
                else:
                    img, lab = paste(cv2.imread(str(src)), list(labels), bank, cells[arm], objs, cam)
                    cv2.imwrite(str(dimg), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    counts[arm] += len(lab) - len(labels)
                (OUT / arm / "labels/train" / f"{name}.txt").write_text("\n".join(yolo_line(c, b) for c, b in lab))
        print(f"  {s} ({split})", flush=True)
    names = {i: f"c{i}" for i in range(80)}
    names.update({0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"})
    for arm in ("plain", "rand", "dsf"):
        (OUT / arm / "data.yaml").write_text(f"path: {OUT / arm}\ntrain: images/train\nval: images/val\nnames:\n" +
                                            "".join(f"  {i}: {n}\n" for i, n in names.items()))
    st = pd.DataFrame(stats)
    st.to_csv(OUT / "paste_stats.csv", index=False)
    print(f"train images {len(st)}, pasted-into {int((st.n > 0).sum())}, frames with sensitive cells "
          f"{float((st.n_sensitive_cells > 0).mean()):.2f}; net label change rand {counts['rand']} dsf {counts['dsf']}")


if __name__ == "__main__":
    main()
