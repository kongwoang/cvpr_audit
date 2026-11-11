#!/usr/bin/env python
"""AV2 data for the fine-tuned baselines (u960-ft, verifier-ft, ESOD, ZoomDet), main venv, CPU.

Training frames come only from AV2 sensor TRAIN logs (the AV2 dev split), with the same log split as av2_sota_labels.py
(md5 first hex digit 0-2 -> val): train = every 5th frame of DEEP's 5 Hz AV2 frames (1 Hz) of the train logs, val =
every 10th of the val logs. Images are the front-centre frames downscaled to long side 1536 (1162 x 1536, ESOD's
training size; every detector trains on the same images). Labels = DEEP's projected GT boxes (height >= 10 px), as on
nuScenes:
  data/av2_ft/images/{train,val}/<log>_<fr>.jpg, labels/{train,val}/*.txt   YOLO, COCO ids (person 0, bicycle 1, car 2)
  data/av2_ft/data.yaml                                                       ultralytics, 80 COCO names
  data/av2_ft/esod/{train,val}/{images -> symlinks, labels}                    ESOD layout, 0 vehicle / 1 pedestrian / 2 cyclist
  data/av2_ft/zoomdet/{train,val}.json                                        COCO json (80 categories)
  data/av2_ft/test/images/<log>_<fr>.jpg -> the native AV2 VAL frames (all, the test set), + test_list.txt (name path)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from vtf import av2 as AV

B = Path(os.environ["B"])
sys.path.insert(0, str(B / "scripts" / "A_method" / "zoomdet"))
from zoomdet_prepare import COCO80                                            # noqa: E402

D = B / "data" / "av2_ft"
COCO = {"vehicle": 2, "person": 0, "cyclist": 1}
ESOD = {"vehicle": 0, "person": 1, "cyclist": 2}
LONG = 1536


def split_of(log):
    return "val" if hashlib.md5(log.encode()).hexdigest()[0] in "012" else "train"


def main():
    logs = sorted(p.stem for p in (B / "datasets" / "av2" / "lists" / "train").glob("*.done"))
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(COCO80)]
    for split, step in (("train", 5), ("val", 10)):
        for sub in (f"images/{split}", f"labels/{split}", f"esod/{split}/images", f"esod/{split}/labels"):
            (D / sub).mkdir(parents=True, exist_ok=True)
        imgs, anns, paths = [], [], []
        sl = [l for l in logs if split_of(l) == split]
        for li, log in enumerate(sl):
            L = AV.AV2Log(B / "datasets" / "av2" / "sensor" / "train", log)
            geom = L.geometry()
            for fr in range(0, len(L.frame_ts), step):
                name = f"{log}_{fr:03d}"
                ip = D / "images" / split / f"{name}.jpg"
                if not ip.exists():
                    img = cv2.imread(str(L.image_path(fr)))
                    if img is None:
                        continue
                    h0, w0 = img.shape[:2]
                    s = LONG / max(h0, w0)
                    cv2.imwrite(str(ip), cv2.resize(img, (int(round(w0 * s)), int(round(h0 * s))), interpolation=cv2.INTER_AREA),
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                g = geom[geom["frame"] == fr] if len(geom) else geom
                g = g[(g["y2"] - g["y1"]) >= 10] if len(g) else g
                W, H = L.W, L.H
                yc, ye, ok = [], [], []
                for o in g:
                    x1, y1, x2, y2 = (float(o[k]) for k in ("x1", "y1", "x2", "y2"))
                    nb = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)
                    yc.append("%d %.6f %.6f %.6f %.6f" % ((COCO[str(o["coarse"])],) + nb))
                    ye.append("%d %.6f %.6f %.6f %.6f" % ((ESOD[str(o["coarse"])],) + nb))
                    ok.append((COCO[str(o["coarse"])], nb))
                (D / "labels" / split / f"{name}.txt").write_text("\n".join(yc) + ("\n" if yc else ""))
                (D / "esod" / split / "labels" / f"{name}.txt").write_text("\n".join(ye) + ("\n" if ye else ""))
                el = D / "esod" / split / "images" / f"{name}.jpg"
                if not el.is_symlink():
                    el.symlink_to(ip)
                paths.append(str(el))
                wi, hi = int(round(W * LONG / max(W, H))), int(round(H * LONG / max(W, H)))
                k = len(imgs)
                imgs.append({"id": k, "file_name": ip.name, "width": wi, "height": hi})
                for c, (cx, cy, bw, bh) in ok:
                    anns.append({"id": len(anns) + 1, "image_id": k, "category_id": c + 1, "iscrowd": 0,
                                 "bbox": [(cx - bw / 2) * wi, (cy - bh / 2) * hi, bw * wi, bh * hi], "area": bw * wi * bh * hi})
            if li % 50 == 0:
                print(f"  {split} [{li + 1}/{len(sl)}] {len(imgs)} images", flush=True)
        (D / "esod" / "split").mkdir(parents=True, exist_ok=True)
        (D / "esod" / "split" / f"{split}.txt").write_text("\n".join(paths) + "\n")
        (D / "zoomdet").mkdir(parents=True, exist_ok=True)
        (D / "zoomdet" / f"{split}.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": cats}))
        print(f"{split}: {len(sl)} logs, {len(imgs)} images, {len(anns)} boxes", flush=True)
    names = {i: n for i, n in enumerate(COCO80)}
    (D / "data.yaml").write_text(f"path: {D}\ntrain: images/train\nval: images/val\nnames:\n" +
                                 "".join(f"  {i}: {n}\n" for i, n in names.items()))
    # test = AV2 sensor VAL, every DEEP frame (as av2_eval_v4 --split val), native resolution
    root = B / "datasets" / "av2" / "sensor" / "val"
    (D / "test" / "images").mkdir(parents=True, exist_ok=True)
    lines = []
    for log in sorted(p.name for p in root.iterdir() if (p / "annotations.feather").exists()):
        L = AV.AV2Log(root, log)
        for fr in range(len(L.frame_ts)):
            src = L.image_path(fr)
            if not Path(src).exists():
                continue
            name = f"{log}_{fr:03d}"
            ln = D / "test" / "images" / f"{name}.jpg"
            if not ln.is_symlink():
                ln.symlink_to(src)
            lines.append(f"{name} {ln}")
    (D / "test_list.txt").write_text("\n".join(lines) + "\n")
    (D / "esod" / "split" / "test.txt").write_text("\n".join(l.split(" ", 1)[1] for l in lines) + "\n")
    print(f"test: {len(lines)} frames")
    print("AV2 FT PREPARE DONE")


if __name__ == "__main__":
    main()
