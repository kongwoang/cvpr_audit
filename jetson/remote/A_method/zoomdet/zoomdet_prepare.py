#!/usr/bin/env python
"""ZoomDet (Wang et al., ISPRS JPRS 2026; official zoomdet_yolo @ c98679a) on nuScenes: COCO-json annotations for the
new-TRAIN / new-VAL keyframes of data/fovea_ft/plain (the 576x960 images every fine-tuned baseline trains on; YOLO labels
in COCO ids person 0 / bicycle 1 / car 2 -> COCO category ids 1 / 2 / 3, all 80 COCO categories listed so the COCO-
pretrained head is kept), and the list of new-TEST keyframe images for inference. Main venv."""
import json
import os
from pathlib import Path

import pandas as pd

from rap.nusc import NuScenesDB
from rap.paths import NUSCENES_TRAINVAL

B = Path(os.environ["B"])
SRC = B / "data" / "fovea_ft" / "plain"
OUT = B / "data" / "zoomdet_nusc"
COCO80 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
          'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
          'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
          'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
          'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
          'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
          'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(COCO80)]
    for split in ("train", "val"):
        imgs, anns = [], []
        for k, ip in enumerate(sorted((SRC / "images" / split).glob("*.jpg"))):
            W, H = 960, 576
            imgs.append({"id": k, "file_name": ip.name, "width": W, "height": H})
            lp = SRC / "labels" / split / (ip.stem + ".txt")
            for line in (lp.read_text().split("\n") if lp.exists() else []):
                if not line.strip():
                    continue
                c, cx, cy, w, h = (float(x) for x in line.split())
                bw, bh = w * W, h * H
                anns.append({"id": len(anns) + 1, "image_id": k, "category_id": int(c) + 1, "iscrowd": 0,
                             "bbox": [cx * W - bw / 2, cy * H - bh / 2, bw, bh], "area": bw * bh})
        (OUT / f"{split}.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": cats}))
        print(split, len(imgs), "images", len(anns), "boxes", flush=True)
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    by = {db.scene_name(t): t for t in db.scenes}
    lines = []
    for s in sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq):
        for fr, tk in enumerate(db.samples(by[s])):
            lines.append(f"{s}_{fr:03d} {db.image_path(tk)}")
    (OUT / "test_list.txt").write_text("\n".join(lines) + "\n")
    print("test", len(lines), "keyframes")
    print("ZOOMDET PREPARE DONE")


if __name__ == "__main__":
    main()
