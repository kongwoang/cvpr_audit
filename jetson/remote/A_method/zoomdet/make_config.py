#!/usr/bin/env python
"""Writes zoomdet_nusc.py from ZoomDet's own VisDrone YOLOv8-s config (configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_
visdrone_saliency.py), changing only: data (nuScenes new-train / new-val COCO json), classes (the 80 COCO classes, so the
COCO-pretrained head is kept), image scale (736x736 square: their deformable_grid_generate assumes square inputs; 736^2 ~ the u960 tensor pixels) and the schedule (15 epochs, batch 8, mosaic
closed for the last 5; LR scaled by their optimizer constructor to the total batch). Everything else is theirs."""
import os
import re
from pathlib import Path

B = Path(os.environ["B"])
REPO = B / "third_party" / "zoomdet_yolo"
src = (REPO / "configs" / "yolov8" / "yolov8_s_syncbn_fast_8xb16-500e_visdrone_saliency.py").read_text()
D = B / "data" / "zoomdet_nusc"
IMG = B / "data" / "fovea_ft" / "plain" / "images"
rep = {
    r"^_base_ = .*$": f"_base_ = ['{REPO}/configs/_base_/default_runtime.py', '{REPO}/configs/_base_/det_p5_tta.py']",
    r"^data_root = .*$": f"data_root = '{B}/'",
    r"^train_ann_file = .*$": f"train_ann_file = '{D.relative_to(B)}/train.json'",
    r"^train_data_prefix = .*$": f"train_data_prefix = '{(IMG / 'train').relative_to(B)}/'",
    r"^val_ann_file = .*$": f"val_ann_file = '{D.relative_to(B)}/val.json'",
    r"^val_data_prefix = .*$": f"val_data_prefix = '{(IMG / 'val').relative_to(B)}/'",
    r"^train_batch_size_per_gpu = .*$": "train_batch_size_per_gpu = 8",
    r"^max_epochs = .*$": "max_epochs = 15  # Maximum training epochs",
    r"^close_mosaic_epochs = .*$": "close_mosaic_epochs = 5",
    r"^img_scale = .*$": "img_scale = (736, 736)  # width, height (their grid code assumes square inputs; 736^2 ~ u960 576x960 pixels)",
    r"^dataset_type = .*$": "dataset_type = 'YOLOv5CocoDataset'",
}
out = src
for k, v in rep.items():
    out, n = re.subn(k, v, out, flags=re.M)
    assert n == 1, (k, n)
# the 80 COCO classes (their file defines class_name = (...) across several lines)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from zoomdet_prepare import COCO80                     # the COCO 2017 names, as mmdet's CocoDataset.METAINFO
out, n = re.subn(r"^class_name = \((.|\n)*?\)\s*(#.*)?$", "class_name = " + repr(tuple(COCO80)), out, count=1, flags=re.M)
assert n == 1
(Path(__file__).parent / "zoomdet_nusc.py").write_text(out)
print("written; 80 COCO classes")
