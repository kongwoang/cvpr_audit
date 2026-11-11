#!/usr/bin/env python
"""Compute of SAHI on KITTI frames (1242x375) vs the deployed pipelines: detector tensor pixels per frame and wall-clock
(torch FP16 for SAHI via its library; the same torch YOLOv8s on the whole frame at 288x960 for a like-for-like ratio)."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".pydeps"))
from sahi import AutoDetectionModel                                              # noqa: E402
from sahi.predict import get_sliced_prediction                                   # noqa: E402
from sahi.slicing import get_slice_bboxes                                        # noqa: E402
from ultralytics import YOLO                                                     # noqa: E402

from rap.paths import KITTI_IMAGES, MODELS                                       # noqa: E402

imgs = sorted((Path(KITTI_IMAGES) / "0000").glob("*.png"))[:120]
H, W = cv2.imread(str(imgs[0])).shape[:2]
sl = get_slice_bboxes(H, W, 640, 640, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
# ultralytics letterboxes each slice to 640 on the long side (stride-32 padding) and the full frame likewise
tens = sum(640 * int(np.ceil((b[3] - b[1]) * 640 / max(b[2] - b[0], b[3] - b[1]) / 32) * 32) for b in sl) \
    + 640 * int(np.ceil(H * 640 / W / 32) * 32)
print(f"image {W}x{H}; {len(sl)} slices + standard prediction; tensor px/frame {tens} = {tens / (W * H):.3f} of native;"
      f" u960 288x960 = {288 * 960 / (W * H):.3f}")
m = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(Path(MODELS) / "yolov8s.pt"),
                                       confidence_threshold=0.25, device="cuda:0", image_size=640)
y = YOLO(str(Path(MODELS) / "yolov8s.pt"))
t_s, t_u = [], []
for i, p in enumerate(imgs):
    im = cv2.imread(str(p))
    torch.cuda.synchronize(); t0 = time.perf_counter()
    get_sliced_prediction(im[:, :, ::-1].copy(), m, slice_height=640, slice_width=640, overlap_height_ratio=0.2,
                          overlap_width_ratio=0.2, perform_standard_pred=True, verbose=0)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    y.predict(im, imgsz=960, half=True, verbose=False, conf=0.25)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    if i >= 20:
        t_s.append(t1 - t0); t_u.append(t2 - t1)
print(f"SAHI torch {1e3 * np.mean(t_s):.1f} ms (p95 {1e3 * np.percentile(t_s, 95):.1f}); "
      f"u960 torch {1e3 * np.mean(t_u):.1f} ms; ratio {np.mean(t_s) / np.mean(t_u):.2f}")
