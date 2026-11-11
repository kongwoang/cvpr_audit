#!/usr/bin/env python
"""Parity of vtf.fastpost.FastDetector with DEEP's GPU post-processing, and per-call time of both (TensorRT, Jetson)."""
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from rap.detect import MODES, Mode, TwoFidelityDetector
from rap.paths import KITTI_IMAGES, MODELS
from vtf import runmeta
from vtf.fastpost import FastDetector

w = str(Path(MODELS) / "yolov8s.pt")
kw = dict(backend="trt")
dets = {}
for name, cls in (("gpu", TwoFidelityDetector), ("cpu", FastDetector)):
    dets[name] = (cls(w, engine_dir=str(Path(MODELS) / "engines"), engine_stem="yolov8s", **kw),
                  cls(w, engine_dir=str(runmeta.ROOT / "engines"), engine_stem="yolov8s_crop", **kw))
imgs = sorted(Path(KITTI_IMAGES).glob("*/*.png"))[::40][:300]
rng = np.random.default_rng(0)
stats = {}
for p in imgs:
    img = cv2.imread(str(p))
    H, W = img.shape[:2]
    x1 = int(rng.integers(0, W - 300)); y1 = int(rng.integers(0, H - 100))
    for mname, which, mode, im in (("cheap_320", 0, MODES["cheap_320"], img), ("full_640", 0, MODES["full_640"], img),
                                   ("288x960", 1, Mode("288x960", 288, 960), img),
                                   ("96x288", 1, Mode("96x288", 96, 288), img[y1:y1 + 100, x1:x1 + 300])):
        out = {}
        for name in ("gpu", "cpu"):
            det = dets[name][which]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            d, _ = det.detect(im, mode)
            torch.cuda.synchronize()
            out[name] = (d, 1e3 * (time.perf_counter() - t0))
        a, b = out["gpu"][0], out["cpu"][0]
        s = stats.setdefault(mname, {"n": 0, "same_count": 0, "max_box": 0.0, "max_conf": 0.0, "coarse_eq": 0, "t_gpu": [], "t_cpu": []})
        s["n"] += 1
        s["t_gpu"].append(out["gpu"][1]); s["t_cpu"].append(out["cpu"][1])
        if len(a["conf"]) == len(b["conf"]):
            s["same_count"] += 1
            if len(a["conf"]):
                oa, ob = np.lexsort(a["xyxy"].T[::-1]), np.lexsort(b["xyxy"].T[::-1])
                s["max_box"] = max(s["max_box"], float(np.abs(a["xyxy"][oa] - b["xyxy"][ob]).max()))
                s["max_conf"] = max(s["max_conf"], float(np.abs(a["conf"][oa] - b["conf"][ob]).max()))
                s["coarse_eq"] += int((a["coarse"][oa] == b["coarse"][ob]).all())
            else:
                s["coarse_eq"] += 1
for m, s in stats.items():
    print(f"{m:10s} frames {s['n']}  same count {s['same_count']}  same classes {s['coarse_eq']}  max |box| {s['max_box']:.4f} px"
          f"  max |conf| {s['max_conf']:.5f}  ms gpu-post {np.median(s['t_gpu'][5:]):.2f}  ms cpu-post {np.median(s['t_cpu'][5:]):.2f}")
