#!/usr/bin/env python
"""K13(c) — do the AV2-fine-tuned detectors still detect the ego hood? On AV2 val (every 4th frame of every log, the
frames of av2_ft_eval): boxes (conf >= 0.25) lying >= 50 % inside the hood band (rows >= 1830) per frame and the share of
frames with at least one, for COCO YOLOv8s at 960 (u960), the AV2-fine-tuned YOLOv8s at 960 (u960-ft, out/av2_ft/
plain960), and the saved whole-frame predictions of ESOD and ZoomDet (YOLO txt, normalised; all their frames).
usage: python k13_ft_hood.py  ->  results/k13/ft_hood.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

B = Path(os.environ["B"])
Y0, H0, W0 = 1830, 2048, 1550


def in_mask(b):
    b = np.asarray(b, float).reshape(-1, 4)
    if not len(b):
        return np.zeros(0, bool)
    return (b[:, 3] - np.maximum(b[:, 1], Y0)).clip(0, None) / np.maximum(b[:, 3] - b[:, 1], 1e-6) >= 0.5


def main():
    from rap.detect import Mode, TwoFidelityDetector
    from vtf import av2 as AV
    root = B / "datasets" / "av2" / "sensor" / "val"
    logs = sorted(p.name for p in root.iterdir() if (p / "annotations.feather").exists())
    dets = {"u960 (COCO)": TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch"),
            "u960-ft": TwoFidelityDetector(str(next((B / "out" / "av2_ft" / "plain960").rglob("best.pt"))), device="cuda:0", backend="torch")}
    m = Mode("av2_f960", 960, 768)
    rows = []
    for log in logs:
        L = AV.AV2Log(root, log)
        for fr in range(0, len(L.frame_ts), 4):
            img = cv2.imread(str(L.image_path(fr)))
            if img is None:
                continue
            for nm, det in dets.items():
                d = det.detect(img, m)[0]
                k = d["conf"] >= 0.25
                rows.append({"method": nm, "log": log, "frame": fr, "n_hood": int(in_mask(d["xyxy"][k]).sum())})
    for nm, folder in (("ESOD (fine-tuned)", B / "out" / "esod" / "av2_val" / "labels"), ("ZoomDet (fine-tuned)", B / "out" / "zoomdet" / "av2_val" / "labels")):
        for f in sorted(folder.glob("*.txt")):
            a = np.loadtxt(f, ndmin=2) if f.stat().st_size else np.zeros((0, 6))
            if len(a):
                a = a[a[:, 5] >= 0.25] if a.shape[1] > 5 else a
                cx, cy, w, h = a[:, 1] * W0, a[:, 2] * H0, a[:, 3] * W0, a[:, 4] * H0
                xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
            else:
                xyxy = np.zeros((0, 4))
            rows.append({"method": nm, "log": f.stem, "frame": -1, "n_hood": int(in_mask(xyxy).sum())})
    d = pd.DataFrame(rows)
    res = d.groupby("method").agg(frames=("n_hood", "size"), hood_boxes_per_frame=("n_hood", "mean"),
                                  frames_with_hood_box=("n_hood", lambda x: float((x > 0).mean()))).reset_index()
    out = B / "results" / "k13"; out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "ft_hood.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
