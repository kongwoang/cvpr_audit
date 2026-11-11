#!/usr/bin/env python
"""Fine-tune COCO YOLOv8s at the cheap input size (320) on one arm's data; same recipe for every arm.
usage: d68_train.py <arm> [dataset=d68] [epochs=15] [model=yolov8s] [imgsz=320] [batch=64] [name=arm]"""
import os, sys
from pathlib import Path
from ultralytics import YOLO

B = Path(os.environ["B"])
arm = sys.argv[1]
ds = sys.argv[2] if len(sys.argv) > 2 else "d68"
ep = int(sys.argv[3]) if len(sys.argv) > 3 else 15
model = sys.argv[4] if len(sys.argv) > 4 else "yolov8s"
imgsz = int(sys.argv[5]) if len(sys.argv) > 5 else 320
batch = int(sys.argv[6]) if len(sys.argv) > 6 else 64
name = sys.argv[7] if len(sys.argv) > 7 else arm
m = YOLO(str(B / "models" / f"{model}.pt"))
m.train(data=str(B / "data" / ds / arm / "data.yaml"), imgsz=imgsz, epochs=ep, batch=batch, workers=int(os.environ.get("WORKERS", 8)), device=0,
        amp=False, plots=False, project=str(B / "out" / ds), name=name, exist_ok=True, seed=int(os.environ.get("SEED", 0)), deterministic=True,
        verbose=False)
