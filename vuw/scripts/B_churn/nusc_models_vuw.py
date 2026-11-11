#!/usr/bin/env python
"""D5 at scale: several detectors at the full mode's input size (384x640) on nuScenes-full CAM_FRONT keyframes.

Caches in DEEP's format under data/cache/nusc_models/<model>/<scene>.npz (one directory per model), so the braking
decisions of every model can be compared with the shared YOLOv8s-320 action history.  Resumable per scene.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

REPO = Path(os.environ["RAP_REPO"])
sys.path.insert(0, str(REPO / "src"))
from rap import cache, mono                                # noqa: E402
from rap.detect import Mode, TwoFidelityDetector           # noqa: E402
from rap.nusc import FRAME_DT, NuScenesDB                  # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL             # noqa: E402

CAM_HEIGHT = 1.51


class RTDETR84(TwoFidelityDetector):
    """ultralytics >= 8.4 returns RT-DETR predictions already decoded: (1, 300, 6) = normalised cx, cy, w, h (of the
    letterboxed input), score, class id.  Same output dict as DEEP's postprocess_detr (no NMS: a set prediction)."""

    def postprocess_detr(self, raw, mode, r, pad, img_shape):
        import numpy as np
        from rap.detect import COCO_TRAFFIC, _empty_dets
        p = raw[0].float().cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw[0])
        cls = p[:, 5].astype(int)
        keep = np.isin(cls, list(COCO_TRAFFIC)) & (p[:, 4] > mode.conf)
        n_raw = int((p[:, 4] > mode.conf).sum())
        if not keep.any():
            return _empty_dets(n_raw, 0)
        q = p[keep][: mode.max_det]
        cx, cy, w, h = q[:, 0] * mode.net_w, q[:, 1] * mode.net_h, q[:, 2] * mode.net_w, q[:, 3] * mode.net_h
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        xyxy[:, [0, 2]] -= pad[0]
        xyxy[:, [1, 3]] -= pad[1]
        xyxy /= r
        h0, w0 = img_shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w0 - 1)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h0 - 1)
        conf = q[:, 4].clip(1e-6, 1 - 1e-6)
        z = np.zeros(len(q), np.float32)
        return {"xyxy": xyxy.astype(np.float32), "conf": q[:, 4].astype(np.float32), "coco_id": q[:, 5].astype(np.int16),
                "coarse": np.array([COCO_TRAFFIC[int(c)] for c in q[:, 5]]), "entropy": z, "margin": z,
                "binent": (-(conf * np.log(conf) + (1 - conf) * np.log(1 - conf))).astype(np.float32),
                "n_cand_raw": n_raw, "n_cand": len(q), "n_post": len(q)}
MODELS = {"yolov8s": "yolo", "yolov8m": "yolo", "yolov8l": "yolo", "yolov8x": "yolo",
          "yolo11s": "yolo", "yolo11m": "yolo", "rtdetr-l": "detr"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--splits", default="val+test")
    ap.add_argument("--set", default="new")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    B = Path(os.environ["B"])
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    scenes = set(sp[(sp["set"] == args.set) & sp.split.isin(args.splits.split("+"))].seq)
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    for m in args.models:
        arch = MODELS[m]
        det = (RTDETR84 if arch == "detr" else TwoFidelityDetector)(str(B / "models" / f"{m}.pt"), device=args.device,
                                                                     backend="torch")
        mode = Mode(f"{m}_640", 640, 640, arch="detr", max_det=300) if arch == "detr" else Mode(f"{m}_640", 384, 640)
        out = Path(CACHE) / "nusc_models" / m
        t0, n = time.time(), 0
        for sc in db.scenes:
            name = db.scene_name(sc)
            if name not in scenes or (out / f"{name}.npz").exists():
                continue
            toks = db.samples(sc)
            calib = db.calib(toks[0])
            dets, geos, prev = [], [], None
            scal = {}
            for tk in toks:
                img = cv2.imread(str(db.image_path(tk)))
                d, _ = det.detect(img, mode)
                geos.append(mono.predicted_geometry(d, prev, calib, CAM_HEIGHT, FRAME_DT))
                dets.append(d)
                for k in ("n_cand", "n_cand_raw", "n_post"):
                    scal.setdefault(k, []).append(d[k])
                prev = d
            cache.save(out / f"{name}.npz", list(range(len(toks))), dets, geos, scal)
            n += 1
        print(f"{m}: {n} scenes in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
