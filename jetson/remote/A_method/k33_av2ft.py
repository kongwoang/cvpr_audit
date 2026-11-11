#!/usr/bin/env python
"""K33 F (table1_design.md §2 F): training data and trainings of the fine-tuned AV2 rows that do not exist yet, from the
rebuilt data/av2_ft (av2_ft_prepare.py: AV2 train logs at 1 Hz, 1162 x 1536 portrait; the images the existing u960-ft /
ESOD / ZoomDet weights were trained on). Recipes = the nuScenes ones (fovea_ft.py, ufp_ft.py, lzu_baseline.py):
  --stage build --arm fovea   every train / val image warped with the official PlainKDEGrid (KDE_I from the image's cheap
                              draft, conf >= 0.25; bandwidth 64, amplitude 1) to the 960 x 736 canvas; GT boxes mapped
                              through the inverse separable grid (exact per axis); YOLO labels -> data/k33_av2ft/fovea
  --stage build --arm ufp     the cheap draft's boxes (detector floor 0.10) -> UnifiedForegroundPacking(1.5) -> mosaic; a
                              GT box is labelled in every chip it overlaps with IoF > 0.9 -> data/k33_av2ft/ufp
  --stage lzusal              LZU's dataset saliency on the 48 x 36 grid (their 27 x 48 transposed to the portrait canvas)
                              from the train labels, canvas 960 x 736 -> results/k33/lzu_saliency_av2.pt
  --stage train --arm lzu     LZU-ft: lzu_baseline.stage_train's recipe (YOLOv8s COCO init, 15 epochs, imgsz 960, batch
                              32, rect, no scale / translate jitter, flip 0.5) on data/av2_ft with the AV2 saliency
FOVEA-ft and UFP-ft train with scripts/A_ext/d68_train.py <arm> k33_av2ft 15 yolov8s 960 32 (the FOVEA-ft recipe).
usage: k33_av2ft.py --stage build|lzusal|train --arm fovea|ufp|lzu
"""
from __future__ import annotations

import argparse
import math
import os
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

B = Path(os.environ["B"])
SRC = B / "data" / "av2_ft"
DST = B / "data" / "k33_av2ft"
W0, H0 = 1162, 1536                                       # the av2_ft images (portrait)
HC, WC = 960, 736                                         # the canvas at long side 960
NAMES = {i: f"c{i}" for i in range(80)}
NAMES.update({0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"})


def labels(p):
    L = np.loadtxt(p, ndmin=2) if p.exists() and p.stat().st_size else np.zeros((0, 5))
    cls = L[:, 0].astype(int)
    cx, cy, w, h = L[:, 1] * W0, L[:, 2] * H0, L[:, 3] * W0, L[:, 4] * H0
    return cls, np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) if len(L) else np.zeros((0, 4))


def yline(c, b, w, h):
    x1, y1, x2, y2 = np.clip(b, [0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1])
    return f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def yaml(arm):
    (DST / arm / "data.yaml").write_text(f"path: {DST / arm}\ntrain: images/train\nval: images/val\nnames:\n" +
                                         "".join(f"  {i}: {n}\n" for i, n in NAMES.items()))


def build(arm):
    from rap.detect import Mode, TwoFidelityDetector
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    CH = Mode("av2_c320", 320, 256)
    if arm == "fovea":
        GG = runpy.run_path(str(B / "scripts" / "A_method" / "fovea_baseline.py"), run_name="fovea_lib")["GG"]
        gen = GG.PlainKDEGrid(output_shape=(HC, WC), grid_shape=(31, 51), separable=True, attraction_fwhm=4, anti_crop=True,
                              bandwidth_scale=64, amplitude_scale=1)
    else:
        UB = runpy.run_path(str(B / "scripts" / "A_method" / "ufp_baseline.py"), run_name="ufp_lib")
    n = 0
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (DST / arm / sub / split).mkdir(parents=True, exist_ok=True)
        for ip in sorted((SRC / "images" / split).glob("*.jpg")):
            if (DST / arm / "labels" / split / f"{ip.stem}.txt").exists():
                continue
            img = cv2.imread(str(ip))
            assert img.shape[:2] == (H0, W0), img.shape
            cls, gt = labels(SRC / "labels" / split / f"{ip.stem}.txt")
            d = det.detect(img, CH)[0]
            if arm == "fovea":
                cheap = np.asarray(d["xyxy"], float)[np.asarray(d["conf"]) >= 0.25]
                x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().cuda()
                with torch.no_grad():
                    grid = gen(x, [{"pad_shape": (H0, W0, 3)}], [torch.tensor(cheap.reshape(-1, 4), dtype=torch.float32, device="cuda")])
                    w_ = F.grid_sample(x, grid, align_corners=True)
                wimg = w_[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()[:, :, ::-1].copy()
                g = grid[0].cpu().numpy()
                xs, ys = (g[0, :, 0] + 1) / 2 * W0, (g[:, 0, 1] + 1) / 2 * H0
                wb = gt.copy()
                wb[:, [0, 2]] = np.interp(wb[:, [0, 2]], xs, np.arange(WC, dtype=float))
                wb[:, [1, 3]] = np.interp(wb[:, [1, 3]], ys, np.arange(HC, dtype=float))
                cv2.imwrite(str(DST / arm / "images" / split / ip.name), wimg, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (DST / arm / "labels" / split / f"{ip.stem}.txt").write_text(
                    "\n".join(yline(k, b, WC, HC) for k, b in zip(cls, wb) if (b[2] - b[0]) > 2 and (b[3] - b[1]) > 2))
            else:
                cxy = np.asarray(d["xyxy"], float).reshape(-1, 4)
                if not len(cxy):
                    continue
                rec, mw, mh = UB["UnifiedForegroundPacking"](cxy.copy(), 1.5, input_shape=[W0, H0])
                if not rec or mw < 1 or mh < 1:
                    continue
                mos = UB["display_merge_result"](rec, img, mw, mh).astype(np.uint8)
                Mh, Mw = mos.shape[:2]
                lines = []
                for c_ in rec:
                    o_x1, o_y1, cw, chh, n_x, n_y, sf = [math.floor(_) for _ in c_]
                    for k, b in zip(cls, gt):
                        if UB["compute_iof"](list(b), [o_x1, o_y1, o_x1 + cw, o_y1 + chh]) > 0.9:
                            nb = np.array([(b[0] - o_x1) * sf + n_x, (b[1] - o_y1) * sf + n_y, (b[2] - o_x1) * sf + n_x, (b[3] - o_y1) * sf + n_y])
                            lines.append(yline(k, nb, Mw, Mh))
                cv2.imwrite(str(DST / arm / "images" / split / ip.name), mos, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (DST / arm / "labels" / split / f"{ip.stem}.txt").write_text("\n".join(lines))
            n += 1
            if n % 1000 == 0:
                print(f"  {arm} {split}: {n} images", flush=True)
    yaml(arm)
    print(f"K33 AV2FT BUILD {arm} DONE ({n} images)", flush=True)


def lzusal():
    FB = runpy.run_path(str(B / "scripts" / "A_method" / "fovea_baseline.py"), run_name="fovea_lib")
    gen = FB["GG"].PlainKDEGrid(output_shape=(HC, WC), grid_shape=(48, 36), separable=True, attraction_fwhm=10,
                                anti_crop=True, bandwidth_scale=64, amplitude_scale=1)
    meta = [{"pad_shape": (HC, WC, 3)}]
    sc = np.array([WC / W0, HC / H0, WC / W0, HC / H0])
    acc = None
    for p in sorted((SRC / "labels" / "train").glob("*.txt")):
        _, gt = labels(p)
        if not len(gt):
            continue
        sal = gen.bbox2sal([torch.tensor(gt * sc, dtype=torch.float32, device="cuda")], meta)
        acc = sal if acc is None else acc + sal
    sal = (acc / acc.sum()).reshape(1, 1, 48, 36).cpu()
    (B / "results" / "k33").mkdir(parents=True, exist_ok=True)
    torch.save(sal, B / "results" / "k33" / "lzu_saliency_av2.pt")
    print("K33 LZU SALIENCY AV2 DONE", flush=True)


def train_lzu():
    sys.path.insert(0, str(B / "scripts" / "common"))
    import lzu_yolo
    lzu_yolo.GRID_SHAPE = (48, 36)
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    sal = torch.load(B / "results" / "k33" / "lzu_saliency_av2.pt")

    class LZUTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return lzu_yolo.patch(super().get_model(cfg, weights, verbose), sal)
    seed = int(os.environ.get("SEED", 0))
    YOLO(str(B / "models" / "yolov8s.pt")).train(
        trainer=LZUTrainer, data=str(SRC / "data.yaml"), imgsz=960, epochs=15, batch=32, workers=4, device=0, amp=False,
        plots=False, project=str(B / "out" / "k33_av2ft"), name="lzu" + (f"_s{seed}" if seed else ""), exist_ok=True, seed=seed,
        deterministic=True, verbose=False, rect=True, scale=0.0, translate=0.0, fliplr=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "lzusal", "train"])
    ap.add_argument("--arm", default="")
    a = ap.parse_args()
    if a.stage == "build":
        build(a.arm)
    elif a.stage == "lzusal":
        lzusal()
    else:
        train_lzu()


if __name__ == "__main__":
    main()
