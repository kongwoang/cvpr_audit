#!/usr/bin/env python
"""ESOD's SAM pseudo-masks for the AV2 fine-tuning set (data/av2_ft/esod, written by av2_ft_prepare.py), with ESOD's own
`gen_mask` exactly as esod_prepare.py does for nuScenes; then ESOD's data yaml (data/av2.yaml).
Run inside envs/esod with cwd = third_party/esod."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

B = Path(os.environ["B"])
sys.path.insert(0, str(B / "third_party" / "esod"))
sys.path.insert(0, str(B / "third_party" / "esod" / "scripts"))
import data_prepare as DP                                                     # noqa: E402  (loads SAM ViT-H on cuda)

_seg = DP.segment_image


def _segment_cpu(*a, **k):
    res, inv = _seg(*a, **k)
    return res.cpu(), (inv.cpu() if hasattr(inv, "cpu") else inv)


DP.segment_image = _segment_cpu
ROOT = B / "data" / "av2_ft" / "esod"


def main():
    print("SAM predictor loaded:", DP.predictor is not None, flush=True)
    for split in ("train", "val"):
        (ROOT / split / "masks").mkdir(parents=True, exist_ok=True)
        lps = sorted((ROOT / split / "labels").glob("*.txt"))
        for i, lp in enumerate(lps):
            mp = ROOT / split / "masks" / f"{lp.stem}.npy"
            if mp.exists():
                continue
            img = cv2.imread(str(ROOT / split / "images" / f"{lp.stem}.jpg"))
            h, w = img.shape[:2]
            if lp.read_text().strip():
                DP.gen_mask(str(lp), img, cls_ratio=False)
            else:
                np.save(mp, np.stack((np.zeros((h, w), np.float16), np.ones((h, w), np.float16)), -1))
            if i % 500 == 0:
                print(f"  {split} {i}/{len(lps)}", flush=True)
    (B / "third_party" / "esod" / "data" / "av2.yaml").write_text(
        f"train: {ROOT / 'split' / 'train.txt'}\nval: {ROOT / 'split' / 'val.txt'}\ntest: {ROOT / 'split' / 'test.txt'}\n"
        "nc: 3\nnames: ['car', 'pedestrian', 'cyclist']\n")
    print("ESOD AV2 MASKS DONE")


if __name__ == "__main__":
    main()
