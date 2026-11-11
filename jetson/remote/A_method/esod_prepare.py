#!/usr/bin/env python
"""ESOD (Liu et al., TIP 2024/25) on nuScenes: dataset in ESOD's layout, with ESOD's own `gen_mask` (SAM pseudo-masks).

Run inside envs/esod with cwd = third_party/esod (ESOD's scripts import `utils.general` relative to it).
Layout: $B/data/esod_nusc/{train,val}/images/<scene>_<frame>.jpg (symlinks to CAM_FRONT keyframes),
labels/*.txt (YOLO: 0 car/vehicle, 1 pedestrian, 2 cyclist — DEEP's projected GT boxes, height >= 10 px),
masks/*.npy (ESOD's heatmap + weight, SAM-refined as in their VisDrone recipe, cls_ratio off); split lists in
$B/data/esod_nusc/split/{train,val}.txt. Train = every 2nd keyframe of the new TRAIN scenes; val = every 4th keyframe
of the new VAL scenes (test scenes are never touched).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

B = Path(os.environ["B"])
sys.path.insert(0, str(B / "code" / "risk-aware-perception" / "src"))
sys.path.insert(0, str(B / "third_party" / "esod"))
sys.path.insert(0, str(B / "third_party" / "esod" / "scripts"))
import data_prepare as DP                                                     # noqa: E402  (loads SAM ViT-H on cuda)

_seg = DP.segment_image


def _segment_cpu(*a, **k):
    """ESOD's gen_mask calls .numpy() on the SAM result when stride == 1, which fails for a cuda tensor; move it to the
    CPU first (their code otherwise unchanged)."""
    res, inv = _seg(*a, **k)
    return res.cpu(), (inv.cpu() if hasattr(inv, "cpu") else inv)


DP.segment_image = _segment_cpu
from rap.nusc import NuScenesDB, make_adapter                                 # noqa: E402

ROOT = B / "data" / "esod_nusc"
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}


def main():
    print("SAM predictor loaded:", DP.predictor is not None, flush=True)
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    db = NuScenesDB(str(B / "datasets" / "nuscenes"), "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    (ROOT / "split").mkdir(parents=True, exist_ok=True)
    for split, step in (("train", 2), ("val", 4)):
        for sub in ("images", "labels", "masks"):
            (ROOT / split / sub).mkdir(parents=True, exist_ok=True)
        paths = []
        for s in sorted(sp[(sp["set"] == "new") & (sp.split == split)].seq):
            g_all = ad.geometry(s)
            toks = db.samples(by[s])
            for fr in range(0, len(toks), step):
                src = db.image_path(toks[fr])
                name = f"{s}_{fr:03d}"
                ip = ROOT / split / "images" / f"{name}.jpg"
                if not ip.exists():
                    ip.symlink_to(src)
                lp = ROOT / split / "labels" / f"{name}.txt"
                g = g_all[(g_all["frame"] == fr) & ((g_all["y2"] - g_all["y1"]) >= 10)]
                img = cv2.imread(str(src))
                h, w = img.shape[:2]
                lines = []
                for o in g:
                    x1, y1, x2, y2 = (float(o[k]) for k in ("x1", "y1", "x2", "y2"))
                    lines.append("%d %.6f %.6f %.6f %.6f" % (CLS[str(o["coarse"])], (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                                                           (x2 - x1) / w, (y2 - y1) / h))
                lp.write_text("\n".join(lines) + ("\n" if lines else ""))
                if not (ROOT / split / "masks" / f"{name}.npy").exists():
                    if lines:
                        DP.gen_mask(str(lp), img, cls_ratio=False)
                    else:                                 # ESOD's gen_mask expects >= 1 label: empty heatmap, unit weight
                        np.save(ROOT / split / "masks" / f"{name}.npy",
                                np.stack((np.zeros((h, w), np.float16), np.ones((h, w), np.float16)), -1))
                paths.append(str(ip))
            print(f"  {split} {s}: {len(paths)}", flush=True)
        (ROOT / "split" / f"{split}.txt").write_text("\n".join(paths) + "\n")
    (B / "third_party" / "esod" / "data" / "nuscenes.yaml").write_text(
        f"train: {ROOT / 'split' / 'train.txt'}\nval: {ROOT / 'split' / 'val.txt'}\ntest: {ROOT / 'split' / 'val.txt'}\n"
        "nc: 3\nnames: ['car', 'pedestrian', 'cyclist']\n")
    print("ESOD PREPARE DONE")


if __name__ == "__main__":
    main()
