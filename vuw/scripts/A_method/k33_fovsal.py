#!/usr/bin/env python
"""K33 A (FOVEA KDE_S; table1_design.md §2 A): dataset-wide saliency = the normalised sum of FOVEA's bbox2sal over every
ground-truth box set of the TRAIN frames (fovea_baseline.dataset_saliency, official PlainKDEGrid, grid 31 x 51), for each
candidate (bandwidth_scale, amplitude_scale) in {32, 64} x {1, 2}.
  nusc   new TRAIN scenes, DEEP's projected boxes (height >= 10 px), pad shape 900 x 1600
  av2    the AV2 train frames of data/av2_ft (YOLO labels of av2_ft_prepare.py, scaled to the native 2048 x 1550 portrait frame)
Argoverse-HD uses AV2's (no training split is used for Argoverse-HD). Output results/k33/fovsal/<ds>_bw<b>_amp<a>.pt
usage: k33_fovsal.py --dataset nusc|av2
"""
from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

B = Path(os.environ["B"])
OUT = B / "results" / "k33" / "fovsal"
CANDS = [(bw, amp) for bw in (32, 64) for amp in (1, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2"])
    a = ap.parse_args()
    FB = runpy.run_path(str(B / "scripts" / "A_method" / "fovea_baseline.py"), run_name="fovea_lib")
    GG = FB["GG"]
    OUT.mkdir(parents=True, exist_ok=True)
    grids = {c: GG.PlainKDEGrid(output_shape=(384, 640), grid_shape=(31, 51), separable=True, attraction_fwhm=4,
                                anti_crop=True, bandwidth_scale=c[0], amplitude_scale=c[1]) for c in CANDS}
    acc = {c: None for c in CANDS}
    if a.dataset == "nusc":
        from rap.nusc import NuScenesDB, make_adapter
        from rap.paths import NUSCENES_TRAINVAL
        db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
        ad = make_adapter(db)
        sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
        meta = [{"pad_shape": (900, 1600, 3)}]

        def boxsets():
            for s in sorted(sp[(sp["set"] == "new") & (sp.split == "train")].seq):
                g = ad.geometry(s)
                for fr in np.unique(g["frame"]):
                    q = g[(g["frame"] == fr) & ((g["y2"] - g["y1"]) >= 10)]
                    if len(q):
                        yield np.stack([q["x1"], q["y1"], q["x2"], q["y2"]], 1).astype(np.float32)
    else:
        Wn, Hn = 1550, 2048
        meta = [{"pad_shape": (Hn, Wn, 3)}]

        def boxsets():
            for f in sorted((B / "data" / "av2_ft" / "labels" / "train").glob("*.txt")):
                L = np.loadtxt(f, ndmin=2)
                if not len(L):
                    continue
                cx, cy, w, h = L[:, 1] * Wn, L[:, 2] * Hn, L[:, 3] * Wn, L[:, 4] * Hn
                yield np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).astype(np.float32)
    n = 0
    for bx in boxsets():
        n += 1
        for c in CANDS:
            sal = grids[c].bbox2sal([torch.tensor(bx, device="cuda")], meta)        # bbox2sal edits its input: fresh tensor each
            acc[c] = sal if acc[c] is None else acc[c] + sal
    for c in CANDS:
        torch.save((acc[c] / acc[c].sum()).cpu(), OUT / f"{a.dataset}_bw{c[0]}_amp{c[1]}.pt")
    print(f"K33 FOVSAL {a.dataset}: {n} frames DONE", flush=True)


if __name__ == "__main__":
    main()
