#!/usr/bin/env python
"""K28 open item (KITTI Planner B mono discrepancy): refiner v3a on KITTI boxes with the camera flag it was trained with for
KITTI (0.0, k23_train_refiner.load) against the flag the Jetson reference path hard-codes (58.RefinerBatch: 1.0).
Data: data/refiner_data__KITTI.npz, splits train / val only (the refiner's own training / selection data; no test unit).
Reports, per split and flag: mean |log(annotated / estimated range)| before and after the correction, mean predicted
log-ratio vs mean target, and the share of boxes the correction moves the wrong way. CPU only.
usage: k28_kitti_flag_check.py [--ckpt out/resnet18_v3a/image_refiner.pt]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "out" / "resnet18_v3a" / "image_refiner.pt"))
    args = ap.parse_args()
    torch.set_num_threads(16)
    s = importlib.util.spec_from_file_location("k23", ROOT / "scripts" / "A_method" / "k23_train_refiner.py")
    K23 = importlib.util.module_from_spec(s); s.loader.exec_module(K23)
    import torchvision
    _rn = torchvision.models.resnet18
    torchvision.models.resnet18 = lambda weights=None, **k: _rn(weights=None, **k)   # no download; weights come from ckpt
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = K23.Res18x3(len(ck["fmu"])); net.load_state_dict(ck["state"]); net.eval()
    z = np.load(ROOT / "data" / "refiner_data__KITTI.npz", allow_pickle=True)
    out = []
    for sp in ("val", "train"):
        m = z["split"] == sp
        X = torch.from_numpy(z["crops"][m]).permute(0, 3, 1, 2).contiguous()
        y = z["target"][m].astype(float)
        ok = np.isfinite(y)
        for flag in (0.0, 1.0):
            Fe = np.c_[z["feats"][m], np.full(int(m.sum()), flag)]
            Fe = torch.from_numpy(((Fe - ck["fmu"]) / ck["fsd"]).astype(np.float32))
            p = []
            with torch.no_grad():
                for b in np.array_split(np.arange(len(X)), max(1, len(X) // 512)):
                    p.append(net(X[b].float() / 255.0, Fe[b])[:, 0].numpy())
            p = np.concatenate(p).astype(float)
            r = {"split": sp, "flag": flag, "boxes": int(ok.sum()),
                 "abs_log_err_lift": float(np.mean(np.abs(y[ok]))),
                 "abs_log_err_refined": float(np.mean(np.abs(y[ok] - p[ok]))),
                 "mean_target": float(np.mean(y[ok])), "mean_pred": float(np.mean(p[ok])),
                 "share_wrong_way": float(np.mean(np.sign(p[ok]) != np.sign(y[ok]))),
                 "share_worse": float(np.mean(np.abs(y[ok] - p[ok]) > np.abs(y[ok])))}
            out.append(r); print(json.dumps(r), flush=True)
    od = ROOT / "results" / "k28" / "p3" / "tables"
    od.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(out).to_csv(od / "kitti_refiner_flag_check.csv", index=False)


if __name__ == "__main__":
    main()
