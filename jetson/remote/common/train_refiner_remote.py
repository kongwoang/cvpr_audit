#!/usr/bin/env python
"""Train image-level range refiners on the VUW GPU server (standalone copy of scripts/35_train_image_refiner.py).

Data: data/refiner_data__KITTI.npz, data/refiner_data__nuScenes.npz (64x64 crops, 12 box features, log range ratio,
split).  Training units only; epoch chosen on validation units; test units scored once.  Architectures:
  small     the 4-block CNN of 35_train_image_refiner.py (from scratch)
  resnet18  torchvision ResNet-18 (ImageNet weights), 64x64 crops upsampled to 128x128
Writes out/<arch>/image_refiner.pt (state, fmu, fsd, arch) and out/refiner_errors.csv.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent


class Small(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.cnn = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 96), blk(96, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(128 + n_feat, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def forward(self, x, f):
        return self.head(torch.cat([self.cnn(x), f], 1)).squeeze(1)


class Res18(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        self.cnn = m
        self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.head = nn.Sequential(nn.Linear(512 + n_feat, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))

    def forward(self, x, f):
        x = Fn.interpolate(x.flip(1), size=128, mode="bilinear", align_corners=False)   # BGR -> RGB
        return self.head(torch.cat([self.cnn((x - self.mu) / self.sd), f], 1)).squeeze(1)


def pick_gpu(max_util=50):
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()
    best = None
    for line in q:
        i, tot, used, util = (int(x) for x in line.split(","))
        if util <= max_util and (best is None or tot - used > best[1]):
            best = (i, tot - used)
    if best is None or best[1] < 4000:
        raise SystemExit(f"no GPU with <= {max_util}% utilisation and >= 4 GB free: {q}")
    return best[0]


def load(names):
    parts = []
    for ds in names:
        z = np.load(ROOT / "data" / f"refiner_data__{ds}.npz", allow_pickle=True)
        n = len(z["target"])
        parts.append(dict(crops=z["crops"], feats=np.c_[z["feats"], np.full(n, float(ds.startswith("nuScenes")))],
                          target=z["target"], split=z["split"], ds=np.full(n, ds)))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", nargs="+", default=["small", "resnet18"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="", help="output goes to out/<arch><tag>/ (the frozen refiner is out/resnet18/)")
    ap.add_argument("--datasets", nargs="+", default=["KITTI", "nuScenes"])
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)       # on GPU the wrapper exposes exactly one free device as cuda:0
    D = load(args.datasets)
    tr, va, te = (D["split"] == s for s in ("train", "val", "test"))
    fmu, fsd = D["feats"][tr].mean(0), D["feats"][tr].std(0) + 1e-6
    X = torch.from_numpy(D["crops"]).permute(0, 3, 1, 2).contiguous()
    F = torch.from_numpy(((D["feats"] - fmu) / fsd).astype(np.float32))
    Y = torch.from_numpy(D["target"].astype(np.float32))
    gbm = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.06, min_samples_leaf=40,
                                        random_state=0).fit(D["feats"][tr], D["target"][tr])
    preds = {"gbm_box_features": gbm.predict(D["feats"])}
    idx_tr = np.flatnonzero(tr)
    for arch in args.arch:
        torch.manual_seed(0)
        net = (Small if arch == "small" else Res18)(F.shape[1]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3 if arch == "small" else 5e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

        def predict(mask):
            net.eval()
            out = []
            with torch.no_grad():
                for b in np.array_split(np.flatnonzero(mask), max(1, int(mask.sum()) // 1024)):
                    out.append(net(X[b].to(dev).float() / 255.0, F[b].to(dev)).float().cpu().numpy())
            return np.concatenate(out)

        best, state = np.inf, None
        for ep in range(args.epochs):
            net.train()
            rng = np.random.default_rng(ep)
            for b in np.array_split(rng.permutation(idx_tr), max(1, len(idx_tr) // 256)):
                x = X[b].to(dev).float() / 255.0
                if rng.random() < 0.5:
                    x = x.flip(3)
                loss = Fn.smooth_l1_loss(net(x, F[b].to(dev)), Y[b].to(dev), beta=0.05)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            v = float(np.mean(np.abs(predict(va) - D["target"][va])))
            print(f"  {arch} epoch {ep}: val |log err| {v:.4f}", flush=True)
            if v < best:
                best, state = v, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
        net.load_state_dict(state)
        (ROOT / "out" / f"{arch}{args.tag}").mkdir(parents=True, exist_ok=True)
        torch.save({"state": state, "fmu": fmu, "fsd": fsd, "arch": arch}, ROOT / "out" / f"{arch}{args.tag}" / "image_refiner.pt")
        preds[arch] = predict(np.ones(len(Y), bool))
        del net, opt
        torch.cuda.empty_cache()
    rows = []
    for ds in ("KITTI", "nuScenes"):
        for sp, m in (("val", va), ("test", te)):
            mm = m & (D["ds"] == ds)
            t = D["target"][mm]
            r = {"dataset": ds, "split": sp, "n": int(mm.sum()), "lift": float(np.mean(np.abs(t)))}
            r.update({k: float(np.mean(np.abs(t - p[mm]))) for k, p in preds.items()})
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "out" / f"refiner_errors{args.tag}.csv", index=False)
    print(res.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
