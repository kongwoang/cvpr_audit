#!/usr/bin/env python
"""B1 step 2: train an image-level range refiner (small CNN on the 64x64 crop + 12 box features) on the Jetson GPU.

Target: log(reference range / lifted range).  Training units only; the epoch is chosen on validation units; the test
units are scored once.  Compared with the box-feature GBM of D40 on the same boxes.
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor

from vtf import runmeta


class Refiner(nn.Module):
    def __init__(self, n_feat=13):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.cnn = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 96), blk(96, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(128 + n_feat, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def forward(self, x, f):
        return self.head(torch.cat([self.cnn(x), f], 1)).squeeze(1)


def load(datasets):
    parts = []
    for ds in datasets:
        z = np.load(sorted(glob.glob(str(runmeta.RAW / f"*_refiner_data_{ds}" / f"refiner_data__{ds}.npz")))[-1], allow_pickle=True)
        n = len(z["target"])
        parts.append(dict(crops=z["crops"], feats=np.c_[z["feats"], np.full(n, float(ds == "nuScenes"))],
                          target=z["target"], split=z["split"], ds=np.full(n, ds)))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["KITTI", "nuScenes"])
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    run = runmeta.new_run("image_refiner", vars(args))
    D = load(args.datasets)
    tr, va, te = (D["split"] == s for s in ("train", "val", "test"))
    fmu, fsd = D["feats"][tr].mean(0), D["feats"][tr].std(0) + 1e-6
    dev = torch.device("cuda")
    X = torch.from_numpy(D["crops"]).permute(0, 3, 1, 2).contiguous()          # uint8 NCHW on CPU
    F = torch.from_numpy(((D["feats"] - fmu) / fsd).astype(np.float32))
    Y = torch.from_numpy(D["target"].astype(np.float32))
    torch.manual_seed(0)
    net = Refiner(F.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    idx_tr = np.flatnonzero(tr)

    def predict(mask):
        net.eval()
        out = []
        with torch.no_grad():
            for b in np.array_split(np.flatnonzero(mask), max(1, int(mask.sum()) // 512)):
                x = X[b].to(dev).float() / 255.0
                out.append(net(x, F[b].to(dev)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    best, best_state = np.inf, None
    for ep in range(args.epochs):
        net.train()
        rng = np.random.default_rng(ep)
        for b in np.array_split(rng.permutation(idx_tr), max(1, len(idx_tr) // 256)):
            x = X[b].to(dev).float() / 255.0
            if rng.random() < 0.5:                                                  # horizontal flip
                x = x.flip(3)
            loss = nn.functional.smooth_l1_loss(net(x, F[b].to(dev)), Y[b].to(dev), beta=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        v = float(np.mean(np.abs(predict(va) - D["target"][va])))
        print(f"  epoch {ep}: val |log err| {v:.4f}", flush=True)
        if v < best:
            best, best_state = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    torch.save({"state": best_state, "fmu": fmu, "fsd": fsd}, run / "image_refiner.pt")
    pred = predict(np.ones(len(Y), bool))
    gbm = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.06, min_samples_leaf=40,
                                        random_state=0).fit(D["feats"][tr], D["target"][tr])
    pg = gbm.predict(D["feats"])
    rows = []
    for ds in args.datasets:
        for sp, m in (("val", va), ("test", te)):
            mm = m & (D["ds"] == ds)
            t = D["target"][mm]
            rows.append({"dataset": ds, "split": sp, "n": int(mm.sum()), "lift_abs_log_err": float(np.mean(np.abs(t))),
                         "gbm_box_features": float(np.mean(np.abs(t - pg[mm]))),
                         "cnn_image_refiner": float(np.mean(np.abs(t - pred[mm])))})
    res = pd.DataFrame(rows)
    res.to_csv(run / "refiner_errors.csv", index=False)
    print(res.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
