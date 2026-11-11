#!/usr/bin/env python
"""K28 family C — the learned temporal cue (method_freeze_draft §2j). Data: data/tdata__{av2,nusc}_*.npz (k28_tdata.py;
AV2 REF-TRAIN / REF-VAL, nuScenes new train with a 1/8 scene hold-out; no dev / test unit). Samples with a target only.
Model TempHead: an ImageNet ResNet-18 shared over the three 64x64 crops (t, t-1, t-2; upsampled to 128, BGR -> RGB), the
three 512-d embeddings concatenated with the normalised extra features (12 box features at t, log h_t / h_{t-1},
log h_{t-1} / h_{t-2}, dt, the camera flag 1.0) -> MLP 256 -> r_hat, the relative closing rate (1 / s). The T4 rate is NOT
an input (C1 replaces it). Loss smooth L1 (beta 0.02) on r clipped to [0, 2]; AdamW 3e-4, wd 1e-4, cosine, 15 epochs,
batch 128, horizontal flip of all three crops with p 0.5, seed 0. Model selection: the epoch with the lowest mean
|r_hat - r| on the validation split (AV2 REF-VAL + the nuScenes hold-out); reported next to the T4 rate's error on the
same samples.  ->  out/k28_temporal/temporal_head.pt, train_report.json
usage: k28_ttrain.py [--epochs 15] [--seed 0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

ROOT = Path("/vol/dl-nguyenb5-solar/users/hoangpc/CVPR")


class TempHead(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        self.cnn = m
        self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.head = nn.Sequential(nn.Linear(3 * 512 + n_feat, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))

    def forward(self, x, f):
        """x: (n, 3, 3, 64, 64) float in [0, 1] (time t, t-1, t-2; BGR channels), f: (n, n_feat)"""
        n = x.shape[0]
        x = x.reshape(n * 3, 3, 64, 64)
        x = Fn.interpolate(x.flip(1), size=128, mode="bilinear", align_corners=False)
        e = self.cnn((x - self.mu) / self.sd).reshape(n, 3 * 512)
        return self.head(torch.cat([e, f], 1))[:, 0]


def extra(feats, hr, dt):
    return np.c_[feats, np.log(np.clip(hr, 1e-3, None)), dt, np.ones(len(dt))].astype(np.float32)


def load():
    parts = sorted((ROOT / "data").glob("tdata__*of*.npz"))
    assert parts
    D = {k: [] for k in ("crops", "feats", "hr", "rT4", "dt", "target", "split", "kind", "src")}
    for f in parts:
        z = np.load(f, allow_pickle=True)
        ok = np.isfinite(z["target"])
        for k in ("crops", "feats", "hr", "rT4", "dt", "target", "split", "kind"):
            D[k].append(z[k][ok])
        D["src"].append(np.full(int(ok.sum()), f.name.split("__")[1].split("_")[0]))
    return {k: np.concatenate(v) for k, v in D.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(8)
    dev = torch.device("cuda:0")
    D = load()
    tr, va = D["split"] == "train", D["split"] == "val"
    E = extra(D["feats"], D["hr"], D["dt"])
    fmu, fsd = E[tr].mean(0), E[tr].std(0) + 1e-6
    X = torch.from_numpy(D["crops"]).permute(0, 1, 4, 2, 3).contiguous()          # n, 3 (time), 3 (ch), 64, 64
    Fe = torch.from_numpy(((E - fmu) / fsd).astype(np.float32))
    Y = torch.from_numpy(np.clip(D["target"], 0, 2).astype(np.float32))
    torch.manual_seed(args.seed)
    net = TempHead(Fe.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    idx_tr, idx_va = np.flatnonzero(tr), np.flatnonzero(va)

    def predict(ix):
        net.eval(); out = []
        with torch.no_grad():
            for b in np.array_split(ix, max(1, len(ix) // 512)):
                out.append(net(X[b].to(dev).float() / 255.0, Fe[b].to(dev)).float().cpu().numpy())
        return np.concatenate(out)
    yv = np.clip(D["target"][idx_va], 0, 2)
    t4v = np.clip(D["rT4"][idx_va], 0, 2)
    base = {"val_mae_T4": float(np.mean(np.abs(t4v - yv))), "val_mae_zero": float(np.mean(np.abs(yv)))}
    for s in ("av2", "nusc"):
        m = D["src"][idx_va] == s
        if m.any():
            base[f"val_mae_T4_{s}"] = float(np.mean(np.abs(t4v[m] - yv[m])))
    log, best, state = [], np.inf, None
    for ep in range(args.epochs):
        net.train()
        rng = np.random.default_rng(1000 * args.seed + ep)
        for b in np.array_split(rng.permutation(idx_tr), max(1, len(idx_tr) // 128)):
            x = X[b].to(dev).float() / 255.0
            if rng.random() < 0.5:
                x = x.flip(4)
            o = net(x, Fe[b].to(dev))
            loss = Fn.smooth_l1_loss(o, Y[b].to(dev), beta=0.02)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        p = np.clip(predict(idx_va), 0, 2)
        s = {"epoch": ep, "val_mae": float(np.mean(np.abs(p - yv)))}
        for src in ("av2", "nusc"):
            m = D["src"][idx_va] == src
            if m.any():
                s[f"val_mae_{src}"] = float(np.mean(np.abs(p[m] - yv[m])))
        log.append(s)
        print(json.dumps(s), flush=True)
        if s["val_mae"] < best:
            best, state = s["val_mae"], {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
    od = ROOT / "out" / "k28_temporal"
    od.mkdir(parents=True, exist_ok=True)
    fn = od / ("temporal_head.pt" if args.seed == 0 else f"temporal_head_s{args.seed}.pt")
    torch.save({"state": state, "fmu": fmu, "fsd": fsd, "n_feat": int(Fe.shape[1]), "seed": args.seed}, fn)
    md5 = hashlib.md5(fn.read_bytes()).hexdigest()
    counts = {f"{s}_{sp}": int(((D["src"] == s) & (D["split"] == sp)).sum()) for s in ("av2", "nusc") for sp in ("train", "val")}
    best_ep = min(log, key=lambda r: r["val_mae"])
    json.dump({"epochs": args.epochs, "seed": args.seed, "samples": counts, "baseline": base, "best": best_ep, "log": log, "md5": md5},
              open(od / f"train_report{'' if args.seed == 0 else f'_s{args.seed}'}.json", "w"), indent=1)
    print("K28 TEMPORAL HEAD best", best_ep, "baseline", base, "md5", md5, flush=True)


if __name__ == "__main__":
    main()
