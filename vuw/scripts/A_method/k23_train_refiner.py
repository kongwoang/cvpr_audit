#!/usr/bin/env python
"""K23 C1 — refiner v3 (method_freeze_draft §2e). v2's architecture and inputs (ImageNet ResNet-18 on the 64x64 crop
upsampled to 128, BGR -> RGB, concatenated with the 13 normalised box features: 12 box / lift features + the camera flag,
1.0 for the nuScenes / AV2 cameras as at inference, 0.0 for KITTI) with a 3-output head:
  out[:, 0]  log(annotated / estimated range)        loss: smooth L1 (beta 0.05) on matched boxes of every dataset
  out[:, 1]  annotated - estimated lateral centre, m  loss: smooth L1 (beta 0.1) on matched AV2 boxes   (variants b, c)
  out[:, 2]  validity logit                           loss: BCE on every AV2 box (matched or not)     (variant c)
Data: KITTI train / val (v2's), nuScenesAll TRAIN scenes only (its validation scenes include nuScenes new val and are
NOT used), AV2 REF-TRAIN / REF-VAL (k23_refdata). Model selection: the epoch with the lowest validation score on KITTI val
+ AV2 REF-VAL: (a) range |log error|; (b) + lateral MAE / 10; (c) + validity BCE / 5. No dev / test unit enters training
or selection. Same optimiser as v2 (AdamW 5e-4, wd 1e-4, cosine, batch 256, horizontal flip with p 0.5, seed 0).
usage: k23_train_refiner.py --variant a|b|c [--epochs 25]  ->  out/resnet18_v3<variant>/image_refiner.pt, train log
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


class Res18x3(nn.Module):
    def __init__(self, n_feat, n_out=3):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        self.cnn = m
        self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.head = nn.Sequential(nn.Linear(512 + n_feat, 256), nn.ReLU(inplace=True), nn.Linear(256, n_out))

    def forward(self, x, f):
        x = Fn.interpolate(x.flip(1), size=128, mode="bilinear", align_corners=False)
        return self.head(torch.cat([self.cnn((x - self.mu) / self.sd), f], 1))


def load():
    parts = []
    z = np.load(ROOT / "data" / "refiner_data__KITTI.npz", allow_pickle=True)
    keep = np.isin(z["split"], ("train", "val"))
    n = int(keep.sum())
    parts.append(dict(crops=z["crops"][keep], feats=np.c_[z["feats"][keep], np.zeros(n)], target=z["target"][keep],
                      lat=np.full(n, np.nan), valid=np.full(n, np.nan), split=z["split"][keep], ds=np.full(n, "KITTI")))
    z = np.load(ROOT / "data" / "refiner_data__nuScenesAll.npz", allow_pickle=True)
    keep = z["split"] == "train"
    n = int(keep.sum())
    parts.append(dict(crops=z["crops"][keep], feats=np.c_[z["feats"][keep], np.ones(n)], target=z["target"][keep],
                      lat=np.full(n, np.nan), valid=np.full(n, np.nan), split=z["split"][keep], ds=np.full(n, "nuScenes")))
    for f in sorted((ROOT / "data").glob("refiner_data__AV2dev_*of*.npz")):
        z = np.load(f, allow_pickle=True)
        n = len(z["target"])
        parts.append(dict(crops=z["crops"], feats=np.c_[z["feats"], np.ones(n)], target=z["target"], lat=z["target_lat"],
                          valid=z["valid"], split=z["split"], ds=np.full(n, "AV2")))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["a", "b", "c"])
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    torch.set_num_threads(8)
    dev = torch.device("cuda:0")
    D = load()
    tr, va = D["split"] == "train", D["split"] == "val"
    has_r = np.isfinite(D["target"]); has_l = np.isfinite(D["lat"]); has_v = np.isfinite(D["valid"])
    use_l, use_v = args.variant in ("b", "c"), args.variant == "c"
    fmu, fsd = D["feats"][tr].mean(0), D["feats"][tr].std(0) + 1e-6
    X = torch.from_numpy(D["crops"]).permute(0, 3, 1, 2).contiguous()
    Fe = torch.from_numpy(((D["feats"] - fmu) / fsd).astype(np.float32))
    Yr = torch.from_numpy(np.nan_to_num(D["target"]).astype(np.float32))
    Yl = torch.from_numpy(np.nan_to_num(D["lat"]).astype(np.float32))
    Yv = torch.from_numpy(np.nan_to_num(D["valid"]).astype(np.float32))
    Mr, Ml, Mv = (torch.from_numpy(m.astype(np.float32)) for m in (has_r, has_l & use_l, has_v & use_v))
    idx_tr = np.flatnonzero(tr & (has_r | (has_l & use_l) | (has_v & use_v)))
    torch.manual_seed(0)
    net = Res18x3(Fe.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    def predict(mask):
        net.eval(); out = []
        with torch.no_grad():
            for b in np.array_split(np.flatnonzero(mask), max(1, int(mask.sum()) // 1024)):
                out.append(net(X[b].to(dev).float() / 255.0, Fe[b].to(dev)).float().cpu().numpy())
        return np.concatenate(out)

    def val_scores(p, m):
        r = {"val_range_abs_log_err": float(np.mean(np.abs(p[has_r[m], 0] - D["target"][m][has_r[m]]))),
             "val_range_abs_log_err_lift": float(np.mean(np.abs(D["target"][m][has_r[m]])))}
        ml = has_l[m]
        if ml.any():
            r["val_lat_mae_m"] = float(np.mean(np.abs(p[ml, 1] - D["lat"][m][ml])))
            r["val_lat_mae_m_lift"] = float(np.mean(np.abs(D["lat"][m][ml])))
        mv = has_v[m]
        if mv.any():
            pv = 1 / (1 + np.exp(-p[mv, 2])); yv = D["valid"][m][mv]
            r["val_valid_bce"] = float(-np.mean(yv * np.log(pv + 1e-6) + (1 - yv) * np.log(1 - pv + 1e-6)))
            r["val_valid_base_rate"] = float(yv.mean())
        return r
    log, best, state = [], np.inf, None
    for ep in range(args.epochs):
        net.train()
        rng = np.random.default_rng(ep)
        for b in np.array_split(rng.permutation(idx_tr), max(1, len(idx_tr) // 256)):
            x = X[b].to(dev).float() / 255.0
            if rng.random() < 0.5:
                x = x.flip(3)
            o = net(x, Fe[b].to(dev))
            mr, ml, mv = Mr[b].to(dev), Ml[b].to(dev), Mv[b].to(dev)
            loss = (Fn.smooth_l1_loss(o[:, 0], Yr[b].to(dev), beta=0.05, reduction="none") * mr).sum() / mr.sum().clamp(min=1)
            if use_l:
                loss = loss + (Fn.smooth_l1_loss(o[:, 1], Yl[b].to(dev), beta=0.1, reduction="none") * ml).sum() / ml.sum().clamp(min=1)
            if use_v:
                loss = loss + (Fn.binary_cross_entropy_with_logits(o[:, 2], Yv[b].to(dev), reduction="none") * mv).sum() / mv.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        p = predict(va)
        s = val_scores(p, va)
        score = s["val_range_abs_log_err"] + (s.get("val_lat_mae_m", 0) / 10 if use_l else 0) + (s.get("val_valid_bce", 0) / 5 if use_v else 0)
        s.update({"epoch": ep, "selection_score": score})
        log.append(s)
        print(json.dumps(s), flush=True)
        if score < best:
            best, state = score, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
    od = ROOT / "out" / f"resnet18_v3{args.variant}"
    od.mkdir(parents=True, exist_ok=True)
    torch.save({"state": state, "fmu": fmu, "fsd": fsd, "arch": "resnet18x3", "variant": args.variant,
                "heads": ["log_range_ratio", "lateral_offset_m", "validity_logit"]}, od / "image_refiner.pt")
    md5 = hashlib.md5((od / "image_refiner.pt").read_bytes()).hexdigest()
    counts = {f"{ds}_{sp}": int(((D["ds"] == ds) & (D["split"] == sp)).sum()) for ds in ("KITTI", "nuScenes", "AV2") for sp in ("train", "val")}
    best_ep = min(log, key=lambda r: r["selection_score"])
    json.dump({"variant": args.variant, "epochs": args.epochs, "samples": counts, "best": best_ep, "log": log, "md5": md5},
              open(od / "train_report.json", "w"), indent=1)
    print("K23 REFINER", args.variant, "best epoch", best_ep["epoch"], "md5", md5, flush=True)


if __name__ == "__main__":
    main()
