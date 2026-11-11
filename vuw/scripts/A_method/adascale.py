#!/usr/bin/env python
"""Baseline: AdaScale (Chin et al., MLSys 2019) re-implemented for YOLOv8s on nuScenes CAM_FRONT.

AdaScale picks, per frame, the input scale at which the detector does best, with a regressor trained on the
"optimal scale" of training frames. Original code is Caffe/R-FCN; this follows the paper's recipe:
  data   run YOLOv8s at long-side scales S = {320, 480, 640, 800, 960} on the new TRAIN scenes; the frame's optimal
         scale = argmin over S of the detection loss 1 - F1@IoU0.5 against DEEP's projected GT (ties -> smaller scale)
  train  a ResNet-18 on the frame's cheap (320) image predicting the optimal scale index (cross-entropy; the paper
         regresses a relative scale from the detector's features — a classifier over the same set is the
         equivalent for a fixed scale set); validated on the new VAL scenes
  eval   new TEST scenes: cheap pass (320, shared by every pipeline) -> predicted scale -> YOLOv8s at that scale on
         the whole frame -> DEEP's mono lift -> braking decision, DEEP's cost, cheap action history. Reports decision
         value, mean scale, network pixels (share of a native 1600x900 frame), vs frozen v3 on the same frames.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import _apply_range_source
from rap.detect import Mode, TwoFidelityDetector
from rap.mono import box_iou, predicted_geometry
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL

B = Path(os.environ["B"])
OUT = B / "results" / "adascale"
SCALES = (320, 480, 640, 800, 960)


def mode_for(s):
    return Mode(f"as{s}", int(round(s * 9 / 16 / 32)) * 32, s)


def scenes(split):
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    return sorted(sp[(sp["set"] == "new") & (sp.split == split)].seq)


def f1_loss(dets, gtb):
    k = dets["conf"] >= 0.25
    bx = dets["xyxy"][k].astype(float)
    if not len(gtb) and not len(bx):
        return 0.0
    if not len(gtb) or not len(bx):
        return 1.0
    iou = box_iou(bx, gtb)
    tp = int((iou.max(0) >= 0.5).sum())
    prec, rec = tp / len(bx), tp / len(gtb)
    return 1.0 - (2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)


def stage_data(args, db, ad, by):
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device=args.device, backend="torch")
    rows = []
    for split in ("train", "val"):
        for s in scenes(split):
            geom = ad.geometry(s)
            for fr, tk in enumerate(db.samples(by[s])):
                g = geom[geom["frame"] == fr]
                g = g[(g["y2"] - g["y1"]) >= 10]
                gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
                img = cv2.imread(str(db.image_path(tk)))
                losses = [f1_loss(det.detect(img, mode_for(sc))[0], gtb) for sc in SCALES]
                rows.append({"seq": s, "frame": fr, "split": split, "path": str(db.image_path(tk)),
                             **{f"loss_{sc}": l for sc, l in zip(SCALES, losses)},
                             "opt": int(np.argmin(np.array(losses) + 1e-6 * np.arange(len(SCALES))))})
            print(f"  {split} {s}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_pickle(OUT / "labels.pkl")
    print(d.groupby("split").opt.value_counts(normalize=True).round(3).to_string())


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(512, len(SCALES))
        self.m = m
        self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):                       # x: B x 3 x 180 x 320, RGB in [0, 1]
        return self.m((x - self.mu) / self.sd)


def small(img):
    return cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()


def stage_train(args, db, ad, by):
    d = pd.read_pickle(OUT / "labels.pkl")
    dev = torch.device(args.device)
    X = {sp: np.stack([small(cv2.imread(p)) for p in d[d.split == sp].path]) for sp in ("train", "val")}
    Y = {sp: d[d.split == sp].opt.to_numpy() for sp in ("train", "val")}
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    w = torch.tensor(1.0 / np.maximum(np.bincount(Y["train"], minlength=len(SCALES)), 1), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=(w / w.sum() * len(SCALES)).to(dev))
    best, rng = (1e9, None), np.random.default_rng(0)
    toT = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).float().div(255).to(dev)       # noqa: E731
    for ep in range(15):
        net.train()
        for b in np.array_split(rng.permutation(len(Y["train"])), max(1, len(Y["train"]) // 64)):
            loss = lossf(net(toT(X["train"][b])), torch.from_numpy(Y["train"][b]).to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = np.concatenate([net(toT(X["val"][b])).argmax(1).cpu().numpy()
                                 for b in np.array_split(np.arange(len(Y["val"])), max(1, len(Y["val"]) // 256))])
        # validation criterion: the mean detection loss of the chosen scales (what AdaScale optimises)
        vl = d[d.split == "val"][[f"loss_{s}" for s in SCALES]].to_numpy()[np.arange(len(pv)), pv].mean()
        print(f"  epoch {ep}: val detection loss at chosen scales {vl:.4f}, acc {(pv == Y['val']).mean():.3f}", flush=True)
        if vl < best[0]:
            best = (vl, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    torch.save(best[1], OUT / "scale_net.pt")


def stage_eval(args, db, ad, by):
    dev = torch.device(args.device)
    net = Net().to(dev)
    net.load_state_dict(torch.load(OUT / "scale_net.pt", map_location=dev))
    net.eval()
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device=args.device, backend="torch")
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    rows = []
    for s in scenes("test"):
        geom, speeds = ad.geometry(s), ad.speeds(s)
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        c = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
        toks = db.samples(by[s])
        prev_act, prev = None, None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            act = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)             # noqa: E731
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]                   # noqa: E731
            dc = c.det(i)
            kc = dc["conf"] >= 0.25
            geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
            cons_c = tuple(np.asarray(a, float) for a in _apply_range_source(dc["xyxy"][kc].astype(float), geo_c, g, "mono", rng, 0.0))
            a_c = act(cons_c)
            img = cv2.imread(str(db.image_path(toks[fr])))
            with torch.no_grad():
                si = int(net(torch.from_numpy(small(img)[None]).permute(0, 3, 1, 2).float().div(255).to(dev)).argmax(1).item())
            sc = SCALES[si]
            d_ = det.detect(img, mode_for(sc))[0]
            k = d_["conf"] >= 0.25
            dd = {"xyxy": d_["xyxy"][k].astype(float), "coarse": d_["coarse"][k]}
            gg = predicted_geometry(dd, prev, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
            prev = dd
            m = mode_for(sc)
            rows.append({"seq": s, "frame": fr, "J_c": cost(a_c), "scale": sc,
                         "net_px": m.net_h * m.net_w / (1600 * 900),
                         "J_ada": cost(act(tuple(np.asarray(gg[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))))})
            prev_act = a_c
        print(f"  {s}", flush=True)
    d = pd.DataFrame(rows)
    d.to_pickle(OUT / "eval_newtest.pkl")
    ref = pd.read_pickle(B / "results" / "nusc_full" / "eval__new_test_v3_confirm.pkl")
    m = d.merge(ref[["seq", "frame", "J_c", "J_u960", "J_spec", "J_spec_ref_resnet18", "px"]], on=["seq", "frame"],
                suffixes=("", "_ref"))
    assert np.allclose(m.J_c, m.J_c_ref)
    L = m.J_c.to_numpy()
    uq = m.seq.unique(); idx = [np.flatnonzero(m.seq.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    V = {"adascale": (m.J_c - m.J_ada).to_numpy(), "u960": (m.J_c - m.J_u960).to_numpy(),
         "v3": (m.J_c - m.J_spec).to_numpy(), "v3_ref": (m.J_c - m.J_spec_ref_resnet18).to_numpy()}
    out = []
    for k_, v_ in V.items():
        r = {"method": k_, "G": v_.sum() / L.sum()}
        if k_ != "adascale":
            dr = [(v_[t_].sum() - V["adascale"][t_].sum()) / L[t_].sum() for t_ in boots]
            r["minus_adascale"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(OUT / "summary_newtest.csv", index=False)
    print(f"AdaScale: mean long side {m.scale.mean():.0f}, scale shares {m.scale.value_counts(normalize=True).round(3).to_dict()}, "
          f"network pixels {m.net_px.mean():.4f} of a native frame (u960 {576 * 960 / (1600 * 900):.4f}; v3 crops "
          f"{(m.px * 0.36).mean():.4f})")
    print(res.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "train", "eval"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    {"data": stage_data, "train": stage_train, "eval": stage_eval}[args.stage](args, db, ad, by)


if __name__ == "__main__":
    main()
