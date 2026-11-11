#!/usr/bin/env python
"""Direction D: real image-space attacks on the cheap detector, placed by the decision-sensitivity field, and
speculative verification as a defence (nuScenes new test frames).

Attack: L_inf-bounded PGD (eps 16/255, 40 steps) confined to a region of the 1600x900 image (a patch-like threat),
optimised through a differentiable letterbox against COCO YOLOv8s at the cheap input (192x320).
  delete_support   region = the box of the cheap scene's highest-demand object (frames whose cheap action >= DECEL);
                   loss = the max class score of the predictions centred in that box (minimised)
  delete_random    the same on a uniformly drawn detected object
  phantom_field    region = the image box of the in-corridor grid cell with the highest insertion demand (frames
                   whose cheap action is KEEP, v > 3 m/s); loss = -(max 'car' score of predictions centred there)
  phantom_random   the same at a uniformly drawn grid cell
Evaluation on the attacked uint8 image through DEEP's normal pipeline (cv2 letterbox, no history): cheap 320, full 640,
full 960 and speculative v3 (proved loop, real crops at 0.6, guard). Success = action moved in the attacker's
direction relative to the same pipeline on the clean image; dJ = decision cost change.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from rap import planner as P
from rap.cache import Camera
from rap.detect import MODES, Mode, TwoFidelityDetector
from rap.mono import predicted_geometry
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import NUSCENES_TRAINVAL
from vtf import speculative as SP

B = Path(os.environ["B"])
EPS, STEPS, ALPHA = 16 / 255, 40, 2 / 255
W, H = 1600, 900


def lift(dd, cam, R, t):
    g = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, None, cam.calib, cam.cam_h, cam.dt,
                           cam_to_ego=(R, t))
    return tuple(np.asarray(g[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))


def keep(d):
    k = d["conf"] >= 0.25
    return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}


def view_loss(net, xa, view, region, target):
    """One view: (crop box in image px or None for the whole image, (nh, nw) network input)."""
    crop, (nh, nw) = view
    cx1, cy1, cx2, cy2 = (0, 0, W, H) if crop is None else (int(crop[0]), int(crop[1]), int(math.ceil(crop[2])),
                                                           int(math.ceil(crop[3])))
    sub = xa[..., cy1:cy2, cx1:cx2]
    h, w = sub.shape[-2:]
    r = min(nh / h, nw / w)
    rh, rw = max(int(round(h * r)), 1), max(int(round(w * r)), 1)
    top, left = (nh - rh) // 2, (nw - rw) // 2
    xr = F.interpolate(sub, size=(rh, rw), mode="bilinear", align_corners=False)
    xl = F.pad(xr, (left, nw - rw - left, top, nh - rh - top), value=114 / 255)
    out = net(xl)
    p = (out[0] if isinstance(out, (list, tuple)) else out)[0]
    x1, y1, x2, y2 = region
    nx1, ny1, nx2, ny2 = (x1 - cx1) * r + left, (y1 - cy1) * r + top, (x2 - cx1) * r + left, (y2 - cy1) * r + top
    inr = (p[0] >= nx1) & (p[0] <= nx2) & (p[1] >= ny1) & (p[1] <= ny2)
    if not inr.any():
        return None
    cls = p[4:, inr]
    return cls[[0, 1, 2, 3, 5, 7]].max() if target == "delete" else -cls[2].max()


def crop_hw(c, scale=0.6):
    w, h = c[2] - c[0], c[3] - c[1]
    return (max(32, int(math.ceil(h * scale / 32)) * 32), max(32, int(math.ceil(w * scale / 32)) * 32))


def pgd(net, img, region, mode_hw, target, dev, eps=None, views=None):
    """img uint8 HxWx3 BGR; region (x1,y1,x2,y2) in image px; target 'delete' or 'phantom'. views: extra views the
    attacker also optimises against (defence-aware attack); the cheap view is always included."""
    eps = EPS if eps is None else eps
    x0 = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().to(dev) / 255.0   # RGB
    x1, y1, x2, y2 = (int(v) for v in region)
    mask = torch.zeros_like(x0)
    mask[..., max(y1, 0):min(y2, H), max(x1, 0):min(x2, W)] = 1
    delta = torch.zeros_like(x0, requires_grad=True)
    allv = [(None, mode_hw)] + list(views or [])
    for _ in range(STEPS):
        xa = (x0 + delta * mask).clamp(0, 1)
        losses = [lv for lv in (view_loss(net, xa, v_, region, target) for v_ in allv) if lv is not None]
        if not losses:
            break
        loss = sum(losses)
        net.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            delta -= (eps / 8) * delta.grad.sign()
            delta.clamp_(-eps, eps)
        delta.grad = None
    xa = (x0 + delta.detach() * mask).clamp(0, 1)
    return (xa[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="frames per attack family")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--adaptive", action="store_true", help="delete family only; eps sweep; defence-aware views")
    args = ap.parse_args()
    dev = torch.device(args.device)
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device=args.device, backend="torch")
    import copy
    net = copy.deepcopy(det.net).float().eval()                  # the detector itself keeps its own precision
    for prm in net.parameters():
        prm.requires_grad_(False)
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    scenes = sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq)
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    frames = [(s, fr) for s in scenes for fr in range(len(db.samples(by[s])))]
    rng.shuffle(frames)
    counts = {"delete": 0, "phantom": 0}
    rows = []
    for s, fr in frames:
        if (counts["delete"] if args.adaptive else min(counts.values())) >= args.n:
            break
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        speeds = ad.speeds(s)
        v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
        g = ad.geometry(s)
        g = g[(g["frame"] == fr) & ((g["y2"] - g["y1"]) >= 10)]
        a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
        img = cv2.imread(str(db.image_path(db.samples(by[s])[fr])))
        act = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)            # noqa: E731
        cost = lambda a: P.decision_cost(a, a_gt, None, pp, cp)["J"]                        # noqa: E731
        ch = keep(det.detect(img, MODES["ns_cheap_320"])[0])
        t_c = lift(ch, cam, R, t)
        a_c = act(t_c)
        fam = "delete" if a_c >= P.DECELERATE else ("phantom" if v > 3 else None)
        if fam is None or counts[fam] >= args.n:
            continue
        counts[fam] += 1
        _, dem = P.required_decel(*t_c, v, pp)
        if fam == "delete":
            regs = {"delete_support": ch["xyxy"][int(np.argmax(dem))], "delete_random": ch["xyxy"][rng.integers(len(ch["conf"]))]}
        else:
            P2 = cam.calib.P2
            fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]

            def cellbox(r_, y_):
                vb, uc, hw, h = cy + fy * cam.cam_h / r_, cx - fx * y_ / r_, fx * SP.OBJ_W / (2 * r_), fy * SP.OBJ_H / r_
                return np.array([uc - hw, vb - h, uc + hw, vb])
            cells = [(r_, y_) for r_ in SP.R_GRID for y_ in SP.Y_GRID if abs(y_) - SP.OBJ_W / 2 < pp.corridor_half_w]
            dm = [P.required_decel(np.array([r_]), np.array([y_ - .9]), np.array([y_ + .9]), np.array([np.inf]), v, pp)[0]
                  for r_, y_ in cells]
            regs = {"phantom_field": cellbox(*cells[int(np.argmax(dm))]),
                    "phantom_random": cellbox(SP.R_GRID[rng.integers(len(SP.R_GRID))], SP.Y_GRID[rng.integers(len(SP.Y_GRID))])}

        def pipelines(im):
            out = {}
            out["cheap"] = act(lift(keep(det.detect(im, MODES["ns_cheap_320"])[0]), cam, R, t))
            out["full640"] = act(lift(keep(det.detect(im, MODES["ns_full_640"])[0]), cam, R, t))
            out["full960"] = act(lift(keep(det.detect(im, Mode("f576x960", 576, 960))[0]), cam, R, t))
            chd = keep(det.detect(im, MODES["ns_cheap_320"])[0])
            chd["mono"] = lift(chd, cam, R, t)

            def dcrop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = im[y1:y2, x1:x2]
                nw_ = max(32, int(math.ceil(sub.shape[1] * 0.6 / 32)) * 32)
                nh_ = max(32, int(math.ceil(sub.shape[0] * 0.6 / 32)) * 32)
                d = keep(det.detect(sub, Mode(f"c{nh_}x{nw_}", nh_, nw_))[0])
                d["xyxy"] = d["xyxy"] + np.array([x1, y1, x1, y1], float)
                return d
            scene, _, _ = SP.run(chd, v, cam, W, H, dcrop, lambda ff: lift(ff, cam, R, t), pp, v_close=10.0, guard=True)
            out["spec"] = act(scene["mono"])
            return out

        clean = pipelines(img)
        attacks = []
        if args.adaptive:
            if fam != "delete":
                counts[fam] -= 1
                continue
            reg = regs["delete_support"]
            # the views the loop would read after the deletion: the field crop of the scene without the target and
            # the target's support crop (context 2x), each at the crop scale 0.6
            ks = int(np.argmax(dem))
            keepm = np.arange(len(ch["conf"])) != ks
            fc = SP.field_crop(tuple(a[keepm] for a in t_c), v, cam, W, H, pp, 10.0)
            bc = SP.box_crop(reg, W, H)
            dviews = [(c_, crop_hw(c_)) for c_ in ([fc] if fc is not None else []) + [bc]]
            for eps in (8 / 255, 16 / 255, 32 / 255):
                attacks.append((f"cheap_only_eps{round(eps * 255)}", reg, eps, None))
                attacks.append((f"adaptive_eps{round(eps * 255)}", reg, eps, dviews))
        else:
            attacks = [(name, reg, EPS, None) for name, reg in regs.items()]
        for name, reg, eps, views in attacks:
            adv = pgd(net, img, reg, (MODES["ns_cheap_320"].net_h, MODES["ns_cheap_320"].net_w),
                      "phantom" if name.startswith("phantom") else "delete", dev, eps, views)
            att = pipelines(adv)
            rec = {"seq": s, "frame": fr, "attack": name, "v": v}
            for p_ in clean:
                moved = att[p_] > clean[p_] if name.startswith("phantom") else att[p_] < clean[p_]
                rec[f"succ_{p_}"] = int(moved)
                rec[f"dJ_{p_}"] = cost(att[p_]) - cost(clean[p_])
                rec[f"J_clean_{p_}"] = cost(clean[p_])
            rows.append(rec)
        if len(rows) % 20 == 0:
            print(f"  {counts}", flush=True)
    d = pd.DataFrame(rows)
    tag = "_adaptive" if args.adaptive else ""
    d.to_pickle(B / "results" / f"d_attack{tag}.pkl")
    out = []
    for name, q in d.groupby("attack"):
        r = {"attack": name, "frames": len(q)}
        for p_ in ("cheap", "full640", "full960", "spec"):
            r[f"success_{p_}"] = q[f"succ_{p_}"].mean()
            r[f"dJ_{p_}"] = q[f"dJ_{p_}"].mean()
        out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(B / "results" / f"d_attack{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
