#!/usr/bin/env python
"""Deployable fixes for the ttc-history loss found by nusc_spec_diag.py (development scenes, old train+val).

The monocular lift takes ttc from the box-height change against the previous frame.  In the loop, a crop-detected
(full-model) box was compared with the previous frame's CHEAP boxes: two detectors' box heights, so the growth ratio is
biased.  Variants (real crops at --scale everywhere, same loop, only the lift's history differs):
  prev_cheap   as shipped (previous cheap boxes)
  inherit      a crop box inherits the ttc of the current cheap box it overlaps most (IoU >= 0.3; cheap-to-cheap
               growth); crop-only objects get prev = previous merged scene
  prev_scene   previous frame's merged scene (crop boxes where it cropped, cheap boxes elsewhere)
  prev_crop    the same crops run on the previous frame's image (full-to-full growth; doubles the crop cost)
  prev_F       previous frame's full-frame 960 detections (oracle, not deployable)
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

from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import _apply_range_source
from rap.detect import Mode, TwoFidelityDetector
from rap.mono import box_iou, predicted_geometry
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL
from vtf import speculative as SP

B = Path(os.environ["B"])
THR, V_CLOSE = 0.25, 10.0
VARS = ("prev_cheap", "inherit", "prev_scene", "prev_crop", "prev_F")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--set", default="old")
    ap.add_argument("--split", default="train+val")
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--guard", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = B / "results" / "nusc_full"
    tab = pd.read_pickle(out / "flips__mono.pkl")
    tab = tab[(tab["set"] == args.set) & tab.split.isin(args.split.split("+"))]
    scenes = sorted(tab.seq.unique())[: args.max_scenes or None]
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device=args.device, backend="torch")
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    adapter = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    rows = []
    for s in scenes:
        geom, speeds = adapter.geometry(s), adapter.speeds(s)
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        c = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
        toks = db.samples(by[s])
        prev_act, prev_cheap, prev960, prev_img = None, None, None, None
        prev_scene = {v_: None for v_ in VARS}
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]              # noqa: E731
            decide = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)    # noqa: E731
            dc = c.det(i)
            kc = dc["conf"] >= THR
            geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
            bxc = dc["xyxy"][kc].astype(float)
            cons_c = tuple(np.asarray(a, float) for a in _apply_range_source(bxc, geo_c, g, "mono", rng, 0.0))
            a_c = decide(cons_c)
            img = cv2.imread(str(db.image_path(toks[fr])))
            H, W = img.shape[:2]
            d9, _ = det.detect(img, Mode("f576x960", 576, 960))
            k9 = d9["conf"] >= THR
            F = {"xyxy": d9["xyxy"][k9].astype(float), "coarse": d9["coarse"][k9]}

            def geo(dets, prev):
                gk = predicted_geometry({"xyxy": dets["xyxy"], "coarse": dets["coarse"]}, prev, cam.calib, cam.cam_h,
                                        cam.dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            a_F = decide(geo(F, prev960))
            rec = {"seq": s, "frame": fr, "a_c": a_c, "a_F": a_F, "J_c": cost(a_c), "J_F": cost(a_F)}
            cheap_d = {"xyxy": bxc, "conf": dc["conf"][kc].astype(float), "coarse": dc["coarse"][kc], "mono": cons_c}
            memo = {}

            def run_crop(im, cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = im[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * args.scale / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * args.scale / 32)) * 32)
                d, _ = det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))
                kk = d["conf"] >= THR
                return {"xyxy": d["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                        "conf": d["conf"][kk].astype(float), "coarse": d["coarse"][kk]}

            for var in VARS:
                prevc = {"xyxy": np.zeros((0, 4)), "coarse": np.zeros(0, "<U8")}

                def detect_crop(cr):
                    key = tuple(np.round(cr, 3))
                    if key not in memo:                          # same crop in another variant: reuse the pass
                        memo[key] = run_crop(img, cr)
                    if var == "prev_crop" and prev_img is not None:
                        pk = ("prev",) + key
                        if pk not in memo:
                            memo[pk] = run_crop(prev_img, cr)
                        for k_ in prevc:
                            prevc[k_] = np.concatenate([prevc[k_], memo[pk][k_]])
                    return memo[key]

                def lift(full):
                    if var == "prev_cheap":
                        return geo(full, prev_cheap)
                    if var == "prev_scene":
                        return geo(full, prev_scene[var])
                    if var == "prev_F":
                        return geo(full, prev960)
                    if var == "prev_crop":
                        return geo(full, prevc if prev_img is not None else None)
                    z = list(geo(full, prev_scene[var]))                        # inherit
                    if len(bxc) and len(full["xyxy"]):
                        iou = box_iou(full["xyxy"], bxc)
                        j = iou.argmax(1)
                        ok = iou.max(1) >= 0.3
                        z[3] = np.where(ok, cons_c[3][j], z[3])
                    return tuple(z)

                scene, crops, rounds = SP.run(cheap_d, v, cam, W, H, detect_crop, lift, pp, v_close=V_CLOSE, guard=args.guard)
                a = decide(scene["mono"])
                rec.update({f"a_{var}": a, f"J_{var}": cost(a), f"crops_{var}": len(crops),
                            f"px_{var}": sum((q[2] - q[0]) * (q[3] - q[1]) for q in crops) / (W * H)})
                prev_scene[var] = {"xyxy": scene["xyxy"]}
            rec["crop_passes"] = sum(1 for k in memo if k[0] != "prev")
            rec["prev_passes"] = sum(1 for k in memo if k[0] == "prev")
            rows.append(rec)
            prev_act, prev_cheap, prev960, prev_img = a_c, {"xyxy": bxc}, {"xyxy": F["xyxy"]}, img
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_pickle(out / f"spec_ttc__{args.tag}.pkl")
    L = d.J_c.sum()
    units = d.seq.to_numpy(); uq = np.unique(units); idx = [np.flatnonzero(units == u) for u in uq]
    G = {"uniform 960 (pi F)": (d.J_c - d.J_F).to_numpy()}
    G.update({v_: (d.J_c - d[f"J_{v_}"]).to_numpy() for v_ in VARS})
    b = np.random.default_rng(0); dr = []
    for _ in range(2000):
        tt = np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))])
        if d.J_c.to_numpy()[tt].sum() > 0:
            dr.append({k: g_[tt].sum() / d.J_c.to_numpy()[tt].sum() for k, g_ in G.items()})
    D = pd.DataFrame(dr)
    print(f"frames {len(d)}, scenes {len(uq)}")
    for k, g_ in G.items():
        a = d.a_F if k.startswith("uniform") else d[f"a_{k}"]
        lo, hi = np.percentile(D[k] - D["uniform 960 (pi F)"], [2.5, 97.5])
        px = "" if k.startswith("uniform") else f"  px {d[f'px_{k}'].mean():.4f}"
        print(f"  {k:20s} G {g_.sum() / L:.4f}  minus pi(F) [{lo:+.3f}, {hi:+.3f}]  agree(pi F) {(a == d.a_F).mean():.4f}{px}")


if __name__ == "__main__":
    main()
