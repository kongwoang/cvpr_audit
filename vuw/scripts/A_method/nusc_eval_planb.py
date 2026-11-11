#!/usr/bin/env python
"""Speculative perception for a second consumer: Planner B (sampling planner, DEEP's 'traj' system with static
obstacles), nuScenes CAM_FRONT, monocular lift.  Uses vtf.spec_general (same loop, guard and scene history as v3,
field and support = Planner B's exact insertion grid and leave-one-out pivots).

Per frame: cheap plan (cache), uniform 640 (cache), uniform 960 (full frame, 576x960), speculative (crops at 0.6
scale).  Plan history = the cheap plan of the previous frame (DEEP's shared history).  Cost = DEEP's executed_cost
against the ground-truth obstacles.  Paired scene bootstrap vs uniform 640 / 960.
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

from rap import planner_b as PB
from rap.cache import Camera, DetCache
from rap.decision import _apply_range_source
from rap.detect import Mode, TwoFidelityDetector
from rap.mono import predicted_geometry
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL
from vtf import spec_general as SG

B = Path(os.environ["B"])
THR = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="old")
    ap.add_argument("--split", default="train+val")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    scenes = sorted(sp[(sp["set"] == args.set) & sp.split.isin(args.split.split("+"))].seq)[: args.max_scenes or None]
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device=args.device, backend="torch")
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    cons = SG.PlannerB()
    pb, cb = cons.pb, cons.cb
    rng = np.random.default_rng(0)
    rows = []
    for s in scenes:
        geom, speeds = ad.geometry(s), ad.speeds(s)
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        cc = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
        cf = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_full_640" / f"{s}.npz")
        toks = db.samples(by[s])
        prev_plan, prev960, prev_scene = None, None, None
        for i, fr in enumerate(cc.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            gt = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc"))
            cost = lambda c: PB.executed_cost(c, *gt, v, prev_plan, pb, cb)["J"]          # noqa: E731
            plan = lambda tt: PB.plan(*tt, v, prev_plan, pb, cb)                           # noqa: E731

            def cache_t(c):
                d = c.det(i)
                k = d["conf"] >= THR
                geo = {kk: vv[k] for kk, vv in c.geo(i).items()}
                bx = d["xyxy"][k].astype(float)
                return d, k, bx, tuple(np.asarray(a, float) for a in _apply_range_source(bx, geo, g, "mono", rng, 0.0))

            dc, kc, bxc, t_c = cache_t(cc)
            _, _, _, t_f = cache_t(cf)
            p_c = plan(t_c)
            img = cv2.imread(str(db.image_path(toks[fr])))
            H, W = img.shape[:2]
            d9, _ = det.detect(img, Mode("f576x960", 576, 960))
            k9 = d9["conf"] >= THR
            dd9 = {"xyxy": d9["xyxy"][k9].astype(float), "coarse": d9["coarse"][k9]}
            g9 = predicted_geometry(dd9, prev960, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
            t9 = tuple(np.asarray(g9[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * args.scale / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * args.scale / 32)) * 32)
                d, _ = det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))
                kk = d["conf"] >= THR
                return {"xyxy": d["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                        "conf": d["conf"][kk].astype(float), "coarse": d["coarse"][kk]}

            def lift(full):
                gk = predicted_geometry({"xyxy": full["xyxy"], "coarse": full["coarse"]}, prev_scene, cam.calib,
                                        cam.cam_h, cam.dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            cons.prev = prev_plan
            cheap_d = {"xyxy": bxc, "conf": dc["conf"][kc].astype(float), "coarse": dc["coarse"][kc], "mono": t_c}
            scene, crops, rounds = SG.run(cheap_d, v, cam, W, H, detect_crop, lift, cons)
            rows.append({"seq": s, "frame": fr, "J_c": cost(p_c), "J_u640": cost(plan(t_f)), "J_u960": cost(plan(t9)),
                         "J_spec": cost(plan(scene["mono"])), "n_crops": len(crops), "rounds": rounds,
                         "px": sum((q[2] - q[0]) * (q[3] - q[1]) for q in crops) / (W * H)})
            prev_plan, prev960, prev_scene = p_c, {"xyxy": dd9["xyxy"]}, {"xyxy": scene["xyxy"]}
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_pickle(B / "results" / "nusc_full" / f"planb__{args.set}_{args.split}{args.tag}.pkl")
    L = d.J_c.to_numpy()
    G = {k: (d.J_c - d[f"J_{k}"]).to_numpy() for k in ("u640", "u960", "spec")}
    uq = d.seq.unique(); idx = [np.flatnonzero(d.seq.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0); dr = []
    for _ in range(2000):
        tt = np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))])
        dr.append({k: g_[tt].sum() / L[tt].sum() for k, g_ in G.items()})
    D = pd.DataFrame(dr)
    print(f"Planner B, {args.set} {args.split}: scenes {len(uq)}, frames {len(d)}, crops/frame {d.n_crops.mean():.2f}, "
          f"pixels {d.px.mean():.4f}")
    for k, g_ in G.items():
        aff = g_ != 0
        print(f"  {k:5s} G {g_.sum() / L.sum():+.4f}  affected {int(aff.sum())}  harmed {float((g_ < 0).sum() / max(aff.sum(), 1)):.2f}"
              f"  - u640 [{np.percentile(D[k] - D.u640, 2.5):+.3f}, {np.percentile(D[k] - D.u640, 97.5):+.3f}]"
              f"  - u960 [{np.percentile(D[k] - D.u960, 2.5):+.3f}, {np.percentile(D[k] - D.u960, 97.5):+.3f}]")


if __name__ == "__main__":
    main()
