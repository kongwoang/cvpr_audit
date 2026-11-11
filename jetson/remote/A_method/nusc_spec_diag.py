#!/usr/bin/env python
"""Where does speculative perception lose value on nuScenes?  Development scenes only (old train+val).

Per frame, all against the full-frame 576x960 detections F (lifted with F's own previous frame):
  real        the proved loop, YOLOv8s on each crop at --scale (as in nusc_full_eval)
  loc         the loop with exact locality: a crop returns the objects of F whose centre lies in it
  loc_near    as loc, with the sensitivity grid extended to near range (2.5-10 m) and +-9 m laterally
  loc_near_tt as loc_near, with the lift's previous frame = F's (isolates the ttc history)
The Proposition says loc_near_tt must reproduce pi(F) wherever the probe covers the true demand; the rest is locality
(real vs loc) and coverage (loc vs loc_near).  Reports agreement with pi(F), value G, pixels.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import _apply_range_source
from rap.detect import Mode, TwoFidelityDetector
from rap.mono import predicted_geometry
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL
from vtf import speculative as SP

B = Path(os.environ["B"])
THR, V_CLOSE = 0.25, 10.0
R_NEAR = np.r_[[2.5, 5.0, 7.5, 10.0], SP.R_GRID]
Y_WIDE = np.arange(-9.0, 9.01, 1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--max_scenes", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = B / "results" / "nusc_full"
    tab = pd.read_pickle(out / "flips__mono.pkl")
    tab = tab[(tab["set"] == "old") & tab.split.isin(["train", "val"])]
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
        prev_act, prev_det, prev960 = None, None, None
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
            F = {"xyxy": d9["xyxy"][k9].astype(float), "conf": d9["conf"][k9].astype(float), "coarse": d9["coarse"][k9]}

            def lifter(prev):
                def lift(full):
                    gk = predicted_geometry({"xyxy": full["xyxy"], "coarse": full["coarse"]}, prev, cam.calib, cam.cam_h,
                                            cam.dt, cam_to_ego=(R, t))
                    return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))
                return lift

            a_F = decide(lifter(prev960)(F))
            rec = {"seq": s, "frame": fr, "v": v, "a_c": a_c, "a_F": a_F, "J_c": cost(a_c), "J_F": cost(a_F),
                   "n_F": len(F["conf"]), "n_c": int(kc.sum())}

            def real_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * args.scale / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * args.scale / 32)) * 32)
                d, _ = det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))
                kk = d["conf"] >= THR
                return {"xyxy": d["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                        "conf": d["conf"][kk].astype(float), "coarse": d["coarse"][kk]}

            def loc_crop(cr):
                m = SP.centres_in(F["xyxy"], [cr])
                return {k: F[k][m] for k in F}

            cheap_d = {"xyxy": bxc, "conf": dc["conf"][kc].astype(float), "coarse": dc["coarse"][kc], "mono": cons_c}
            variants = {"real": (real_crop, None, prev_det, None), "loc": (loc_crop, None, prev_det, None),
                        "loc_near": (loc_crop, R_NEAR, prev_det, Y_WIDE), "loc_near_tt": (loc_crop, R_NEAR, prev960, Y_WIDE)}
            for name, (fn, rg, prev, yg) in variants.items():
                y_old = SP.Y_GRID
                if yg is not None:
                    SP.Y_GRID = yg
                try:
                    scene, crops, rounds = SP.run(cheap_d, v, cam, W, H, fn, lifter(prev), pp, v_close=V_CLOSE, r_grid=rg)
                finally:
                    SP.Y_GRID = y_old
                a = decide(scene["mono"])
                rec.update({f"a_{name}": a, f"J_{name}": cost(a), f"px_{name}": sum((q[2] - q[0]) * (q[3] - q[1]) for q in crops) / (W * H),
                            f"n_crops_{name}": len(crops), f"rounds_{name}": rounds})
            rows.append(rec)
            prev_act, prev_det, prev960 = a_c, {"xyxy": dc["xyxy"][kc]}, {"xyxy": F["xyxy"]}
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_pickle(out / "spec_diag__dev.pkl")
    L = d.J_c.sum()
    print(f"frames {len(d)}; G(pi(F)) = {(d.J_c - d.J_F).sum() / L:.4f}")
    for n in ("real", "loc", "loc_near", "loc_near_tt"):
        dis = d[f"a_{n}"] != d.a_F
        print(f"  {n:12s} G {(d.J_c - d[f'J_{n}']).sum() / L:.4f}  agree(pi F) {1 - dis.mean():.4f}  "
              f"disagree n {int(dis.sum())} (a_c==a_F among them {int((dis & (d.a_c == d.a_F)).sum())})  "
              f"px {d[f'px_{n}'].mean():.4f}  crops {d[f'n_crops_{n}'].mean():.2f}  "
              f"lost value {(d[f'J_{n}'] - d.J_F)[dis].sum() / L:+.4f}")
    print(pd.crosstab(d.a_F, d.a_real, rownames=["a_F"], colnames=["a_real"]).to_string())
    print(pd.crosstab(d.a_F, d.a_loc_near_tt, rownames=["a_F"], colnames=["a_loc_near_tt"]).to_string())


if __name__ == "__main__":
    main()
