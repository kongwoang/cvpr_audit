#!/usr/bin/env python
"""Per-object look at frames where the loop with prev_crop disagrees with pi(F) (development scenes).

For each such frame: the crops, F's objects (box, z, ttc, demand) and the loop's merged objects under prev_crop and
prev_F, plus whether each crop box touches an interior crop edge (truncation).
"""
import math, os, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import _apply_range_source
from rap.detect import Mode, TwoFidelityDetector
from rap.mono import predicted_geometry
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL
from vtf import speculative as SP

B = Path(os.environ["B"])
d = pd.read_pickle(B / "results" / "nusc_full" / "spec_ttc__dev.pkl")
bad = d[(d.a_prev_crop != d.a_F)]
print("disagreements prev_crop:", len(bad), pd.crosstab(bad.a_F, bad.a_prev_crop).to_string(), sep="\n")
pick = bad.groupby("seq").head(2).head(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cpu", backend="torch")
db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
adapter = make_adapter(db)
by = {db.scene_name(t): t for t in db.scenes}
pp = P.PlannerParams()
rng = np.random.default_rng(0)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
for _, r in pick.iterrows():
    s, fr = r.seq, int(r.frame)
    geom, speeds = adapter.geometry(s), adapter.speeds(s)
    cam = Camera("nuScenes", s); R, t = cam.transform("ego")
    c = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
    toks = db.samples(by[s])
    i = list(map(int, c.frames)).index(fr)
    v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
    g = geom[geom["frame"] == fr]; g = g[(g["y2"] - g["y1"]) >= 10]
    dc = c.det(i); kc = dc["conf"] >= 0.25
    geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
    bxc = dc["xyxy"][kc].astype(float)
    cons_c = tuple(np.asarray(a, float) for a in _apply_range_source(bxc, geo_c, g, "mono", rng, 0.0))
    imgs = [cv2.imread(str(db.image_path(toks[f]))) for f in (fr - 1, fr)]
    H, W = imgs[1].shape[:2]

    def full(im):
        d9, _ = det.detect(im, Mode("f576x960", 576, 960)); k9 = d9["conf"] >= 0.25
        return {"xyxy": d9["xyxy"][k9].astype(float), "coarse": d9["coarse"][k9]}

    Fp, F = full(imgs[0]), full(imgs[1])
    geo = lambda dd, prev: predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, prev, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))  # noqa: E731

    def show(tag, dd, gk):
        a_max, a_obj = P.required_decel(gk["z"], gk["lat_min"], gk["lat_max"], gk["ttc"], v, pp)
        print(f"   {tag}: action {P.discrete_action(a_max, pp)}")
        for k in np.argsort(-np.asarray(a_obj))[:4]:
            print(f"      box {dd['xyxy'][k]} z {gk['z'][k]:.1f} lat [{gk['lat_min'][k]:.1f},{gk['lat_max'][k]:.1f}] ttc {gk['ttc'][k]:.2f} demand {a_obj[k]:.2f}")

    def crop_det(im, cr):
        x1, y1 = int(cr[0]), int(cr[1]); x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
        sub = im[y1:y2, x1:x2]
        nw = max(32, int(math.ceil(sub.shape[1] * 0.6 / 32)) * 32); nh = max(32, int(math.ceil(sub.shape[0] * 0.6 / 32)) * 32)
        dd, _ = det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw)); kk = dd["conf"] >= 0.25
        return {"xyxy": dd["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float), "conf": dd["conf"][kk].astype(float), "coarse": dd["coarse"][kk]}

    print(f"== {s} frame {fr} v {v:.1f}  a_c {r.a_c} a_F {r.a_F} a_prev_crop {r.a_prev_crop} a_prev_F {r.a_prev_F}  gt-J: F {r.J_F:.2f} spec {r.J_prev_crop:.2f}")
    show("F (prev F)", F, geo(F, Fp))
    prevc = {"xyxy": np.zeros((0, 4)), "coarse": np.zeros(0, "<U8")}

    def dcrop(cr):
        pc = crop_det(imgs[0], cr)
        for k in prevc: prevc[k] = np.concatenate([prevc[k], pc[k]])
        return crop_det(imgs[1], cr)

    cheap_d = {"xyxy": bxc, "conf": dc["conf"][kc].astype(float), "coarse": dc["coarse"][kc], "mono": cons_c}
    lift = lambda ff: tuple(np.asarray(geo(ff, prevc)[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))  # noqa: E731
    scene, crops, rounds = SP.run(cheap_d, v, cam, W, H, dcrop, lift, pp, v_close=10.0)
    print(f"   crops {np.array(crops)} rounds {rounds}")
    ff = {"xyxy": scene["xyxy"], "coarse": scene["coarse"]}
    gk = dict(zip(("z", "lat_min", "lat_max", "ttc"), scene["mono"]))
    show("loop prev_crop", ff, gk)
    fb = scene["xyxy"][scene["from_full"]]
    trunc = [any((abs(b[0] - cr[0]) < 3 and cr[0] > 1) or (abs(b[2] - cr[2]) < 3 and cr[2] < W - 2) or
                 (abs(b[1] - cr[1]) < 3 and cr[1] > 1) or (abs(b[3] - cr[3]) < 3 and cr[3] < H - 2) for cr in crops) for b in fb]
    print(f"   crop boxes touching an interior crop edge: {sum(trunc)}/{len(fb)}; prev-crop boxes {len(prevc['xyxy'])}")
    show("cheap", {"xyxy": bxc}, dict(zip(("z", "lat_min", "lat_max", "ttc"), cons_c)))
