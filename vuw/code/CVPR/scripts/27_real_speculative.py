#!/usr/bin/env python
"""D47-real: speculative perception with real detections on every crop (DSF discovery crop + pivot crops).

One round: from the cheap draft (mono geometry for the field, consumer geometry for the decision), build the discovery
crop (insertion-sensitive ground cells, projected) and a crop of twice the box around each pivot of the draft braking
decision.  Run YOLOv8s on each crop at the full mode's scale, lift with DEEP's ego-frame monocular lift, and replace the
draft's detections whose centre lies in any crop.  Braking controller, DEEP's shared history.  KITTI or nuScenes.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import KittiAdapter, _apply_range_source
from rap.detect import Mode, TwoFidelityDetector
from rap.mono import predicted_geometry
from rap.paths import CACHE, KITTI_IMAGES, MODELS, NUSCENES_TRAINVAL
from rap.risk import RiskConfig
from vtf import runmeta

Y_GRID = np.array([-6.0, -4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0])
R_GRID = np.array([12.5, 17.5, 22.5, 27.5, 35.0, 45.0, 55.0, 70.0])
OBJ_W, OBJ_H, CTX = 1.8, 1.6, 2.0
DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"


def cell_boxes(cam, img_w, img_h):
    P2 = cam.calib.P2
    fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
    rr, yy = np.meshgrid(R_GRID, Y_GRID, indexing="ij")
    vb = cy + fy * cam.cam_h / rr
    vt = vb - fy * OBJ_H / rr
    uc = cx - fx * yy / rr
    hw = fx * OBJ_W / (2 * rr)
    return np.stack([np.clip(uc - hw, 0, img_w), np.clip(vt, 0, img_h), np.clip(uc + hw, 0, img_w),
                     np.clip(vb, 0, img_h)], -1)


def centres_in(xyxy, crops):
    if not len(xyxy) or not crops:
        return np.zeros(len(xyxy), bool)
    u, v = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
    m = np.zeros(len(xyxy), bool)
    for c in crops:
        m |= (u >= c[0]) & (u <= c[2]) & (v >= c[1]) & (v <= c[3])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["KITTI", "nuScenes"], required=True)
    ap.add_argument("--geometry", choices=["mono", "oracle"], default="mono")
    ap.add_argument("--split", choices=["all", "test"], default="all")
    ap.add_argument("--no_discovery", action="store_true")
    ap.add_argument("--no_pivots", action="store_true")
    ap.add_argument("--scale", type=float, default=None, help="crop detection scale (default: the full mode's)")
    args = ap.parse_args()
    tag = ("_nodisc" if args.no_discovery else "") + ("_nopiv" if args.no_pivots else "") + \
          (f"_s{args.scale:.3f}" if args.scale else "")
    run = runmeta.new_run(f"realspec{tag}_{args.dataset}_{args.geometry}", vars(args))
    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    det = TwoFidelityDetector(str(Path(MODELS) / "yolov8s.pt"), backend="torch")
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[(dv.track == args.dataset) & (dv.geometry == args.geometry) & (dv.system == "brake")].copy()
    dv["frame"] = dv.frame.astype(int)
    split = dict(zip(dv.seq, dv.split))
    if args.dataset == "KITTI":
        det_dir, cm, fm, adapter, scale = Path(CACHE) / "det", "cheap_320", "full_640", KittiAdapter, 640 / 1242
        img_of = lambda s, fr: Path(KITTI_IMAGES) / s / f"{fr:06d}.png"                   # noqa: E731
    else:
        from rap.nusc import NuScenesDB, make_adapter
        db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
        det_dir, cm, fm, adapter, scale = Path(CACHE) / "nusc_det_tv", "ns_cheap_320", "ns_full_640", make_adapter(db), 0.4
        by_name = {db.scene_name(t): t for t in db.scenes}
        img_of = lambda s, fr: db.image_path(db.samples(by_name[s])[fr])                    # noqa: E731
    if args.scale:
        scale = args.scale
    seqs = [p.stem for p in sorted((det_dir / cm).glob("*.npz"))]
    if args.split == "test":
        seqs = [s for s in seqs if split.get(s) == "test"]
    rows = []
    for s in seqs:
        geom, speeds = adapter.geometry(s), adapter.speeds(s)
        cam = Camera(args.dataset, s)
        R, t = cam.transform("ego")
        c, f = DetCache(det_dir / cm / f"{s}.npz"), DetCache(det_dir / fm / f"{s}.npz")
        prev_act = None
        prev_det = None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            sc = c.scalars(i)
            img_w, img_h = float(sc["img_w"]), float(sc["img_h"])
            dc = c.det(i)
            kc = dc["conf"] >= cfg.thr("cheap")
            geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
            bxc = dc["xyxy"][kc].astype(float)
            cons_c = tuple(np.asarray(a, float) for a in _apply_range_source(bxc, geo_c, g, args.geometry, rng, 0.0))
            mono_c = tuple(np.asarray(geo_c[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))
            decide = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)    # noqa: E731
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]              # noqa: E731
            a_c = decide(cons_c)
            crops = []
            if not args.no_discovery:
                base = decide(mono_c)
                boxes = cell_boxes(cam, img_w, img_h)
                sens = np.zeros((len(R_GRID), len(Y_GRID)), bool)
                for ai, r in enumerate(R_GRID):
                    for bi, y in enumerate(Y_GRID):
                        tt = tuple(np.r_[m_, val] for m_, val in zip(mono_c, (r, y - OBJ_W / 2, y + OBJ_W / 2, np.inf)))
                        sens[ai, bi] = decide(tt) != base
                if sens.any():
                    b = boxes[sens]
                    crops.append(np.array([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()]))
            if not args.no_pivots:
                for k in range(len(bxc)):
                    if decide(tuple(np.delete(a, k) for a in cons_c)) != a_c:
                        x1, y1, x2, y2 = bxc[k]
                        cw, ch = (x2 - x1) * CTX, (y2 - y1) * CTX
                        cx_, cy_ = (x1 + x2) / 2, (y1 + y2) / 2
                        crops.append(np.array([max(cx_ - cw / 2, 0), max(cy_ - ch / 2, 0),
                                               min(cx_ + cw / 2, img_w), min(cy_ + ch / 2, img_h)]))
            px = 0
            J_spec = cost(a_c)
            if crops:
                img = cv2.imread(str(img_of(s, fr)))
                allb, allc, allk = [], [], []
                for cr in crops:
                    x1, y1 = int(math.floor(cr[0])), int(math.floor(cr[1]))
                    x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                    sub = img[y1:y2, x1:x2]
                    nw = max(32, int(math.ceil(sub.shape[1] * scale / 32)) * 32)
                    nh = max(32, int(math.ceil(sub.shape[0] * scale / 32)) * 32)
                    px += nh * nw
                    d, _ = det.detect(sub, Mode(f"crop_{nh}x{nw}", nh, nw))
                    keep = d["conf"] >= cfg.thr("full")
                    allb.append(d["xyxy"][keep].astype(float) + np.array([x1, y1, x1, y1], float))
                    allc.append(d["conf"][keep])
                    allk.append(d["coarse"][keep])
                bx = np.concatenate(allb) if allb else np.zeros((0, 4))
                dd = {"xyxy": bx, "coarse": np.concatenate(allk) if allk else np.zeros(0, dtype="<U8")}
                if len(bx):
                    gk = predicted_geometry(dd, prev_det, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                    cons_k = tuple(np.asarray(a, float) for a in _apply_range_source(bx, gk, g, args.geometry, rng, 0.0))
                else:
                    cons_k = tuple(np.zeros(0) for _ in range(4))
                out = centres_in(bxc, crops)
                fused = tuple(np.r_[a[~out], b] for a, b in zip(cons_c, cons_k))
                J_spec = cost(decide(fused))
            rows.append({"seq": s, "frame": fr, "J_c": cost(a_c), "J_spec": J_spec, "px": px, "n_crops": len(crops)})
            prev_act = a_c
            prev_det = {"xyxy": dc["xyxy"][kc]}
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows).merge(dv[["seq", "frame", "split", "V", "J_cheap"]], on=["seq", "frame"])
    assert np.allclose(d.J_c, d.J_cheap, atol=1e-9)
    d.to_pickle(run / "realspec.pkl")
    full_px = 192 * 640 if args.dataset == "KITTI" else 384 * 640
    out = []
    for sp in ("all", "test"):
        q = d if sp == "all" else d[d.split == "test"]
        if not len(q):
            continue
        Lc = float(q.J_cheap.sum())
        out.append({"cell": f"{args.dataset} {args.geometry} brake", "variant": tag or "discovery+pivots", "split": sp,
                    "G_all_full": float(q.V.sum()) / Lc, "G_speculative_real": float((q.J_c - q.J_spec).sum()) / Lc,
                    "net_pixels_vs_full": float(q.px.mean()) / full_px, "crops_per_frame": float(q.n_crops.mean())})
    res = pd.DataFrame(out)
    res.to_csv(run / "realspec_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
