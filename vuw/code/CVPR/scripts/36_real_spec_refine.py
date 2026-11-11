#!/usr/bin/env python
"""B1 step 3: speculative v1 with real crops, closed-form field/pivots and the image-level refiner (braking).

cheap draft (cached cheap detections, as DEEP's cheap branch) -> closed-form DSF crop + pivot crops -> YOLOv8s on each
crop at the 960-px scale -> lift -> fuse -> pivots of the fused decision that come from crops get their range and
lateral extent rescaled by the image refiner (trained on training units only) -> decision.  Also reports the same
system without the refiner, and uniform escalation from DEEP's tables.
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import KittiAdapter, _apply_range_source
from rap.detect import Mode, TwoFidelityDetector
from rap.mono import predicted_geometry
from rap.paths import CACHE, KITTI_IMAGES, MODELS, NUSCENES_TRAINVAL
from vtf import dsf, runmeta

import importlib.util
_spec = importlib.util.spec_from_file_location("tr", str(runmeta.ROOT / "scripts" / "35_train_image_refiner.py"))
_tr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_tr)
_dsm = importlib.util.spec_from_file_location("ds", str(runmeta.ROOT / "scripts" / "34_refiner_dataset.py"))
_ds = importlib.util.module_from_spec(_dsm); _dsm.loader.exec_module(_ds)

R_GRID = np.array([12.5, 17.5, 22.5, 27.5, 35.0, 45.0, 55.0, 70.0])
Y_GRID = np.array([-6.0, -4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0])
OBJ_W, OBJ_H, CTX, THR = 1.8, 1.6, 2.0, 0.25
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["KITTI", "nuScenes"], required=True)
    ap.add_argument("--geometry", choices=["mono", "oracle"], default="mono")
    ap.add_argument("--split", choices=["all", "test"], default="test")
    args = ap.parse_args()
    run = runmeta.new_run(f"real_spec_refine_{args.dataset}_{args.geometry}", vars(args))
    ck = torch.load(sorted(glob.glob(str(runmeta.RAW / "*_image_refiner" / "image_refiner.pt")))[-1])
    net = _tr.Refiner(len(ck["fmu"])).cuda().eval()
    net.load_state_dict(ck["state"])
    det = TwoFidelityDetector(str(Path(MODELS) / "yolov8s.pt"), backend="torch")
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[(dv.track == args.dataset) & (dv.geometry == args.geometry) & (dv.system == "brake")].copy()
    dv["frame"] = dv.frame.astype(int)
    split = dict(zip(dv.seq, dv.split))
    if args.dataset == "KITTI":
        det_dir, cm, adapter, scale = Path(CACHE) / "det", "cheap_320", KittiAdapter, 960 / 1242
        img_of = lambda s, fr: Path(KITTI_IMAGES) / s / f"{fr:06d}.png"                   # noqa: E731
    else:
        from rap.nusc import NuScenesDB, make_adapter
        db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
        det_dir, cm, adapter, scale = Path(CACHE) / "nusc_det_tv", "ns_cheap_320", make_adapter(db), 0.6
        by_name = {db.scene_name(t): t for t in db.scenes}
        img_of = lambda s, fr: db.image_path(db.samples(by_name[s])[fr])                    # noqa: E731
    is_ns = float(args.dataset == "nuScenes")
    seqs = sorted(s for s in split if args.split == "all" or split[s] == "test")
    rows = []
    for s in seqs:
        geom, speeds = adapter.geometry(s), adapter.speeds(s)
        cam = Camera(args.dataset, s)
        R, t = cam.transform("ego")
        P2 = cam.calib.P2
        fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
        c = DetCache(det_dir / cm / f"{s}.npz")
        prev_act, prev_det = None, None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            dc = c.det(i)
            kc = dc["conf"] >= THR
            geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
            bxc = dc["xyxy"][kc].astype(float)
            cons_c = tuple(np.asarray(a, float) for a in _apply_range_source(bxc, geo_c, g, args.geometry, rng, 0.0))
            mono_c = tuple(np.asarray(geo_c[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]              # noqa: E731
            decide = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)    # noqa: E731
            a_c = decide(cons_c)
            img = None
            sc = c.scalars(i)
            W, H = float(sc["img_w"]), float(sc["img_h"])
            crops = []
            sens = dsf.insertion_grid(mono_c, v, R_GRID, Y_GRID, OBJ_W, pp)
            if sens.any():
                rr, yy = np.meshgrid(R_GRID, Y_GRID, indexing="ij")
                rr, yy = rr[sens], yy[sens]
                vb = cy + fy * cam.cam_h / rr
                uc, hw = cx - fx * yy / rr, fx * OBJ_W / (2 * rr)
                crops.append([max((uc - hw).min(), 0), max((vb - fy * OBJ_H / rr).min(), 0), min((uc + hw).max(), W), min(vb.max(), H)])
            for k in dsf.brake_pivots(cons_c, v, pp):
                x1, y1, x2, y2 = bxc[k]
                cw, ch = (x2 - x1) * CTX, (y2 - y1) * CTX
                crops.append([max((x1 + x2) / 2 - cw / 2, 0), max((y1 + y2) / 2 - ch / 2, 0),
                              min((x1 + x2) / 2 + cw / 2, W), min((y1 + y2) / 2 + ch / 2, H)])
            J_spec = J_ref = cost(a_c)
            n_ref = 0
            if crops:
                img = cv2.imread(str(img_of(s, fr)))
                bx, cf, cl = [], [], []
                for cr in crops:
                    x1, y1 = int(cr[0]), int(cr[1])
                    x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                    sub = img[y1:y2, x1:x2]
                    nw = max(32, int(math.ceil(sub.shape[1] * scale / 32)) * 32)
                    nh = max(32, int(math.ceil(sub.shape[0] * scale / 32)) * 32)
                    d, _ = det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))
                    kk = d["conf"] >= THR
                    bx.append(d["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float))
                    cf.append(d["conf"][kk]); cl.append(d["coarse"][kk])
                bx, cf, cl = np.concatenate(bx), np.concatenate(cf), np.concatenate(cl)
                u, w_ = (bxc[:, 0] + bxc[:, 2]) / 2, (bxc[:, 1] + bxc[:, 3]) / 2
                inside = np.zeros(len(bxc), bool)
                for cr in crops:
                    inside |= (u >= cr[0]) & (u <= cr[2]) & (w_ >= cr[1]) & (w_ <= cr[3])
                if len(bx):
                    gk = predicted_geometry({"xyxy": bx, "coarse": cl}, prev_det, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                    cons_k = [np.asarray(a, float) for a in _apply_range_source(bx, gk, g, args.geometry, rng, 0.0)]
                else:
                    gk, cons_k = None, [np.zeros(0) for _ in range(4)]
                n_out = int((~inside).sum())
                fused = [np.r_[a[~inside], b] for a, b in zip(cons_c, cons_k)]
                J_spec = cost(decide(tuple(fused)))
                # refine the fused decision's pivots that come from the crops
                refined = [a.copy() for a in fused]
                for p_ in dsf.brake_pivots(tuple(fused), v, pp):
                    if p_ < n_out or gk is None:
                        continue
                    o = p_ - n_out
                    x1, y1, x2, y2 = bx[o]
                    z, zg, zh = float(gk["z"][o]), float(gk["z_ground"][o]), float(gk["z_height"][o])
                    f = np.array([[x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(cf[o]),
                                   float(CLS[str(cl[o])]), z, zg, zh, float(np.log(max(zg, .5) / max(zh, .5))), is_ns]], np.float32)
                    f = (f - ck["fmu"]) / ck["fsd"]
                    x = torch.from_numpy(_ds.crop64(img, bx[o])).permute(2, 0, 1)[None].float().cuda() / 255.0
                    with torch.no_grad():
                        corr = float(np.exp(net(x, torch.from_numpy(f.astype(np.float32)).cuda()).item()))
                    for a in refined[:3]:
                        a[p_] *= corr
                    n_ref += 1
                J_ref = cost(decide(tuple(refined)))
            rows.append({"seq": s, "frame": fr, "J_c": cost(a_c), "J_spec": J_spec, "J_spec_ref": J_ref,
                         "n_crops": len(crops), "n_refined": n_ref})
            prev_act, prev_det = a_c, {"xyxy": dc["xyxy"][kc]}
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows).merge(dv[["seq", "frame", "unit", "split", "V", "J_cheap"]], on=["seq", "frame"])
    assert np.allclose(d.J_c, d.J_cheap, atol=1e-9)
    d.to_pickle(run / "real_spec_refine.pkl")
    units = d.unit.to_numpy(); uq = np.unique(units); idx = [np.flatnonzero(units == u) for u in uq]
    b = np.random.default_rng(0)
    out = []
    for name, col in (("speculative", "J_spec"), ("speculative + image refiner", "J_spec_ref")):
        G = (d.J_c - d[col]).to_numpy(); V = d.V.to_numpy(); L = d.J_c.to_numpy()
        diffs = []
        for _ in range(2000):
            tt = np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))])
            if L[tt].sum() > 0:
                diffs.append((G[tt].sum() - V[tt].sum()) / L[tt].sum())
        out.append({"cell": f"{args.dataset} {args.geometry} brake", "split": args.split, "method": name,
                    "G": G.sum() / L.sum(), "G_uniform640": V.sum() / L.sum(),
                    "diff_lo": np.percentile(diffs, 2.5), "diff_hi": np.percentile(diffs, 97.5),
                    "crops_per_frame": d.n_crops.mean(), "refined_per_frame": d.n_refined.mean()})
    res = pd.DataFrame(out)
    res.to_csv(run / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
