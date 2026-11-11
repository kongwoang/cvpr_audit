#!/usr/bin/env python
"""D5 at scale: decision churn of detector upgrades on nuScenes CAM_FRONT (new val + test scenes).

Every model's cached boxes (data/cache/nusc_models/<m>, 384x640; RT-DETR 640x640) are lifted with DEEP's ego-frame
monocular lift and the model's own previous frame (the cache's stored geometry is camera-frame, so it is recomputed).
Operating points: 0.25 for YOLO models; RT-DETR's threshold calibrated on the VAL scenes to emit as many boxes as
YOLOv8s. Consumers: braking and Planner B (static obstacles), DEEP's costs, the OLD model's action history (DEEP's
shared-history convention). Per upgrade pair on the TEST scenes: recall / false positives, frames whose decision
changes, harmed share, net loss change, and the change a decision-congruent upgrade would keep (only improvements).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap import planner_b as PB
from rap.cache import Camera, DetCache
from rap.mono import box_iou, predicted_geometry
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL

from rap import cache as _rc                                # new cache roots are nuScenes CAM_FRONT caches too
_rc.NUSC_CACHES = type(_rc.NUSC_CACHES)(set(_rc.NUSC_CACHES) | {"nusc_models", "nusc_det_ft"})

B = Path(os.environ["B"])
MODELS = ["yolov8s", "yolov8m", "yolov8l", "yolov8x", "yolo11s", "yolo11m", "rtdetr-l"]
PAIRS = [("yolov8s", "yolov8m"), ("yolov8s", "yolov8l"), ("yolov8s", "yolov8x"), ("yolov8m", "yolov8l"),
         ("yolov8l", "yolov8x"), ("yolov8s", "yolo11s"), ("yolo11s", "yolo11m"), ("yolov8s", "rtdetr-l"),
         ("yolov8m", "yolo11m")]


def main():
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    sp = sp[(sp["set"] == "new") & sp.split.isin(["val", "test"])]
    have = [m for m in MODELS if (Path(CACHE) / "nusc_models" / m).exists()]
    scenes = [s for s in sp.seq if all((Path(CACHE) / "nusc_models" / m / f"{s}.npz").exists() for m in have)]
    split = dict(zip(sp.seq, sp.split))
    # RT-DETR threshold: equal box count with YOLOv8s on the val scenes
    thr = {m: 0.25 for m in have}
    if "rtdetr-l" in have:
        val = [s for s in scenes if split[s] == "val"]
        n_ref = sum(int((DetCache(Path(CACHE) / "nusc_models" / "yolov8s" / f"{s}.npz").det(i)["conf"] >= 0.25).sum())
                    for s in val for i in range(len(DetCache(Path(CACHE) / "nusc_models" / "yolov8s" / f"{s}.npz").frames)))
        confs = np.concatenate([DetCache(Path(CACHE) / "nusc_models" / "rtdetr-l" / f"{s}.npz").det(i)["conf"]
                                for s in val for i in range(len(DetCache(Path(CACHE) / "nusc_models" / "rtdetr-l" / f"{s}.npz").frames))])
        thr["rtdetr-l"] = float(np.sort(confs)[::-1][min(n_ref, len(confs) - 1)])
    print("thresholds", thr, "scenes", len(scenes), flush=True)
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    pp, cp = P.PlannerParams(), P.CostParams()
    pb, cb = PB.PARAMS_B["static_obstacles"], PB.COSTS_B["default"]
    rows = []
    for s in scenes:
        if split[s] != "test":
            continue
        geom, speeds = ad.geometry(s), ad.speeds(s)
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        caches = {m: DetCache(Path(CACHE) / "nusc_models" / m / f"{s}.npz") for m in have}
        prev = {m: None for m in have}
        n = min(len(c.frames) for c in caches.values())
        per_frame = []
        for i in range(n):
            fr = int(caches[have[0]].frames[i])
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            gt = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc"))
            gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
            a_gt, _ = P.required_decel(*gt, v, pp)
            rec = {"seq": s, "frame": fr, "v": v, "a_gt": a_gt, "gt": gt, "n_gt": len(gtb)}
            for m in have:
                d = caches[m].det(i)
                k = d["conf"] >= thr[m]
                dd = {"xyxy": d["xyxy"][k].astype(float), "coarse": d["coarse"][k]}
                gg = predicted_geometry(dd, prev[m], cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                prev[m] = dd
                tt = tuple(np.asarray(gg[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))
                iou = box_iou(gtb, dd["xyxy"]) if len(gtb) and len(dd["xyxy"]) else np.zeros((len(gtb), len(dd["xyxy"])))
                rec[f"hit_{m}"] = int((iou.max(1) >= 0.5).sum()) if iou.size else 0
                rec[f"fp_{m}"] = int((iou.max(0) < 0.3).sum()) if iou.size else len(dd["xyxy"])
                rec[f"brake_{m}"] = P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)
                rec[f"t_{m}"] = tt
            per_frame.append(rec)
        # costs with the OLD model's history, per pair
        for old, new in PAIRS:
            if old not in have or new not in have:
                continue
            pa_b, pa_p = None, None
            for rec in per_frame:
                v, gt = rec["v"], rec["gt"]
                Jb = {m: P.decision_cost(rec[f"brake_{m}"], rec["a_gt"], pa_b, pp, cp)["J"] for m in (old, new)}
                pl = {m: PB.plan(*rec[f"t_{m}"], v, pa_p, pb, cb) for m in (old, new)}
                Jp = {m: PB.executed_cost(pl[m], *gt, v, pa_p, pb, cb)["J"] for m in (old, new)}
                rows.append({"seq": rec["seq"], "frame": rec["frame"], "pair": f"{old}->{new}",
                             "Jb_old": Jb[old], "Jb_new": Jb[new], "Jp_old": Jp[old], "Jp_new": Jp[new],
                             "hit_old": rec[f"hit_{old}"], "hit_new": rec[f"hit_{new}"], "fp_old": rec[f"fp_{old}"],
                             "fp_new": rec[f"fp_{new}"], "n_gt": rec["n_gt"]})
                pa_b, pa_p = rec[f"brake_{old}"], pl[old]
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_pickle(B / "results" / "d5_scale_nuscenes.pkl")
    out = []
    for pair, q in d.groupby("pair", sort=False):
        r = {"pair": pair, "frames": len(q), "recall_old": q.hit_old.sum() / q.n_gt.sum(),
             "recall_new": q.hit_new.sum() / q.n_gt.sum(), "fp_change": q.fp_new.sum() / max(q.fp_old.sum(), 1) - 1}
        for c, lab in (("b", "brake"), ("p", "planB")):
            V = (q[f"J{c}_old"] - q[f"J{c}_new"]).to_numpy()
            L = q[f"J{c}_old"].sum()
            aff = np.abs(V) > 1e-9
            r.update({f"{lab}_affected": int(aff.sum()), f"{lab}_affected_share": aff.mean(),
                      f"{lab}_harmed": float((V < -1e-9).sum() / max(aff.sum(), 1)),
                      f"{lab}_net_change": -V.sum() / L, f"{lab}_congruent_change": -np.clip(V, 0, None).sum() / L})
        out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(B / "results" / "d5_scale_nuscenes.csv", index=False)
    pd.set_option("display.width", 300)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
