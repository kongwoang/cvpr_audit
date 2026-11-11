#!/usr/bin/env python
"""K30 V8e (pre-registered in handoff/K30/STATUS.md): corner-case objects of classes the COCO detector does not have (or
that the benchmark does not count), from the RAW annotations of the test units, projected exactly as the benchmark adapters
project their classes (rap.nusc.NuScenesDB.sequence_geometry; vtf.av2.AV2Log.geometry):
  nuScenes new test   movable_object.barrier, movable_object.trafficcone, vehicle.construction (the last one IS in the
                      benchmark, as "vehicle")
  AV2 val             CONSTRUCTION_CONE, CONSTRUCTION_BARREL, BOLLARD
Per frame: every such object with its image box, near range and lateral extent, and whether it lies in the ego corridor
(in_corridor(lat_min, lat_max, 1.2 m), 0 < long_near <= 80 m). CPU, no image read.
results/k30/corner/<ds>.pkl (DataFrame: unit, frame, cls, x1..y2, long_near, lat_min, lat_max, in_corridor)
usage: k30_corner.py --dataset nusc|av2
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", B))
_k24 = importlib.util.spec_from_file_location("k24t", B / "scripts" / "A_method" / "k24_test.py")
K24 = importlib.util.module_from_spec(_k24); _k24.loader.exec_module(K24)
from rap import planner as P                                                     # noqa: E402

NUSC_CAT = ("movable_object.barrier", "movable_object.trafficcone", "vehicle.construction")
AV2_CAT = ("CONSTRUCTION_CONE", "CONSTRUCTION_BARREL", "BOLLARD")


def box_from(allc_cam, K, W, H):
    front = allc_cam[:, 2] > 0.5
    if front.sum() < 4:
        return None
    uv = (K @ allc_cam[front].T).T
    uv = uv[:, :2] / uv[:, 2:3]
    x1, y1 = uv.min(0); x2, y2 = uv.max(0)
    x1, x2 = np.clip([x1, x2], 0, W - 1); y1, y2 = np.clip([y1, y2], 0, H - 1)
    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return None
    return float(x1), float(y1), float(x2), float(y2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2"])
    args = ap.parse_args()
    pp = P.PlannerParams()
    rows = []
    if args.dataset == "nusc":
        from rap.nusc import NuScenesDB
        from rap.paths import NUSCENES_TRAINVAL
        db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
        by = {db.scene_name(t): t for t in db.scenes}
        sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
        for s in sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq):
            for fi, tk in enumerate(db.samples(by[s])):
                Rg, tg = db._global_to_ego(tk)
                Rc, tc = db._cam_to_ego(tk)
                K = db.calib(tk).P2[:3, :3]
                for a in db._ann_by_sample.get(tk, []):
                    cat = db._t["category"][db._t["instance"][a["instance_token"]]["category_token"]]["name"]
                    if not cat.startswith(NUSC_CAT) or a["num_lidar_pts"] + a["num_radar_pts"] < 1:
                        continue
                    centre = np.array(a["translation"], float)
                    w, l, h = (float(x) for x in a["size"])
                    Rb = Rotation.from_quat(np.roll(a["rotation"], -1)).as_matrix()
                    corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
                    foot = np.stack([np.concatenate([Rb[:2, :2] @ c, [-h / 2]]) + centre for c in corners])
                    foot_ego = (Rg.T @ (foot - tg).T).T
                    top = foot.copy(); top[:, 2] += h
                    cam = (Rc.T @ ((Rg.T @ (np.vstack([foot, top]) - tg).T).T - tc).T).T
                    bx = box_from(cam, K, 1600, 900)
                    if bx is None:
                        continue
                    ln, lo, hi = foot_ego[:, 0].min(), foot_ego[:, 1].min(), foot_ego[:, 1].max()
                    rows.append({"unit": s, "frame": fi, "cls": cat, "x1": bx[0], "y1": bx[1], "x2": bx[2], "y2": bx[3],
                                 "long_near": float(ln), "lat_min": float(lo), "lat_max": float(hi)})
            print(" ", s, flush=True)
    else:
        from vtf import av2 as AV
        from vtf.av2 import _R
        root = B / "datasets" / "av2" / "sensor" / "val"
        for log in sorted(p.name for p in root.iterdir() if (p / "annotations.feather").exists()):
            L = AV.AV2Log(root, log)
            for fr, t in enumerate(L.frame_ts):
                a = L.ann[(L.ann.timestamp_ns == t) & L.ann.category.isin(list(AV2_CAT)) & (L.ann.num_interior_pts >= 1)]
                if not len(a):
                    continue
                Rl, tl = L._pose(t)
                Rm, tm = L._pose(L.image_ts(fr))
                for r in a.itertuples():
                    Rb = _R(r.qw, r.qx, r.qy, r.qz)
                    c = np.array([r.tx_m, r.ty_m, r.tz_m])
                    l, w, h = r.length_m, r.width_m, r.height_m
                    corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
                    foot = np.stack([Rb @ np.array([x, y, -h / 2]) + c for x, y in corners])
                    top = foot + Rb @ np.array([0, 0, h])
                    comp = lambda p: ((Rl @ p.T).T + tl - tm) @ Rm                  # noqa: E731
                    foot_e, allc_e = comp(foot), comp(np.vstack([foot, top]))
                    cam = (allc_e - L.tc) @ L.Rc
                    bx = box_from(cam, L.K, L.W, L.H)
                    if bx is None:
                        continue
                    rows.append({"unit": log, "frame": fr, "cls": r.category, "x1": bx[0], "y1": bx[1], "x2": bx[2], "y2": bx[3],
                                 "long_near": float(foot_e[:, 0].min()), "lat_min": float(foot_e[:, 1].min()),
                                 "lat_max": float(foot_e[:, 1].max())})
            print(" ", log, flush=True)
    d = pd.DataFrame(rows)
    if len(d):
        d["in_corridor"] = P.in_corridor(d.lat_min, d.lat_max, pp.corridor_half_w) & (d.long_near > 0) & (d.long_near <= pp.max_range)
    od = REAL / "results" / "k30" / "corner"
    od.mkdir(parents=True, exist_ok=True)
    d.to_pickle(od / f"{args.dataset}.pkl")
    print(f"K30 CORNER {args.dataset} DONE: {len(d)} objects, in corridor {int(d.in_corridor.sum()) if len(d) else 0}, "
          f"frames with one in the corridor {d[d.in_corridor].groupby(['unit', 'frame']).ngroups if len(d) else 0}", flush=True)


if __name__ == "__main__":
    main()
