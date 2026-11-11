#!/usr/bin/env python
"""K30 ground-truth pass (CPU, no image is read, no pipeline is run). Per frame of the K28 test units (k24_test.units:
nuScenes new test 228 scenes, AV2 val 150 logs, Argoverse-HD val 24 sequences), from the annotations only:
  a_gt        continuous required deceleration (rap.planner.required_decel on the annotated geometry, default PlannerParams)
  act_gt      its discrete action; act2_gt the TTC-warning consumer's (instr_eval.c2_action)
  dec_*       the decisive annotated object (argmax of the per-object requirement, if a_gt > 0): range (long_near), lateral
              centre and extent, ttc, image box, class
  rel_*       every decision-relevant annotated object (requirement >= thr_decel): ranges, lateral centres, boxes
  yaw_rate    ego yaw rate (rad / s, central difference of the ego heading over the neighbouring frames); curvature = yaw / v
The ego-mask protocol of AV2 changes predicted geometry only, never the annotations, so it does not enter here.
results/k30/gt/<ds>.pkl (a pandas DataFrame)
usage: k30_gt.py --dataset nusc|av2|ahd
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
_k24 = importlib.util.spec_from_file_location("k24t", B / "scripts" / "A_method" / "k24_test.py")
K24 = importlib.util.module_from_spec(_k24); _k24.loader.exec_module(K24)
IE = K24.IE
from rap import planner as P                                                     # noqa: E402


def heading(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def yaw_rates(Rs, ts):
    h = np.unwrap(np.array([heading(R) for R in Rs]))
    ts = np.asarray(ts, float)
    out = np.zeros(len(h))
    for i in range(len(h)):
        a, b = max(0, i - 1), min(len(h) - 1, i + 1)
        out[i] = (h[b] - h[a]) / max(ts[b] - ts[a], 1e-3) if b > a else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2", "ahd"])
    ap.add_argument("--max_units", type=int, default=0)
    args = ap.parse_args()
    ds = args.dataset
    pp = P.PlannerParams()
    db = None
    if ds == "nusc":
        from rap.nusc import NuScenesDB
        from rap.paths import NUSCENES_TRAINVAL
        db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
        by = {db.scene_name(t): t for t in db.scenes}
    rows = []
    for ui, (unit, cam, gen) in enumerate(K24.units(ds, args.max_units)):
        fr_rows = []
        for fr, path, v, g, cached in gen():
            a_gt, a_obj = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, np.zeros(0))
            gt_t = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc")) if len(g) else tuple(np.zeros(0) for _ in range(4))
            a_obj = np.asarray(a_obj, float)
            rel = np.flatnonzero(a_obj >= pp.thr_decel) if len(g) else np.zeros(0, int)
            rec = {"unit": unit, "frame": int(fr), "v": float(v), "gt": not isinstance(cached, str), "a_gt": float(a_gt),
                   "act_gt": P.discrete_action(a_gt, pp), "act2_gt": int(IE.c2_action(gt_t, pp)), "n_relevant": int(len(rel)),
                   "n_objects": int(len(g))}
            if a_gt > 0 and len(g):
                k = int(np.argmax(a_obj))
                r = {c: g[c][k] for c in g.dtype.names} if hasattr(g, "dtype") and g.dtype.names else g.iloc[k]
                rec.update({"dec_range": float(r["long_near"]), "dec_lat": float((r["lat_min"] + r["lat_max"]) / 2),
                            "dec_lat_min": float(r["lat_min"]), "dec_lat_max": float(r["lat_max"]), "dec_ttc": float(r["ttc"]),
                            "dec_box": np.array([r["x1"], r["y1"], r["x2"], r["y2"]], float), "dec_cls": str(r["coarse"]),
                            "dec_class": str(r["class"]) if "class" in r else str(r["coarse"])})
            else:
                rec.update({"dec_range": np.nan, "dec_lat": np.nan, "dec_lat_min": np.nan, "dec_lat_max": np.nan, "dec_ttc": np.nan,
                            "dec_box": None, "dec_cls": "", "dec_class": ""})
            col = (lambda c, k: g[c][k]) if hasattr(g, "dtype") and g.dtype.names else (lambda c, k: g.iloc[k][c])
            rec["rel_range"] = [float(col("long_near", k)) for k in rel]
            rec["rel_lat"] = [float((col("lat_min", k) + col("lat_max", k)) / 2) for k in rel]
            rec["rel_box"] = [np.array([col(c, k) for c in ("x1", "y1", "x2", "y2")], float) for k in rel]
            fr_rows.append(rec)
        # ego heading per frame -> yaw rate
        frs = [r["frame"] for r in fr_rows]
        if ds == "nusc":
            toks = db.samples(by[unit])
            poses = [db._global_to_ego(toks[f]) for f in frs]
            ts = [db._t["sample"][toks[f]]["timestamp"] / 1e6 for f in frs]
        else:
            L = cam
            ts = [L.frame_ts[f] * 1e-9 for f in frs]
            poses = [L._pose(L.frame_ts[f]) for f in frs]
        yr = yaw_rates([p[0] for p in poses], ts) if len(poses) else np.zeros(0)
        for r, y in zip(fr_rows, yr):
            r["yaw_rate"] = float(y)
            r["curvature"] = float(y / r["v"]) if r["v"] > 1.0 else np.nan
        rows += fr_rows
        print(f"  [{ui}] {unit}: {len(fr_rows)} frames", flush=True)
    od = REAL / "results" / "k30" / "gt"
    od.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    with open(od / f"{ds}.pkl", "wb") as fh:
        pickle.dump(df, fh)
    print(f"K30 GT {ds} DONE: {len(df)} frames, {df.unit.nunique()} units", flush=True)


if __name__ == "__main__":
    main()
