#!/usr/bin/env python
"""K30 V5 (post-hoc): the best published method on nuScenes new test, SAHI (official library) at scale 1.0 (sahi_s10, K28
phase 3 §4: G 0.130, the highest prior-work G there), re-run with sahi_baseline.py's code unchanged, storing per frame what
the stored run did not: the action, the lifted geometry (z, lat_min, lat_max, ttc) of SAHI's and of the cheap draft's
objects, the continuous a_gt, and the previous cheap action (the cost's jerk term). This lets K30 re-score SAHI under other
loss weights (V5a) and consumer parameters (V5b: SAHI has no refiner or support, so its geometry does not depend on the
consumer). Parity: J_sahi_s10 and J_c equal the stored results/sahi/eval_newtest.pkl.
results/k30/prior/nusc_sahi_s10.pkl
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.slicing import get_slice_bboxes

from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import _apply_range_source
from rap.mono import predicted_geometry
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL

B = Path(os.environ["B"])
W, H = 1600, 900
COARSE = {0: "person", 1: "cyclist", 2: "vehicle", 3: "cyclist", 5: "vehicle", 7: "vehicle"}
VARIANTS = {"sahi_s10": 640}


def main():
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    scenes = sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq)
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    models = {k: AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(B / "models" / "yolov8s.pt"),
                                                     confidence_threshold=0.25, device="cuda:0", image_size=sz)
              for k, sz in VARIANTS.items()}
    n_slices = len(get_slice_bboxes(H, W, 640, 640, overlap_height_ratio=0.2, overlap_width_ratio=0.2))
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    rows = []
    for s in scenes:
        geom, speeds = ad.geometry(s), ad.speeds(s)
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        c = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
        toks = db.samples(by[s])
        prev_act, prev = None, {k: None for k in VARIANTS}
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            act = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)             # noqa: E731
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]                   # noqa: E731
            dc = c.det(i)
            kc = dc["conf"] >= 0.25
            geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
            cons_c = tuple(np.asarray(a, float) for a in _apply_range_source(dc["xyxy"][kc].astype(float), geo_c, g, "mono", rng, 0.0))
            a_c = act(cons_c)
            rec = {"seq": s, "frame": fr, "v": v, "J_c": cost(a_c), "a_c": a_c, "a_gt": float(a_gt), "prev_cheap": prev_act,
                   "geo_c": cons_c}
            path = str(db.image_path(toks[fr]))
            for k, m in models.items():
                res = get_sliced_prediction(path, m, slice_height=640, slice_width=640, overlap_height_ratio=0.2,
                                            overlap_width_ratio=0.2, perform_standard_pred=True, verbose=0)
                ob = [o for o in res.object_prediction_list if o.category.id in COARSE]
                xyxy = np.array([o.bbox.to_xyxy() for o in ob], float).reshape(-1, 4)
                dd = {"xyxy": xyxy, "coarse": np.array([COARSE[o.category.id] for o in ob]) if ob else np.zeros(0, "<U8")}
                gg = predicted_geometry(dd, prev[k], cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                prev[k] = dd
                tt_ = tuple(np.asarray(gg[q], float) for q in ("z", "lat_min", "lat_max", "ttc"))
                rec[f"a_{k}"] = act(tt_)
                rec[f"J_{k}"] = cost(rec[f"a_{k}"])
                rec[f"geo_{k}"] = tt_
            rows.append(rec)
            prev_act = a_c
        print(f"  {s}", flush=True)
    d = pd.DataFrame(rows)
    out = B / "results" / "k30" / "prior"
    out.mkdir(parents=True, exist_ok=True)
    d.to_pickle(out / "nusc_sahi_s10.pkl")
    ref = pd.read_pickle(B / "results" / "sahi" / "eval_newtest.pkl")
    m = d.merge(ref[["seq", "frame", "J_c", "J_sahi_s10"]], on=["seq", "frame"], suffixes=("", "_stored"))
    print("frames", len(d), "merged", len(m), "J_c identical", float(np.isclose(m.J_c, m.J_c_stored).mean()),
          "J_sahi_s10 identical", float(np.isclose(m.J_sahi_s10, m.J_sahi_s10_stored).mean()), flush=True)
    print("K30 PRIOR SAHI NUSC DONE", flush=True)


if __name__ == "__main__":
    main()
