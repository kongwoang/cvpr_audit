#!/usr/bin/env python
"""Baseline: SAHI (Akyon et al., ICIP 2022) — official library (sahi 0.12.7), inference only, same YOLOv8s (COCO).

Uniform overlapping slices of the native 1600x900 frame (640x640, overlap 0.2 each way, plus SAHI's standard
full-image prediction, merged by SAHI's postprocess), at two detector input scales:
  sahi_s06   each slice detected at 384 px (0.6 of native: the same magnification as the v3 crops)
  sahi_s10   each slice detected at 640 px (1.0: SAHI's default use)
Then DEEP's mono lift (own previous frame), braking decision, cost with the cheap history; compared with the
confirmatory frozen v3 numbers. Detector tensor pixels per frame are reported (slices x input^2 + full-image pass).
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
VARIANTS = {"sahi_s06": 384, "sahi_s10": 640}


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
            rec = {"seq": s, "frame": fr, "J_c": cost(a_c)}
            path = str(db.image_path(toks[fr]))
            for k, m in models.items():
                res = get_sliced_prediction(path, m, slice_height=640, slice_width=640, overlap_height_ratio=0.2,
                                            overlap_width_ratio=0.2, perform_standard_pred=True, verbose=0)
                ob = [o for o in res.object_prediction_list if o.category.id in COARSE]
                xyxy = np.array([o.bbox.to_xyxy() for o in ob], float).reshape(-1, 4)
                dd = {"xyxy": xyxy, "coarse": np.array([COARSE[o.category.id] for o in ob]) if ob else np.zeros(0, "<U8")}
                gg = predicted_geometry(dd, prev[k], cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                prev[k] = dd
                rec[f"J_{k}"] = cost(act(tuple(np.asarray(gg[q], float) for q in ("z", "lat_min", "lat_max", "ttc"))))
            rows.append(rec)
            prev_act = a_c
        print(f"  {s}", flush=True)
    d = pd.DataFrame(rows)
    out = B / "results" / "sahi"
    out.mkdir(parents=True, exist_ok=True)
    d.to_pickle(out / "eval_newtest.pkl")
    ref = pd.read_pickle(B / "results" / "nusc_full" / "eval__new_test_v3_confirm.pkl")
    m = d.merge(ref[["seq", "frame", "J_c", "J_u960", "J_spec", "J_spec_ref_resnet18", "px"]], on=["seq", "frame"], suffixes=("", "_ref"))
    assert np.allclose(m.J_c, m.J_c_ref)
    L = m.J_c.to_numpy()
    uq = m.seq.unique(); idx = [np.flatnonzero(m.seq.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    V = {k: (m.J_c - m[f"J_{k}"]).to_numpy() for k in VARIANTS}
    V.update({"u960": (m.J_c - m.J_u960).to_numpy(), "v3": (m.J_c - m.J_spec).to_numpy(),
              "v3_ref": (m.J_c - m.J_spec_ref_resnet18).to_numpy()})
    tensor_px = {k: (n_slices * sz * sz + sz * math.ceil(sz * H / W / 32) * 32) / (W * H) for k, sz in VARIANTS.items()}
    tensor_px.update({"u960": 576 * 960 / (W * H), "v3": float((m.px * 0.36).mean()), "v3_ref": float((m.px * 0.36).mean())})
    res = []
    for k, v_ in V.items():
        r = {"method": k, "G": v_.sum() / L.sum(), "tensor_px_share": tensor_px[k]}
        for ref_k in ("v3", "v3_ref"):
            dr = [(V[ref_k][t_].sum() - v_[t_].sum()) / L[t_].sum() for t_ in boots]
            r[f"{ref_k}_minus"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        res.append(r)
    res = pd.DataFrame(res)
    res.to_csv(out / "summary_newtest.csv", index=False)
    print(f"slices per frame {n_slices} (+ standard full-image prediction)")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
