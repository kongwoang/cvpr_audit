#!/usr/bin/env python
"""ESOD baseline, decision side (main venv): read ESOD's saved predictions for the new test keyframes (YOLO txt with
confidence, from their `test.py --task test --save-txt --save-conf`), lift with DEEP's monocular ego-frame lift (own
previous frame), braking decision and cost with the cheap action history; compare with frozen v3 from the
confirmatory run. usage: esod_eval.py <predictions_dir> [<measure_log>]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import _apply_range_source
from rap.mono import predicted_geometry
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL

B = Path(os.environ["B"])
COARSE = {0: "vehicle", 1: "person", 2: "cyclist"}
W, H = 1600, 900


def main():
    pred = Path(sys.argv[1])
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    scenes = sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq)
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    rows = []
    for s in scenes:
        geom, speeds = ad.geometry(s), ad.speeds(s)
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        c = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
        prev_act, prev = None, None
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
            f = pred / f"{s}_{fr:03d}.txt"
            arr = np.loadtxt(f, ndmin=2) if f.exists() and f.stat().st_size else np.zeros((0, 6))
            arr = arr[arr[:, 5] >= 0.25] if len(arr) else arr
            xyxy = np.stack([(arr[:, 1] - arr[:, 3] / 2) * W, (arr[:, 2] - arr[:, 4] / 2) * H,
                             (arr[:, 1] + arr[:, 3] / 2) * W, (arr[:, 2] + arr[:, 4] / 2) * H], 1) if len(arr) else np.zeros((0, 4))
            dd = {"xyxy": xyxy, "coarse": np.array([COARSE[int(k)] for k in arr[:, 0]]) if len(arr) else np.zeros(0, "<U8")}
            gg = predicted_geometry(dd, prev, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
            prev = dd
            rows.append({"seq": s, "frame": fr, "J_c": cost(a_c),
                         "J_esod": cost(act(tuple(np.asarray(gg[q], float) for q in ("z", "lat_min", "lat_max", "ttc"))))})
            prev_act = a_c
    d = pd.DataFrame(rows)
    out = B / "results" / os.environ.get("EVAL_OUT", "esod")     # reused for ZoomDet (same prediction format)
    out.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("ESOD_TAG", "")                   # e.g. _s1 for another training seed
    d.to_pickle(out / f"eval_newtest{tag}.pkl")
    ref = pd.read_pickle(B / "results" / "nusc_full" / "eval__new_test_v3_confirm.pkl")
    m = d.merge(ref[["seq", "frame", "J_c", "J_u960", "J_spec", "J_spec_ref_resnet18"]], on=["seq", "frame"], suffixes=("", "_ref"))
    assert np.allclose(m.J_c, m.J_c_ref)
    ft = pd.read_pickle(B / "results" / "fovea" / "eval_ft_newtest.pkl")      # ESOD is trained on nuScenes: fine-tuned refs
    m = m.merge(ft[["seq", "frame", "J_u960_ft", "J_v3_ftverifier"]], on=["seq", "frame"])
    L = m.J_c.to_numpy()
    uq = m.seq.unique(); idx = [np.flatnonzero(m.seq.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    mn = os.environ.get("METHOD", "esod")                          # name in the summary (zoomdet when reused)
    V = {mn: (m.J_c - m.J_esod).to_numpy(), "u960": (m.J_c - m.J_u960).to_numpy(), "v3": (m.J_c - m.J_spec).to_numpy(),
         "v3_ref": (m.J_c - m.J_spec_ref_resnet18).to_numpy(), "u960_ft": (m.J_c - m.J_u960_ft).to_numpy(),
         "v3_ftverifier": (m.J_c - m.J_v3_ftverifier).to_numpy()}
    res = []
    for k_, v_ in V.items():
        r = {"method": k_, "G": v_.sum() / L.sum()}
        if k_ != mn:
            dr = [(v_[t_].sum() - V[mn][t_].sum()) / L[t_].sum() for t_ in boots]
            r[f"minus_{mn}"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        res.append(r)
    res = pd.DataFrame(res)
    res.to_csv(out / f"summary_newtest{tag}.csv", index=False)
    print(res.to_string(index=False))
    if len(sys.argv) > 2:
        print("ESOD compute (their measure task):", *[l.strip() for l in open(sys.argv[2]) if "FLOP" in l or "FPS" in l][-3:])


if __name__ == "__main__":
    main()
