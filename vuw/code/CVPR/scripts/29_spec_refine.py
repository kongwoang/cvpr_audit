#!/usr/bin/env python
"""D47+D40 (exploratory): speculative perception with the learned pivot range refiner inside the loop. KITTI mono.

Draft (cheap) -> discovery crop (DSF insertion probes) + pivot crops -> merge full detections in the crops (proxy) ->
the merged decision's pivot, if it is a full-resolution detection, gets its range rescaled by the refiner (GBM on box
features, fitted on the training units only, as in 20_learned_refiner.py) -> decision.  Braking and Planner B.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from rap import planner as P
from rap import planner_b as B
from rap.cache import Camera, DetCache
from rap.decision import KittiAdapter
from rap.mono import box_iou
from rap.paths import CACHE
from rap.risk import RiskConfig
from vtf import flips, runmeta
from vtf import headroom as HR

Y_GRID = np.array([-6.0, -4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0])
R_GRID = np.array([12.5, 17.5, 22.5, 27.5, 35.0, 45.0, 55.0, 70.0])
OBJ_W, OBJ_H, CTX = 1.8, 1.6, 2.0
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"


def feats(xyxy, conf, cls, mono_row, img_w, img_h):
    x1, y1, x2, y2 = xyxy
    z, zg, zh = mono_row
    return [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h, conf, cls, z, zg, zh,
            float(np.log(max(zg, .5) / max(zh, .5)))]


def main():
    run = runmeta.new_run("spec_refine_KITTI_mono", {})
    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()
    pb, cb = B.PARAMS_B["static_obstacles"], B.COSTS_B["default"]
    rng0 = np.random.default_rng(0)
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str})
    dv = dv[(dv.track == "KITTI") & (dv.geometry == "mono")].copy()
    dv["frame"] = dv.frame.astype(int)
    split = dict(zip(dv.seq, dv.split))
    seqs = sorted(set(dv.seq))
    X, y = [], []
    for s in [q for q in seqs if split[q] == "train"]:
        geom = KittiAdapter.geometry(s)
        f = DetCache(Path(CACHE) / "det" / "full_640" / f"{s}.npz")
        c = DetCache(Path(CACHE) / "det" / "cheap_320" / f"{s}.npz")
        for i, fr in enumerate(f.frames):
            g = geom[geom["frame"] == int(fr)]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            if not len(g):
                continue
            sc = c.scalars(c.index[int(fr)])
            full = flips.Branch(f.det(i), f.geo(i), cfg.thr("full"), g, "mono", rng0)
            if not len(full):
                continue
            iou = box_iou(full.xyxy, np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float))
            for k in np.flatnonzero(iou.max(1) >= 0.5):
                zt = float(g["long_near"][int(np.argmax(iou[k]))])
                if zt > 0.5 and full.mono["z"][k] > 0.5:
                    X.append(feats(full.xyxy[k], full.conf[k], CLS[str(full.coarse[k])],
                                   (full.mono["z"][k], full.mono["z_ground"][k], full.mono["z_height"][k]),
                                   sc["img_w"], sc["img_h"]))
                    y.append(np.log(zt / full.mono["z"][k]))
    ref = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.06, min_samples_leaf=40,
                                        random_state=0).fit(np.asarray(X), np.asarray(y))
    rows = []
    for s in seqs:
        geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s)
        cam = Camera("KITTI", s)
        P2 = cam.calib.P2
        c = DetCache(Path(CACHE) / "det" / "cheap_320" / f"{s}.npz")
        f = DetCache(Path(CACHE) / "det" / "full_640" / f"{s}.npz")
        prev_act = prev_plan = None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            j = f.index[fr]
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            sc = c.scalars(i)
            img_w, img_h = float(sc["img_w"]), float(sc["img_h"])
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            truth = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc"))
            cheap = flips.Branch(c.det(i), c.geo(i), cfg.thr("cheap"), g, "mono", rng0)
            full = flips.Branch(f.det(j), f.geo(j), cfg.thr("full"), g, "mono", rng0)
            rec = {"seq": s, "frame": fr}
            for name in ("brake", "traj"):
                if name == "brake":
                    decide = lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp)    # noqa: E731
                    cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]            # noqa: E731
                else:
                    decide = lambda t, pu=prev_plan: B.plan(*t, v, pu, pb, cb)                  # noqa: E731
                    cost = lambda a, pu=prev_plan: B.executed_cost(a, *truth, v, pu, pb, cb)["J"]  # noqa: E731
                a_c = decide(cheap.cons)
                # discovery crop
                fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
                crops = []
                sens_boxes = []
                for r in R_GRID:
                    for yv in Y_GRID:
                        t = tuple(np.r_[m_, val] for m_, val in zip(cheap.mono_t, (r, yv - OBJ_W / 2, yv + OBJ_W / 2, np.inf)))
                        if decide(t) != a_c:
                            vb = cy + fy * cam.cam_h / r
                            uc, hw = cx - fx * yv / r, fx * OBJ_W / (2 * r)
                            sens_boxes.append([max(uc - hw, 0), max(vb - fy * OBJ_H / r, 0), min(uc + hw, img_w), min(vb, img_h)])
                if sens_boxes:
                    b = np.asarray(sens_boxes)
                    crops.append([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()])
                for k in range(len(cheap)):
                    if decide(tuple(np.delete(a, k) for a in cheap.cons)) != a_c:
                        x1, y1, x2, y2 = cheap.xyxy[k]
                        cw, ch = (x2 - x1) * CTX, (y2 - y1) * CTX
                        crops.append([(x1 + x2) / 2 - cw / 2, (y1 + y2) / 2 - ch / 2, (x1 + x2) / 2 + cw / 2, (y1 + y2) / 2 + ch / 2])

                def cin(xy):
                    if not len(xy) or not crops:
                        return np.zeros(len(xy), bool)
                    u, w_ = (xy[:, 0] + xy[:, 2]) / 2, (xy[:, 1] + xy[:, 3]) / 2
                    return np.any([(u >= cr[0]) & (u <= cr[2]) & (w_ >= cr[1]) & (w_ <= cr[3]) for cr in crops], axis=0)
                ic, jf = cin(cheap.xyxy), cin(full.xyxy)
                cons = [np.r_[a[~ic], b_[jf]] for a, b_ in zip(cheap.cons, full.cons)]
                a_spec = decide(tuple(cons))
                # refine the merged decision's pivots that are full-resolution detections
                n_c = int((~ic).sum())
                fidx = np.flatnonzero(jf)
                cons_r = [a.copy() for a in cons]
                for kk in range(len(cons[0])):
                    if kk < n_c:
                        continue
                    if decide(tuple(np.delete(a, kk) for a in cons)) != a_spec:
                        jj = fidx[kk - n_c]
                        corr = float(np.exp(ref.predict(np.asarray([feats(full.xyxy[jj], full.conf[jj], CLS[str(full.coarse[jj])],
                                                                          (full.mono["z"][jj], full.mono["z_ground"][jj], full.mono["z_height"][jj]),
                                                                          img_w, img_h)]))[0]))
                        for a in cons_r[:3]:
                            a[kk] *= corr
                rec.update({f"{name}_J_c": cost(a_c), f"{name}_J_f": cost(decide(full.cons)),
                            f"{name}_J_spec": cost(a_spec), f"{name}_J_spec_ref": cost(decide(tuple(cons_r)))})
            prev_act = P.discrete_action(P.required_decel(*cheap.cons_raw, v, pp)[0], pp)
            prev_plan = B.plan(*cheap.cons_raw, v, prev_plan, pb, cb)
            rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_pickle(run / "spec_refine.pkl")
    out = []
    for name in ("brake", "traj"):
        m = d.merge(dv[dv.system == name][["seq", "frame", "split", "V", "J_cheap"]], on=["seq", "frame"])
        assert np.allclose(m[f"{name}_J_c"] - m[f"{name}_J_f"], m.V, atol=1e-9)
        for sp in ("all", "test"):
            q = m if sp == "all" else m[m.split == "test"]
            Lc = float(q.J_cheap.sum())
            out.append({"cell": f"KITTI mono {name}", "split": sp, "G_all_full": float(q.V.sum()) / Lc,
                        "G_speculative": float((q[f"{name}_J_c"] - q[f"{name}_J_spec"]).sum()) / Lc,
                        "G_speculative_plus_refiner": float((q[f"{name}_J_c"] - q[f"{name}_J_spec_ref"]).sum()) / Lc})
    res = pd.DataFrame(out)
    res.to_csv(run / "spec_refine_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
