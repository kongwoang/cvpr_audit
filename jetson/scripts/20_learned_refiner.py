#!/usr/bin/env python
"""D40 (exploratory): a learned range refiner for the decision's pivot, in place of the oracle.

Refiner v0: gradient boosting on box-level features of the full-resolution detection that confirms the pivot (a
second look at 640 px: box geometry, class, confidence, both monocular range cues), predicting log(true range / lifted
range).  Fitted on reference-matched full-mode detections of the training units only.  At test time the cheap
decision's pivot is looked at again: if a full detection confirms it (IoU >= 0.3), its range and lateral extent are
rescaled by the refiner; if none confirms it, it is left alone (asymmetric: never delete).  KITTI mono.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from rap import planner as P
from rap import planner_b as B
from rap.cache import DetCache
from rap.decision import KittiAdapter
from rap.mono import box_iou
from rap.paths import CACHE
from rap.risk import RiskConfig
from vtf import flips, runmeta
from vtf import headroom as HR

DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}


def box_feats(br: flips.Branch, k: int, img_w: float, img_h: float) -> list[float]:
    x1, y1, x2, y2 = br.xyxy[k]
    m = br.mono
    return [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h,
            float(br.conf[k]), float(CLS[str(br.coarse[k])]), float(m["z"][k]), float(m["z_ground"][k]),
            float(m["z_height"][k]), float(np.log(max(m["z_ground"][k], .5) / max(m["z_height"][k], .5)))]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["KITTI", "nuScenes"], default="KITTI")
    args = ap.parse_args()
    run = runmeta.new_run(f"learned_refiner_{args.dataset}", vars(args))
    global KittiAdapter
    if args.dataset == "KITTI":
        det_root, cm, fm = Path(CACHE) / "det", "cheap_320", "full_640"
        match_iou = 0.5
        consumers_wanted = ("brake", "traj")
    else:
        from rap.nusc import NuScenesDB, make_adapter
        from rap.paths import NUSCENES_TRAINVAL
        KittiAdapter = make_adapter(NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval"))
        det_root, cm, fm = Path(CACHE) / "nusc_det_tv", "ns_cheap_320", "ns_full_640"
        match_iou = 0.3
        consumers_wanted = ("brake",)
    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()
    pb, cb = B.PARAMS_B["static_obstacles"], B.COSTS_B["default"]
    rng0 = np.random.default_rng(0)
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[(dv.track == args.dataset) & (dv.geometry == "mono")].copy()
    dv["frame"] = dv.frame.astype(int)
    split = dict(zip(dv.seq, dv.split))
    seqs = sorted(set(dv.seq))

    # ---- training data: full-mode detections matched to reference boxes, training units only
    X, y = [], []
    for s in [q for q in seqs if split[q] == "train"]:
        geom = KittiAdapter.geometry(s)
        f = DetCache(det_root / fm / f"{s}.npz")
        c = DetCache(det_root / cm / f"{s}.npz")
        for i, fr in enumerate(f.frames):
            fr = int(fr)
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            if not len(g):
                continue
            sc = c.scalars(c.index[fr])
            full = flips.Branch(f.det(i), f.geo(i), cfg.thr("full"), g, "mono", rng0)
            if not len(full):
                continue
            gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
            iou = box_iou(full.xyxy, gt)
            for k in np.flatnonzero(iou.max(1) >= match_iou):
                zt = float(g["long_near"][int(np.argmax(iou[k]))])
                if zt > 0.5 and full.mono["z"][k] > 0.5:
                    X.append(box_feats(full, k, sc["img_w"], sc["img_h"]))
                    y.append(np.log(zt / full.mono["z"][k]))
    X, y = np.asarray(X), np.asarray(y)
    ref = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.06, min_samples_leaf=40,
                                        random_state=0).fit(X, y)
    print(f"  refiner fitted on {len(y)} boxes; train |log err| lift {np.mean(np.abs(y)):.3f} -> "
          f"refined {np.mean(np.abs(y - ref.predict(X))):.3f}", flush=True)

    rows, errs = [], []
    for s in seqs:
        geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s)
        c = DetCache(det_root / cm / f"{s}.npz")
        f = DetCache(det_root / fm / f"{s}.npz")
        prev_act = prev_plan = None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            j = f.index[fr]
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            sc = c.scalars(i)
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            truth = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc"))
            cheap = flips.Branch(c.det(i), c.geo(i), cfg.thr("cheap"), g, "mono", rng0)
            full = flips.Branch(f.det(j), f.geo(j), cfg.thr("full"), g, "mono", rng0)
            rec = {"seq": s, "frame": fr}
            consumers = {"brake": (lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp),
                                   lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]),
                         "traj": (lambda t, pu=prev_plan: B.plan(*t, v, pu, pb, cb),
                                  lambda a, pu=prev_plan: B.executed_cost(a, *truth, v, pu, pb, cb)["J"])}
            for name, (decide, cost) in consumers.items():
                if name not in consumers_wanted:
                    continue
                rec[f"{name}_J_c"] = cost(decide(cheap.cons_raw))
                S = HR.Obj(cheap.xyxy, cheap.cons)
                piv = HR.pivot_of(S, decide, v, pp) if len(cheap) else None
                J_learn = J_full_geo = rec[f"{name}_J_c"]
                if piv is not None and len(full):
                    iou = box_iou(cheap.xyxy[piv:piv + 1], full.xyxy)[0]
                    if iou.max() >= 0.3:
                        b = int(np.argmax(iou))
                        corr = float(np.exp(ref.predict(np.asarray([box_feats(full, b, sc["img_w"], sc["img_h"])]))[0]))
                        zf = float(full.mono["z"][b])
                        t_l = [a.copy() for a in S.t]
                        t_f = [a.copy() for a in S.t]
                        for tt, zz, lo_, hi_ in ((t_l, zf * corr, full.mono["lat_min"][b] * corr, full.mono["lat_max"][b] * corr),
                                                 (t_f, zf, full.mono["lat_min"][b], full.mono["lat_max"][b])):
                            tt[0][piv], tt[1][piv], tt[2][piv] = zz, lo_, hi_
                        J_learn = cost(decide(tuple(t_l)))
                        J_full_geo = cost(decide(tuple(t_f)))
                        if name == "brake" and len(g):
                            gi = box_iou(full.xyxy[b:b + 1], np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float))[0]
                            if gi.max() >= match_iou:
                                zt = float(g["long_near"][int(np.argmax(gi))])
                                errs.append({"split": split[s], "lift_cheap": abs(np.log(max(cheap.mono['z'][piv], .5) / zt)),
                                             "lift_full": abs(np.log(max(zf, .5) / zt)),
                                             "refined": abs(np.log(max(zf * corr, .5) / zt))})
                rec[f"{name}_J_learn"] = J_learn
                rec[f"{name}_J_fullgeo"] = J_full_geo
            prev_act = P.discrete_action(P.required_decel(*cheap.cons_raw, v, pp)[0], pp)
            if "traj" in consumers_wanted:
                prev_plan = B.plan(*cheap.cons_raw, v, prev_plan, pb, cb)
            rows.append(rec)
    d = pd.DataFrame(rows)
    e = pd.DataFrame(errs)
    print("  pivot |log range error| by split:\n", e.groupby("split").mean().round(3).to_string(), flush=True)
    out = []
    for name in consumers_wanted:
        r_ = dv[dv.system == name][["seq", "frame", "split", "V", "J_cheap"]]
        m = d.merge(r_, on=["seq", "frame"])
        assert np.allclose(m[f"{name}_J_c"], m.J_cheap, atol=1e-9)
        for sp in ("all", "test"):
            q = m if sp == "all" else m[m.split == "test"]
            Lc = float(q.J_cheap.sum())
            out.append({"cell": f"{args.dataset} mono {name}", "split": sp, "G_all_full": float(q.V.sum()) / Lc,
                        "G_pivot_full_geometry": float((q[f"{name}_J_c"] - q[f"{name}_J_fullgeo"]).sum()) / Lc,
                        "G_pivot_learned_refiner": float((q[f"{name}_J_c"] - q[f"{name}_J_learn"]).sum()) / Lc})
    res = pd.DataFrame(out)
    d.to_pickle(run / "learned_refiner.pkl")
    res.to_csv(run / "learned_refiner_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
