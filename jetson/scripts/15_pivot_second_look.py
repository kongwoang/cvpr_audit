#!/usr/bin/env python
"""D9 (exploratory): a real second look at the decision's pivot — the full-resolution detector on a crop around it.

No reference state enters the rule.  For the cheap decision's pivot (the object whose removal alone changes the
decision; else the one the braking rule reacts to most), the full mode is run on a crop of twice the pivot box's size
(proxy: the cached full-frame detections whose centre falls in that crop).  If a full detection overlaps the pivot
(IoU >= 0.3) the pivot is confirmed and takes the full detection's geometry; otherwise it is dropped.  Up to K pivots.
Cost = crop area / image area at full resolution (compared with 1.0 for escalating the frame).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap import planner_b as B
from rap.cache import DetCache
from rap.decision import KittiAdapter
from rap.mono import box_iou
from rap.paths import CACHE, NUSCENES_TRAINVAL
from rap.risk import RiskConfig
from rap.scoring import topk_expect
from vtf import flips, runmeta
from vtf import headroom as HR

CONFIRM_IOU = 0.3
CTX = 2.0
DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"


def second_look(cheap: flips.Branch, full: flips.Branch, decide, v, pp, K, img_w, img_h, geo_source="full"):
    S = HR.Obj(cheap.xyxy, cheap.cons)
    area = 0.0
    for _ in range(K):
        i = HR.pivot_of(S, decide, v, pp)
        if i is None:
            break
        x1, y1, x2, y2 = S.xyxy[i]
        cw, ch = (x2 - x1) * CTX, (y2 - y1) * CTX
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        crop = np.array([max(cx - cw / 2, 0), max(cy - ch / 2, 0), min(cx + cw / 2, img_w), min(cy + ch / 2, img_h)])
        area += (crop[2] - crop[0]) * (crop[3] - crop[1]) / (img_w * img_h)
        if len(full):
            u = (full.xyxy[:, 0] + full.xyxy[:, 2]) / 2
            w_ = (full.xyxy[:, 1] + full.xyxy[:, 3]) / 2
            inside = (u >= crop[0]) & (u <= crop[2]) & (w_ >= crop[1]) & (w_ <= crop[3])
            iou = box_iou(S.xyxy[i:i + 1], full.xyxy)[0] * inside
        else:
            iou = np.zeros(0)
        if not len(iou) or iou.max() < CONFIRM_IOU:
            S.drop(i)
            continue
        b = int(np.argmax(iou))
        if geo_source == "full":
            for t_, src in zip(S.t, full.cons):
                t_[i] = src[b]
        S.verified[i] = True
    return S, area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["KITTI", "nuScenes"], required=True)
    ap.add_argument("--geometry", choices=["mono", "oracle"], required=True)
    args = ap.parse_args()
    run = runmeta.new_run(f"secondlook_{args.dataset}_{args.geometry}", vars(args))
    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()
    pb, cb = B.PARAMS_B["static_obstacles"], B.COSTS_B["default"]
    rng0 = np.random.default_rng(0)
    if args.dataset == "KITTI":
        det_dir, cm, fm, adapter, with_traj = Path(CACHE) / "det", "cheap_320", "full_640", KittiAdapter, True
    else:
        from rap.nusc import NuScenesDB, make_adapter
        det_dir, cm, fm = Path(CACHE) / "nusc_det_tv", "ns_cheap_320", "ns_full_640"
        adapter, with_traj = make_adapter(NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")), False
    seqs = [p.stem for p in sorted((det_dir / cm).glob("*.npz"))]
    rows = []
    for s in seqs:
        geom, speeds = adapter.geometry(s), adapter.speeds(s)
        c, f = DetCache(det_dir / cm / f"{s}.npz"), DetCache(det_dir / fm / f"{s}.npz")
        prev_act = prev_plan = None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            j = f.index[fr]
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            sc = c.scalars(i)
            img_w, img_h = float(sc["img_w"]), float(sc["img_h"])
            cheap = flips.Branch(c.det(i), c.geo(i), cfg.thr("cheap"), g, args.geometry, rng0)
            full = flips.Branch(f.det(j), f.geo(j), cfg.thr("full"), g, args.geometry, rng0)
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            truth = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc"))
            rec = {"seq": s, "frame": fr}
            consumers = {"brake": (lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp),
                                   lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"])}
            if with_traj:
                pu = prev_plan
                consumers["traj"] = (lambda t: B.plan(*t, v, pu, pb, cb),
                                     lambda a: B.executed_cost(a, *truth, v, pu, pb, cb)["J"])
            for name, (decide, cost) in consumers.items():
                rec[f"{name}_J_c"] = cost(decide(cheap.cons_raw))
                rec[f"{name}_J_f"] = cost(decide(full.cons_raw))
                for K in (1, 2):
                    S, area = second_look(cheap, full, decide, v, pp, K, img_w, img_h)
                    rec[f"{name}_J_sl{K}"] = cost(decide(S.cons()))
                    rec[f"{name}_area_sl{K}"] = area
                S, _ = second_look(cheap, full, decide, v, pp, 1, img_w, img_h, geo_source="cheap")
                rec[f"{name}_J_sl1_existence_only"] = cost(decide(S.cons()))
            prev_act = P.discrete_action(P.required_decel(*cheap.cons_raw, v, pp)[0], pp)
            if with_traj:
                prev_plan = B.plan(*cheap.cons_raw, v, prev_plan, pb, cb)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_pickle(run / f"secondlook__{args.dataset}__{args.geometry}.pkl")

    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[dv.track == args.dataset].copy()
    dv["frame"] = dv.frame.astype(int)
    out = []
    for name in (["brake", "traj"] if with_traj else ["brake"]):
        ref = dv[(dv.geometry == args.geometry) & (dv.system == name)][["seq", "frame", "split", "V", "J_cheap"]]
        d = df.merge(ref, on=["seq", "frame"])
        assert np.allclose(d[f"{name}_J_c"] - d[f"{name}_J_f"], d.V, atol=1e-9)
        for sp in ("all", "test"):
            e = d if sp == "all" else d[d.split == "test"]
            Lc, V = float(e.J_cheap.sum()), e.V.to_numpy(float)
            r = {"cell": f"{args.dataset} {args.geometry} {name}", "split": sp, "G_all_full": V.sum() / Lc}
            for K in (1, 2):
                a = float(e[f"{name}_area_sl{K}"].mean())
                k = max(int(round(a * len(e))), 1)
                r[f"G_second_look_K{K}"] = float((e[f"{name}_J_c"] - e[f"{name}_J_sl{K}"]).sum()) / Lc
                r[f"mean_pixel_cost_K{K}"] = a
                r[f"G_frames_random_equal_pixels_K{K}"] = a * V.sum() / Lc
                r[f"G_frames_oracle_equal_pixels_K{K}"] = topk_expect(V, [V], [k])[0][0][0] / Lc
            r["G_second_look_K1_existence_only"] = float((e[f"{name}_J_c"] - e[f"{name}_J_sl1_existence_only"]).sum()) / Lc
            out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(run / f"secondlook_summary__{args.dataset}__{args.geometry}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).T.to_string(), flush=True)


if __name__ == "__main__":
    main()
