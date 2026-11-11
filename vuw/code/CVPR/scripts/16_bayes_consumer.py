#!/usr/bin/env python
"""D15 (exploratory): a Blackwell-consistent consumer — the braking decision as a Bayes action under existence uncertainty.

Each kept detection gets a calibrated probability of being real, p(conf bin, apparent-height bin, mode), estimated on
the training units by matching to reference boxes.  Objects are independent; the true requirement is the largest
required deceleration among the real ones.  The consumer picks the action with the least expected loss under that
distribution (the same loss DEEP scores), instead of thresholding the maximum over all detections.  Question: does
escalation still harm, and what does the Bayes consumer do to the all-cheap loss itself?
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap.cache import DetCache
from rap.decision import KittiAdapter, _apply_range_source
from rap.mono import box_iou
from rap.paths import CACHE, NUSCENES_TRAINVAL
from rap.risk import RiskConfig
from vtf import runmeta

CB = np.array([0.25, 0.35, 0.5, 0.7, 1.01])
HB = np.array([0, 15, 25, 40, 70, 1e9])
DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"


def calibrate(caches, seqs, adapter, cfg):
    """P(real | conf bin, height bin) per mode, from the training units."""
    out = {}
    for tag, (dd, mode) in caches.items():
        hit = np.zeros((len(CB) - 1, len(HB) - 1))
        tot = np.zeros_like(hit)
        for s in seqs:
            geom = adapter.geometry(s)
            c = DetCache(dd / mode / f"{s}.npz")
            for i, fr in enumerate(c.frames):
                g = geom[geom["frame"] == int(fr)]
                d = c.det(i)
                k = d["conf"] >= cfg.thr("cheap")
                if not k.any():
                    continue
                bx = d["xyxy"][k].astype(float)
                gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
                real = box_iou(bx, gt).max(1) >= 0.5 if len(gt) else np.zeros(len(bx), bool)
                ci = np.clip(np.digitize(d["conf"][k], CB) - 1, 0, len(CB) - 2)
                hi = np.clip(np.digitize(bx[:, 3] - bx[:, 1], HB) - 1, 0, len(HB) - 2)
                np.add.at(tot, (ci, hi), 1)
                np.add.at(hit, (ci, hi), real)
        out[tag] = (hit + 1) / (tot + 2)
    return out


def bayes_action(a_obj, p, prev, pp, cp):
    """argmin_a E[J(a)] with a_true = max over real objects of a_obj (independent existence)."""
    o = np.argsort(-a_obj)
    a_s, p_s = a_obj[o], p[o]
    surv = np.cumprod(np.r_[1.0, 1 - p_s])
    probs = np.r_[p_s * surv[:-1], surv[-1]]
    vals = np.r_[a_s, 0.0]
    costs = [sum(w * P.decision_cost(a, x, prev, pp, cp)["J"] for w, x in zip(probs, vals) if w > 1e-12)
             for a in range(3)]
    return int(np.argmin(costs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["KITTI", "nuScenes"], required=True)
    ap.add_argument("--geometry", choices=["mono", "oracle"], required=True)
    args = ap.parse_args()
    run = runmeta.new_run(f"bayes_{args.dataset}_{args.geometry}", vars(args))
    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    if args.dataset == "KITTI":
        det_dir, cm, fm, adapter = Path(CACHE) / "det", "cheap_320", "full_640", KittiAdapter
    else:
        from rap.nusc import NuScenesDB, make_adapter
        det_dir, cm, fm = Path(CACHE) / "nusc_det_tv", "ns_cheap_320", "ns_full_640"
        adapter = make_adapter(NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval"))
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[(dv.track == args.dataset) & (dv.geometry == args.geometry) & (dv.system == "brake")].copy()
    dv["frame"] = dv.frame.astype(int)
    train = sorted(set(dv[dv.split == "train"].seq))
    cal = calibrate({"c": (det_dir, cm), "f": (det_dir, fm)}, train, adapter, cfg)
    print("  P(real) cheap:\n", np.round(cal["c"], 2), "\n  full:\n", np.round(cal["f"], 2), flush=True)
    seqs = [p.stem for p in sorted((det_dir / cm).glob("*.npz"))]
    rows = []
    for s in seqs:
        geom, speeds = adapter.geometry(s), adapter.speeds(s)
        c, f = DetCache(det_dir / cm / f"{s}.npz"), DetCache(det_dir / fm / f"{s}.npz")
        prev_t = prev_b = None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            j = f.index[fr]
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            rec = {"seq": s, "frame": fr}
            acts = {}
            for tag, cache, idx in (("c", c, i), ("f", f, j)):
                d, geo = cache.det(idx), cache.geo(idx)
                k = d["conf"] >= cfg.thr("cheap")
                bx = d["xyxy"][k].astype(float)
                z, lo, hi, tt = _apply_range_source(bx, {kk: vv[k] for kk, vv in geo.items()}, g, args.geometry, rng, 0.0)
                a_max, a_obj = P.required_decel(z, lo, hi, tt, v, pp)
                ci = np.clip(np.digitize(d["conf"][k], CB) - 1, 0, len(CB) - 2)
                hi_ = np.clip(np.digitize(bx[:, 3] - bx[:, 1], HB) - 1, 0, len(HB) - 2)
                p = cal[tag][ci, hi_] if len(bx) else np.zeros(0)
                act_t = P.discrete_action(a_max, pp)
                act_b = bayes_action(np.asarray(a_obj, float), p, prev_b, pp, cp) if len(bx) else \
                    bayes_action(np.zeros(0), np.zeros(0), prev_b, pp, cp)
                rec[f"J_thr_{tag}"] = P.decision_cost(act_t, a_gt, prev_t, pp, cp)["J"]
                rec[f"J_bayes_{tag}"] = P.decision_cost(act_b, a_gt, prev_b, pp, cp)["J"]
                acts[tag] = (act_t, act_b)
            prev_t, prev_b = acts["c"]
            rows.append(rec)
    d = pd.DataFrame(rows).merge(dv[["seq", "frame", "split", "J_cheap", "J_full"]], on=["seq", "frame"])
    assert np.allclose(d.J_thr_c, d.J_cheap, atol=1e-9) and np.allclose(d.J_thr_f, d.J_full, atol=1e-9)
    d.to_pickle(run / f"bayes__{args.dataset}__{args.geometry}.pkl")
    out = []
    for sp in ("all", "test"):
        e = d if sp == "all" else d[d.split == "test"]
        L0 = float(e.J_thr_c.sum())
        for name in ("thr", "bayes"):
            V = (e[f"J_{name}_c"] - e[f"J_{name}_f"]).to_numpy()
            aff = np.abs(V) > 1e-9
            out.append({"cell": f"{args.dataset} {args.geometry} brake", "split": sp, "consumer": name,
                        "all_cheap_loss_vs_threshold_consumer": float(e[f"J_{name}_c"].sum()) / L0,
                        "all_full_loss_vs_threshold_cheap": float(e[f"J_{name}_f"].sum()) / L0,
                        "affected": int(aff.sum()), "harmed_share": float((V < -1e-9).sum() / max(aff.sum(), 1)),
                        "harm_ratio": float(-V[V < 0].sum() / max(V[V > 0].sum(), 1e-9))})
    res = pd.DataFrame(out)
    res.to_csv(run / "bayes_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
