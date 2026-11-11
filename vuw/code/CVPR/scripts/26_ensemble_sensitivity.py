#!/usr/bin/env python
"""D62 (exploratory): ensemble decision sensitivity, borrowed from targeted observations in weather forecasting.

Numerical weather prediction places extra observations where an ensemble of forecasts is most sensitive.  Here: from
the cheap scene (mono geometry), sample M plausible scenes — each object's range scaled by a log-normal lift error
(sigma fitted on training units), each object kept with its calibrated probability of being real (confidence bins,
training units), and with the full mode's recall gain per range band a missed obstacle inserted in the corridor — and
run the braking controller on each.  Frame score = probability that the decision differs from the cheap decision
(decision instability).  Scored as an allocator (nDG@20, test split) against ego speed and the probe VOI (D4).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap.cache import DetCache
from rap.decision import KittiAdapter
from rap.mono import box_iou
from rap.paths import CACHE, NUSCENES_TRAINVAL
from rap.risk import RiskConfig
from rap.scoring import topk_expect
from vtf import runmeta

EDGES = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80], float)
CB = np.array([0.25, 0.35, 0.5, 0.7, 1.01])
M = 48
DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["KITTI", "nuScenes"], required=True)
    args = ap.parse_args()
    run = runmeta.new_run(f"ensemble_sens_{args.dataset}", vars(args))
    cfg = RiskConfig()
    pp = P.PlannerParams()
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str})
    dv = dv[dv.track == args.dataset].copy()
    dv["frame"] = dv.frame.astype(int)
    split = dict(zip(dv.seq, dv.split))
    if args.dataset == "KITTI":
        det_dir, cm, fm, adapter = Path(CACHE) / "det", "cheap_320", "full_640", KittiAdapter
    else:
        from rap.nusc import NuScenesDB, make_adapter
        det_dir, cm, fm = Path(CACHE) / "nusc_det_tv", "ns_cheap_320", "ns_full_640"
        adapter = make_adapter(NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval"))
    seqs = [p.stem for p in sorted((det_dir / cm).glob("*.npz"))]
    train = [s for s in seqs if split.get(s) == "train"]
    # ---- training-unit statistics: lift error, P(real | conf), recall gain per range band
    errs, hit, tot = [], np.zeros(len(CB) - 1), np.zeros(len(CB) - 1)
    rg_hit = {m: np.zeros(len(EDGES) - 1) for m in ("c", "f")}
    rg_tot = np.zeros(len(EDGES) - 1)
    for s in train:
        geom = adapter.geometry(s)
        c, f = DetCache(det_dir / cm / f"{s}.npz"), DetCache(det_dir / fm / f"{s}.npz")
        for i, fr in enumerate(c.frames):
            g = geom[geom["frame"] == int(fr)]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
            d, geo = c.det(i), c.geo(i)
            k = d["conf"] >= cfg.thr("cheap")
            if k.any():
                iou = box_iou(d["xyxy"][k].astype(float), gt) if len(gt) else np.zeros((k.sum(), 0))
                real = iou.max(1) >= 0.5 if len(gt) else np.zeros(k.sum(), bool)
                ci = np.clip(np.digitize(d["conf"][k], CB) - 1, 0, len(CB) - 2)
                np.add.at(tot, ci, 1)
                np.add.at(hit, ci, real)
                for o in np.flatnonzero(real):
                    zt = float(g["long_near"][int(np.argmax(iou[o]))])
                    if zt > 1:
                        errs.append(np.log(max(geo["z"][k][o], .5) / zt))
            if len(g):
                b = np.clip(np.digitize(g["long_near"], EDGES) - 1, 0, len(EDGES) - 2)
                np.add.at(rg_tot, b, 1)
                for m, cache, idx, role in (("c", c, i, "cheap"), ("f", f, f.index[int(fr)], "full")):
                    dd = cache.det(idx)
                    kk = dd["conf"] >= cfg.thr(role)
                    if kk.any():
                        np.add.at(rg_hit[m], b[box_iou(gt, dd["xyxy"][kk].astype(float)).max(1) >= 0.5], 1)
    sigma = float(np.std(np.clip(errs, -1, 1)))
    p_real = (hit + 1) / (tot + 2)
    miss = np.clip((rg_hit["f"] - rg_hit["c"]) / np.maximum(rg_tot, 1), 0, None)
    print(f"  lift sigma {sigma:.3f}; P(real|conf) {np.round(p_real, 2)}; recall gain {np.round(miss, 3)}", flush=True)
    mids = (EDGES[:-1] + EDGES[1:]) / 2
    rows = []
    rng = np.random.default_rng(0)
    for s in seqs:
        speeds = adapter.speeds(s)
        c = DetCache(det_dir / cm / f"{s}.npz")
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            d, geo = c.det(i), c.geo(i)
            k = d["conf"] >= cfg.thr("cheap")
            z, lo, hi, tt = (np.asarray(geo[x][k], float) for x in ("z", "lat_min", "lat_max", "ttc"))
            pr = p_real[np.clip(np.digitize(d["conf"][k], CB) - 1, 0, len(CB) - 2)] if k.any() else np.zeros(0)
            a0 = P.discrete_action(P.required_decel(z, lo, hi, tt, v, pp)[0], pp)
            diff_rng = diff_ex = diff_miss = diff_all = 0
            for _ in range(M):
                sc = np.exp(rng.normal(0, sigma, len(z)))
                keep = rng.random(len(z)) < pr
                ins = rng.random(len(mids)) < miss * 0.5            # at most one missed object per band, halved
                zi = mids[ins] + rng.uniform(-2, 2, ins.sum())
                yi = rng.uniform(-3, 3, ins.sum())
                t_r = (z * sc, lo * sc, hi * sc, tt)
                t_e = (z[keep], lo[keep], hi[keep], tt[keep])
                t_m = (np.r_[z, zi], np.r_[lo, yi - .9], np.r_[hi, yi + .9], np.r_[tt, np.full(ins.sum(), np.inf)])
                t_a = (np.r_[(z * sc)[keep], zi], np.r_[(lo * sc)[keep], yi - .9], np.r_[(hi * sc)[keep], yi + .9],
                       np.r_[tt[keep], np.full(ins.sum(), np.inf)])
                dec = lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp)   # noqa: E731
                diff_rng += dec(t_r) != a0
                diff_ex += dec(t_e) != a0
                diff_miss += dec(t_m) != a0
                diff_all += dec(t_a) != a0
            rows.append({"seq": s, "frame": fr, "ens_range": diff_rng / M, "ens_exist": diff_ex / M,
                         "ens_miss": diff_miss / M, "ens_all": diff_all / M, "v": v})
    e = pd.DataFrame(rows)
    e.to_pickle(run / "ensemble.pkl")
    pr_ = pd.read_pickle(sorted(glob.glob(str(runmeta.RAW / f"*_voi_probe_{args.dataset}" / f"probe__{args.dataset}.pkl")))[-1])
    out = []
    for (geo, system), ref in dv[dv.system == "brake"].groupby(["geometry", "system"]):
        q = ref.merge(e, on=["seq", "frame"]).merge(pr_[["seq", "frame", "voi_brake"]], on=["seq", "frame"])
        te = q[q.split == "test"]
        V = te.V.to_numpy(float)
        k = max(int(round(0.2 * len(te))), 1)
        prize = topk_expect(V, [V], [k])[0][0][0]
        row = {"cell": f"{args.dataset} {geo} brake"}
        for n in ("v", "voi_brake", "ens_range", "ens_exist", "ens_miss", "ens_all"):
            row[n] = topk_expect(te[n].to_numpy(float) + 1e-6 * te.v.to_numpy(float), [V], [k])[0][0][0] / prize
        out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(run / "ensemble_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
