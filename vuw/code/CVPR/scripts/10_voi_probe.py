#!/usr/bin/env python
"""D4 (exploratory): a pre-escalation value-of-information score by probing the consumer with hypothetical objects.

Before escalating, insert one hypothetical static obstacle in the ego corridor at range r, on the cheap set (mono
geometry, as every allocator in DEEP sees), and ask whether the consumer's decision would change.  Weight each range
by how much more often the full mode detects an object at that range than the cheap mode (recall gain, estimated on
the training units only).  The score is the expected number of decision-changing objects the escalation could reveal.

    VOI_probe(i) = sum_r  drecall(r) * 1[ decision(cheap_i + obstacle at r) != decision(cheap_i) ] * width(r)

Scored as an allocator with DEEP's scorer on the test split, alone and inside a gradient-boosted gate.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from rap import planner as P
from rap import planner_b as B
from rap.cache import DetCache
from rap.decision import KittiAdapter
from rap.mono import box_iou
from rap.paths import CACHE, NUSCENES_TRAINVAL
from rap.risk import RiskConfig
from rap.scoring import topk_expect
from vtf import runmeta

EDGES = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80], float)
MIDS = (EDGES[:-1] + EDGES[1:]) / 2
DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"
RAW_RAP = runmeta.RAP_REPO / "results" / "raw"


def recall_gain(det_dir, cm, fm, seqs, adapter, cfg):
    """Full minus cheap recall of reference objects, per range band (training units only)."""
    hit = {m: np.zeros(len(MIDS)) for m in ("c", "f")}
    tot = np.zeros(len(MIDS))
    for s in seqs:
        geom = adapter.geometry(s)
        c, f = DetCache(det_dir / cm / f"{s}.npz"), DetCache(det_dir / fm / f"{s}.npz")
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            if not len(g):
                continue
            gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
            b = np.clip(np.digitize(g["long_near"], EDGES) - 1, 0, len(MIDS) - 1)
            np.add.at(tot, b, 1)
            for m, cache, idx, role in (("c", c, i, "cheap"), ("f", f, f.index[fr], "full")):
                d = cache.det(idx)
                k = d["conf"] >= cfg.thr(role)
                if k.any():
                    got = box_iou(gt, d["xyxy"][k].astype(float)).max(1) >= 0.5
                    np.add.at(hit[m], b[got], 1)
    return np.clip((hit["f"] - hit["c"]) / np.maximum(tot, 1), 0, None), tot


def probe_row(z, lo, hi, tt, v, pp, pb, cb, prev_plan, with_traj):
    base_a = P.discrete_action(P.required_decel(z, lo, hi, tt, v, pp)[0], pp)
    fl_b = np.zeros(len(MIDS))
    fl_t = np.zeros(len(MIDS))
    base_p = B.plan(z, lo, hi, tt, v, prev_plan, pb, cb) if with_traj else None
    for k, r in enumerate(MIDS):
        z2, lo2, hi2, tt2 = (np.r_[z, r], np.r_[lo, -0.9], np.r_[hi, 0.9], np.r_[tt, np.inf])
        fl_b[k] = P.discrete_action(P.required_decel(z2, lo2, hi2, tt2, v, pp)[0], pp) != base_a
        if with_traj:
            fl_t[k] = B.plan(z2, lo2, hi2, tt2, v, prev_plan, pb, cb) != base_p
    return base_a, base_p, fl_b, fl_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["KITTI", "nuScenes"], required=True)
    args = ap.parse_args()
    run = runmeta.new_run(f"voi_probe_{args.dataset}", vars(args))
    cfg = RiskConfig()
    pp = P.PlannerParams()
    pb, cb = B.PARAMS_B["static_obstacles"], B.COSTS_B["default"]
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[dv.track == args.dataset].copy()
    dv["frame"] = dv.frame.astype(int)
    if args.dataset == "KITTI":
        det_dir, cm, fm, adapter, with_traj = Path(CACHE) / "det", "cheap_320", "full_640", KittiAdapter, True
    else:
        from rap.nusc import NuScenesDB, make_adapter
        det_dir, cm, fm = Path(CACHE) / "nusc_det_tv", "ns_cheap_320", "ns_full_640"
        adapter, with_traj = make_adapter(NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")), False
    split = dv[dv.system == "brake"].drop_duplicates(["seq", "frame"])[["seq", "frame", "split", "unit"]]
    seqs = [p.stem for p in sorted((det_dir / cm).glob("*.npz"))]
    train_seqs = sorted(set(split[split.split == "train"].seq))
    dr, tot = recall_gain(det_dir, cm, fm, train_seqs, adapter, cfg)
    print("  recall gain by band:", np.round(dr, 3), "n:", tot.astype(int), flush=True)
    w = dr * np.diff(EDGES)

    rows = []
    for s in seqs:
        speeds = adapter.speeds(s)
        c = DetCache(det_dir / cm / f"{s}.npz")
        prev_plan = None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            d, geo = c.det(i), c.geo(i)
            k = d["conf"] >= cfg.thr("cheap")
            z, lo, hi, tt = (np.asarray(geo[x][k], float) for x in ("z", "lat_min", "lat_max", "ttc"))
            a, p, fb, ft = probe_row(z, lo, hi, tt, v, pp, pb, cb, prev_plan, with_traj)
            rec = {"seq": s, "frame": fr, "voi_brake": float(fb @ w), "reach_brake": float(fb @ np.diff(EDGES)),
                   "n_cheap": int(k.sum()), "act_c": a,
                   "conf_mean": float(d["conf"][k].mean()) if k.any() else 0.0,
                   "a_c": float(P.required_decel(z, lo, hi, tt, v, pp)[0])}
            if with_traj:
                rec.update({"voi_traj": float(ft @ w), "reach_traj": float(ft @ np.diff(EDGES))})
                prev_plan = p
            rows.append(rec)
    pr = pd.DataFrame(rows)
    pr.to_pickle(run / f"probe__{args.dataset}.pkl")

    # ---- score as allocators on each cell's test split
    gates = sorted(RAW_RAP.glob("*_budget_gate_scores_ego"))[-1]
    routers = sorted(RAW_RAP.glob("*_routers_r1_ego"))[-1]
    out = []
    for (geo, system, target), ref in dv.groupby(["geometry", "system", "target"]):
        d = ref.merge(pr, on=["seq", "frame"], how="left", validate="one_to_one")
        d = d.merge(pd.read_csv(Path(CACHE) / "nusc_v_ego_causal.csv", dtype={"seq": str})[["seq", "frame", "v_ego_causal"]]
                    if args.dataset == "nuScenes" else d[["seq", "frame"]].assign(v_ego_causal=np.nan),
                    on=["seq", "frame"], how="left")
        if args.dataset == "KITTI":
            d["v_ego_causal"] = d.seq.map(lambda s_: 0.0)
            spd = {s_: KittiAdapter.speeds(s_) for s_ in d.seq.unique()}
            d["v_ego_causal"] = [float(spd[a][b]) if b < len(spd[a]) else float(spd[a][-1])
                                 for a, b in zip(d.seq, d.frame)]
        fam = "traj" if system == "traj" else "brake"
        voi = f"voi_{fam}" if f"voi_{fam}" in d else "voi_brake"
        feats_base = ["v_ego_causal", "n_cheap", "conf_mean", "a_c", "act_c"]
        feats_probe = feats_base + [voi, voi.replace("voi", "reach")]
        te = d[d.split == "test"].reset_index(drop=True)
        nt = d[d.split != "test"]
        V, Lc = te.V.to_numpy(float), float(te.J_cheap.sum())
        kk = max(int(round(0.2 * len(te))), 1)
        prize = topk_expect(V, [V], [kk])[0][0][0]
        scores = {"random": None, "ego speed": te.v_ego_causal.to_numpy(float),
                  "probe VOI": te[voi].to_numpy(float),
                  "probe VOI x speed": te[voi].to_numpy(float) * (1 + te.v_ego_causal.to_numpy(float))}
        for name, fs in (("GBM base", feats_base), ("GBM base+probe", feats_probe)):
            m = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.08, min_samples_leaf=40,
                                              l2_regularization=1.0, max_bins=64, early_stopping=False,
                                              random_state=0).fit(nt[fs].to_numpy(float), nt.V.to_numpy(float))
            scores[name] = m.predict(te[fs].to_numpy(float))
        key = f"{args.dataset}__{geo}__{system}__{target}"
        gz = np.load(gates / f"gates__{key}.npz", allow_pickle=True)
        gk = pd.DataFrame({"seq": gz["key_seq"].astype(str), "frame": gz["key_frame"].astype(int),
                           "DEEP gate GBM": gz["gate_gbm"]})
        rz = np.load(routers / f"scores__{key}.npz", allow_pickle=True)
        rk = pd.DataFrame({"seq": rz["seq"].astype(str), "frame": rz["frame"].astype(int),
                           "DEEP router MLP V>0": rz["R1_mlp_clf"], "DEEP router GBM V>0": rz["R1_gbm_clf"]})
        mm = te[["seq", "frame"]].merge(gk, how="left").merge(rk, how="left")
        for n in ("DEEP gate GBM", "DEEP router MLP V>0", "DEEP router GBM V>0"):
            scores[n] = mm[n].to_numpy(float)
        row = {"cell": f"{args.dataset} {geo} {system}", "n_test": len(te), "oracle_share": prize / Lc}
        for n, sc in scores.items():
            g = kk / len(V) * V.sum() if sc is None else topk_expect(sc, [V], [kk])[0][0][0]
            row[n] = g / prize if prize > 0 else np.nan
        out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(run / f"voi_probe_ndg20__{args.dataset}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
