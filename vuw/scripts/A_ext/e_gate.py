#!/usr/bin/env python
"""Direction E: the training-free ensemble decision-sensitivity gate (D62/D62b) on the NEW nuScenes scenes.

Statistics (lift error, P(real | confidence), full-mode recall gain per range band) and the ridge mixture are fitted
on the new TRAIN scenes; everything is scored on the new TEST scenes (never used for fitting). Allocator quality =
nDG@q (expected value captured by escalating the top-q frames / the oracle's top-q value), for q = 10, 20, 30%, cells
braking mono and braking oracle geometry (DEEP's per-frame V from our tables). Paired scene bootstrap vs random.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap.cache import DetCache
from rap.mono import box_iou
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL

B = Path(os.environ["B"])
EDGES = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80], float)
CB = np.array([0.25, 0.35, 0.5, 0.7, 1.01])
M = 48
DET = Path(CACHE) / "nusc_det_tv"


def ndg(score, V, q, rng):
    k = max(int(round(q * len(V))), 1)
    top = np.lexsort((rng.random(len(V)), -score))[:k]
    prize = np.sort(V)[::-1][:k].clip(0).sum()
    return V[top].sum() / prize if prize > 0 else np.nan


def main():
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    sp = sp[sp["set"] == "new"]
    split = dict(zip(sp.seq, sp.split))
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    pp = P.PlannerParams()
    train = [s for s in sp.seq if split[s] == "train"]
    errs, hit, tot = [], np.zeros(len(CB) - 1), np.zeros(len(CB) - 1)
    rg_hit = {m: np.zeros(len(EDGES) - 1) for m in ("c", "f")}
    rg_tot = np.zeros(len(EDGES) - 1)
    for s in train:
        geom = ad.geometry(s)
        c, f = DetCache(DET / "ns_cheap_320" / f"{s}.npz"), DetCache(DET / "ns_full_640" / f"{s}.npz")
        for i, fr in enumerate(c.frames):
            g = geom[geom["frame"] == int(fr)]
            g = g[(g["y2"] - g["y1"]) >= 10]
            gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
            d, geo = c.det(i), c.geo(i)
            k = d["conf"] >= 0.25
            if k.any() and len(gt):
                iou = box_iou(d["xyxy"][k].astype(float), gt)
                real = iou.max(1) >= 0.5
                ci = np.clip(np.digitize(d["conf"][k], CB) - 1, 0, len(CB) - 2)
                np.add.at(tot, ci, 1)
                np.add.at(hit, ci, real)
                for o in np.flatnonzero(real):
                    zt = float(g["long_near"][int(np.argmax(iou[o]))])
                    if zt > 1:
                        errs.append(np.log(max(geo["z"][k][o], .5) / zt))
            elif k.any():
                ci = np.clip(np.digitize(d["conf"][k], CB) - 1, 0, len(CB) - 2)
                np.add.at(tot, ci, 1)
            if len(g):
                b = np.clip(np.digitize(g["long_near"], EDGES) - 1, 0, len(EDGES) - 2)
                np.add.at(rg_tot, b, 1)
                for m, cache, idx in (("c", c, i), ("f", f, f.index[int(fr)])):
                    dd = cache.det(idx)
                    kk = dd["conf"] >= 0.25
                    if kk.any():
                        np.add.at(rg_hit[m], b[box_iou(gt, dd["xyxy"][kk].astype(float)).max(1) >= 0.5], 1)
    sigma = float(np.std(np.clip(errs, -1, 1)))
    p_real = (hit + 1) / (tot + 2)
    miss = np.clip((rg_hit["f"] - rg_hit["c"]) / np.maximum(rg_tot, 1), 0, None)
    print(f"sigma {sigma:.3f}; P(real|conf) {np.round(p_real, 2)}; recall gain {np.round(miss, 3)}", flush=True)
    mids = (EDGES[:-1] + EDGES[1:]) / 2
    rng = np.random.default_rng(0)
    rows = []
    for s in sp.seq:
        if split[s] not in ("train", "test"):
            continue
        speeds = ad.speeds(s)
        c = DetCache(DET / "ns_cheap_320" / f"{s}.npz")
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            d, geo = c.det(i), c.geo(i)
            k = d["conf"] >= 0.25
            z, lo, hi, tt = (np.asarray(geo[x][k], float) for x in ("z", "lat_min", "lat_max", "ttc"))
            pr = p_real[np.clip(np.digitize(d["conf"][k], CB) - 1, 0, len(CB) - 2)] if k.any() else np.zeros(0)
            dec = lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp)       # noqa: E731
            a0 = dec((z, lo, hi, tt))
            cnt = np.zeros(4)
            for _ in range(M):
                sc = np.exp(rng.normal(0, sigma, len(z)))
                keep = rng.random(len(z)) < pr
                ins = rng.random(len(mids)) < miss * 0.5
                zi = mids[ins] + rng.uniform(-2, 2, ins.sum())
                yi = rng.uniform(-3, 3, ins.sum())
                inf = np.full(ins.sum(), np.inf)
                cnt += [dec((z * sc, lo * sc, hi * sc, tt)) != a0, dec((z[keep], lo[keep], hi[keep], tt[keep])) != a0,
                        dec((np.r_[z, zi], np.r_[lo, yi - .9], np.r_[hi, yi + .9], np.r_[tt, inf])) != a0,
                        dec((np.r_[(z * sc)[keep], zi], np.r_[(lo * sc)[keep], yi - .9], np.r_[(hi * sc)[keep], yi + .9],
                             np.r_[tt[keep], inf])) != a0]
            rows.append({"seq": s, "frame": fr, "split": split[s], "v": v, "ens_range": cnt[0] / M,
                         "ens_exist": cnt[1] / M, "ens_miss": cnt[2] / M, "ens_all": cnt[3] / M})
        print(f"  {s}", flush=True)
    e = pd.DataFrame(rows)
    e.to_pickle(B / "results" / "e_gate_scores.pkl")
    feats = ["ens_range", "ens_exist", "ens_miss", "ens_all", "v"]
    out = []
    for geo in ("mono", "oracle"):
        t = pd.read_pickle(B / "results" / "nusc_full" / f"flips__{geo}.pkl")[["seq", "frame", "brake_V"]]
        q = e.merge(t, on=["seq", "frame"])
        tr, te = q[q.split == "train"], q[q.split == "test"].reset_index(drop=True)
        X = tr[feats].to_numpy(float)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
        w = np.linalg.solve(Xs.T @ Xs + 1.0 * np.eye(Xs.shape[1]), Xs.T @ tr.brake_V.to_numpy(float))
        te["ridge"] = np.c_[np.ones(len(te)), (te[feats].to_numpy(float) - mu) / sd] @ w
        V = te.brake_V.to_numpy(float)
        uq = te.seq.unique()
        idx = [np.flatnonzero(te.seq.to_numpy() == u) for u in uq]
        for qq in (0.1, 0.2, 0.3):
            for sname in feats + ["ridge"]:
                sc = te[sname].to_numpy(float) + 1e-6 * te.v.to_numpy(float)
                r = np.random.default_rng(1)
                point = ndg(sc, V, qq, r)
                rand = np.mean([ndg(r.random(len(V)), V, qq, r) for _ in range(50)])
                dr = []
                for _ in range(500):
                    tt_ = np.concatenate([idx[i] for i in r.integers(0, len(uq), len(uq))])
                    dr.append(ndg(sc[tt_], V[tt_], qq, r) - ndg(r.random(len(tt_)), V[tt_], qq, r))
                out.append({"cell": f"nuScenes-new {geo} brake", "q": qq, "score": sname, "nDG": point, "random": rand,
                            "minus_random_lo": np.nanpercentile(dr, 2.5), "minus_random_hi": np.nanpercentile(dr, 97.5)})
        print(f"  ridge weights ({geo}): {dict(zip(['b'] + feats, np.round(w, 4)))}", flush=True)
    res = pd.DataFrame(out)
    res.to_csv(B / "results" / "e_gate.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
