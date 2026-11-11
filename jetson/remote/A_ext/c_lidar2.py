#!/usr/bin/env python
"""Direction C, round 2: decision-steered LiDAR budget WITH learned discovery.

Clusters: returned points 0.3-2.5 m above ground, bird's-eye occupancy grid (0.3 m), 8-connected components with >= 3
points. A gradient-boosted classifier (cluster size, extents, height stats, range, density, image position) says
whether a cluster is a road user; trained on the NEW TRAIN scenes with labels from DEEP's ground-truth footprints
(centroid inside [long_near, 2*long_cen - long_near] x [lat_min, lat_max] + 0.5 m), on point clouds subsampled at
rates 1-100% so it sees sparse clusters. Discovery = clusters with p >= 0.5 outside every camera box, added as static
objects. Pipelines at budgets 1/2/5/10%: uniform / field-steered, each with re-ranging only and with discovery.
Scored on the old 85 scenes and the new test scenes (both), braking, DEEP's cost, scene bootstrap.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier

from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import _apply_range_source
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL
from vtf import speculative as SP

B = Path(os.environ["B"])
sys.path.insert(0, str(B))
import d72_lidar as M                                                         # noqa: E402

QS = (0.01, 0.02, 0.05, 0.10)
CELL = 0.3


def clusters(pts):
    """pts: (n, 5) u, v, x, y, z. Returns list of index arrays and a feature matrix."""
    m = np.flatnonzero((pts[:, 4] > 0.3) & (pts[:, 4] < 2.5))
    if len(m) < 3:
        return [], np.zeros((0, 12))
    q = pts[m]
    ij = np.floor(q[:, 2:4] / CELL).astype(int)
    ij -= ij.min(0)
    grid = np.zeros(ij.max(0) + 1, bool)
    grid[ij[:, 0], ij[:, 1]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
    pl = lab[ij[:, 0], ij[:, 1]]
    groups, feats = [], []
    for c in range(1, n + 1):
        g = np.flatnonzero(pl == c)
        if len(g) < 3:
            continue
        r = q[g]
        ext = r[:, 2:5].max(0) - r[:, 2:5].min(0)
        feats.append([len(g), *ext, r[:, 4].max(), r[:, 4].mean(), r[:, 2].min(), np.hypot(r[:, 2].mean(), r[:, 3].mean()),
                      len(g) / max(ext[0] * ext[1], 0.09), r[:, 0].mean() / 1600, r[:, 1].mean() / 900, r[:, 3].mean()])
        groups.append(m[g])
    return groups, np.asarray(feats, float).reshape(-1, 12)


def labels(pts, groups, g):
    if not len(g):
        return np.zeros(len(groups), int)
    x0, x1 = np.asarray(g["long_near"], float), 2 * np.asarray(g["long_cen"], float) - np.asarray(g["long_near"], float)
    y0, y1 = np.asarray(g["lat_min"], float) - 0.5, np.asarray(g["lat_max"], float) + 0.5
    out = []
    for gi in groups:
        cx, cy = pts[gi, 2].mean(), pts[gi, 3].mean()
        out.append(int(((cx >= x0 - 0.5) & (cx <= x1 + 0.5) & (cy >= y0) & (cy <= y1)).any()))
    return np.array(out, int)


def discover(bx, t, pts, clf):
    groups, F = clusters(pts)
    if not len(groups):
        return t
    p = clf.predict_proba(F)[:, 1]
    new = []
    for gi, pi in zip(groups, p):
        if pi < 0.5:
            continue
        r = pts[gi]
        inside = any(((r[:, 0] >= x1) & (r[:, 0] <= x2) & (r[:, 1] >= y1) & (r[:, 1] <= y2)).mean() > 0.5
                     for x1, y1, x2, y2 in bx)
        if not inside:
            new.append((r[:, 2].min(), r[:, 3].min(), r[:, 3].max()))
    if not new:
        return t
    z, lo, hi = (np.array(a) for a in zip(*new))
    return (np.r_[t[0], z], np.r_[t[1], lo], np.r_[t[2], hi], np.r_[t[3], np.full(len(z), np.inf)])


def setup_db():
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    lid = {k for k, v in db._t["sensor"].items() if v["channel"] == "LIDAR_TOP"}
    db._lidar_by_sample = {sd["sample_token"]: sd for sd in db._t["sample_data"].values() if sd["is_key_frame"] and
                           db._t["calibrated_sensor"][sd["calibrated_sensor_token"]]["sensor_token"] in lid}
    return db


def train_classifier(db, ad, by, scenes, rng):
    X, Y = [], []
    for s in scenes:
        geom = ad.geometry(s)
        toks = db.samples(by[s])
        for fr in range(0, len(toks), 2):
            pts = M.lidar_in_cam(db, toks[fr])
            g = geom[geom["frame"] == fr]
            for rate in (1.0, 0.1, 0.05, 0.02):
                sub = pts if rate == 1.0 else pts[rng.random(len(pts)) < rate]
                groups, F = clusters(sub)
                if len(groups):
                    X.append(F)
                    Y.append(labels(sub, groups, g))
    X, Y = np.concatenate(X), np.concatenate(Y)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X, Y)
    print(f"classifier: {len(Y)} clusters, {Y.mean():.3f} positive; train acc {clf.score(X, Y):.3f}", flush=True)
    return clf


def evaluate(db, ad, by, scenes, clf, rng, tag):
    pp, cp = P.PlannerParams(), P.CostParams()
    rows = []
    for s in scenes:
        geom, speeds = ad.geometry(s), ad.speeds(s)
        cam = Camera("nuScenes", s)
        cc = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
        toks = db.samples(by[s])
        prev_act = None
        for i, fr in enumerate(cc.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            act = lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp)             # noqa: E731
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]                  # noqa: E731
            d = cc.det(i)
            k = d["conf"] >= 0.25
            geo = {kk: vv[k] for kk, vv in cc.geo(i).items()}
            bxc = d["xyxy"][k].astype(float)
            t_c = tuple(np.asarray(a, float) for a in _apply_range_source(bxc, geo, g, "mono", np.random.default_rng(0), 0.0))
            a_c = act(t_c)
            pts = M.lidar_in_cam(db, toks[fr])
            rec = {"seq": s, "frame": fr, "J_c": cost(a_c), "J_all_disc": cost(act(discover(bxc, M.rerange(bxc, t_c, pts), pts, clf)))}
            region = []
            fc = SP.field_crop(t_c, v, cam, 1600, 900, pp, 10.0)
            if fc is not None:
                region.append(fc)
            region += [SP.box_crop(bxc[kk], 1600, 900) for kk in SP.support(t_c, v, pp)]
            inreg = SP.centres_in(np.c_[pts[:, 0], pts[:, 1], pts[:, 0], pts[:, 1]], region) if region else np.zeros(len(pts), bool)
            for q in QS:
                n = max(int(round(q * len(pts))), 1)
                uni = pts[rng.choice(len(pts), min(n, len(pts)), replace=False)]
                ridx = np.flatnonzero(inreg)
                fld = pts[ridx if len(ridx) <= n else rng.choice(ridx, n, replace=False)]
                for nm, sub in (("uni", uni), ("fld", fld)):
                    tr = M.rerange(bxc, t_c, sub)
                    rec[f"J_{nm}_{q}"] = cost(act(tr))
                    rec[f"J_{nm}disc_{q}"] = cost(act(discover(bxc, tr, sub, clf)))
            rows.append(rec)
            prev_act = a_c
        print(f"  {tag} {s}", flush=True)
    d = pd.DataFrame(rows)
    d.to_pickle(B / "results" / f"c_lidar2__{tag}.pkl")
    L = d.J_c.to_numpy()
    uq = d.seq.unique(); idx = [np.flatnonzero(d.seq.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    boots = [np.concatenate([idx[j] for j in b.integers(0, len(uq), len(uq))]) for _ in range(1000)]
    out = [{"set": tag, "pipeline": "all points + discovery", "G": (d.J_c - d.J_all_disc).sum() / L.sum()}]
    for q in QS:
        for nm in ("uni", "fld", "unidisc", "flddisc"):
            V = (d.J_c - d[f"J_{nm}_{q}"]).to_numpy()
            r = {"set": tag, "pipeline": f"{q:.0%} {nm}", "G": V.sum() / L.sum()}
            if nm in ("fld", "flddisc"):
                base = (d.J_c - d[f"J_uni{'disc' if 'disc' in nm else ''}_{q}"]).to_numpy()
                dr = [(V[t].sum() - base[t].sum()) / L[t].sum() for t in boots]
                r["minus_uniform_ci"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
                aff = np.abs(V) > 1e-9
                r["harmed"] = float((V < -1e-9).sum() / max(aff.sum(), 1))
            out.append(r)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_train", type=int, default=120)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    db = setup_db()
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    train = sorted(sp[(sp["set"] == "new") & (sp.split == "train")].seq)
    clf = train_classifier(db, ad, by, list(rng.permutation(train))[: args.max_train], rng)
    res = pd.concat([evaluate(db, ad, by, sorted(sp[sp["set"] == "old"].seq), clf, rng, "old85"),
                     evaluate(db, ad, by, sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq), clf, rng, "newtest")])
    res.to_csv(B / "results" / "c_lidar2.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
