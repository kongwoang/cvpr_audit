#!/usr/bin/env python
"""D72: a LiDAR point budget steered by the decision-sensitivity field (nuScenes, CAM_FRONT + LIDAR_TOP keyframes).

A steerable LiDAR (or a LiDAR-processing budget) returns only a fraction q of its points. The cheap camera draft
detects objects; each detection's range is re-measured from the returned points that project inside its (shrunk)
box: near range = 10th percentile of the points' ego-x (>= 3 points), lateral extent rescaled by the range ratio.
  uniform   q of all points, drawn uniformly
  field     the same NUMBER of points, drawn only from points projecting into the braking field's footprint or the
            support boxes (v_close 10; if the region holds fewer points, all of them)
  all       every point (upper bound for LiDAR ranging of the cheap boxes)
Reference: uniform escalation of the camera to 640 (DEEP's cached full mode). Braking, DEEP's cost, cheap action
history. Exploratory (not pre-registered).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import _apply_range_source
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL
from vtf import speculative as SP

B = Path(os.environ["B"])
QS = (0.01, 0.02, 0.05, 0.10)


def T(rec):
    R = Rotation.from_quat(np.roll(rec["rotation"], -1)).as_matrix()
    return R, np.array(rec["translation"], float)


def lidar_in_cam(db, stok):
    """LIDAR_TOP keyframe points -> (u, v, ego_x, ego_y) in the CAM_FRONT image, ego frame at the camera time."""
    t = db._t
    sd_l = db._lidar_by_sample[stok]
    pts = np.fromfile(db.root / sd_l["filename"], dtype=np.float32).reshape(-1, 5)[:, :3].astype(float)
    Rl, tl = T(t["calibrated_sensor"][sd_l["calibrated_sensor_token"]])
    Rpl, tpl = T(t["ego_pose"][sd_l["ego_pose_token"]])
    sd_c = db._cam_by_sample[stok]
    Rpc, tpc = T(t["ego_pose"][sd_c["ego_pose_token"]])
    Rc, tc = T(t["calibrated_sensor"][sd_c["calibrated_sensor_token"]])
    p_ego_l = pts @ Rl.T + tl
    p_glob = p_ego_l @ Rpl.T + tpl
    p_ego = (p_glob - tpc) @ Rpc                          # ego frame at the camera timestamp
    p_cam = (p_ego - tc) @ Rc
    K = np.array(t["calibrated_sensor"][sd_c["calibrated_sensor_token"]]["camera_intrinsic"], float)
    ok = p_cam[:, 2] > 1.0
    uvw = p_cam[ok] @ K.T
    u, v = uvw[:, 0] / uvw[:, 2], uvw[:, 1] / uvw[:, 2]
    inside = (u >= 0) & (u < 1600) & (v >= 0) & (v < 900)
    return np.c_[u[inside], v[inside], p_ego[ok][inside, 0], p_ego[ok][inside, 1], p_ego[ok][inside, 2]]


def rerange(bx, t, pts):
    z, lo, hi, tt = (a.copy() for a in t)
    for k, (x1, y1, x2, y2) in enumerate(bx):
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 0.8, (y2 - y1) * 0.8
        m = (pts[:, 0] >= cx - w / 2) & (pts[:, 0] <= cx + w / 2) & (pts[:, 1] >= cy - h / 2) & (pts[:, 1] <= cy + h / 2)
        if m.sum() >= 3:
            zn = float(np.percentile(pts[m, 2], 10))
            if zn > 0.5 and z[k] > 0.5:
                s = zn / z[k]
                z[k], lo[k], hi[k] = zn, lo[k] * s, hi[k] * s
    return z, lo, hi, tt


def discover(bx, t, pts, cell=0.6, min_pts=4):
    """Add objects from returned points above the ground (0.3-2.5 m) that fall in no camera box: 2-D grid clusters
    in ego x-y; each cluster becomes a static object (near range, lateral extent)."""
    if not len(pts):
        return t
    m = (pts[:, 4] > 0.3) & (pts[:, 4] < 2.5)
    for x1, y1, x2, y2 in bx:
        m &= ~((pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2))
    q = pts[m]
    if len(q) < min_pts:
        return t
    key = np.floor(q[:, 2:4] / cell).astype(int)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    new = []
    for c in np.flatnonzero(cnt >= min_pts):
        r = q[inv.ravel() == c]
        new.append((r[:, 2].min(), r[:, 3].min() - 0.2, r[:, 3].max() + 0.2))
    if not new:
        return t
    z, lo, hi = (np.array(a) for a in zip(*new))
    return (np.r_[t[0], z], np.r_[t[1], lo], np.r_[t[2], hi], np.r_[t[3], np.full(len(z), np.inf)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="new")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    scenes = sorted(sp[(sp["set"] == args.set) & sp.split.isin(args.split.split("+"))].seq)[: args.max_scenes or None]
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    lid = {k for k, v in db._t["sensor"].items() if v["channel"] == "LIDAR_TOP"}
    db._lidar_by_sample = {sd["sample_token"]: sd for sd in db._t["sample_data"].values() if sd["is_key_frame"] and
                           db._t["calibrated_sensor"][sd["calibrated_sensor_token"]]["sensor_token"] in lid}
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    rows = []
    for s in scenes:
        geom, speeds = ad.geometry(s), ad.speeds(s)
        cam = Camera("nuScenes", s)
        cc = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
        cf = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_full_640" / f"{s}.npz")
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

            def cache_t(c, idx):
                d = c.det(idx)
                k = d["conf"] >= 0.25
                geo = {kk: vv[k] for kk, vv in c.geo(idx).items()}
                bx = d["xyxy"][k].astype(float)
                return bx, tuple(np.asarray(a, float) for a in _apply_range_source(bx, geo, g, "mono", rng, 0.0))

            bxc, t_c = cache_t(cc, i)
            _, t_f = cache_t(cf, cf.index[fr])
            a_c = act(t_c)
            try:
                pts = lidar_in_cam(db, toks[fr])
            except FileNotFoundError:
                continue
            rec = {"seq": s, "frame": fr, "J_c": cost(a_c), "J_u640": cost(act(t_f)), "n_pts": len(pts)}
            rec["J_all"] = cost(act(rerange(bxc, t_c, pts)))
            # the field footprint + support boxes of the cheap draft
            region = []
            fc = SP.field_crop(t_c, v, cam, 1600, 900, pp, 10.0)
            if fc is not None:
                region.append(fc)
            region += [SP.box_crop(bxc[k], 1600, 900) for k in SP.support(t_c, v, pp)]
            inreg = SP.centres_in(np.c_[pts[:, 0], pts[:, 1], pts[:, 0], pts[:, 1]], region) if region else \
                np.zeros(len(pts), bool)
            rec["pts_region"] = int(inreg.sum())
            for q in QS:
                n = max(int(round(q * len(pts))), 1)
                uni = pts[rng.choice(len(pts), min(n, len(pts)), replace=False)]
                ridx = np.flatnonzero(inreg)
                fld = pts[ridx if len(ridx) <= n else rng.choice(ridx, n, replace=False)]
                rec[f"J_uni_{q}"] = cost(act(rerange(bxc, t_c, uni)))
                rec[f"J_fld_{q}"] = cost(act(rerange(bxc, t_c, fld)))
                rec[f"J_flddisc_{q}"] = cost(act(discover(bxc, rerange(bxc, t_c, fld), fld)))
                rec[f"J_unidisc_{q}"] = cost(act(discover(bxc, rerange(bxc, t_c, uni), uni)))
            rows.append(rec)
            prev_act = a_c
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_pickle(B / "results" / f"d72_lidar__{args.set}_{args.split}{args.tag}.pkl")
    L = d.J_c.sum()
    uq = d.seq.unique(); idx = [np.flatnonzero(d.seq.to_numpy() == u) for u in uq]
    print(f"{args.set} {args.split}: scenes {len(uq)}, frames {len(d)}, points in image {d.n_pts.mean():.0f}/frame, "
          f"in field+support region {d.pts_region.mean():.0f} ({(d.pts_region / d.n_pts).mean():.1%})")
    out = [{"pipeline": "uniform camera 640", "G": (d.J_c - d.J_u640).sum() / L},
           {"pipeline": "LiDAR, all points", "G": (d.J_c - d.J_all).sum() / L}]
    b = np.random.default_rng(0)
    for q in QS:
        dr = []
        for _ in range(1000):
            tt = np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))])
            dr.append(((d[f"J_uni_{q}"].to_numpy()[tt] - d[f"J_fld_{q}"].to_numpy()[tt]).sum()) / d.J_c.to_numpy()[tt].sum())
        out.append({"pipeline": f"LiDAR {q:.0%} uniform", "G": (d.J_c - d[f"J_uni_{q}"]).sum() / L})
        out.append({"pipeline": f"LiDAR {q:.0%} field-steered", "G": (d.J_c - d[f"J_fld_{q}"]).sum() / L,
                    "field_minus_uniform_ci": f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"})
        out.append({"pipeline": f"LiDAR {q:.0%} uniform + discovery", "G": (d.J_c - d[f"J_unidisc_{q}"]).sum() / L})
        out.append({"pipeline": f"LiDAR {q:.0%} field-steered + discovery", "G": (d.J_c - d[f"J_flddisc_{q}"]).sum() / L})
    res = pd.DataFrame(out)
    res.to_csv(B / "results" / f"d72_lidar__{args.set}_{args.split}{args.tag}.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
