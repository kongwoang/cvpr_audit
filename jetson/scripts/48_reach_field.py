#!/usr/bin/env python
"""D60: a reachability-tube field — does the field at time t already cover the objects that will matter later?

Fields computed on the cheap scene at frame t (KITTI mono, all units):
  static     insertion field, static probe (v_close 0), y in +-6 m
  closing    insertion field, probe closing at 10 m/s (the frozen v3 field)
  tube       closing at 15 m/s on y in +-9 m, then dilated laterally by v_lat * T = 2 m/s * 1.5 s (cut-ins and
             crossing pedestrians within the horizon): a superset in the spirit of backward reachability
Anticipation recall: among ground-truth objects that are decision-relevant at frame t+k (in the braking corridor,
their own demand >= the DECELERATE threshold) and visible at t (same track id), the share whose box centre at t lies
inside the field footprint at t. k = 5 frames (0.5 s) and 10 (1 s). Also footprint pixels.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import KittiAdapter
from rap.paths import CACHE
from vtf import dsf, runmeta, speculative as SP

pp = P.PlannerParams()
Y_WIDE = np.arange(-9.0, 9.01, 1.5)


def footprint(sens, rg, yg, cam, W, H):
    if not sens.any():
        return None
    P2 = cam.calib.P2
    fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
    rr, yy = np.meshgrid(rg, yg, indexing="ij")
    rr, yy = rr[sens], yy[sens]
    vb = cy + fy * cam.cam_h / rr
    uc, hw = cx - fx * yy / rr, fx * SP.OBJ_W / (2 * rr)
    return np.array([max((uc - hw).min(), 0), max((vb - fy * SP.OBJ_H / rr).min(), 0), min((uc + hw).max(), W),
                     min(vb.max(), H)])


def main():
    dv = pd.read_csv(runmeta.RAP_REPO / "results/final/benchmark_decision_values.csv.gz", dtype={"seq": str, "frame": str})
    dv = dv[(dv.track == "KITTI") & (dv.geometry == "mono") & (dv.system == "brake")]
    rows = []
    for s in sorted(set(dv.seq)):
        geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s)
        cam = Camera("KITTI", s)
        c = DetCache(Path(CACHE) / "det" / "cheap_320" / f"{s}.npz")
        fps = {}
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            sc = c.scalars(i)
            W, H = float(sc["img_w"]), float(sc["img_h"])
            d, gg = c.det(i), c.geo(i)
            k = d["conf"] >= 0.25
            t = tuple(np.asarray(gg[x][k], float) for x in ("z", "lat_min", "lat_max", "ttc"))
            f_static = footprint(dsf.insertion_grid(t, v, SP.R_GRID, SP.Y_GRID, SP.OBJ_W, pp, 0.0), SP.R_GRID, SP.Y_GRID, cam, W, H)
            f_close = footprint(dsf.insertion_grid(t, v, SP.R_GRID, SP.Y_GRID, SP.OBJ_W, pp, 10.0), SP.R_GRID, SP.Y_GRID, cam, W, H)
            sw = dsf.insertion_grid(t, v, SP.R_GRID, Y_WIDE, SP.OBJ_W, pp, 15.0)
            dil = int(round(2.0 * 1.5 / 1.5))                                  # cells of 1.5 m
            tube = sw.copy()
            for sh in range(1, dil + 1):
                tube[:, sh:] |= sw[:, :-sh]
                tube[:, :-sh] |= sw[:, sh:]
            f_tube = footprint(tube, SP.R_GRID, Y_WIDE, cam, W, H)
            fps[fr] = {"static": f_static, "closing": f_close, "tube": f_tube, "W": W, "H": H, "v": v}
        frames = sorted(fps)
        for fr in frames:
            g_t = geom[geom["frame"] == fr]
            rec = {"seq": s, "frame": fr}
            for name in ("static", "closing", "tube"):
                f = fps[fr][name]
                rec[f"px_{name}"] = 0.0 if f is None else (f[2] - f[0]) * (f[3] - f[1]) / (fps[fr]["W"] * fps[fr]["H"])
            for kk in (5, 10):
                if fr + kk not in fps:
                    continue
                g_f = geom[geom["frame"] == fr + kk]
                v_f = fps[fr + kk]["v"]
                _, dem = P.required_decel(g_f["long_near"], g_f["lat_min"], g_f["lat_max"], g_f["ttc"], v_f, pp)
                rel = g_f["track_id"][np.asarray(dem) >= pp.thr_decel] if len(g_f) else []
                cov = {n: [] for n in ("static", "closing", "tube")}
                for tid in rel:
                    o = g_t[g_t["track_id"] == tid]
                    if not len(o):
                        continue
                    cxy = np.array([[(o["x1"][0] + o["x2"][0]) / 2, (o["y1"][0] + o["y2"][0]) / 2] * 2])
                    for n in cov:
                        f = fps[fr][n]
                        cov[n].append(bool(f is not None and SP.centres_in(cxy, [f])[0]))
                for n in cov:
                    rec[f"cov_{n}_{kk}"] = np.mean(cov[n]) if cov[n] else np.nan
                    rec[f"ncov_{kk}"] = len(cov[n])
            rows.append(rec)
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_pickle(runmeta.FINAL / "d60_reach_field_kitti.pkl")
    out = []
    for n in ("static", "closing", "tube"):
        r = {"field": n, "pixels_mean": d[f"px_{n}"].mean()}
        for kk in (5, 10):
            w = d[f"ncov_{kk}"].fillna(0)
            r[f"anticipation_recall_{kk}f"] = float((d[f"cov_{n}_{kk}"].fillna(0) * w).sum() / max(w.sum(), 1))
        out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(runmeta.FINAL / "d60_reach_field_kitti.csv", index=False)
    print(f"relevant future objects: {int(d.ncov_5.fillna(0).sum())} (0.5 s), {int(d.ncov_10.fillna(0).sum())} (1 s)")
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
