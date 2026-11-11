#!/usr/bin/env python
"""K8 plot data (CSV only) from existing results (CPU): (i) G vs network px per method (from K1 tables), (ii) per-frame
network px of ours (histogram), (iii) field area (share of the frame, draft-round field footprint of ours) vs ego speed,
(iv) operating points of ours on nuScenes new test. Output: results/handoff_K8/."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
R = B / "results"
OUT = R / "handoff_K8"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    from rap import planner as P
    from rap.mono import predicted_geometry
    from vtf import speculative as SP
    pp = P.PlannerParams()
    # (i)
    rows = []
    for ds, f in (("nuScenes new test", "table_main_nusc.csv"), ("AV2 val", "table_main_av2.csv")):
        t = pd.read_csv(R / "handoff_K1" / f)
        for r in t.itertuples():
            rows.append({"dataset": ds, "method": r.method, "G": r.G, "network_px_mean": r.network_px_mean, "network_px_p95": r.network_px_p95})
    pd.DataFrame(rows).to_csv(OUT / "plot_G_vs_px.csv", index=False)
    # (ii)
    nat = pd.read_pickle(R / "native" / "nusc.pkl")
    px_n = 320 * 192 / 1.44e6 + nat.px.to_numpy() * 0.36
    fin = pd.read_pickle(R / "av2" / "eval__val_v4_final.pkl")
    px_a = 320 * 256 / (1550 * 2048) + fin.netpx_v4lite.to_numpy()
    bins = np.r_[np.arange(0, 0.6, 0.01), 10]
    rows = []
    for ds, px in (("nuScenes new test (v3)", px_n), ("AV2 val (v4-lite)", px_a)):
        h, e = np.histogram(px, bins=bins)
        rows += [{"dataset": ds, "bin_lo": e[i], "bin_hi": e[i + 1], "frames": int(h[i]), "share": h[i] / len(px)} for i in range(len(h))]
    pd.DataFrame(rows).to_csv(OUT / "plot_px_histogram.csv", index=False)
    # (iii) field area vs ego speed: footprint of the draft-round field of ours (v_close of the configuration)
    rows = []
    from rap.cache import Camera, DetCache
    from rap.decision import _apply_range_source
    from rap.nusc import NuScenesDB, make_adapter
    from rap.paths import CACHE, NUSCENES_TRAINVAL
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval"); ad = make_adapter(db)
    sp = pd.read_csv(R / "nusc_full" / "split.csv")
    rng = np.random.default_rng(0)
    for s in sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq):
        cam = Camera("nuScenes", s); c = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
        geom, speeds = ad.geometry(s), ad.speeds(s)
        for i, fr in enumerate(c.frames):
            fr = int(fr); v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            dc = c.det(i); kc = dc["conf"] >= 0.25
            geo = {kk: vv[kc] for kk, vv in c.geo(i).items()}
            g = geom[geom["frame"] == fr]; g = g[(g["y2"] - g["y1"]) >= 10]
            cons = tuple(np.asarray(a, float) for a in _apply_range_source(dc["xyxy"][kc].astype(float), geo, g, "mono", rng, 0.0))
            fc = SP.field_crop(cons, v, cam, 1600, 900, pp, 10.0)
            rows.append({"dataset": "nuScenes new test", "v": v, "field_area": 0.0 if fc is None else (fc[2] - fc[0]) * (fc[3] - fc[1]) / 1.44e6})
    from vtf import av2 as AV
    bx = pickle.load(open(R / "av2_sota" / "boxes__test.pkl", "rb"))
    fr_t = pd.read_pickle(R / "av2_sota" / "frames__test.pkl")
    root = B / "datasets" / "av2" / "sensor" / "val"
    for log, grp in fr_t.groupby("log", sort=False):
        L = AV.AV2Log(root, log); R_, t_ = L.transform("ego"); prev = None
        for r in grp.sort_values("frame").itertuples():
            b = bx[(log, int(r.frame))]
            k = b["weak_conf"] >= 0.25
            dd = {"xyxy": b["weak_xyxy"][k], "coarse": b["weak_cls"][k]}
            gk = predicted_geometry(dd, prev, L.calib, L.cam_h, L.dt, cam_to_ego=(R_, t_))
            prev = dd
            cons = tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))
            fc = SP.field_crop(cons, float(r.v_ego), L, L.W, L.H, pp, 0.0)
            rows.append({"dataset": "AV2 val", "v": float(r.v_ego), "field_area": 0.0 if fc is None else (fc[2] - fc[0]) * (fc[3] - fc[1]) / (L.W * L.H)})
    fa = pd.DataFrame(rows)
    fa["speed_bin"] = pd.cut(fa.v, [0, 0.5, 2.5, 5, 7.5, 10, 12.5, 15, 20, 40], include_lowest=True)
    t = fa.groupby(["dataset", "speed_bin"], observed=True).field_area.agg(["count", "mean", lambda x: (x > 0).mean(), lambda x: x.quantile(0.9)])
    t.columns = ["frames", "field_area_mean", "share_frames_with_field", "field_area_p90"]
    t.reset_index().to_csv(OUT / "plot_field_area_vs_speed.csv", index=False)
    # (iv) operating points, nuScenes new test
    rows = []
    for f in sorted((R / "nusc_full").glob("eval__new_test_curve_*.pkl")) + [R / "nusc_full" / "eval__new_test_v3_confirm.pkl"]:
        d = pd.read_pickle(f)
        name = f.stem.replace("eval__new_test_", "")
        if name == "v3_confirm":
            vc, rr, sc = 10, 3, 0.6
        else:
            p = name.replace("curve_", "").split("_")
            vc, rr, sc = float(p[0][2:]), int(p[1][1:]), float(p[2][1:])
        L = d.J_c.sum()
        row = {"source": f"nusc_full/{f.name}", "v_close": vc, "max_rounds": rr, "crop_scale": sc,
               "crop_area_mean": round(d.px.mean(), 4), "network_px_mean": round(320 * 192 / 1.44e6 + (d.px * sc ** 2).mean(), 4),
               "G": round((d.J_c - d.J_spec).sum() / L, 3), "G_refiner_v1": round((d.J_c - d.J_spec_ref_resnet18).sum() / L, 3)}
        if "J_spec_ref_resnet18_v2" in d:
            row["G_refiner_v2"] = round((d.J_c - d.J_spec_ref_resnet18_v2).sum() / L, 3)
        rows.append(row)
    pd.DataFrame(rows).sort_values(["v_close", "max_rounds", "crop_scale"]).to_csv(OUT / "plot_operating_points_nusc.csv", index=False)
    print("K8 PLOT DATA DONE")


if __name__ == "__main__":
    main()
