#!/usr/bin/env python
"""Uniform whole-frame baselines up to NATIVE resolution (missing so far: u960 downsamples nuScenes / AV2 / KITTI;
on KITTI the native-width pass dominates every speculative variant, directions_log 2026-10-02).

  --dataset nusc   nuScenes new TEST keyframes (CAM_FRONT 1600x900): long side 1280 / 1600 (native)
  --dataset av2    AV2 sensor VAL (ring_front_center 1550x2048): long side 1280 / 1536 / 2048 (native)
YOLOv8s (COCO), letterbox, conf >= 0.25, mono lift with the arm's own history, braking, DEEP's cost, cheap action
history — exactly the conventions of the confirmatory runs (nuScenes: router_baseline.py stage_labels / the
confirmatory pickle; AV2: av2_eval_v4.py). The cheap cost is asserted equal to the confirmatory run's.
Writes results/native/<dataset>.pkl and a summary with paired bootstraps of our configuration minus each arm.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rap import planner as P
from rap.detect import Mode, TwoFidelityDetector
from rap.mono import predicted_geometry

B = Path(os.environ["B"])
OUT = B / "results" / "native"
THR = 0.25


def modes(dataset):
    if dataset == "nusc":            # landscape 1600x900: net_h = 9/16 of the long side, rounded up to 32
        return {s: Mode(f"ns_u{s}", int(np.ceil(s * 9 / 16 / 32)) * 32, s) for s in (1280, 1600)}
    return {s: Mode(f"av2_u{s}", s, int(np.ceil(s * 1550 / 2048 / 32)) * 32) for s in (1280, 1536, 2048)}


def run_nusc(det, args):
    from rap.cache import Camera, DetCache
    from rap.decision import _apply_range_source
    from rap.nusc import NuScenesDB, make_adapter
    from rap.paths import CACHE, NUSCENES_TRAINVAL
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    pp, cp = P.PlannerParams(), P.CostParams()
    M = modes("nusc")
    rows = []
    for s in sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq):
        geom, speeds = ad.geometry(s), ad.speeds(s)
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        c = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
        toks = db.samples(by[s])
        rng = np.random.default_rng(0)
        prev_act, prev = None, {k: None for k in M}
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]              # noqa: E731
            decide = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)    # noqa: E731
            dc = c.det(i)
            kc = dc["conf"] >= THR
            geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
            cons_c = tuple(np.asarray(a, float) for a in _apply_range_source(dc["xyxy"][kc].astype(float), geo_c, g, "mono", rng, 0.0))
            a_c = decide(cons_c)
            img = cv2.imread(str(db.image_path(toks[fr])))
            rec = {"seq": s, "frame": fr, "J_c": cost(a_c)}
            for sc, m in M.items():
                d, _ = det.detect(img, m)
                k = d["conf"] >= THR
                dd = {"xyxy": d["xyxy"][k].astype(float), "coarse": d["coarse"][k]}
                gg = predicted_geometry(dd, prev[sc], cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                prev[sc] = dd
                rec[f"J_u{sc}"] = cost(decide(tuple(np.asarray(gg[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))))
                rec[f"px_u{sc}"] = m.net_h * m.net_w / (1600 * 900)
            rows.append(rec)
            prev_act = a_c
        print(f"  {s}", flush=True)
        if args.max_units and len({r["seq"] for r in rows}) >= args.max_units:
            break
    d = pd.DataFrame(rows)
    ref = pd.read_pickle(B / "results" / "nusc_full" / "eval__new_test_v3_confirm.pkl")
    m = d.merge(ref[["seq", "frame", "J_c", "J_u960", "J_spec", "J_spec_ref_resnet18", "px"]], on=["seq", "frame"], suffixes=("", "_ref"))
    assert len(m) == len(d) and np.allclose(m.J_c, m.J_c_ref), "cheap cost differs from the confirmatory run"
    ours = {"v3 + ref": "J_spec_ref_resnet18", "v3": "J_spec"}
    px_ours = 320 * 192 / (1600 * 900) + float((m.px * 0.36).mean())
    return m.rename(columns={"seq": "unit"}), ours, px_ours, {"u960": 576 * 960 / (1600 * 900)}


def run_av2(det, args):
    from vtf import av2 as AV
    root = B / "datasets" / "av2" / "sensor" / "val"
    logs = sorted(p.name for p in root.iterdir() if (p / "annotations.feather").exists())[: args.max_units or None]
    pp, cp = P.PlannerParams(), P.CostParams()
    M = modes("av2")
    M_CHEAP = Mode("av2_c320", 320, 256)
    rows = []
    for li, log in enumerate(logs):
        L = AV.AV2Log(root, log)
        geom, speeds = L.geometry(), L.speeds()
        R, t = L.transform("ego")
        prev, prev_c, prev_act = {k: None for k in M}, None, None

        def lift(dd, pv):
            g = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, L.calib, L.cam_h, L.dt, cam_to_ego=(R, t))
            return tuple(np.asarray(g[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

        def keep(d):
            k = d["conf"] >= THR
            return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}
        for fr in range(len(L.frame_ts)):
            img = cv2.imread(str(L.image_path(fr)))
            if img is None:
                continue
            v = float(speeds[fr])
            g = geom[geom["frame"] == fr] if len(geom) else geom
            g = g[(g["y2"] - g["y1"]) >= 10] if len(g) else g
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
            act = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)            # noqa: E731
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]                  # noqa: E731
            ch = keep(det.detect(img, M_CHEAP)[0])
            a_c = act(lift(ch, prev_c))
            rec = {"log": log, "frame": fr, "J_c": cost(a_c)}
            for sc, m in M.items():
                dd = keep(det.detect(img, m)[0])
                rec[f"J_u{sc}"] = cost(act(lift(dd, prev[sc])))
                rec[f"px_u{sc}"] = m.net_h * m.net_w / (1550 * 2048)
                prev[sc] = dd
            rows.append(rec)
            prev_c, prev_act = ch, a_c
        print(f"  [{li + 1}/{len(logs)}] {log}", flush=True)
    d = pd.DataFrame(rows)
    ref = pd.read_pickle(B / "results" / "av2" / "eval__val_v4_final.pkl")
    m = d.merge(ref[["log", "frame", "J_c", "J_u960", "J_spec_ref_v4lite", "J_spec_ref_v3", "netpx_v4lite"]], on=["log", "frame"], suffixes=("", "_ref"))
    assert len(m) == len(d) and np.allclose(m.J_c, m.J_c_ref), "cheap cost differs from av2_eval_v4"
    px_ours = 320 * 256 / (1550 * 2048) + float(m.netpx_v4lite.mean())
    return m.rename(columns={"log": "unit"}), {"v4-lite + ref": "J_spec_ref_v4lite", "v3 + ref": "J_spec_ref_v3"}, px_ours, \
        {"u960": 960 * 768 / (1550 * 2048)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_units", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device=args.device, backend="torch")
    m, ours, px_ours, px_ref = (run_nusc if args.dataset == "nusc" else run_av2)(det, args)
    OUT.mkdir(parents=True, exist_ok=True)
    m.to_pickle(OUT / f"{args.dataset}{args.tag}.pkl")
    L = m.J_c.to_numpy()
    uq = m.unit.unique(); idx = [np.flatnonzero(m.unit.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    arms = {k: (m.J_c - m[c]).to_numpy() for k, c in ours.items()}
    unif = {c[2:]: (m.J_c - m[c]).to_numpy() for c in m.columns if c.startswith("J_u")}
    o0 = list(ours)[0]
    res = []
    for k, v_ in {**arms, **unif}.items():
        px = px_ours if k in ours else (px_ref.get(k) if k in px_ref else float(m[f"px_{k}"].mean()))
        r = {"method": k, "G": v_.sum() / L.sum(), "network px": px}
        if k != o0:
            dr = [(arms[o0][t_].sum() - v_[t_].sum()) / L[t_].sum() for t_ in boots]
            r[f"{o0} minus"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        res.append(r)
    res = pd.DataFrame(res)
    res.to_csv(OUT / f"summary_{args.dataset}{args.tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{len(uq)} units, {len(m)} frames")
    print(res.to_string(index=False))
    print("NATIVE UNIFORM DONE")


if __name__ == "__main__":
    main()
