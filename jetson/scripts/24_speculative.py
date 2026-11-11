#!/usr/bin/env python
"""D47 (exploratory): speculative perception — draft with the cheap mode, verify only what the decision depends on.

Borrowed from speculative decoding: a small model drafts, the large model verifies.  Here the cheap detector drafts the
scene and the decision; the full detector is run only on
  (a) the discovery crop: the image region of ground cells where an unseen obstacle would change the draft decision
      (decision-sensitivity field, insertion probes on the draft's mono geometry), and
  (b) the verification crops: a box around each current pivot (object whose removal alone changes the decision).
Full detections inside the crops replace the draft's there (proxy: cached full-frame 640 detections by box centre).
If the merged decision differs from the draft's, the loop runs once more on the merged scene (new pivots, new field).

Claim under test ("decision-lossless"): any decision change caused by a single element of the cheap→full difference is
caught — an added element lies in an insertion-sensitive cell, a removed/moved one is a pivot — so the speculative
decision equals the full decision on almost every frame, at a small fraction of the full mode's pixels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap import planner_b as B
from rap.cache import Camera, DetCache
from rap.decision import KittiAdapter
from rap.paths import CACHE, NUSCENES_TRAINVAL
from rap.risk import RiskConfig
from rap.scoring import topk_expect
from vtf import flips, runmeta
from vtf import headroom as HR

Y_GRID = np.array([-6.0, -4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0])
OBJ_W, OBJ_H, CTX = 1.8, 1.6, 2.0
DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"


def cell_boxes(cam, r_grid, img_w, img_h):
    P2 = cam.calib.P2
    fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
    rr, yy = np.meshgrid(r_grid, Y_GRID, indexing="ij")
    vb = cy + fy * cam.cam_h / rr
    vt = vb - fy * OBJ_H / rr
    uc = cx - fx * yy / rr
    hw = fx * OBJ_W / (2 * rr)
    return np.stack([np.clip(uc - hw, 0, img_w), np.clip(vt, 0, img_h), np.clip(uc + hw, 0, img_w),
                     np.clip(vb, 0, img_h)], -1)


def in_any(xyxy, crops):
    if not len(xyxy) or not crops:
        return np.zeros(len(xyxy), bool)
    u, v = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
    m = np.zeros(len(xyxy), bool)
    for c in crops:
        m |= (u >= c[0]) & (u <= c[2]) & (v >= c[1]) & (v <= c[3])
    return m


def union_area(crops, img_w, img_h, cell=8):
    if not crops:
        return 0.0
    W, H = int(np.ceil(img_w / cell)), int(np.ceil(img_h / cell))
    mask = np.zeros((H, W), bool)
    for c in crops:
        mask[int(c[1] // cell):int(np.ceil(c[3] / cell)), int(c[0] // cell):int(np.ceil(c[2] / cell))] = True
    return float(mask.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["KITTI", "nuScenes"], required=True)
    ap.add_argument("--geometry", choices=["mono", "oracle"], default="mono")
    ap.add_argument("--rmin", type=float, default=10.0, help="nearest probed range (the cheap mode sees near objects)")
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()
    run = runmeta.new_run(f"speculative_{args.dataset}_{args.geometry}_r{int(args.rmin)}", vars(args))
    r_grid = np.array([r for r in (5.0, 7.5, 12.5, 17.5, 22.5, 27.5, 35.0, 45.0, 55.0, 70.0) if r >= args.rmin])
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
        cam = Camera(args.dataset, s)
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
            boxes = cell_boxes(cam, r_grid, img_w, img_h)
            cheap = flips.Branch(c.det(i), c.geo(i), cfg.thr("cheap"), g, args.geometry, rng0)
            full = flips.Branch(f.det(j), f.geo(j), cfg.thr("full"), g, args.geometry, rng0)
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            truth = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc"))
            rec = {"seq": s, "frame": fr}
            consumers = {"brake": (lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp),
                                   lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"])}
            if with_traj:
                consumers["traj"] = (lambda t, pu=prev_plan: B.plan(*t, v, pu, pb, cb),
                                     lambda a, pu=prev_plan: B.executed_cost(a, *truth, v, pu, pb, cb)["J"])
            for name, (decide, cost) in consumers.items():
                a_c, a_f = decide(cheap.cons), decide(full.cons)
                crops = []
                # the draft: detections and their (consumer, mono) geometry
                bx, cons, mono = cheap.xyxy, cheap.cons, cheap.mono_t
                a_d = a_c
                for _ in range(args.rounds):
                    # (a) discovery crop from insertion probes on the draft's mono geometry
                    base = decide(mono)
                    sens = np.zeros((len(r_grid), len(Y_GRID)), bool)
                    for ai, r in enumerate(r_grid):
                        for bi, y in enumerate(Y_GRID):
                            t = tuple(np.r_[m_, val] for m_, val in zip(mono, (r, y - OBJ_W / 2, y + OBJ_W / 2, np.inf)))
                            sens[ai, bi] = decide(t) != base
                    if sens.any():
                        b = boxes[sens]
                        crops.append(np.array([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()]))
                    # (b) verification crops around the draft decision's pivots
                    S = HR.Obj(bx, cons)
                    for k in range(len(S)):
                        if decide(tuple(np.delete(a, k) for a in S.t)) != a_d:
                            x1, y1, x2, y2 = bx[k]
                            cw, ch = (x2 - x1) * CTX, (y2 - y1) * CTX
                            cx_, cy_ = (x1 + x2) / 2, (y1 + y2) / 2
                            crops.append(np.array([max(cx_ - cw / 2, 0), max(cy_ - ch / 2, 0),
                                                   min(cx_ + cw / 2, img_w), min(cy_ + ch / 2, img_h)]))
                    # merge: full detections replace the draft inside every crop
                    ic, jf = in_any(cheap.xyxy, crops), in_any(full.xyxy, crops)
                    bx = np.r_[cheap.xyxy[~ic], full.xyxy[jf]]
                    cons = tuple(np.r_[a[~ic], b_[jf]] for a, b_ in zip(cheap.cons, full.cons))
                    mono = tuple(np.r_[a[~ic], b_[jf]] for a, b_ in zip(cheap.mono_t, full.mono_t))
                    a_new = decide(cons)
                    if a_new == a_d:
                        break
                    a_d = a_new
                rec.update({f"{name}_J_c": cost(a_c), f"{name}_J_f": cost(a_f), f"{name}_J_spec": cost(a_d),
                            f"{name}_agree_full": int(a_d == a_f), f"{name}_flip": int(a_c != a_f),
                            f"{name}_cost": union_area(crops, img_w, img_h), f"{name}_n_crops": len(crops)})
            prev_act = P.discrete_action(P.required_decel(*cheap.cons_raw, v, pp)[0], pp)
            if with_traj:
                prev_plan = B.plan(*cheap.cons_raw, v, prev_plan, pb, cb)
            rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_pickle(run / "speculative.pkl")
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[(dv.track == args.dataset) & (dv.geometry == args.geometry)].copy()
    dv["frame"] = dv.frame.astype(int)
    out = []
    for name in (["brake", "traj"] if with_traj else ["brake"]):
        m = d.merge(dv[dv.system == name][["seq", "frame", "split", "V", "J_cheap"]], on=["seq", "frame"])
        assert np.allclose(m[f"{name}_J_c"] - m[f"{name}_J_f"], m.V, atol=1e-9)
        for sp in ("all", "test"):
            q = m if sp == "all" else m[m.split == "test"]
            Lc, V = float(q.J_cheap.sum()), q.V.to_numpy(float)
            b = float(q[f"{name}_cost"].mean())
            k = max(int(round(b * len(q))), 1)
            fl = q[f"{name}_flip"] == 1
            out.append({"cell": f"{args.dataset} {args.geometry} {name}", "split": sp, "rmin": args.rmin,
                        "mean_pixel_cost": b, "decision_agrees_with_full": float(q[f"{name}_agree_full"].mean()),
                        "agrees_on_frames_where_full_flips": float(q.loc[fl, f"{name}_agree_full"].mean()),
                        "G_all_full": float(V.sum()) / Lc,
                        "G_speculative": float((q[f"{name}_J_c"] - q[f"{name}_J_spec"]).sum()) / Lc,
                        "G_frames_oracle_equal_pixels": topk_expect(V, [V], [k])[0][0][0] / Lc,
                        "G_frames_random_equal_pixels": b * V.sum() / Lc})
    res = pd.DataFrame(out)
    res.to_csv(run / "speculative_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
