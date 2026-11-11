#!/usr/bin/env python
"""D3 (exploratory): escalate the image region the decision is sensitive to, instead of the whole frame.

Decision-sensitivity field (DSF), computed before escalation from the cheap set (mono geometry) and the consumer
alone: for each cell (range r, lateral y) of a ground grid, insert a hypothetical static obstacle (1.8 m wide) and
record whether the decision changes.  Cells where the full mode adds recall (r >= R_MIN) and the decision is sensitive
are projected into the image (pinhole, flat ground, 1.6 m object height); the escalated crop is their bounding box.

Proxy for running the full detector on that crop: full-mode detections whose box centre lies in the crop replace the
cheap detections there.  Cost = crop area / image area (pixels at the full scale).  Compared at the same mean pixel
cost with frame escalation (random, DEEP's cached allocators, oracle) and with a static horizon crop of equal area.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P
from rap import planner_b as B
from rap.cache import Camera, DetCache
from rap.decision import KittiAdapter, _apply_range_source
from rap.paths import CACHE, NUSCENES_TRAINVAL
from rap.risk import RiskConfig
from rap.scoring import topk_expect
from vtf import runmeta

R_GRID = np.array([12.5, 17.5, 22.5, 27.5, 35.0, 45.0, 55.0, 70.0])
Y_GRID = np.array([-6.0, -4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0])
OBJ_W, OBJ_H = 1.8, 1.6
DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"
RAW_RAP = runmeta.RAP_REPO / "results" / "raw"


def cell_boxes(cam: Camera, img_w, img_h):
    """Image box of an obstacle standing on each grid cell (camera-frame pinhole, flat ground)."""
    P2 = cam.calib.P2
    fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
    rr, yy = np.meshgrid(R_GRID, Y_GRID, indexing="ij")
    vb = cy + fy * cam.cam_h / rr
    vt = vb - fy * OBJ_H / rr
    uc = cx - fx * yy / rr
    hw = fx * OBJ_W / (2 * rr)
    return np.stack([np.clip(uc - hw, 0, img_w), np.clip(vt, 0, img_h), np.clip(uc + hw, 0, img_w),
                     np.clip(vb, 0, img_h)], -1)                                 # (nr, ny, 4)


def crop_of(mask, boxes):
    if not mask.any():
        return None
    b = boxes[mask]
    return np.array([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()])


def in_crop(xyxy, crop):
    if crop is None or not len(xyxy):
        return np.zeros(len(xyxy), bool)
    u = (xyxy[:, 0] + xyxy[:, 2]) / 2
    v = (xyxy[:, 1] + xyxy[:, 3]) / 2
    return (u >= crop[0]) & (u <= crop[2]) & (v >= crop[1]) & (v <= crop[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["KITTI", "nuScenes"], required=True)
    ap.add_argument("--geometry", choices=["mono", "oracle"], required=True)
    ap.add_argument("--crop_mode", default=None, help="KITTI only: detect the crop with this cached mode, e.g. full_960")
    args = ap.parse_args()
    tag = f"_{args.crop_mode}" if args.crop_mode else ""
    run = runmeta.new_run(f"region{tag}_{args.dataset}_{args.geometry}", vars(args))
    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()
    pb, cb = B.PARAMS_B["static_obstacles"], B.COSTS_B["default"]
    rng = np.random.default_rng(0)
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
        fc = DetCache(det_dir / args.crop_mode / f"{s}.npz") if args.crop_mode else f
        prev_act = prev_plan = None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            j = f.index[fr]
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            sc = c.scalars(i)
            img_w, img_h = float(sc["img_w"]), float(sc["img_h"])
            boxes = cell_boxes(cam, img_w, img_h)
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            truth = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc"))

            br = {}
            for tag, cache, idx in (("c", c, i), ("f", f, j), ("x", fc, fc.index[fr])):
                d, geo = cache.det(idx), cache.geo(idx)
                k = d["conf"] >= cfg.thr("cheap" if tag == "c" else "full")
                geo_k = {kk: vv[k] for kk, vv in geo.items()}
                bx = d["xyxy"][k].astype(float)
                cons = tuple(np.asarray(a, float) for a in _apply_range_source(bx, geo_k, g, args.geometry, rng, 0.0))
                mono = tuple(np.asarray(geo_k[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))
                br[tag] = (bx, cons, mono)
            rec = {"seq": s, "frame": fr}
            consumers = {"brake": (lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp),
                                   lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"])}
            if with_traj:
                pu = prev_plan
                consumers["traj"] = (lambda t: B.plan(*t, v, pu, pb, cb),
                                     lambda a: B.executed_cost(a, *truth, v, pu, pb, cb)["J"])
            for name, (decide, cost) in consumers.items():
                bxc, consc, monoc = br["c"]
                bxf, consf, _ = br["f"]
                bxx, consx, _ = br["x"]
                base = decide(monoc)
                sens = np.zeros((len(R_GRID), len(Y_GRID)), bool)
                for a_, r in enumerate(R_GRID):
                    for b_, y in enumerate(Y_GRID):
                        t = tuple(np.r_[m_, val] for m_, val in zip(monoc, (r, y - OBJ_W / 2, y + OBJ_W / 2, np.inf)))
                        sens[a_, b_] = decide(t) != base
                crop = crop_of(sens, boxes)
                area = 0.0 if crop is None else (crop[2] - crop[0]) * (crop[3] - crop[1]) / (img_w * img_h)
                ic, jf = in_crop(bxc, crop), in_crop(bxx, crop)
                cons_r = tuple(np.r_[a[~ic], b[jf]] for a, b in zip(consc, consx))
                rec.update({f"{name}_J_c": cost(decide(consc)), f"{name}_J_f": cost(decide(consf)),
                            f"{name}_J_r": cost(decide(cons_r)), f"{name}_area": area,
                            f"{name}_dsf_mass": float(sens.sum())})
                # static horizon crop: centred band around the vanishing row, filled in later at matched area
                rec[f"{name}_crop"] = crop
            rec["img_w"], rec["img_h"] = img_w, img_h
            rec["cy"], rec["cx"] = float(cam.calib.P2[1, 2]), float(cam.calib.P2[0, 2])
            rec["_c"] = (br["c"][0], br["c"][1])
            rec["_f"] = (br["f"][0], br["f"][1])
            prev_act = P.discrete_action(P.required_decel(*br["c"][1], v, pp)[0], pp)
            if with_traj:
                prev_plan = B.plan(*br["c"][1], v, prev_plan, pb, cb)
            rec["_ctx"] = (v, a_gt, truth, prev_act, prev_plan)
            rows.append(rec)
    df = pd.DataFrame(rows)

    # ---- static crop of the same mean area: a band centred on the principal point's column, top at the horizon row
    out = []
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[dv.track == args.dataset].copy()
    dv["frame"] = dv.frame.astype(int)
    for name in (["brake", "traj"] if with_traj else ["brake"]):
        mean_area = float(df[f"{name}_area"].mean())
        sys_ = "brake" if name == "brake" else "traj"
        ref = dv[(dv.geometry == args.geometry) & (dv.system == sys_)][["seq", "frame", "split", "V", "J_cheap"]]
        d = df.merge(ref, on=["seq", "frame"], how="inner")
        assert np.allclose(d[f"{name}_J_c"] - d[f"{name}_J_f"], d.V, atol=1e-9), "V mismatch"
        # static crop J, recomputed with the saved branches and consumer context
        Js = []
        for _, r in d.iterrows():
            w, h = r.img_w, r.img_h
            side = np.sqrt(mean_area) if mean_area > 0 else 0.0
            cw, ch = min(w, side * w * 2.0), min(h, side * h / 2.0)       # a wide band, 4:1 relative aspect
            crop = np.array([r.cx - cw / 2, r.cy - ch / 3, r.cx + cw / 2, r.cy + 2 * ch / 3])
            bxc, consc = r["_c"]
            bxf, consf = r["_f"]
            ic, jf = in_crop(bxc, crop), in_crop(bxf, crop)
            cons_s = tuple(np.r_[a[~ic], b[jf]] for a, b in zip(consc, consf))
            v, a_gt, truth, _, _ = r["_ctx"]
            Js.append((cons_s, v, a_gt, truth))
        # recompute with the right history: the previous cheap action/plan of the same sequence
        Jst = []
        prev = {}
        for (_, r), (cons_s, v, a_gt, truth) in zip(d.iterrows(), Js):
            pa = prev.get(("a", r.seq))
            pl = prev.get(("p", r.seq))
            if name == "brake":
                a = P.discrete_action(P.required_decel(*cons_s, v, pp)[0], pp)
                Jst.append(P.decision_cost(a, a_gt, pa, pp, cp)["J"])
            else:
                pl_s = B.plan(*cons_s, v, pl, pb, cb)
                Jst.append(B.executed_cost(pl_s, *truth, v, pl, pb, cb)["J"])
            bxc, consc = r["_c"]
            prev[("a", r.seq)] = P.discrete_action(P.required_decel(*consc, v, pp)[0], pp)
            prev[("p", r.seq)] = B.plan(*consc, v, pl, pb, cb) if name == "traj" else None
        d["J_static"] = Jst
        for sp in ("all", "test"):
            e = d if sp == "all" else d[d.split == "test"]
            Lc = float(e.J_cheap.sum())
            V = e.V.to_numpy(float)
            Vr = (e[f"{name}_J_c"] - e[f"{name}_J_r"]).to_numpy(float)
            Vs = (e[f"{name}_J_c"] - e.J_static).to_numpy(float)
            b = float(e[f"{name}_area"].mean())
            k = max(int(round(b * len(e))), 1)
            orc = topk_expect(V, [V], [k])[0][0][0]
            out.append({"cell": f"{args.dataset} {args.geometry} {name}", "split": sp, "n": len(e),
                        "mean_crop_area": b, "share_frames_with_crop": float((e[f"{name}_area"] > 0).mean()),
                        "G_full_all_frames": V.sum() / Lc,
                        "G_frames_random_at_b": b * V.sum() / Lc, "G_frames_oracle_at_b": orc / Lc,
                        "G_dsf_region": Vr.sum() / Lc, "G_static_region": Vs.sum() / Lc,
                        "dsf_region_harm_share": float((Vr < -1e-9).sum() / max((np.abs(Vr) > 1e-9).sum(), 1)),
                        "frame_harm_share": float((V < -1e-9).sum() / max((np.abs(V) > 1e-9).sum(), 1))})
    res = pd.DataFrame(out)
    df.drop(columns=["_c", "_f", "_ctx"]).to_pickle(run / f"region__{args.dataset}__{args.geometry}.pkl")
    res.to_csv(run / f"region_summary__{args.dataset}__{args.geometry}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
