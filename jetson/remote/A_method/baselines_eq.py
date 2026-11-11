#!/usr/bin/env python
"""Equal-pixel-budget crop baselines (answers the survey's objections 1-2), nuScenes, braking, monocular lift.

For every frame the budget is the pixel area the frozen v3 loop used on that frame (from its confirmatory eval).
Each baseline picks crops of that total area by its own rule, then the SAME machinery runs: YOLOv8s on each crop at
scale 0.6 with the truncation guard, detections replace the cheap ones whose centres lie in the crops, lift with the
previous merged scene as ttc history, braking decision, DEEP's cost.
  random        crops of the cheap scene's box size distribution (2x context) at random image positions
  corridor      the image footprint of the ego corridor (|y| <= 2.5 m) from 10 m to the range that fills the budget
  uncertainty   2x-context boxes of the cheap detections with the lowest confidence first, then the highest entropy
  seen_cf       counterfactual relevance of DETECTED objects only: crops around the objects whose removal changes the
                action (pivots) and the support — the loop without the unseen-object field
  field_only    the unseen-object field footprint alone (no support crops)
  v3            the frozen loop (field + support, iterated) — re-run here as the reference
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import _apply_range_source
from rap.detect import Mode, TwoFidelityDetector
from rap.mono import box_iou, predicted_geometry
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL
from vtf import dsf
from vtf import speculative as SP

B = Path(os.environ["B"])
W, H, THR, V_CLOSE = 1600, 900, 0.25, 10.0
ARMS = ("random", "corridor", "uncertainty", "seen_cf", "field_only", "v3")
ALL_ARMS = ARMS + ("fovea", "corridor_fit", "traj_support", "temporal_crit")


def area(c):
    return max(c[2] - c[0], 0) * max(c[3] - c[1], 0)


STRICT = False


def fill(cands, budget):
    """Candidates in priority order until the budget is used. Default: the crop that crosses the budget is kept (the
    first runs); STRICT: a crop is taken only if it fits in what is left (later, smaller candidates may still fit)."""
    out, used = [], 0.0
    for c in cands:
        if STRICT:
            if used + area(c) <= budget:
                out.append(c)
                used += area(c)
            continue
        if used >= budget:
            break
        out.append(c)
        used += area(c)
    return out


def merge_once(cheap, crops, gdet, lift):
    full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
    grown = []
    for c in crops:
        d, ex = gdet(c)
        grown += ex
        full = {k: np.concatenate([full[k], np.asarray(d[k])]) for k in full}
    if len(full["conf"]) > 1:
        o = np.argsort(-full["conf"], kind="stable")
        iou = box_iou(full["xyxy"], full["xyxy"])
        keep = []
        for i in o:
            if not keep or iou[i, keep].max() < 0.5:
                keep.append(i)
        full = {k: full[k][np.sort(np.array(keep))] for k in full}
    out = SP.centres_in(cheap["xyxy"], crops) if crops else np.zeros(len(cheap["xyxy"]), bool)
    if grown and len(full["conf"]) and len(cheap["xyxy"]):
        out |= box_iou(np.asarray(cheap["xyxy"], float), full["xyxy"]).max(1) >= 0.5
    fz = lift(full) if len(full["conf"]) else tuple(np.zeros(0) for _ in range(4))
    return {"xyxy": np.r_[cheap["xyxy"][~out], full["xyxy"]].reshape(-1, 4),
            "mono": tuple(np.r_[a[~out], b] for a, b in zip(cheap["mono"], fz))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="new")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ref", default="eval__new_test_v3_confirm.pkl", help="the v3 run whose per-frame pixels set the budget")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ALL_ARMS))
    ap.add_argument("--strict", action="store_true", help="never exceed the per-frame budget (see fill)")
    ap.add_argument("--budget_scale", type=float, default=1.0, help="budget = scale x v3's per-frame pixels (curves)")
    args = ap.parse_args()
    arms = tuple(args.arms) if "v3" in args.arms else tuple(args.arms) + ("v3",)
    torch.set_num_threads(args.threads)
    ref = pd.read_pickle(B / "results" / "nusc_full" / args.ref)[["seq", "frame", "px", "J_spec"]]
    global STRICT
    STRICT = args.strict
    budget = {(s, int(f)): args.budget_scale * float(p) * W * H for s, f, p in zip(ref.seq, ref.frame, ref.px)}
    tab = pd.read_pickle(B / "results" / "nusc_full" / "flips__mono.pkl")
    tab = tab[(tab["set"] == args.set) & tab.split.isin(args.split.split("+"))]
    scenes = sorted(tab.seq.unique())[: args.max_scenes or None]
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0" if torch.cuda.is_available() else "cpu",
                              backend="torch")
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    rows = []
    for s in scenes:
        geom, speeds = ad.geometry(s), ad.speeds(s)
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        P2 = cam.calib.P2
        fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
        c = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
        toks = db.samples(by[s])
        prev_act, prev_act_o = None, None
        prev_scene = {a: None for a in arms}
        prev_state = {a: None for a in arms}                 # (xyxy, mono) of each arm's previous merged scene
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            act = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)             # noqa: E731
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]                   # noqa: E731
            dc = c.det(i)
            kc = dc["conf"] >= THR
            geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
            bxc = dc["xyxy"][kc].astype(float)
            cons_c = tuple(np.asarray(a, float) for a in _apply_range_source(bxc, geo_c, g, "mono", rng, 0.0))
            cheap = {"xyxy": bxc, "conf": dc["conf"][kc].astype(float), "coarse": dc["coarse"][kc], "mono": cons_c,
                     "entropy": dc["entropy"][kc].astype(float)}
            a_c = act(cons_c)
            # DEEP's oracle-range brake cell: every detection takes the range of its matched GT (cheap-oracle history)
            oracle = lambda xyxy, mono: tuple(np.asarray(a, float) for a in _apply_range_source(       # noqa: E731
                np.asarray(xyxy, float).reshape(-1, 4), dict(zip(("z", "lat_min", "lat_max", "ttc"), mono)), g, "gt", rng, 0.0))
            cost_o = lambda a: P.decision_cost(a, a_gt, prev_act_o, pp, cp)["J"]              # noqa: E731
            to_c = oracle(bxc, cons_c)
            a_c_o = act(to_c)
            cost_o_c = cost_o(a_c_o)
            img = cv2.imread(str(db.image_path(toks[fr])))
            bud = budget.get((s, fr), 0.0)
            rec = {"seq": s, "frame": fr, "J_c": cost(a_c), "budget_px": bud / (W * H)}

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * 0.6 / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * 0.6 / 32)) * 32)
                d, _ = det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))
                kk = d["conf"] >= THR
                return {"xyxy": d["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                        "conf": d["conf"][kk].astype(float), "coarse": d["coarse"][kk]}
            gdet = SP.guarded(detect_crop, W, H)
            for arm in arms:
                def lift(full, arm=arm):
                    gk = predicted_geometry({"xyxy": full["xyxy"], "coarse": full["coarse"]}, prev_scene[arm], cam.calib,
                                            cam.cam_h, cam.dt, cam_to_ego=(R, t))
                    return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))
                if arm == "v3":
                    scene, crops, _ = SP.run(cheap, v, cam, W, H, detect_crop, lift, pp, v_close=V_CLOSE, guard=True)
                else:
                    if arm == "random":
                        cands = []
                        for _ in range(40):
                            if len(bxc):
                                b = bxc[rng.integers(len(bxc))]
                                w_, h_ = min((b[2] - b[0]) * 2, W), min((b[3] - b[1]) * 2, H)
                            else:
                                w_, h_ = 200.0, 120.0
                            h_ = min(h_, H * 0.65)
                            x0, y0 = rng.uniform(0, max(W - w_, 1)), rng.uniform(H * 0.35, max(H - h_, H * 0.35 + 1))
                            cands.append(np.array([x0, y0, x0 + w_, y0 + h_]))
                    elif arm == "corridor":
                        cands = []
                        for r_far in (70, 55, 45, 35, 27.5, 22.5, 17.5, 12.5):
                            vb_n, vb_f = cy + fy * cam.cam_h / 10.0, cy + fy * cam.cam_h / r_far
                            top = vb_f - fy * 1.6 / r_far
                            half = fx * 2.5 / 10.0
                            cands.append(np.array([max(cx - half, 0), max(top, 0), min(cx + half, W), min(vb_n, H)]))
                        cands = sorted(cands, key=lambda q: abs(area(q) - bud))[:1]
                    elif arm == "fovea":                                   # objectness prior: most confident first
                        cands = [SP.box_crop(bxc[k], W, H) for k in np.argsort(-cheap["conf"])]
                    elif arm == "corridor_fit":                            # corridor widened until it fills the budget
                        cands, vb_n = [], min(cy + fy * cam.cam_h / 10.0, H)
                        top = max(cy + fy * cam.cam_h / 70.0 - fy * 1.6 / 70.0, 0)
                        for half_m in np.arange(1.0, 12.01, 0.25):
                            half = fx * half_m / 10.0
                            cc_ = np.array([max(cx - half, 0), top, min(cx + half, W), vb_n])
                            cands = [cc_]
                            if area(cc_) >= bud:
                                break
                    elif arm == "traj_support":                          # VeteranAD-style: look along the path + support
                        sup = [SP.box_crop(bxc[k], W, H) for k in SP.support(cons_c, v, pp)]
                        left = max(bud - sum(area(q) for q in sup), 0.0)
                        tube = []
                        if left > 0:
                            vb_n, top = min(cy + fy * cam.cam_h / 10.0, H), max(cy + fy * cam.cam_h / 70.0 - fy * 1.6 / 70.0, 0)
                            for half_m in np.arange(1.0, 12.01, 0.25):
                                half = fx * half_m / 10.0
                                tube = [np.array([max(cx - half, 0), top, min(cx + half, W), vb_n])]
                                if area(tube[0]) >= left:
                                    break
                        cands = sup + tube
                    elif arm == "temporal_crit":                         # Self-Cueing-style: previous objects by criticality
                        cands = []
                        if prev_state[arm] is not None:
                            pxy, pmono = prev_state[arm]
                            if len(pxy):
                                _, dem = P.required_decel(*pmono, v, pp)
                                order = np.lexsort((pmono[0], -np.asarray(dem)))
                                cands = [SP.box_crop(np.r_[pxy[k][:2] - 0.25 * (pxy[k][2:] - pxy[k][:2]),
                                                            pxy[k][2:] + 0.25 * (pxy[k][2:] - pxy[k][:2])], W, H) for k in order]
                    elif arm == "uncertainty":
                        order = np.lexsort((-cheap["entropy"], cheap["conf"]))
                        cands = [SP.box_crop(bxc[k], W, H) for k in order]
                    elif arm == "seen_cf":
                        piv = dsf.brake_pivots(cons_c, v, pp)
                        idx = list(dict.fromkeys(list(np.atleast_1d(piv).astype(int)) + list(SP.support(cons_c, v, pp))))
                        cands = [SP.box_crop(bxc[k], W, H) for k in idx]
                    else:                                                            # field_only
                        fc = SP.field_crop(cons_c, v, cam, W, H, pp, V_CLOSE)
                        cands = [fc] if fc is not None else []
                    crops = fill(cands, bud) if arm in ("random", "uncertainty", "fovea", "temporal_crit") else cands
                    scene = merge_once(cheap, crops, gdet, lift)
                rec[f"J_{arm}"] = cost(act(scene["mono"]))
                rec[f"Jbo_{arm}"] = cost_o(act(oracle(scene["xyxy"], scene["mono"])))
                rec[f"px_{arm}"] = sum(area(q) for q in crops) / (W * H)
                prev_scene[arm] = {"xyxy": scene["xyxy"]}
                prev_state[arm] = (np.asarray(scene["xyxy"], float).reshape(-1, 4), tuple(np.asarray(a, float) for a in scene["mono"]))
            rows.append(rec)
            prev_act, prev_act_o = a_c, a_c_o
            rec["Jbo_c"] = cost_o_c
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_pickle(B / "results" / f"baselines_eq__{args.set}_{args.split}{args.tag}.pkl")
    L = d.J_c.to_numpy()
    uq = d.seq.unique(); idx = [np.flatnonzero(d.seq.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    boots = [np.concatenate([idx[j] for j in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    Vv3 = (d.J_c - d.J_v3).to_numpy()
    Lo = d.Jbo_c.to_numpy()
    Vv3o = (d.Jbo_c - d.Jbo_v3).to_numpy()
    out = []
    for arm in arms:
        V = (d.J_c - d[f"J_{arm}"]).to_numpy()
        aff = np.abs(V) > 1e-9
        r = {"arm": arm, "G": V.sum() / L.sum(), "pixels": d[f"px_{arm}"].mean(),
             "harmed": float((V < -1e-9).sum() / max(aff.sum(), 1))}
        if arm != "v3":
            dr = [(Vv3[t].sum() - V[t].sum()) / L[t].sum() for t in boots]
            r["v3_minus_arm"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        Vo = (d.Jbo_c - d[f"Jbo_{arm}"]).to_numpy()
        r["G_oracle"] = Vo.sum() / Lo.sum()
        if arm != "v3":
            dr = [(Vv3o[t].sum() - Vo[t].sum()) / Lo[t].sum() for t in boots]
            r["v3_minus_arm_oracle"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(B / "results" / f"baselines_eq__{args.set}_{args.split}{args.tag}.csv", index=False)
    print(f"scenes {len(uq)}, frames {len(d)}, v3 budget {d.budget_px.mean():.4f}")
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
