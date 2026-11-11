#!/usr/bin/env python
"""SOTA baselines on DEEP's four KITTI cells (brake / Planner B x mono / oracle range), Jetson, TensorRT FP16 where the
deployed engines exist (cheap 96x320, full 288x960, crop canvas 96x288), same frames (DEEP's KITTI test split).

Per frame, all with the same COCO YOLOv8s and DEEP's conventions (cheap action / plan history, monocular ego lift):
  u960            the whole frame at 288x960 (reference)
  v3 / v3_ref     the frozen method (v_close 10, 3 rounds, truncation guard, merged-scene history) + image refiner
  uncertainty     cheap boxes by (confidence ascending, entropy descending), crops strictly within v3's per-frame pixels
  temporal_crit   previous merged objects by braking demand, crops strictly within v3's pixels (Self-Cueing-style)
  fovea           FOVEA KDE_I (official grid generator, cheap draft saliency) warped to the 288x960 canvas, unwarped
  sahi            SAHI (official library, torch): 640x640 slices, overlap 0.2, + standard prediction
Crop baselines use v3's machinery (guard, merge rule, crop canvas, own merged-scene history).
Writes results/raw/<run>/cells.pkl and cells_bootstrap.csv (paired sequence bootstrap, 2,000).
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".pydeps"))
_s = importlib.util.spec_from_file_location("e2e", ROOT / "scripts" / "40_e2e_v2.py")
E = importlib.util.module_from_spec(_s)
_s.loader.exec_module(E)
P, PB, SP, runmeta = E.P, E.PB, E.SP, E.runmeta
from rap.cache import Camera                                                     # noqa: E402
from rap.decision import KittiAdapter, _apply_range_source                       # noqa: E402
from rap.detect import MODES, Mode                                               # noqa: E402
from rap.mono import box_iou, predicted_geometry                                 # noqa: E402
from rap.paths import KITTI_IMAGES, MODELS                                       # noqa: E402
from vtf.fastpost import FastDetector                                            # noqa: E402

THR = E.THR
SAHI_COARSE = {0: "person", 1: "cyclist", 2: "vehicle", 3: "cyclist", 5: "vehicle", 7: "vehicle"}
ARMS = ("c", "u960", "spec", "spec_ref", "uncertainty", "temporal_crit", "fovea", "sahi")


def fovea_gg():
    if not hasattr(np, "float"):
        np.float = float

    class _Reg:
        def __init__(self, name): self.name = name
        def register_module(self, *a, **k): return (lambda c: c)
    mm, mu = types.ModuleType("mmcv"), types.ModuleType("mmcv.utils")
    mu.Registry = _Reg
    md, mdm, mdb = (types.ModuleType(n) for n in ("mmdet", "mmdet.models", "mmdet.models.builder"))
    mdb.build = lambda *a, **k: None
    pkg, ut, vis = types.ModuleType("fovea_official"), types.ModuleType("fovea_official.utils"), types.ModuleType("fovea_official.utils.vis")
    pkg.__path__ = [str(ROOT / "third_party" / "fovea" / "fovea")]
    vis.vis_batched_imgs = lambda *a, **k: None
    for n, m_ in {"mmcv": mm, "mmcv.utils": mu, "mmdet": md, "mmdet.models": mdm, "mmdet.models.builder": mdb,
                  "fovea_official": pkg, "fovea_official.utils": ut, "fovea_official.utils.vis": vis,
                  "fovea_official.models": types.ModuleType("fovea_official.models")}.items():
        sys.modules.setdefault(n, m_)
    spec = importlib.util.spec_from_file_location("fovea_official.models.grid_generator",
                                                  ROOT / "third_party" / "fovea" / "fovea" / "models" / "grid_generator.py")
    gg = importlib.util.module_from_spec(spec)
    gg.__package__ = "fovea_official.models"
    spec.loader.exec_module(gg)
    return gg


def area(c):
    return max(c[2] - c[0], 0) * max(c[3] - c[1], 0)


def fill_strict(cands, budget):
    out, used = [], 0.0
    for c in cands:
        if used + area(c) <= budget:
            out.append(c)
            used += area(c)
    return out


def merge_once(cheap, crops, gdet, lift):
    """baselines_eq.merge_once: crop detections (guarded) replace the cheap boxes whose centres fall in a crop."""
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
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    run = runmeta.new_run("kitti_baseline_cells", vars(args))
    ref = E.TRTRefiner(runmeta.ROOT / "engines" / "image_refiner.engine", runmeta.ROOT / "engines" / "norm.npz")
    w = str(Path(MODELS) / "yolov8s.pt")
    base = FastDetector(w, backend="trt", engine_dir=str(Path(MODELS) / "engines"), engine_stem="yolov8s")
    crop = FastDetector(w, backend="trt", engine_dir=str(runmeta.ROOT / "engines"), engine_stem="yolov8s_crop")
    sahi = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=w, confidence_threshold=THR,
                                              device="cuda:0", image_size=640)
    gg = fovea_gg()
    m_cheap, m_960, m_canvas = MODES["cheap_320"], Mode("288x960", 288, 960), Mode("96x288", 96, 288)
    pp, cp = P.PlannerParams(), P.CostParams()
    pbp, pbc = PB.PARAMS_B["static_obstacles"], PB.COSTS_B["default"]
    rng = np.random.default_rng(0)
    dv = pd.read_csv(E.DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[(dv.track == "KITTI") & (dv.geometry == "mono") & (dv.system == "brake")].copy()
    dv["frame"] = dv.frame.astype(int)
    seqs = sorted(set(dv[dv.split == args.split].seq))
    rows, n = [], 0
    for s in seqs:
        geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s)
        cam = Camera("KITTI", s)
        R, t = cam.transform("ego")
        hist = {k: None for k in ARMS}
        prev_state = None
        prev_act, prev_act_o, prev_plan, prev_plan_o = None, None, None, None
        gen = None
        for fr in sorted(int(f) for f in dv[dv.seq == s].frame):
            img = cv2.imread(str(Path(KITTI_IMAGES) / s / f"{fr:06d}.png"))
            H, W = img.shape[:2]
            if gen is None:
                gen = gg.PlainKDEGrid(output_shape=(288, 960), grid_shape=(31, 51), separable=True, attraction_fwhm=4,
                                      anti_crop=True, bandwidth_scale=64, amplitude_scale=1)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            gt_t = tuple(np.asarray(g[k_], float) for k_ in ("long_near", "lat_min", "lat_max", "ttc"))
            rec = {"seq": s, "frame": fr}

            def lift(xyxy, coarse, pv):
                gk = predicted_geometry({"xyxy": xyxy, "coarse": coarse}, pv, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def score(name, xyxy, tt):
                """brake mono + DEEP's other three KITTI cells, cheap histories (as 40_e2e_v2.cells)."""
                rec[f"Jbm_{name}"] = P.decision_cost(P.discrete_action(P.required_decel(*tt, v, pp)[0], pp), a_gt, prev_act, pp, cp)["J"]
                geo = dict(zip(("z", "lat_min", "lat_max", "ttc"), (np.asarray(a, float) for a in tt)))
                to = tuple(np.asarray(a, float) for a in _apply_range_source(np.asarray(xyxy, float).reshape(-1, 4), geo, g, "gt", rng, 0.0))
                rec[f"Jbo_{name}"] = P.decision_cost(P.discrete_action(P.required_decel(*to, v, pp)[0], pp), a_gt, prev_act_o, pp, cp)["J"]
                for tag, t_, pv in (("Jtm", tt, prev_plan), ("Jto", to, prev_plan_o)):
                    rec[f"{tag}_{name}"] = PB.executed_cost(PB.plan(*t_, v, pv, pbp, pbc), *gt_t, v, pv, pbp, pbc)["J"]
                return to

            def keep(d):
                k = d["conf"] >= THR
                return {kk: np.asarray(vv)[k] for kk, vv in d.items() if kk in ("xyxy", "conf", "coarse", "entropy")}

            # cheap draft
            dc = keep(base.detect(img, m_cheap)[0])
            dc["xyxy"] = dc["xyxy"].astype(float)
            dc["mono"] = lift(dc["xyxy"], dc["coarse"], hist["c"])
            to_c = score("c", dc["xyxy"], dc["mono"])
            # u960
            d9 = keep(crop.detect(img, m_960)[0])
            score("u960", d9["xyxy"].astype(float), lift(d9["xyxy"].astype(float), d9["coarse"], hist["u960"]))
            hist["u960"] = {"xyxy": d9["xyxy"].astype(float), "coarse": d9["coarse"]}

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                dk = keep(crop.detect(img[y1:y2, x1:x2], m_canvas)[0])
                return {"xyxy": dk["xyxy"].astype(float) + np.array([x1, y1, x1, y1], float), "conf": dk["conf"].astype(float),
                        "coarse": dk["coarse"]}
            # v3 (+ refiner)
            scene, crops, _ = SP.run({k: dc[k] for k in ("xyxy", "conf", "coarse", "mono")}, v, cam, W, H, detect_crop,
                                     lambda full: lift(full["xyxy"], full["coarse"], hist["spec"]), pp, v_close=10.0,
                                     max_rounds=3, guard=True)
            budget = sum(area(c) for c in crops)
            rec["px_spec"] = budget / (W * H)
            score("spec", scene["xyxy"], scene["mono"])
            refined = [np.asarray(a, float).copy() for a in scene["mono"]]
            for p_ in SP.support(scene["mono"], v, pp):
                if scene["from_full"][p_]:
                    corr = ref(img, scene["xyxy"][p_], scene["conf"][p_], scene["coarse"][p_], cam, R, t, W, H)
                    for a in refined[:3]:
                        a[p_] *= corr
            score("spec_ref", scene["xyxy"], tuple(refined))
            hist["spec"] = {"xyxy": scene["xyxy"]}
            # crop baselines within v3's pixels
            gdet = SP.guarded(detect_crop, W, H)
            for arm in ("uncertainty", "temporal_crit"):
                if arm == "uncertainty":
                    order = np.lexsort((-dc["entropy"], dc["conf"]))
                    cands = [SP.box_crop(dc["xyxy"][k], W, H) for k in order]
                else:
                    cands = []
                    if prev_state is not None and len(prev_state[0]):
                        pxy, pmono = prev_state
                        _, dem = P.required_decel(*pmono, v, pp)
                        order = np.lexsort((pmono[0], -np.asarray(dem)))
                        cands = [SP.box_crop(np.r_[pxy[k][:2] - 0.25 * (pxy[k][2:] - pxy[k][:2]),
                                                    pxy[k][2:] + 0.25 * (pxy[k][2:] - pxy[k][:2])], W, H) for k in order]
                cr_ = fill_strict(cands, budget)
                sc = merge_once(dc, cr_, gdet, lambda full, arm=arm: lift(full["xyxy"], full["coarse"], hist[arm]))
                score(arm, sc["xyxy"], sc["mono"])
                rec[f"px_{arm}"] = sum(area(c) for c in cr_) / (W * H)
                hist[arm] = {"xyxy": sc["xyxy"]}
                if arm == "temporal_crit":
                    prev_state = (np.asarray(sc["xyxy"], float).reshape(-1, 4), tuple(np.asarray(a, float) for a in sc["mono"]))
            # FOVEA KDE_I
            x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().cuda()
            with torch.no_grad():
                grid = gen(x, [{"pad_shape": (H, W, 3)}], [torch.tensor(dc["xyxy"], dtype=torch.float32, device="cuda").reshape(-1, 4)])
                wimg = F.grid_sample(x, grid, align_corners=True)[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()[:, :, ::-1].copy()
            df = keep(crop.detect(wimg, m_960)[0])
            bb = gg.unwarp_bboxes(torch.tensor(df["xyxy"], dtype=torch.float32, device="cuda"), grid[0], (H, W)).cpu().numpy() \
                if len(df["conf"]) else np.zeros((0, 4))
            score("fovea", bb.astype(float), lift(bb.astype(float), df["coarse"], hist["fovea"]))
            hist["fovea"] = {"xyxy": bb.astype(float), "coarse": df["coarse"]}
            # SAHI
            res = get_sliced_prediction(img[:, :, ::-1].copy(), sahi, slice_height=640, slice_width=640, overlap_height_ratio=0.2,
                                        overlap_width_ratio=0.2, perform_standard_pred=True, verbose=0)
            ob = [o for o in res.object_prediction_list if o.category.id in SAHI_COARSE]
            sx = np.array([o.bbox.to_xyxy() for o in ob], float).reshape(-1, 4)
            scls = np.array([SAHI_COARSE[o.category.id] for o in ob]) if ob else np.zeros(0, "<U8")
            score("sahi", sx, lift(sx, scls, hist["sahi"]))
            hist["sahi"] = {"xyxy": sx, "coarse": scls}
            # DEEP's cheap histories
            hist["c"] = {"xyxy": dc["xyxy"], "coarse": dc["coarse"]}
            prev_act = P.discrete_action(P.required_decel(*dc["mono"], v, pp)[0], pp)
            prev_act_o = P.discrete_action(P.required_decel(*to_c, v, pp)[0], pp)
            prev_plan = PB.plan(*dc["mono"], v, prev_plan, pbp, pbc)
            prev_plan_o = PB.plan(*to_c, v, prev_plan_o, pbp, pbc)
            rows.append(rec)
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
        print(f"  {s} done ({n} frames)", flush=True)
        if args.max_frames and n >= args.max_frames:
            break
    d = pd.DataFrame(rows)
    d.to_pickle(run / "cells.pkl")
    uq = d.seq.unique(); idx = [np.flatnonzero(d.seq.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    out = []
    for cell, tag in (("brake mono", "Jbm"), ("brake oracle", "Jbo"), ("traj mono", "Jtm"), ("traj oracle", "Jto")):
        L = d[f"{tag}_c"].to_numpy()
        V = {a: (d[f"{tag}_c"] - d[f"{tag}_{a}"]).to_numpy() for a in ARMS if a != "c"}
        for a, v_ in V.items():
            r = {"cell": cell, "method": a, "G": v_.sum() / L.sum()}
            for rk in ("spec", "spec_ref"):
                if a != rk:
                    dr = [(V[rk][t_].sum() - v_[t_].sum()) / L[t_].sum() for t_ in boots]
                    r[f"{rk}_minus"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
            out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(run / "cells_bootstrap.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"frames {len(d)}, v3 px {d.px_spec.mean():.4f}, uncertainty px {d.px_uncertainty.mean():.4f}, "
          f"temporal_crit px {d.px_temporal_crit.mean():.4f}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
