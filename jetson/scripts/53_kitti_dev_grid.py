#!/usr/bin/env python
"""KITTI per-benchmark tuning on the DEV split only (DEEP's KITTI train + val sequences; test untouched).

Same conventions as 50_kitti_baseline_cells.py (Jetson, TRT FP16 engines, COCO YOLOv8s, DEEP's four KITTI cells with
cheap action / plan histories, monocular lift, merged-scene history per arm). Arms:
  c, u960, sahi                         references on dev
  b_vc{V}_r{R}                          braking field (vtf.speculative.run, guard), v_close V, at most R rounds
  pb_r{R}                               Planner B's own field + pivots (vtf.spec_general.PlannerB, guard)
  un_vc{V}_r{R}                         union of both consumers' crops per round
each also + refiner ("_ref"; refiner on the arm's consumer support objects that came from crops).
Selection (declared before the run): the configuration with the highest mean G over the four dev cells (+ refiner);
"matched" = braking-field arm on the brake cells and Planner-B arm on the traj cells is evaluated as a combination.
Writes results/raw/<run>/cells.pkl and dev_grid.csv (G per cell and arm, sequence bootstrap CI vs the frozen v3+ref).
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".pydeps"))
_s = importlib.util.spec_from_file_location("bc", ROOT / "scripts" / "50_kitti_baseline_cells.py")
BC = importlib.util.module_from_spec(_s)
_s.loader.exec_module(BC)
E, P, PB, SP, runmeta, THR = BC.E, BC.P, BC.PB, BC.SP, BC.runmeta, BC.THR
from rap.cache import Camera                                                     # noqa: E402
from rap.decision import KittiAdapter, _apply_range_source                       # noqa: E402
from rap.detect import MODES, Mode                                               # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from rap.paths import KITTI_IMAGES, MODELS                                       # noqa: E402
from vtf import spec_general as SG                                               # noqa: E402
from vtf.fastpost import FastDetector                                            # noqa: E402

BRAKE = [(10.0, 3), (0.0, 3), (5.0, 3), (15.0, 3), (10.0, 5), (0.0, 5)]
PLANB = [3, 5]
UNION = [(10.0, 3), (0.0, 3)]
CELLS = (("brake mono", "Jbm"), ("brake oracle", "Jbo"), ("traj mono", "Jtm"), ("traj oracle", "Jto"))


class Union:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def field(self, t, v, rg, yg):
        return self.a.field(t, v, rg, yg) | self.b.field(t, v, rg, yg)

    def support(self, t, v):
        return np.union1d(self.a.support(t, v), self.b.support(t, v)).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "train", "val"])
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--no_sahi", action="store_true")
    ap.add_argument("--final2", action="store_true",
                    help="the ONE test run after dev round 2 (declared choice b_vc0_r3_nat_ref), round-2 arm set on KITTI test")
    ap.add_argument("--round2", action="store_true",
                    help="dev round 2: native-resolution crop canvases (_nat), u1248 (+ refiner) next to the round-1 choices")
    ap.add_argument("--final", action="store_true",
                    help="the ONE test run of the dev-selected configuration (un_vc0_r3 + ref) next to frozen v3 / u960 / SAHI")
    args = ap.parse_args()
    global BRAKE, PLANB, UNION
    if args.final:
        BRAKE, PLANB, UNION = [(10.0, 3)], [], [(0.0, 3)]
    if args.final2:
        args.round2, args.final = True, True
    if args.round2:
        BRAKE, PLANB, UNION = [(0.0, 3), (10.0, 3)], [], [(0.0, 3)]
        args.no_sahi = True
    run = runmeta.new_run("kitti_final2" if args.final2 else "kitti_final" if args.final else "kitti_dev_round2" if args.round2 else "kitti_dev_grid", vars(args))
    ref = E.TRTRefiner(runmeta.ROOT / "engines" / "image_refiner.engine", runmeta.ROOT / "engines" / "norm.npz")
    w = str(Path(MODELS) / "yolov8s.pt")
    base = FastDetector(w, backend="trt", engine_dir=str(Path(MODELS) / "engines"), engine_stem="yolov8s")
    crop = FastDetector(w, backend="trt", engine_dir=str(runmeta.ROOT / "engines"), engine_stem="yolov8s_crop")
    if not args.no_sahi:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        sahi = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=w, confidence_threshold=THR,
                                                  device="cuda:0", image_size=640)
    m_cheap, m_960, m_canvas = MODES["cheap_320"], Mode("288x960", 288, 960), Mode("96x288", 96, 288)
    m_1248 = Mode("384x1248", 384, 1248)
    # native policy: the smallest crop canvas that holds the crop at scale >= 1 (else the largest)
    CANVAS = [Mode(f"{h}x{w}", h, w) for h, w in ((96, 288), (192, 576), (288, 864), (384, 1248))]
    pp, cp = P.PlannerParams(), P.CostParams()
    pbp, pbc = PB.PARAMS_B["static_obstacles"], PB.COSTS_B["default"]
    rng = np.random.default_rng(0)
    dv = pd.read_csv(E.DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[(dv.track == "KITTI") & (dv.geometry == "mono") & (dv.system == "brake")].copy()
    dv["frame"] = dv.frame.astype(int)
    splits = ["test"] if args.final else ["train", "val"] if args.split == "dev" else [args.split]
    assert args.final or "test" not in splits
    seqs = sorted(set(dv[dv.split.isin(splits)].seq))
    arms = ["c", "u960"] + ([] if args.no_sahi else ["sahi"]) + (["u1248", "u1248_ref"] if args.round2 else [])
    spec_arms = [f"b_vc{vc:g}_r{r}" for vc, r in BRAKE] + [f"pb_r{r}" for r in PLANB] + [f"un_vc{vc:g}_r{r}" for vc, r in UNION]
    if args.round2:
        spec_arms += [a + "_nat" for a in spec_arms]
    arms += [a + sfx for a in spec_arms for sfx in ("", "_ref")]
    rows, n = [], 0
    for s in seqs:
        geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s)
        cam = Camera("KITTI", s)
        R, t = cam.transform("ego")
        hist = {k: None for k in ["c", "u960", "sahi", "u1248"] + spec_arms}
        prev_act, prev_act_o, prev_plan, prev_plan_o = None, None, None, None
        for fr in sorted(int(f) for f in dv[dv.seq == s].frame):
            img = cv2.imread(str(Path(KITTI_IMAGES) / s / f"{fr:06d}.png"))
            H, W = img.shape[:2]
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            gt_t = tuple(np.asarray(g[k_], float) for k_ in ("long_near", "lat_min", "lat_max", "ttc"))
            rec = {"seq": s, "frame": fr, "split": "val" if s in set(dv[dv.split == "val"].seq) else "train"}

            def lift(xyxy, coarse, pv):
                gk = predicted_geometry({"xyxy": xyxy, "coarse": coarse}, pv, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def score(name, xyxy, tt):
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

            dc = keep(base.detect(img, m_cheap)[0])
            dc["xyxy"] = dc["xyxy"].astype(float)
            dc["mono"] = lift(dc["xyxy"], dc["coarse"], hist["c"])
            to_c = score("c", dc["xyxy"], dc["mono"])
            d9 = keep(crop.detect(img, m_960)[0])
            score("u960", d9["xyxy"].astype(float), lift(d9["xyxy"].astype(float), d9["coarse"], hist["u960"]))
            hist["u960"] = {"xyxy": d9["xyxy"].astype(float), "coarse": d9["coarse"]}

            def detect_crop(cr, native=False):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                mc = next((m for m in CANVAS if y2 - y1 <= m.net_h and x2 - x1 <= m.net_w), CANVAS[-1]) if native else m_canvas
                dk = keep(crop.detect(img[y1:y2, x1:x2], mc)[0])
                return {"xyxy": dk["xyxy"].astype(float) + np.array([x1, y1, x1, y1], float), "conf": dk["conf"].astype(float),
                        "coarse": dk["coarse"]}

            cheap = {k: dc[k] for k in ("xyxy", "conf", "coarse", "mono")}
            planb = SG.PlannerB(pbp, pbc)
            planb.prev = prev_plan
            if args.round2:          # uniform at KITTI's native width; + refiner on its decision support (brake ∪ Planner B)
                du = keep(crop.detect(img, m_1248)[0])
                du["xyxy"] = du["xyxy"].astype(float)
                tu = lift(du["xyxy"], du["coarse"], hist["u1248"])
                score("u1248", du["xyxy"], tu)
                hist["u1248"] = {"xyxy": du["xyxy"], "coarse": du["coarse"]}
                ru = [np.asarray(a, float).copy() for a in tu]
                for p_ in Union(SG.Brake(pp, 0.0), planb).support(tu, v):
                    corr = ref(img, du["xyxy"][p_], du["conf"][p_], du["coarse"][p_], cam, R, t, W, H)
                    for a in ru[:3]:
                        a[p_] *= corr
                score("u1248_ref", du["xyxy"], tuple(ru))
            for arm in spec_arms:
                lf = (lambda full, arm=arm: lift(full["xyxy"], full["coarse"], hist[arm]))
                nat = arm.endswith("_nat")
                core = arm[:-4] if nat else arm
                dcrop = (lambda cr: detect_crop(cr, native=True)) if nat else detect_crop
                if core.startswith("b_"):
                    vc, r = float(core.split("_")[1][2:]), int(core.split("_r")[1])
                    scene, crops, _ = SP.run(cheap, v, cam, W, H, dcrop, lf, pp, v_close=vc, max_rounds=r, guard=True)
                    sup = SP.support(scene["mono"], v, pp)
                else:
                    r = int(core.split("_r")[1])
                    cons = planb if core.startswith("pb_") else Union(SG.Brake(pp, float(core.split("_")[1][2:])), planb)
                    scene, crops, _ = SG.run(cheap, v, cam, W, H, dcrop, lf, cons, max_rounds=r, guard=True)
                    sup = cons.support(scene["mono"], v)
                rec[f"px_{arm}"] = sum(BC.area(c) for c in crops) / (W * H)
                score(arm, scene["xyxy"], scene["mono"])
                refined = [np.asarray(a, float).copy() for a in scene["mono"]]
                for p_ in sup:
                    if scene["from_full"][p_]:
                        corr = ref(img, scene["xyxy"][p_], scene["conf"][p_], scene["coarse"][p_], cam, R, t, W, H)
                        for a in refined[:3]:
                            a[p_] *= corr
                score(arm + "_ref", scene["xyxy"], tuple(refined))
                hist[arm] = {"xyxy": scene["xyxy"]}
            if not args.no_sahi:
                res = get_sliced_prediction(img[:, :, ::-1].copy(), sahi, slice_height=640, slice_width=640, overlap_height_ratio=0.2,
                                            overlap_width_ratio=0.2, perform_standard_pred=True, verbose=0)
                ob = [o for o in res.object_prediction_list if o.category.id in BC.SAHI_COARSE]
                sx = np.array([o.bbox.to_xyxy() for o in ob], float).reshape(-1, 4)
                scls = np.array([BC.SAHI_COARSE[o.category.id] for o in ob]) if ob else np.zeros(0, "<U8")
                score("sahi", sx, lift(sx, scls, hist["sahi"]))
                hist["sahi"] = {"xyxy": sx, "coarse": scls}
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
        d = pd.DataFrame(rows)
        d.to_pickle(run / "cells.pkl")
        if args.max_frames and n >= args.max_frames:
            break
    uq = d.seq.unique(); idx = [np.flatnonzero(d.seq.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    out = []
    for cell, tag in CELLS:
        L = d[f"{tag}_c"].to_numpy()
        V = {a: (d[f"{tag}_c"] - d[f"{tag}_{a}"]).to_numpy() for a in arms if a != "c"}
        for a, v_ in V.items():
            pk = "px_" + (a[:-4] if a.endswith("_ref") else a)
            r = {"cell": cell, "method": a, "G": v_.sum() / L.sum(), "px": d[pk].mean() if pk in d else np.nan}
            if a != "b_vc10_r3_ref":
                dr = [(v_[t_].sum() - V["b_vc10_r3_ref"][t_].sum()) / L[t_].sum() for t_ in boots]
                r["minus_v3ref"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
            if "u1248" in V and a != "u1248":
                dr = [(v_[t_].sum() - V["u1248"][t_].sum()) / L[t_].sum() for t_ in boots]
                r["minus_u1248"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
            out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(run / "dev_grid.csv", index=False)
    piv = res.pivot(index="method", columns="cell", values="G")
    piv["mean4"] = piv.mean(1)
    pd.set_option("display.width", 220)
    print(f"frames {len(d)} ({', '.join(splits)})")
    print(piv.sort_values("mean4", ascending=False).round(3).to_string())
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
