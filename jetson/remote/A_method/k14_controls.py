#!/usr/bin/env python
"""K14 — "+ refiner" controls (post-hoc analysis; frozen configurations, nothing tuned). Is ours' gain the planner-driven
selection or the range refiner alone?  Per frame and pipeline: action and decision cost WITHOUT and WITH the dataset's
refiner (v1 out/resnet18 on nuScenes, v2 out/resnet18_v2 on AV2 / Argoverse-HD), tensor detector px (canvases incl.
guard re-detections, 32-px rounding), crops.  The refiner is applied exactly as for ours (instr_eval.refine): to the
objects of the pipeline's final scene that are in the support of its action AND come from a high-resolution pass
(crop or whole-frame detection; for a whole-frame pipeline every object).
  ours            frozen loop (speculative.run): nuScenes v3 (v_close 10, crops 0.6), AV2 / A-HD v4-lite (v_close 0,
                  crops 960 per long side); its per-frame crop area is the budget of the crop heuristics
  uncertainty, temporal_crit, random   instr_eval's heuristics at ours' per-frame crop area (strict fill)
  traj_support    support crops + the ego-corridor tube widened until the rest of the budget is filled (baselines_eq)
  corridor        the ego-corridor tube alone (10-70 m, widened until it reaches the budget; baselines_eq corridor_fit)
  u960, native    whole frame (u1280 too on Argoverse-HD / AV2 is not needed here)
AV2 runs inside the K10 shadow tree with the ego mask (k10_masked_run.py, K10_MASK_Y0=1830).
Also stored (for K19): the required deceleration a_gt and the required action per frame.
usage: k14_controls.py --dataset nusc|av2|ahd [--shard i/n]
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pickle
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("ie", B / "scripts" / "A_method" / "instr_eval.py")
IE = importlib.util.module_from_spec(_s); _s.loader.exec_module(IE)
_h = importlib.util.spec_from_file_location("hood", B / "scripts" / "A_method" / "k10_hood.py")
HOOD = importlib.util.module_from_spec(_h); _h.loader.exec_module(HOOD)
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

OUT = B / "results" / "k14"
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
HEUR = ("uncertainty", "temporal_crit", "random", "traj_support", "corridor")


def tube(cam, W, H, budget, left=None):
    """the ego-corridor image footprint (10-70 m), widened in 0.25 m steps until its area reaches `left`"""
    P2 = cam.calib.P2
    fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
    left = budget if left is None else left
    if left <= 0:
        return []
    vb_n, top = min(cy + fy * cam.cam_h / 10.0, H), max(cy + fy * cam.cam_h / 70.0 - fy * 1.6 / 70.0, 0)
    t = []
    for half_m in np.arange(1.0, 12.01, 0.25):
        half = fx * half_m / 10.0
        t = [np.array([max(cx - half, 0), top, min(cx + half, W), vb_n])]
        if IE.area(t[0]) >= left:
            break
    return t


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2", "ahd"])
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    ds = args.dataset
    si, sn = (int(x) for x in args.shard.split("/"))
    sys.path.insert(0, str(B / "scripts" / "common"))
    import train_refiner_remote as TR
    from nusc_refiner_crop import crop64
    dev = torch.device("cuda:0")
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    ck = torch.load(B / "out" / ("resnet18" if ds == "nusc" else "resnet18_v2") / "image_refiner.pt", map_location=dev,
                    weights_only=False)
    rnet = TR.Res18(len(ck["fmu"])); rnet.load_state_dict(ck["state"]); rnet = rnet.to(dev).eval()
    fmu, fsd = ck["fmu"], ck["fsd"]
    pp, cp = P.PlannerParams(), P.CostParams()
    if ds == "nusc":
        VC, SCALE, CH = 10.0, 0.6, None
        M = {"u960": Mode("f576x960", 576, 960), "native": Mode("ns_u1600", 928, 1600)}
        cheap_px = 192 * 320
    elif ds == "av2":
        VC, SCALE, CH = 0.0, 960 / 2048, Mode("av2_c320", 320, 256)
        M = {"u960": Mode("av2_f960", 960, 768), "native": Mode("av2_u2048", 2048, 1568)}
        cheap_px = 320 * 256
    else:
        VC, SCALE, CH = 0.0, 960 / 1920, Mode("ahd_c320", 224, 320)
        M = {"u960": Mode("ahd_u960", 608, 960), "native": Mode("ahd_u1920", 1216, 1920)}
        cheap_px = 224 * 320
    OUT.mkdir(parents=True, exist_ok=True)
    for ui, (unit, cam, gen) in enumerate(HOOD.units(ds)):
        if ui % sn != si:
            continue
        fo = OUT / f"{ds}__{unit}.pkl"
        if fo.exists():
            continue
        R, t = cam.transform("ego")
        hist, prev_act, prev_c, rows = {}, None, None, []
        for fr, path, v, g, extra in gen():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            NATIVE = W * H
            a_gt, a_gt_obj = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, np.zeros(0))
            rec = {"unit": unit, "frame": fr, "v": v, "gt": extra != "nogt", "a_gt": float(a_gt),
                   "act_gt": P.discrete_action(a_gt, pp), "prev_act": prev_act,
                   "n_relevant": int((np.asarray(a_gt_obj) >= pp.thr_decel).sum()) if len(g) else 0}

            def lift(dd, pv):
                gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, cam.dt,
                                        cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def keep(d):
                k = d["conf"] >= IE.THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k],
                        "entropy": np.asarray(d.get("entropy", np.zeros(len(d["conf"]))))[k]}

            def act(tt):
                return P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)

            def cost(a):
                return P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]

            tpx = [0.0]

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * SCALE / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * SCALE / 32)) * 32)
                tpx[0] += nh * nw
                d = keep(det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))[0])
                d["xyxy"] = d["xyxy"] + np.array([x1, y1, x1, y1], float)
                return {k: d[k] for k in ("xyxy", "conf", "coarse")}

            def refine(scene):
                refined = [np.asarray(a, float).copy() for a in scene["mono"]]
                for p_ in SP.support(scene["mono"], v, pp):
                    if not scene["from_full"][p_]:
                        continue
                    x1, y1, x2, y2 = scene["xyxy"][p_]
                    zz = predicted_geometry({"xyxy": scene["xyxy"][p_:p_ + 1], "coarse": scene["coarse"][p_:p_ + 1]}, None,
                                            cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                    zg, zh = float(zz["z_ground"][0]), float(zz["z_height"][0])
                    f = np.array([[x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(scene["conf"][p_]),
                                   float(CLS[str(scene["coarse"][p_])]), float(scene["mono"][0][p_]), zg, zh,
                                   float(np.log(max(zg, .5) / max(zh, .5))), 1.0]], np.float32)
                    f = ((f - fmu) / fsd).astype(np.float32)
                    x = torch.from_numpy(crop64(img, scene["xyxy"][p_])).permute(2, 0, 1)[None].float().to(dev) / 255.0
                    with torch.no_grad():
                        corr = float(np.exp(rnet(x, torch.from_numpy(f).to(dev)).item()))
                    for a in refined[:3]:
                        a[p_] *= corr
                return tuple(refined)

            def record(name, scene, px, **extra_):
                a, ar = act(scene["mono"]), act(refine(scene))
                rec.update({f"a_{name}": a, f"J_{name}": cost(a), f"a_{name}_ref": ar, f"J_{name}_ref": cost(ar),
                            f"px_{name}": px, **{f"{k}_{name}": x for k, x in extra_.items()}})
            # ---- cheap draft (cached on nuScenes, online elsewhere)
            if ds == "nusc":
                ch = extra
            else:
                ch = keep(det.detect(img, CH)[0]); ch["mono"] = lift(ch, prev_c)
                prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            base = {k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}
            a_c = act(ch["mono"])
            rec.update({"a_cheap": a_c, "J_cheap": cost(a_c), "px_cheap": cheap_px / NATIVE})
            # ---- whole frame (+ refiner on its support: every object is high-resolution)
            for nm, m in M.items():
                d = keep(det.detect(img, m)[0])
                tt = lift(d, hist.get(nm)); hist[nm] = d
                record(nm, {"xyxy": d["xyxy"], "conf": d["conf"], "coarse": d["coarse"], "mono": tt,
                            "from_full": np.ones(len(d["conf"]), bool)}, m.net_h * m.net_w / NATIVE, n=len(d["conf"]))
            # ---- ours (frozen)
            tpx[0] = 0.0
            scene, crops, rounds = SP.run(base, v, cam, W, H, detect_crop, lambda full: lift(full, hist.get("scene_full")),
                                          pp, v_close=VC, guard=True)
            budget = sum(IE.area(c) for c in crops)
            record("full", scene, (cheap_px + tpx[0]) / NATIVE, crops=len(crops), rounds=rounds, crop_area=budget / NATIVE)
            hist["scene_full"] = {"xyxy": scene["xyxy"]}
            # ---- crop heuristics at ours' per-frame crop area
            gdet = SP.guarded(detect_crop, W, H)
            for hn in HEUR:
                if hn == "uncertainty":
                    order = np.lexsort((-np.asarray(ch.get("entropy", np.zeros(len(ch["conf"])))), ch["conf"]))
                    sel = IE.fill_strict([SP.box_crop(ch["xyxy"][k], W, H) for k in order], budget)
                elif hn == "temporal_crit":
                    cands = []
                    ps = hist.get("state_temporal_crit")
                    if ps is not None and len(ps[0]):
                        pxy, pmono = ps
                        _, dem = P.required_decel(*pmono, v, pp)
                        order = np.lexsort((pmono[0], -np.asarray(dem)))
                        cands = [SP.box_crop(np.r_[pxy[k][:2] - 0.25 * (pxy[k][2:] - pxy[k][:2]),
                                                   pxy[k][2:] + 0.25 * (pxy[k][2:] - pxy[k][:2])], W, H) for k in order]
                    sel = IE.fill_strict(cands, budget)
                elif hn == "random":
                    rr = np.random.default_rng(zlib.crc32(f"{unit}/{fr}".encode()))
                    cands = []
                    for _ in range(40):
                        if len(ch["xyxy"]):
                            b = ch["xyxy"][rr.integers(len(ch["xyxy"]))]
                            w_, h_ = min((b[2] - b[0]) * 2, W), min((b[3] - b[1]) * 2, H)
                        else:
                            w_, h_ = 200.0, 120.0
                        h_ = min(h_, H * 0.65)
                        x0, y0 = rr.uniform(0, max(W - w_, 1)), rr.uniform(H * 0.35, max(H - h_, H * 0.35 + 1))
                        cands.append(np.array([x0, y0, x0 + w_, y0 + h_]))
                    sel = IE.fill_strict(cands, budget)
                elif hn == "traj_support":
                    sup = [SP.box_crop(ch["xyxy"][k], W, H) for k in SP.support(ch["mono"], v, pp)] if budget > 0 else []
                    sel = sup + tube(cam, W, H, budget, max(budget - sum(IE.area(q) for q in sup), 0.0))
                else:
                    sel = tube(cam, W, H, budget) if budget > 0 else []
                tpx[0] = 0.0
                sc_h, cr_h = IE.merge_once(base, sel, gdet, lambda full, hn=hn: lift(full, hist.get(f"scene_{hn}")))
                record(hn, sc_h, (cheap_px + tpx[0]) / NATIVE, crops=len(cr_h), crop_area=sum(IE.area(c) for c in sel) / NATIVE)
                hist[f"scene_{hn}"] = {"xyxy": sc_h["xyxy"]}
                if hn == "temporal_crit":
                    hist["state_temporal_crit"] = (np.asarray(sc_h["xyxy"], float).reshape(-1, 4),
                                                   tuple(np.asarray(a, float) for a in sc_h["mono"]))
            prev_act = a_c
            rows.append(rec)
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{ui}] {unit}: {len(rows)} frames", flush=True)
    print("K14 CONTROLS DONE", flush=True)


if __name__ == "__main__":
    main()
