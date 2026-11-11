#!/usr/bin/env python
"""K18(A) — crop heuristics matched on TENSOR px (post-hoc analysis; AV2 with the K10 ego mask via k10_masked_run.py).
Per frame, ours (frozen loop) is run first and its crop tensor px (canvases incl. guard re-detections, 32-px rounding)
is the reference; each heuristic then adds crops in its own order while the summed canvas px of its crops stays
<= b x ours' crop tensor px on that frame (b in {0.5, 1, 2, 4}); the tube heuristics (ego corridor, traj_support's tube)
take the widest tube whose canvas fits. Guard re-detections of the heuristic are counted in its reported px (so its
mean px can exceed the budget slightly; reported as is). Heuristics as in instr_eval / baselines_eq: uncertainty,
temporal_crit, random, traj_support (support crops + corridor tube), corridor (tube only).
Per frame and arm: action, decision cost without / with refiner, tensor px.
usage: k18_heur.py --dataset nusc|av2|ahd [--budgets 0.5,1,2,4] [--shard i/n]   (ahd: frames without 3D labels are
run for the histories and flagged gt = False)
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
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
HEUR = ("uncertainty", "temporal_crit", "random", "traj_support", "corridor")
BUDGETS = (0.5, 1.0, 2.0, 4.0)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2", "ahd"])
    ap.add_argument("--budgets", default="0.5,1,2,4")
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
    global BUDGETS
    BUDGETS = tuple(float(x) for x in args.budgets.split(","))
    if ds == "nusc":
        VC, SCALE, CH, cheap_px, units = 10.0, 0.6, None, 192 * 320, IE.units_nusc("test")
    elif ds == "av2":
        VC, SCALE, CH, cheap_px, units = 0.0, 960 / 2048, Mode("av2_c320", 320, 256), 320 * 256, IE.units_av2("test")
    else:                                               # Argoverse-HD val: the AV2 configuration, untuned (K13 / K14)
        _h = importlib.util.spec_from_file_location("hood", B / "scripts" / "A_method" / "k10_hood.py")
        HOOD = importlib.util.module_from_spec(_h); _h.loader.exec_module(HOOD)
        VC, SCALE, CH, cheap_px, units = 0.0, 960 / 1920, Mode("ahd_c320", 224, 320), 224 * 320, HOOD.units("ahd")
    out = B / "results" / "k18_heur"
    out.mkdir(parents=True, exist_ok=True)
    for ui, (unit, cam, frames) in enumerate(units):
        if ui % sn != si:
            continue
        fo = out / f"{ds}__{unit}.pkl"
        if fo.exists():
            continue
        R, t = cam.transform("ego")
        P2 = cam.calib.P2
        fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
        hist, prev_act, prev_c, rows = {}, None, None, []
        for fr, path, v, g, cheap_cached in frames():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            NATIVE = W * H
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
            rec = {"unit": unit, "frame": fr, "act_gt": P.discrete_action(a_gt, pp), "gt": not isinstance(cheap_cached, str)}

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

            def canvas(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = min(max(int(math.ceil(cr[2])), x1 + 8), W), min(max(int(math.ceil(cr[3])), y1 + 8), H)
                return max(32, int(math.ceil((x2 - x1) * SCALE / 32)) * 32) * max(32, int(math.ceil((y2 - y1) * SCALE / 32)) * 32)

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

            def fill_px(cands, bud):
                sel, used = [], 0
                for c in cands:
                    cv_ = canvas(c)
                    if used + cv_ <= bud:
                        sel.append(c); used += cv_
                return sel, used

            def tube_px(bud):
                vb_n, top = min(cy + fy * cam.cam_h / 10.0, H), max(cy + fy * cam.cam_h / 70.0 - fy * 1.6 / 70.0, 0)
                best = []
                for half_m in np.arange(0.5, 12.01, 0.25):
                    half = fx * half_m / 10.0
                    c = np.array([max(cx - half, 0), top, min(cx + half, W), vb_n])
                    if canvas(c) > bud:
                        break
                    best = [c]
                return best
            if cheap_cached is not None and not isinstance(cheap_cached, str):
                ch = cheap_cached
            else:
                ch = keep(det.detect(img, CH)[0]); ch["mono"] = lift(ch, prev_c)
                prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            base = {k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}
            a_c = act(ch["mono"])
            rec.update({"a_cheap": a_c, "J_cheap": cost(a_c), "px_cheap": cheap_px / NATIVE})
            tpx[0] = 0.0
            sc, crops, rounds = SP.run(base, v, cam, W, H, detect_crop, lambda full: lift(full, hist.get("scene_full")), pp,
                                       v_close=VC, guard=True)
            ours_crop_px = tpx[0]
            a, ar = act(sc["mono"]), act(refine(sc))
            rec.update({"a_full": a, "J_full": cost(a), "a_full_ref": ar, "J_full_ref": cost(ar),
                        "px_full": (cheap_px + ours_crop_px) / NATIVE})
            hist["scene_full"] = {"xyxy": sc["xyxy"]}
            gdet = SP.guarded(detect_crop, W, H)
            ent = np.asarray(ch.get("entropy", np.zeros(len(ch["conf"]))))
            for bx in BUDGETS:
                bud = bx * ours_crop_px
                for hn in HEUR:
                    nm = f"{hn}_x{bx:g}"
                    if hn == "uncertainty":
                        order = np.lexsort((-ent, ch["conf"]))
                        sel, _ = fill_px([SP.box_crop(ch["xyxy"][k], W, H) for k in order], bud)
                    elif hn == "temporal_crit":
                        cands = []
                        ps_ = hist.get(f"state_{nm}")
                        if ps_ is not None and len(ps_[0]):
                            pxy, pmono = ps_
                            _, dem = P.required_decel(*pmono, v, pp)
                            order = np.lexsort((pmono[0], -np.asarray(dem)))
                            cands = [SP.box_crop(np.r_[pxy[k][:2] - 0.25 * (pxy[k][2:] - pxy[k][:2]),
                                                       pxy[k][2:] + 0.25 * (pxy[k][2:] - pxy[k][:2])], W, H) for k in order]
                        sel, _ = fill_px(cands, bud)
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
                        sel, _ = fill_px(cands, bud)
                    elif hn == "traj_support":
                        sup, used = fill_px([SP.box_crop(ch["xyxy"][k], W, H) for k in SP.support(ch["mono"], v, pp)], bud) if bud > 0 else ([], 0)
                        sel = sup + (tube_px(bud - used) if bud - used > 0 else [])
                    else:
                        sel = tube_px(bud) if bud > 0 else []
                    tpx[0] = 0.0
                    sh, cr_h = IE.merge_once(base, sel, gdet, lambda full, nm=nm: lift(full, hist.get(f"scene_{nm}")))
                    a, ar = act(sh["mono"]), act(refine(sh))
                    rec.update({f"a_{nm}": a, f"J_{nm}": cost(a), f"a_{nm}_ref": ar, f"J_{nm}_ref": cost(ar),
                                f"px_{nm}": (cheap_px + tpx[0]) / NATIVE})
                    hist[f"scene_{nm}"] = {"xyxy": sh["xyxy"]}
                    if hn == "temporal_crit":
                        hist[f"state_{nm}"] = (np.asarray(sh["xyxy"], float).reshape(-1, 4), tuple(np.asarray(q, float) for q in sh["mono"]))
            prev_act = a_c
            rows.append(rec)
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{ui}] {unit}: {len(rows)} frames", flush=True)
    print("K18 HEUR DONE", flush=True)


if __name__ == "__main__":
    main()
