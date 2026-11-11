#!/usr/bin/env python
"""K16 — which approximation causes each disagreement of ours with pi(F)? (post-hoc analysis; frozen configurations,
ours WITHOUT refiner, F = u960, AV2 with the K10 ego mask via k10_masked_run.py; full test splits.)
Per frame:
  ours      the frozen loop with real crops (run_g with the frozen parameters = speculative.run; parity checked offline
            against instr_eval's a_full), its crops + guard-grown crops (= the verified region R), rounds and whether
            it stopped at the round cap WITHOUT having converged (a further round would still add a crop)
  chain     counterfactual chain, one intervention added at a time (cumulative), the crop detector replaced by F:
              a  exact locality (F's objects whose centre lies in a crop; guard as in instr_eval's loc diagnostic)
              b  + F's ttc history (lift with F's previous detections)
              c  + closing probe (nuScenes v_close 10 -> 20; AV2 0 -> 10 (c1) -> 20 (c2))
              d  + extended grid (ranges 8, 12.5, ..., 70, 80 m; lateral -9 ... 9 m in 1.5 m steps)
              e  + probe templates (pedestrian 0.6 x 1.8, car 1.8 x 1.6, truck 2.6 x 3.6 m; footprint = union)
              f  + round cap 10 instead of 3
  coverage  for ours' final action a, every F object whose demand reaches tau_{a+1}: range, lateral centre, class,
            whether its box centre lies in R, whether it is F's deciding object
usage: k16_attr.py --dataset nusc|av2 [--shard i/n]
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pickle
from pathlib import Path

import cv2
import numpy as np

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("ie", B / "scripts" / "A_method" / "instr_eval.py")
IE = importlib.util.module_from_spec(_s); _s.loader.exec_module(IE)
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import box_iou, predicted_geometry                                 # noqa: E402
from vtf import dsf                                                              # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

OUT = B / "results" / "k16"
CAR = (SP.OBJ_W, SP.OBJ_H)
PROBES3 = ((0.6, 1.8), CAR, (2.6, 3.6))
R_EXT = np.array([8.0, 12.5, 17.5, 22.5, 27.5, 35.0, 45.0, 55.0, 70.0, 80.0])
Y_EXT = np.arange(-9.0, 9.01, 1.5)


def field_crop_g(mono, v, cam, W, H, pp, v_closes, r_grid, y_grid, probes):
    """speculative.field_crop generalised to several closing speeds / probe sizes (footprint = bounding box of the
    union); with one speed, the car probe and the frozen grid it is speculative.field_crop."""
    P2 = cam.calib.P2
    fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
    lo_u, lo_v, hi_u, hi_v = [], [], [], []
    for vc in v_closes:
        for (w, h) in probes:
            sens = dsf.insertion_grid(mono, v, r_grid, y_grid, w, pp, vc)
            if not sens.any():
                continue
            rr, yy = np.meshgrid(r_grid, y_grid, indexing="ij")
            rr, yy = rr[sens], yy[sens]
            vb = cy + fy * cam.cam_h / rr
            uc, hw = cx - fx * yy / rr, fx * w / (2 * rr)
            lo_u.append((uc - hw).min()); hi_u.append((uc + hw).max())
            lo_v.append((vb - fy * h / rr).min()); hi_v.append(vb.max())
    if not lo_u:
        return None
    return np.array([max(min(lo_u), 0), max(min(lo_v), 0), min(max(hi_u), W), min(max(hi_v), H)])


def run_g(cheap, v, cam, W, H, detect_crop, lift, pp, max_rounds=3, v_closes=(0.0,), r_grid=SP.R_GRID, y_grid=SP.Y_GRID,
          probes=(CAR,), guard=True):
    """speculative.run (guard=True, sequential) with a generalised field; also returns the guard-grown crops and
    whether the loop converged (False = stopped at the cap while a further round would still add a crop)."""
    gdet = SP.guarded(detect_crop, W, H) if guard else None
    crops, grown, full = [], [], {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
    scene, rounds = dict(cheap), 0

    def new_crops(scene):
        new = []
        fc = field_crop_g(scene["mono"], v, cam, W, H, pp, v_closes, r_grid, y_grid, probes)
        if fc is not None and not SP.covered(fc, crops):
            new.append(fc)
        for k in SP.support(scene["mono"], v, pp):
            bc = SP.box_crop(scene["xyxy"][k], W, H)
            if not SP.covered(bc, crops + grown + new):
                new.append(bc)
        return new
    converged = False
    for _ in range(max_rounds):
        new = new_crops(scene)
        if not new:
            converged = True
            break
        rounds += 1
        for c in new:
            if gdet is None:
                d = detect_crop(c)
            else:
                d, ex = gdet(c)
                grown += ex
            full = {k: np.concatenate([full[k], np.asarray(d[k])]) for k in full}
        crops += new
        if len(full["conf"]) > 1:
            o = np.argsort(-full["conf"], kind="stable")
            iou = box_iou(full["xyxy"], full["xyxy"])
            keep = []
            for i in o:
                if not keep or iou[i, keep].max() < 0.5:
                    keep.append(i)
            keep = np.sort(np.array(keep))
            full = {k: full[k][keep] for k in full}
        out = SP.centres_in(cheap["xyxy"], crops)
        if grown and len(full["conf"]) and len(cheap["xyxy"]):
            out |= box_iou(np.asarray(cheap["xyxy"], float), full["xyxy"]).max(1) >= 0.5
        fz = lift(full) if len(full["conf"]) else tuple(np.zeros(0) for _ in range(4))
        scene = {"xyxy": np.r_[cheap["xyxy"][~out], full["xyxy"]], "conf": np.r_[cheap["conf"][~out], full["conf"]],
                 "coarse": np.r_[cheap["coarse"][~out], full["coarse"]],
                 "mono": tuple(np.r_[a[~out], b] for a, b in zip(cheap["mono"], fz))}
    if not converged:
        converged = not new_crops(scene)
    return scene, crops, grown, rounds, converged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2"])
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    ds = args.dataset
    si, sn = (int(x) for x in args.shard.split("/"))
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    pp, cp = P.PlannerParams(), P.CostParams()
    if ds == "nusc":
        VC, SCALE, CH, F = 10.0, 0.6, None, Mode("f576x960", 576, 960)
        units = IE.units_nusc("test")
        chain_c = [("c", (20.0,))]
    else:
        VC, SCALE, CH, F = 0.0, 960 / 2048, Mode("av2_c320", 320, 256), Mode("av2_f960", 960, 768)
        units = IE.units_av2("test")
        chain_c = [("c1", (10.0,)), ("c2", (20.0,))]
    vc_final = chain_c[-1][1]
    steps = [("a", dict(v_closes=(VC,)), False), ("b", dict(v_closes=(VC,)), True)] + \
        [(n, dict(v_closes=vcs), True) for n, vcs in chain_c] + \
        [("d", dict(v_closes=vc_final, r_grid=R_EXT, y_grid=Y_EXT), True),
         ("e", dict(v_closes=vc_final, r_grid=R_EXT, y_grid=Y_EXT, probes=PROBES3), True),
         ("f", dict(v_closes=vc_final, r_grid=R_EXT, y_grid=Y_EXT, probes=PROBES3, max_rounds=10), True)]
    OUT.mkdir(parents=True, exist_ok=True)
    for ui, (unit, cam, frames) in enumerate(units):
        if ui % sn != si:
            continue
        fo = OUT / f"{ds}__{unit}.pkl"
        if fo.exists():
            continue
        R, t = cam.transform("ego")
        hist, prev_act, prev_c, rows = {}, None, None, []
        for fr, path, v, g, cheap_cached in frames():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
            rec = {"unit": unit, "frame": fr, "v": v, "a_gt": float(a_gt), "act_gt": P.discrete_action(a_gt, pp)}

            def lift(dd, pv):
                gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, cam.dt,
                                        cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def keep(d):
                k = d["conf"] >= IE.THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}

            def act(tt):
                return P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)

            def cost(a):
                return P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]
            if cheap_cached is not None:
                ch = cheap_cached
            else:
                ch = keep(det.detect(img, CH)[0]); ch["mono"] = lift(ch, prev_c)
                prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            base = {k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}
            a_c = act(ch["mono"])
            rec.update({"a_cheap": a_c, "J_cheap": cost(a_c)})
            # ---- F = u960 (its own history)
            F960 = keep(det.detect(img, F)[0])
            ttF = lift(F960, hist.get("u960"))
            aF = act(ttF)
            rec.update({"a_F": aF, "J_F": cost(aF)})
            # ---- ours, real crops (frozen parameters)

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * SCALE / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * SCALE / 32)) * 32)
                d = keep(det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))[0])
                d["xyxy"] = d["xyxy"] + np.array([x1, y1, x1, y1], float)
                return d
            sc, crops, grown, rounds, conv = run_g(base, v, cam, W, H, detect_crop,
                                                   lambda full: lift(full, hist.get("scene_ours")), pp, v_closes=(VC,))
            a_o = act(sc["mono"])
            rec.update({"a_ours": a_o, "J_ours": cost(a_o), "rounds": rounds, "converged": conv, "crops": len(crops),
                        "grown": len(grown)})
            hist["scene_ours"] = {"xyxy": sc["xyxy"]}
            # ---- coverage of F's objects that would raise ours' action
            a_max_F, dem_F = P.required_decel(*ttF, v, pp)
            dem_F = np.asarray(dem_F, float)
            drvF = int(np.argmax(dem_F)) if aF != P.KEEP and len(dem_F) else -1
            cov = []
            if a_o < P.HARD_BRAKE and len(dem_F):
                thr = pp.thr_decel if a_o == P.KEEP else pp.thr_hard
                inR = SP.centres_in(F960["xyxy"], crops + grown)
                for j in np.flatnonzero(dem_F >= thr):
                    cov.append({"z": float(ttF[0][j]), "y": float((ttF[1][j] + ttF[2][j]) / 2), "cls": str(F960["coarse"][j]),
                                "inR": bool(inR[j]), "is_F_driver": j == drvF, "demand": float(dem_F[j])})
            rec["cov"] = cov
            # ---- counterfactual chain (crop detector replaced by F's objects)

            def detect_crop_F(cr):
                k = SP.centres_in(F960["xyxy"], [cr]) if len(F960["xyxy"]) else np.zeros(0, bool)
                return {"xyxy": F960["xyxy"][k], "conf": F960["conf"][k], "coarse": F960["coarse"][k]}
            for sn_, kw, fhist in steps:
                pv = hist.get("u960") if fhist else hist.get(f"scene_{sn_}")
                sd, cr_, gr_, rd, cv = run_g(base, v, cam, W, H, detect_crop_F, lambda full, pv=pv: lift(full, pv), pp, **kw)
                a_s = act(sd["mono"])
                rec.update({f"a_{sn_}": a_s, f"J_{sn_}": cost(a_s), f"rounds_{sn_}": rd, f"conv_{sn_}": cv})
                hist[f"scene_{sn_}"] = {"xyxy": sd["xyxy"]}
            hist["u960"] = F960
            prev_act = a_c
            rows.append(rec)
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{ui}] {unit}: {len(rows)} frames", flush=True)
    print("K16 ATTR DONE", flush=True)


if __name__ == "__main__":
    main()
