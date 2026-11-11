#!/usr/bin/env python
"""K23 — development-only evaluation of v5 candidates (method_freeze_draft §2e). nuScenes new val (152 scenes; cached
draft) and AV2 CFG-DEV (300 train logs of results/k23/av2_split.csv; ego mask through k10_masked_run.py). Nothing here
touches a test split.
An ARM is '+'-joined tokens on top of the frozen configuration (nuScenes v3: scale 0.6, v_close 10; AV2 v4-lite: crops
at 960 px / long side, v_close 0; context 2, no field margin, <= 3 rounds, guard, merged-scene history):
  s<x>      crop scale (nuScenes: factor, e.g. s0.8; AV2: long side in px, e.g. s1280)
  ctx<x>    support-crop context          fm<x>   field-crop margin           vc<x>   v_close (m/s)
  ts        C3 temporal support: crops at the previous frame's support objects that have no match in the draft
  b<N>      C5 bounded loop: <= N detector rounds, guard growth deferred to the next round
  c<x>      C2 evidence threshold: a crop detection without draft counterpart that raises the action needs conf >= x
  p         C2 persistence: a DECELERATE raised only by such detections needs a match in the previous frame's scene
Every arm is scored without refiner and with: the frozen refiner (v1 nuScenes / v2 AV2), v3a, v3b, v3c with the
validity veto at thresholds 0.1 / 0.2 / 0.3 / 0.5 (C1). Baselines on the same frames: u960 and uncertainty crops (at
the base arm's crop area), each without refiner and with every refiner. Crop detections are cached per frame and crop
geometry (tensor px are still counted per arm). Refiners are applied as in instr_eval: to the support objects of the
final scene that come from a high-resolution pass.
usage: k23_dev.py --dataset nusc|av2 --arms base,s0.8,... [--tag t] [--shard i/n]
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
_s = importlib.util.spec_from_file_location("ie", B / "scripts" / "A_method" / "instr_eval.py")
IE = importlib.util.module_from_spec(_s); _s.loader.exec_module(IE)
_l = importlib.util.spec_from_file_location("kl", B / "scripts" / "A_method" / "k_loop.py")
KL = importlib.util.module_from_spec(_l); _l.loader.exec_module(KL)
_t = importlib.util.spec_from_file_location("kt", B / "scripts" / "A_method" / "k23_train_refiner.py")
KT = importlib.util.module_from_spec(_t); _t.loader.exec_module(KT)
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import box_iou, predicted_geometry                                 # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
VETO = (0.1, 0.2, 0.3, 0.5)


def parse(arm, ds):
    loop = {"scale": 0.6 if ds == "nusc" else 960 / 2048, "ctx": 2.0, "fm": 0.0, "vc": 10.0 if ds == "nusc" else 0.0,
            "ts": False, "b": None}
    post = {"c": 0.25, "p": False}
    for tok in arm.split("+"):
        if tok == "base":
            continue
        if tok == "ts":
            loop["ts"] = True
        elif tok == "p":
            post["p"] = True
        elif tok.startswith("ctx"):
            loop["ctx"] = float(tok[3:])
        elif tok.startswith("fm"):
            loop["fm"] = float(tok[2:])
        elif tok.startswith("vc"):
            loop["vc"] = float(tok[2:])
        elif tok.startswith("s"):
            x = float(tok[1:])
            loop["scale"] = x if ds == "nusc" else x / 2048
        elif tok.startswith("b"):
            loop["b"] = int(tok[1:])
        elif tok.startswith("c"):
            post["c"] = float(tok[1:])
        else:
            raise ValueError(tok)
    return loop, post


def units(ds):
    if ds == "nusc":
        yield from IE.units_nusc("val")
        return
    from vtf import av2 as AV
    sp = pd.read_csv(REAL / "results" / "k23" / "av2_split.csv")
    root = B / "datasets" / "av2" / "sensor" / "train"
    for log in sorted(sp[sp.role == "CFG-DEV"].log):
        L = AV.AV2Log(root, log)

        def frames(L=L):
            geom, speeds = L.geometry(), L.speeds()
            for fr in range(len(L.frame_ts)):
                g = geom[geom["frame"] == fr] if len(geom) else geom
                g = g[(g["y2"] - g["y1"]) >= 10] if len(g) else g
                yield fr, str(L.image_path(fr)), float(speeds[fr]), g, None
        yield log, L, frames


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2"])
    ap.add_argument("--arms", required=True)
    ap.add_argument("--tag", default="alone")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    ds, arms = args.dataset, args.arms.split(",")
    si, sn = (int(x) for x in args.shard.split("/"))
    sys.path.insert(0, str(B / "scripts" / "common"))
    import train_refiner_remote as TR
    from nusc_refiner_crop import crop64
    dev = torch.device("cuda:0")
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    nets = {}
    ck = torch.load(B / "out" / ("resnet18" if ds == "nusc" else "resnet18_v2") / "image_refiner.pt", map_location=dev, weights_only=False)
    n1 = TR.Res18(len(ck["fmu"])); n1.load_state_dict(ck["state"]); nets["frozen"] = (n1.to(dev).eval(), ck["fmu"], ck["fsd"], 1)
    for v in ("a", "b", "c"):
        if not (REAL / "out" / f"resnet18_v3{v}" / "image_refiner.pt").exists():
            print(f"refiner v3{v} missing: skipped (smoke runs only)", flush=True)
            continue
        ck = torch.load(REAL / "out" / f"resnet18_v3{v}" / "image_refiner.pt", map_location=dev, weights_only=False)
        n3 = KT.Res18x3(len(ck["fmu"])); n3.load_state_dict(ck["state"]); nets[f"v3{v}"] = (n3.to(dev).eval(), ck["fmu"], ck["fsd"], 3)
    pp, cp = P.PlannerParams(), P.CostParams()
    if ds == "nusc":
        CH, F, cheap_px = None, Mode("f576x960", 576, 960), 192 * 320
    else:
        CH, F, cheap_px = Mode("av2_c320", 320, 256), Mode("av2_f960", 960, 768), 320 * 256
    specs = {a: parse(a, ds) for a in arms}
    loops = {}
    for a, (lp, _) in specs.items():
        loops.setdefault(tuple(sorted(lp.items())), lp)
    od = REAL / "results" / "k23" / f"{ds}_{args.tag}"
    od.mkdir(parents=True, exist_ok=True)
    for ui, (unit, cam, gen) in enumerate(units(ds)):
        if ui % sn != si:
            continue
        fo = od / f"{unit}.pkl"
        if fo.exists():
            continue
        R, t = cam.transform("ego")
        hist, prev_act, prev_c, rows = {}, None, None, []
        for fr, path, v, g, cached in gen():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            NATIVE = W * H
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
            rec = {"unit": unit, "frame": fr, "act_gt": P.discrete_action(a_gt, pp)}

            def lift(dd, pv):
                gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def keep(d):
                k = d["conf"] >= IE.THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k],
                        "entropy": np.asarray(d.get("entropy", np.zeros(len(d["conf"]))))[k]}

            def act(tt):
                return P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)

            def cost(a):
                return P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]
            cache, tpx = {}, [0.0]

            def make_detect(scale):
                def detect_crop(cr):
                    x1, y1 = int(cr[0]), int(cr[1])
                    x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                    sub = img[y1:y2, x1:x2]
                    nw = max(32, int(math.ceil(sub.shape[1] * scale / 32)) * 32)
                    nh = max(32, int(math.ceil(sub.shape[0] * scale / 32)) * 32)
                    tpx[0] += nh * nw
                    key = (x1, y1, x2, y2, nh, nw)
                    if key not in cache:
                        d = keep(det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))[0])
                        d["xyxy"] = d["xyxy"] + np.array([x1, y1, x1, y1], float)
                        cache[key] = {k: d[k] for k in ("xyxy", "conf", "coarse")}
                    return {k: x.copy() for k, x in cache[key].items()}
                return detect_crop

            def ref_outputs(scene, idx, name):
                net, fmu, fsd, nout = nets[name]
                if not len(idx):
                    return np.zeros((0, 3))
                zz = predicted_geometry({"xyxy": scene["xyxy"][idx], "coarse": scene["coarse"][idx]}, None, cam.calib, cam.cam_h,
                                        cam.dt, cam_to_ego=(R, t))
                Fm = []
                for n, p_ in enumerate(idx):
                    x1, y1, x2, y2 = scene["xyxy"][p_]
                    zg, zh = float(zz["z_ground"][n]), float(zz["z_height"][n])
                    Fm.append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(scene["conf"][p_]),
                               float(CLS[str(scene["coarse"][p_])]), float(scene["mono"][0][p_]), zg, zh,
                               float(np.log(max(zg, .5) / max(zh, .5))), 1.0])
                Fm = ((np.array(Fm, np.float32) - fmu) / fsd).astype(np.float32)
                x = torch.from_numpy(np.stack([crop64(img, scene["xyxy"][p_]) for p_ in idx])).permute(0, 3, 1, 2).float().to(dev) / 255.0
                with torch.no_grad():
                    o = net(x, torch.from_numpy(Fm).to(dev)).float().cpu().numpy()
                o = o.reshape(len(idx), -1)
                return np.c_[o, np.zeros((len(idx), 3 - o.shape[1]))] if o.shape[1] < 3 else o

            def score_all(prefix, scene):
                """actions / costs of a final scene without refiner and with every refiner"""
                a0 = act(scene["mono"])
                rec[f"a_{prefix}|none"], rec[f"J_{prefix}|none"] = a0, cost(a0)
                sup = [p_ for p_ in SP.support(scene["mono"], v, pp) if scene["from_full"][p_]]
                for name in nets:
                    o = ref_outputs(scene, sup, name)
                    tt = [np.asarray(a, float).copy() for a in scene["mono"]]
                    for n, p_ in enumerate(sup):
                        c = float(np.exp(o[n, 0]))
                        lo0, hi0 = tt[1][p_], tt[2][p_]
                        for a in tt[:3]:
                            a[p_] *= c
                        if name in ("v3b", "v3c"):
                            # the lateral head was trained on (annotated - ESTIMATED) centre of the unrefined box: the
                            # corrected centre is the estimated centre + offset; the width scales with the range
                            ctr, w = (lo0 + hi0) / 2 + o[n, 1], (hi0 - lo0) * c
                            tt[1][p_], tt[2][p_] = ctr - w / 2, ctr + w / 2
                    a1 = act(tuple(tt))
                    rec[f"a_{prefix}|{name}"], rec[f"J_{prefix}|{name}"] = a1, cost(a1)
                    if name == "v3c":
                        val = 1 / (1 + np.exp(-o[:, 2])) if len(sup) else np.zeros(0)
                        for thr in VETO:
                            t2 = [a.copy() for a in tt]
                            while True:
                                ac = act(tuple(t2))
                                vetoed = False
                                for n, p_ in enumerate(sup):
                                    if val[n] < thr and np.isfinite(t2[0][p_]):
                                        t3 = [a.copy() for a in t2]; t3[0][p_] = np.inf
                                        if act(tuple(t3)) < ac:
                                            t2 = t3; vetoed = True; break
                                if not vetoed:
                                    break
                            a2 = act(tuple(t2))
                            rec[f"a_{prefix}|v3c_veto{thr:g}"], rec[f"J_{prefix}|v3c_veto{thr:g}"] = a2, cost(a2)
            # ---- draft
            if cached is not None:
                ch = cached
            else:
                ch = keep(det.detect(img, CH)[0]); ch["mono"] = lift(ch, prev_c)
            if ds != "nusc":
                prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            base = {k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}
            a_c = act(ch["mono"])
            rec.update({"a_cheap": a_c, "J_cheap": cost(a_c), "px_cheap": cheap_px / NATIVE})
            # ---- u960 baseline (+ every refiner)
            d9 = keep(det.detect(img, F)[0]); t9 = lift(d9, hist.get("u960")); hist["u960"] = d9
            score_all("u960", {"xyxy": d9["xyxy"], "conf": d9["conf"], "coarse": d9["coarse"], "mono": t9,
                               "from_full": np.ones(len(d9["conf"]), bool)})
            # ---- loops (shared by arms with the same loop parameters)
            res = {}
            for key, lp in loops.items():
                seed = None
                if lp["ts"] and hist.get(f"sup_{key}") is not None and len(hist[f"sup_{key}"]):
                    pb = hist[f"sup_{key}"]
                    m = box_iou(pb, ch["xyxy"]).max(1) < 0.3 if len(ch["xyxy"]) else np.ones(len(pb), bool)
                    seed = [KL.box_crop_ctx(b, W, H, 2.0) for b in pb[m]]
                tpx[0] = 0.0
                sc, crops, grown, rounds, conv = KL.run_g(base, v, cam, W, H, make_detect(lp["scale"]),
                                                          lambda full, key=key: lift(full, hist.get(f"scene_{key}")), pp,
                                                          v_closes=(lp["vc"],), ctx=lp["ctx"], fmargin=lp["fm"], seed=seed,
                                                          max_calls=lp["b"])
                res[key] = (sc, crops + grown, tpx[0], rounds)
                hist[f"scene_{key}"] = {"xyxy": sc["xyxy"]}
                sup = [p_ for p_ in SP.support(sc["mono"], v, pp)]
                hist[f"sup_{key}"] = sc["xyxy"][sup] if sup else np.zeros((0, 4))
            # ---- arms: post-filters (C2) on the loop's scene, then every refiner
            for arm, (lp, post) in specs.items():
                key = tuple(sorted(lp.items()))
                sc, crops, px, rounds = res[key]
                sc = dict(sc)
                if post["c"] > 0.25 or post["p"]:
                    new = sc["from_full"] & ((box_iou(sc["xyxy"], ch["xyxy"]).max(1) < 0.5) if len(ch["xyxy"]) and len(sc["xyxy"]) else True)
                    a_now = act(sc["mono"])
                    while a_now > P.KEEP:
                        _, dem = P.required_decel(*sc["mono"], v, pp)
                        thr = pp.thr_hard if a_now == P.HARD_BRAKE else pp.thr_decel
                        raisers = np.flatnonzero(np.asarray(dem) >= thr)
                        drop = [j for j in raisers if new[j] and sc["conf"][j] < post["c"]]
                        if post["p"] and a_now == P.DECELERATE and not drop and len(raisers) and all(new[raisers]):
                            pv = hist.get(f"final_{arm}")
                            drop = [j for j in raisers if pv is None or not len(pv) or box_iou(sc["xyxy"][j:j + 1], pv).max() < 0.3]
                        if not drop:
                            break
                        k_ = np.ones(len(sc["conf"]), bool); k_[drop] = False
                        sc = {kk: (vv[k_] if not isinstance(vv, tuple) else tuple(a[k_] for a in vv)) for kk, vv in sc.items()}
                        new = new[k_]
                        a_now = act(sc["mono"])
                hist[f"final_{arm}"] = sc["xyxy"]
                score_all(arm, sc)
                rec[f"px_{arm}"] = (cheap_px + px) / NATIVE
                rec[f"rounds_{arm}"] = rounds
                if arm == arms[0]:
                    budget = sum(IE.area(c) for c in crops)
            # ---- uncertainty crops at the first arm's crop area (+ every refiner)
            order = np.lexsort((-ch.get("entropy", np.zeros(len(ch["conf"]))), ch["conf"]))
            sel = IE.fill_strict([SP.box_crop(ch["xyxy"][k], W, H) for k in order], budget)
            tpx[0] = 0.0
            sh, _ = IE.merge_once(base, sel, SP.guarded(make_detect(specs[arms[0]][0]["scale"]), W, H),
                                  lambda full: lift(full, hist.get("scene_unc")))
            hist["scene_unc"] = {"xyxy": sh["xyxy"]}
            score_all("uncertainty", sh)
            rec["px_uncertainty"] = (cheap_px + tpx[0]) / NATIVE
            rec["px_u960"] = F.net_h * F.net_w / NATIVE
            prev_act = a_c
            rows.append(rec)
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{ui}] {unit}: {len(rows)} frames", flush=True)
    print("K23 DEV DONE", flush=True)


if __name__ == "__main__":
    main()
