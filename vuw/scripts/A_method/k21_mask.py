#!/usr/bin/env python
"""K21(3) — ego-mask sensitivity on AV2 val (post-hoc analysis). The K10 mask removes from the decision every detection
whose box lies at least a fraction c inside the full-width band of rows y >= y0 (rap.mono.predicted_geometry returns
+inf range / ttc for it, exactly as k10_masked_run.py). Here y0 in {1780, 1830, 1880} x c in {0.3, 0.5, 0.7}, all nine
settings in one pass (detections of the draft / whole-frame passes and of identical crops are shared across settings;
every setting keeps its own lift history). Per frame and setting: action and decision cost of cheap, u960, native,
uncertainty crops at ours' crop area (instr_eval's rule), ours without / with refiner v2; tensor px of ours.
Run WITHOUT k10_masked_run.py (the mask is applied here).
usage: k21_mask.py [--shard i/n]  ->  results/k21_mask/av2__<log>.pkl
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

import rap.mono as RM

_ORIG = RM.predicted_geometry
MASK = {"y0": None, "c": 0.5}


def masked_geometry(dets, *a, **k):
    g = _ORIG(dets, *a, **k)
    if MASK["y0"] is None or not len(dets["xyxy"]):
        return g
    b = np.asarray(dets["xyxy"], float).reshape(-1, 4)
    h = np.maximum(b[:, 3] - b[:, 1], 1e-6)
    m = (b[:, 3] - np.maximum(b[:, 1], MASK["y0"])).clip(0, None) / h >= MASK["c"]
    if m.any():
        g = dict(g)
        for key in ("z", "z_ground", "z_height", "ttc"):
            if key in g:
                arr = np.asarray(g[key], float).copy(); arr[m] = np.inf; g[key] = arr
    return g


RM.predicted_geometry = masked_geometry
B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("ie", B / "scripts" / "A_method" / "instr_eval.py")
IE = importlib.util.module_from_spec(_s); _s.loader.exec_module(IE)
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
SETTINGS = [(y0, c) for y0 in (1780, 1830, 1880) for c in (0.3, 0.5, 0.7)]


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))
    sys.path.insert(0, str(B / "scripts" / "common"))
    import train_refiner_remote as TR
    from nusc_refiner_crop import crop64
    dev = torch.device("cuda:0")
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    ck = torch.load(B / "out" / "resnet18_v2" / "image_refiner.pt", map_location=dev, weights_only=False)
    rnet = TR.Res18(len(ck["fmu"])); rnet.load_state_dict(ck["state"]); rnet = rnet.to(dev).eval()
    fmu, fsd = ck["fmu"], ck["fsd"]
    pp, cp = P.PlannerParams(), P.CostParams()
    VC, SCALE, CH = 0.0, 960 / 2048, Mode("av2_c320", 320, 256)
    M = {"u960": Mode("av2_f960", 960, 768), "native": Mode("av2_u2048", 2048, 1568)}
    out = B / "results" / "k21_mask"
    out.mkdir(parents=True, exist_ok=True)
    for ui, (unit, cam, frames) in enumerate(IE.units_av2("test")):
        if ui % sn != si:
            continue
        fo = out / f"av2__{unit}.pkl"
        if fo.exists():
            continue
        R, t = cam.transform("ego")
        hist = {s: {} for s in SETTINGS}
        prev = {s: {"act": None, "c": None} for s in SETTINGS}
        rows = []
        for fr, path, v, g, _ in frames():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            NATIVE = W * H
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
            rec = {"unit": unit, "frame": fr, "act_gt": P.discrete_action(a_gt, pp)}

            def keep(d):
                k = d["conf"] >= IE.THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k],
                        "entropy": np.asarray(d.get("entropy", np.zeros(len(d["conf"]))))[k]}
            dch = keep(det.detect(img, CH)[0])
            dwf = {nm: keep(det.detect(img, m)[0]) for nm, m in M.items()}
            cache = {}
            tpx = [0.0]

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * SCALE / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * SCALE / 32)) * 32)
                tpx[0] += nh * nw
                key = (x1, y1, x2, y2)
                if key not in cache:
                    d = keep(det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))[0])
                    d["xyxy"] = d["xyxy"] + np.array([x1, y1, x1, y1], float)
                    cache[key] = {k: d[k] for k in ("xyxy", "conf", "coarse")}
                return {k: x.copy() for k, x in cache[key].items()}
            for (y0, c) in SETTINGS:
                MASK["y0"], MASK["c"] = y0, c
                hs, ps = hist[(y0, c)], prev[(y0, c)]
                tag = f"{y0}_{c:g}"

                def lift(dd, pv):
                    gk = RM.predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, cam.dt,
                                               cam_to_ego=(R, t))
                    return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

                def act(tt):
                    return P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)

                def cost(a):
                    return P.decision_cost(a, a_gt, ps["act"], pp, cp)["J"]

                def refine(scene):
                    refined = [np.asarray(a, float).copy() for a in scene["mono"]]
                    for p_ in SP.support(scene["mono"], v, pp):
                        if not scene["from_full"][p_]:
                            continue
                        x1, y1, x2, y2 = scene["xyxy"][p_]
                        zz = RM.predicted_geometry({"xyxy": scene["xyxy"][p_:p_ + 1], "coarse": scene["coarse"][p_:p_ + 1]},
                                                   None, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
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
                ch = dict(dch); ch["mono"] = lift(ch, ps["c"])
                a_c = act(ch["mono"])
                rec.update({f"a_cheap_{tag}": a_c, f"J_cheap_{tag}": cost(a_c)})
                for nm, d in dwf.items():
                    a = act(lift(d, hs.get(nm))); hs[nm] = d
                    rec.update({f"a_{nm}_{tag}": a, f"J_{nm}_{tag}": cost(a)})
                base = {k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}
                tpx[0] = 0.0
                sc, crops, _ = SP.run(base, v, cam, W, H, detect_crop, lambda full: lift(full, hs.get("scene")), pp,
                                      v_close=VC, guard=True)
                a, ar = act(sc["mono"]), act(refine(sc))
                rec.update({f"a_full_{tag}": a, f"J_full_{tag}": cost(a), f"a_full_ref_{tag}": ar, f"J_full_ref_{tag}": cost(ar),
                            f"px_full_{tag}": (CH.net_h * CH.net_w + tpx[0]) / NATIVE})
                hs["scene"] = {"xyxy": sc["xyxy"]}
                budget = sum(IE.area(q) for q in crops)
                order = np.lexsort((-ch["entropy"], ch["conf"]))
                sel = IE.fill_strict([SP.box_crop(ch["xyxy"][k], W, H) for k in order], budget)
                sh, _ = IE.merge_once(base, sel, SP.guarded(detect_crop, W, H), lambda full: lift(full, hs.get("scene_unc")))
                a = act(sh["mono"])
                rec.update({f"a_uncertainty_{tag}": a, f"J_uncertainty_{tag}": cost(a)})
                hs["scene_unc"] = {"xyxy": sh["xyxy"]}
                ps["act"], ps["c"] = a_c, {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            rows.append(rec)
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{ui}] {unit}: {len(rows)} frames", flush=True)
    print("K21 MASK DONE", flush=True)


if __name__ == "__main__":
    main()
