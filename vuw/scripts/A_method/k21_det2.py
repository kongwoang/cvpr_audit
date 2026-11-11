#!/usr/bin/env python
"""K21(2) — a second detector family with the frozen procedure, no re-tuning (post-hoc analysis): YOLO11s (COCO, not a
YOLOv8 variant) as BOTH the cheap draft (long side 320) and the target detector (whole frame and crops). nuScenes new
test (draft computed online here, lifted like every other pass) and AV2 val (ego mask via k10_masked_run.py).
Per frame: action / decision cost of cheap, u640, u960, native, ours without refiner, ours with the existing refiner
(v1 nuScenes, v2 AV2; applied as is — the transfer check of the K5 follow-up), tensor px.
usage: k21_det2.py --dataset nusc|av2 [--weights yolo11s.pt] [--shard i/n]
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

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("ie", B / "scripts" / "A_method" / "instr_eval.py")
IE = importlib.util.module_from_spec(_s); _s.loader.exec_module(IE)
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

CLS = {"vehicle": 0, "person": 1, "cyclist": 2}


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2"])
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    ds = args.dataset
    si, sn = (int(x) for x in args.shard.split("/"))
    sys.path.insert(0, str(B / "scripts" / "common"))
    import train_refiner_remote as TR
    from nusc_refiner_crop import crop64
    dev = torch.device("cuda:0")
    det = TwoFidelityDetector(str(B / "models" / args.weights), device="cuda:0", backend="torch")
    ck = torch.load(B / "out" / ("resnet18" if ds == "nusc" else "resnet18_v2") / "image_refiner.pt", map_location=dev,
                    weights_only=False)
    rnet = TR.Res18(len(ck["fmu"])); rnet.load_state_dict(ck["state"]); rnet = rnet.to(dev).eval()
    fmu, fsd = ck["fmu"], ck["fsd"]
    pp, cp = P.PlannerParams(), P.CostParams()
    if ds == "nusc":
        VC, SCALE, CH = 10.0, 0.6, Mode("ns_c320", 192, 320)
        M = {"u640": Mode("f384x640", 384, 640), "u960": Mode("f576x960", 576, 960), "native": Mode("ns_u1600", 928, 1600)}
        units = IE.units_nusc("test")
    else:
        VC, SCALE, CH = 0.0, 960 / 2048, Mode("av2_c320", 320, 256)
        M = {"u640": Mode("av2_f640", 640, 512), "u960": Mode("av2_f960", 960, 768), "native": Mode("av2_u2048", 2048, 1568)}
        units = IE.units_av2("test")
    out = B / "results" / "k21_det2" / Path(args.weights).stem
    out.mkdir(parents=True, exist_ok=True)
    for ui, (unit, cam, frames) in enumerate(units):
        if ui % sn != si:
            continue
        fo = out / f"{ds}__{unit}.pkl"
        if fo.exists():
            continue
        R, t = cam.transform("ego")
        hist, prev_act, prev_c, rows = {}, None, None, []
        for fr, path, v, g, _ in frames():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            NATIVE = W * H
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
            rec = {"unit": unit, "frame": fr, "act_gt": P.discrete_action(a_gt, pp)}

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
                return d

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
            ch = keep(det.detect(img, CH)[0]); ch["mono"] = lift(ch, prev_c)
            prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            a_c = act(ch["mono"])
            rec.update({"a_cheap": a_c, "J_cheap": cost(a_c), "px_cheap": CH.net_h * CH.net_w / NATIVE})
            for nm, m in M.items():
                d = keep(det.detect(img, m)[0])
                a = act(lift(d, hist.get(nm))); hist[nm] = d
                rec.update({f"a_{nm}": a, f"J_{nm}": cost(a), f"px_{nm}": m.net_h * m.net_w / NATIVE})
            tpx[0] = 0.0
            sc, crops, rounds = SP.run({k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}, v, cam, W, H, detect_crop,
                                       lambda full: lift(full, hist.get("scene")), pp, v_close=VC, guard=True)
            a, ar = act(sc["mono"]), act(refine(sc))
            rec.update({"a_full": a, "J_full": cost(a), "a_full_ref": ar, "J_full_ref": cost(ar),
                        "px_full": (CH.net_h * CH.net_w + tpx[0]) / NATIVE, "crops_full": len(crops), "rounds_full": rounds})
            hist["scene"] = {"xyxy": sc["xyxy"]}
            prev_act = a_c
            rows.append(rec)
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{ui}] {unit}: {len(rows)} frames", flush=True)
    print("K21 DET2 DONE", flush=True)


if __name__ == "__main__":
    main()
