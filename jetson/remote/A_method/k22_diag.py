#!/usr/bin/env python
"""K22(3) — diagnostic only (post-hoc analysis, no new configuration): K17's group d on AV2 val (ego mask) and
Argoverse-HD val: frames where FULL and SUPPORT ONLY act differently and the field-crop detection that matters matches
an annotated but NON-relevant object. Each unit containing such frames is re-run from its start with the frozen loop
(k_loop.run_g, identical to speculative.run), support only, F = u960 and the refiner (v2), histories as in K17.
On every group-d frame (K17's field_cat == "d"):
  decisive object   the object with the largest required deceleration in ours' final scene (with refiner) when ours'
                    action is above support-only's; whether it comes from a crop and is absent from support-only's scene
  annotated match   the annotated object with the largest IoU (>= 0.3, any class) with it: annotated range, lateral
                    extent, ttc, class, and its own required deceleration in the annotated scene
  errors            range (estimated - annotated, and ratio), lateral (inside the corridor in ours, outside in the
                    annotation), closing speed (range / ttc), class (incompatible coarse classes)
  attribution       which single substitution of the annotated value (range, lateral extent, ttc) for ours' estimate
                    brings ours' action back down (one object, everything else unchanged)
  F                 whether u960 has a detection of the same object (IoU >= 0.5) that is decisive for F's action
usage: k22_diag.py --dataset av2|ahd [--shard i/n]   (AV2 through k10_masked_run.py)
"""
from __future__ import annotations

import argparse
import glob
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
_h = importlib.util.spec_from_file_location("hood", B / "scripts" / "A_method" / "k10_hood.py")
HOOD = importlib.util.module_from_spec(_h); _h.loader.exec_module(HOOD)
_l = importlib.util.spec_from_file_location("kl", B / "scripts" / "A_method" / "k_loop.py")
KL = importlib.util.module_from_spec(_l); _l.loader.exec_module(KL)
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import box_iou, predicted_geometry                                 # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
COMPAT = {"vehicle": {"vehicle"}, "person": {"person", "cyclist"}, "cyclist": {"cyclist", "person"}}


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["av2", "ahd"])
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    ds = args.dataset
    si, sn = (int(x) for x in args.shard.split("/"))
    want = {}
    for f in glob.glob(str(B / "results" / "k17" / f"{ds}_test_base" / "*.pkl")):
        for r in pickle.load(open(f, "rb")):
            if r.get("field_cat") == "d":
                want.setdefault(r["unit"], set()).add(r["frame"])
    sys.path.insert(0, str(B / "scripts" / "common"))
    import train_refiner_remote as TR
    from nusc_refiner_crop import crop64
    dev = torch.device("cuda:0")
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    ck = torch.load(B / "out" / "resnet18_v2" / "image_refiner.pt", map_location=dev, weights_only=False)
    rnet = TR.Res18(len(ck["fmu"])); rnet.load_state_dict(ck["state"]); rnet = rnet.to(dev).eval()
    fmu, fsd = ck["fmu"], ck["fsd"]
    pp = P.PlannerParams()
    if ds == "av2":
        SCALE, CH, F = 960 / 2048, Mode("av2_c320", 320, 256), Mode("av2_f960", 960, 768)
        units = IE.units_av2("test")
    else:
        SCALE, CH, F = 960 / 1920, Mode("ahd_c320", 224, 320), Mode("ahd_u960", 608, 960)
        units = HOOD.units("ahd")
    out = B / "results" / "k22_diag"
    out.mkdir(parents=True, exist_ok=True)
    for ui, (unit, cam, gen) in enumerate(units):
        if unit not in want or ui % sn != si:
            continue
        fo = out / f"{ds}__{unit}.pkl"
        if fo.exists():
            continue
        R, t = cam.transform("ego")
        hist, prev_c, rows = {}, None, []
        last = max(want[unit])
        for fr, path, v, g, extra in gen():
            if fr > last:
                break
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]

            def lift(dd, pv):
                gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, cam.dt,
                                        cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def keep(d):
                k = d["conf"] >= IE.THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}

            def act(tt):
                return P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * SCALE / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * SCALE / 32)) * 32)
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
            base = {k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}
            d9 = keep(det.detect(img, F)[0]); t9 = lift(d9, hist.get("u960")); hist["u960"] = d9
            sf, _, _, _, _ = KL.run_g(base, v, cam, W, H, detect_crop, lambda full: lift(full, hist.get("scene_full")), pp, v_closes=(0.0,))
            so, _, _, _, _ = KL.run_g(base, v, cam, W, H, detect_crop, lambda full: lift(full, hist.get("scene_so")), pp, field=False)
            hist["scene_full"], hist["scene_so"] = {"xyxy": sf["xyxy"]}, {"xyxy": so["xyxy"]}
            if fr not in want[unit]:
                continue
            tf, ts = refine(sf), refine(so)
            af, aso, a9 = act(tf), act(ts), act(t9)
            rec = {"unit": unit, "frame": fr, "v": v, "a_full_ref": af, "a_so_ref": aso, "a_u960": a9}
            _, dem = P.required_decel(*tf, v, pp)
            dem = np.asarray(dem, float)
            if af > aso and len(dem):
                j = int(np.argmax(dem))
                rec.update({"dec_from_crop": bool(sf["from_full"][j]),
                            "dec_new_vs_so": bool(not len(so["xyxy"]) or box_iou(sf["xyxy"][j:j + 1], so["xyxy"]).max() < 0.5),
                            "est_z": tf[0][j], "est_lmin": tf[1][j], "est_lmax": tf[2][j], "est_ttc": tf[3][j],
                            "est_cls": str(sf["coarse"][j]), "est_conf": float(sf["conf"][j]), "est_demand": dem[j],
                            "est_in_corridor": bool(P.in_corridor(np.array([tf[1][j]]), np.array([tf[2][j]]), pp.corridor_half_w)[0])})
                if len(g):
                    gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
                    iou = box_iou(sf["xyxy"][j:j + 1], gtb)[0]
                    k = int(np.argmax(iou))
                    _, gdem = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
                    if iou[k] >= 0.3:
                        gz, glo, ghi, gtt = (float(np.asarray(g[c])[k]) for c in ("long_near", "lat_min", "lat_max", "ttc"))
                        gcl = str(np.asarray(g["coarse"])[k])
                        rec.update({"gt_iou": float(iou[k]), "gt_z": gz, "gt_lmin": glo, "gt_lmax": ghi, "gt_ttc": gtt, "gt_cls": gcl,
                                    "gt_demand": float(np.asarray(gdem)[k]),
                                    "gt_in_corridor": bool(P.in_corridor(np.array([glo]), np.array([ghi]), pp.corridor_half_w)[0]),
                                    "class_mismatch": gcl not in COMPAT.get(rec["est_cls"], {rec["est_cls"]})})
                        for nm, idx_, val in (("range", 0, gz), ("lateral", (1, 2), (glo, ghi)), ("ttc", 3, gtt)):
                            tt = [np.asarray(a, float).copy() for a in tf]
                            if isinstance(idx_, tuple):
                                tt[1][j], tt[2][j] = val
                            else:
                                tt[idx_][j] = val
                            rec[f"fix_{nm}"] = act(tuple(tt)) < af
                if len(d9["xyxy"]):
                    io9 = box_iou(sf["xyxy"][j:j + 1], d9["xyxy"])[0]
                    m9 = int(np.argmax(io9))
                    _, dem9 = P.required_decel(*t9, v, pp)
                    dem9 = np.asarray(dem9, float)
                    rec.update({"F_has_object": bool(io9[m9] >= 0.5), "F_object_decisive": bool(io9[m9] >= 0.5 and len(dem9) and
                                                                                                int(np.argmax(dem9)) == m9 and a9 >= af),
                                "F_object_demand": float(dem9[m9]) if io9[m9] >= 0.5 else np.nan})
            rows.append(rec)
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{ui}] {unit}: {len(rows)} group-d frames", flush=True)
    print("K22 DIAG DONE", flush=True)


if __name__ == "__main__":
    main()
