#!/usr/bin/env python
"""K23 step 1 — AV2 development split and refiner-v3 training data (development data only; method_freeze_draft §2e).
  --split    writes results/k23/av2_split.csv: the 700 AV2 sensor-train logs, sorted, permuted with
             numpy.random.default_rng(20261010): first 300 CFG-DEV, next 340 REF-TRAIN, last 60 REF-VAL
  (default)  for REF-TRAIN / REF-VAL logs (every 2nd frame): the frozen v4-lite loop (ego mask through
             k10_masked_run.py) and u960; every detection of a crop (after the guard) and of u960 whose estimated
             position lies within 80 m and within 2 m of the ego corridor (|lateral| <= half width + 2 m) becomes a
             sample: 64x64 crop (crop64, as v2), the 13 box features of the frozen refiner (as instr_eval.refine; the
             estimated range is the lifted range of the box in its own scene), and targets
               target_logr   log(annotated range / estimated range) for detections matched to an annotation
                             (class-compatible IoU >= 0.3, Hungarian), NaN otherwise
               target_lat    annotated minus estimated lateral centre (m), matched only
               valid         1 if matched AND the annotated object is inside the ego corridor within 80 m, else 0
             -> data/refiner_data__AV2dev_<shard>.npz (crops, feats, target, target_lat, valid, split, seq, kind)
usage: k23_refdata.py --split | k23_refdata.py [--shard i/n]   (data runs through k10_masked_run.py, B = shadow)
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}


def split():
    logs = sorted(p.stem for p in (REAL / "datasets" / "av2" / "lists" / "train").glob("*.done"))
    perm = np.random.default_rng(20261010).permutation(len(logs))
    role = np.empty(len(logs), object)
    role[perm[:300]], role[perm[300:640]], role[perm[640:]] = "CFG-DEV", "REF-TRAIN", "REF-VAL"
    out = REAL / "results" / "k23"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"log": logs, "role": role}).to_csv(out / "av2_split.csv", index=False)
    print(pd.Series(role).value_counts().to_dict(), "->", out / "av2_split.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    if args.split:
        return split()
    import torch
    _s = importlib.util.spec_from_file_location("ie", B / "scripts" / "A_method" / "instr_eval.py")
    IE = importlib.util.module_from_spec(_s); _s.loader.exec_module(IE)
    from rap import planner as P
    from rap.detect import Mode, TwoFidelityDetector
    from rap.mono import box_iou, predicted_geometry
    from vtf import av2 as AV
    from vtf import speculative as SP
    sys.path.insert(0, str(B / "scripts" / "common"))
    from nusc_refiner_crop import crop64
    si, sn = (int(x) for x in args.shard.split("/"))
    sp = pd.read_csv(REAL / "results" / "k23" / "av2_split.csv")
    roles = dict(zip(sp.log, sp.role))
    logs = sorted(l for l, r in roles.items() if r in ("REF-TRAIN", "REF-VAL"))
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    pp = P.PlannerParams()
    CH, F, SCALE = Mode("av2_c320", 320, 256), Mode("av2_f960", 960, 768), 960 / 2048
    root = B / "datasets" / "av2" / "sensor" / "train"
    out = REAL / "data"
    acc = {k: [] for k in ("crops", "feats", "target", "target_lat", "valid", "split", "seq", "kind")}
    for li, log in enumerate(logs):
        if li % sn != si:
            continue
        L = AV.AV2Log(root, log)
        R, t = L.transform("ego")
        geom, speeds = L.geometry(), L.speeds()
        hist, prev_c = {}, None
        split_name = "train" if roles[log] == "REF-TRAIN" else "val"
        for fr in range(len(L.frame_ts)):
            img = cv2.imread(str(L.image_path(fr)))
            if img is None:
                continue
            H, W = img.shape[:2]
            v = float(speeds[fr])
            g = geom[geom["frame"] == fr] if len(geom) else geom
            g = g[(g["y2"] - g["y1"]) >= 10] if len(g) else g

            def lift(dd, pv):
                gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, L.calib, L.cam_h, L.dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def keep(d):
                k = d["conf"] >= IE.THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}
            crop_dets = []

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * SCALE / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * SCALE / 32)) * 32)
                d = keep(det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))[0])
                d["xyxy"] = d["xyxy"] + np.array([x1, y1, x1, y1], float)
                return d
            ch = keep(det.detect(img, CH)[0]); ch["mono"] = lift(ch, prev_c)
            prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            sc, crops, _ = SP.run({k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}, v, L, W, H, detect_crop,
                                  lambda full: lift(full, hist.get("scene")), pp, v_close=0.0, guard=True)
            hist["scene"] = {"xyxy": sc["xyxy"]}
            d9 = keep(det.detect(img, F)[0]); t9 = lift(d9, hist.get("u960")); hist["u960"] = d9
            if fr % 2:
                continue
            gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
            gtc = np.asarray(g["coarse"]).astype(str) if len(g) else np.zeros(0, str)
            for kind, xyxy, conf, coarse, mono in (("crop", sc["xyxy"][sc["from_full"]], sc["conf"][sc["from_full"]],
                                                    sc["coarse"][sc["from_full"]], tuple(a[sc["from_full"]] for a in sc["mono"])),
                                                   ("u960", d9["xyxy"], d9["conf"], d9["coarse"], t9)):
                if not len(xyxy):
                    continue
                z, lo, hi, _ = mono
                ok = np.isfinite(z) & (z > 0) & (z <= 80) & (hi > -pp.corridor_half_w - 2) & (lo < pp.corridor_half_w + 2)
                if not ok.any():
                    continue
                idx = np.flatnonzero(ok)
                mm = IE.matched(gtb, gtc, xyxy[idx], coarse[idx]) if len(gtb) else -np.ones(0, int)
                gt_of = -np.ones(len(idx), int)
                for gi, dj in enumerate(mm):
                    if dj >= 0:
                        gt_of[dj] = gi
                zz = predicted_geometry({"xyxy": xyxy[idx], "coarse": coarse[idx]}, None, L.calib, L.cam_h, L.dt, cam_to_ego=(R, t))
                for n, j in enumerate(idx):
                    x1, y1, x2, y2 = xyxy[j]
                    zg, zh = float(zz["z_ground"][n]), float(zz["z_height"][n])
                    acc["feats"].append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(conf[j]),
                                         float(CLS[str(coarse[j])]), float(z[j]), zg, zh, float(np.log(max(zg, .5) / max(zh, .5)))])
                    acc["crops"].append(crop64(img, xyxy[j]))
                    gi = gt_of[n]
                    if gi >= 0:
                        gz = float(np.asarray(g["long_near"])[gi]); glo, ghi = float(np.asarray(g["lat_min"])[gi]), float(np.asarray(g["lat_max"])[gi])
                        acc["target"].append(float(np.log(max(gz, 0.5) / max(z[j], 0.5))))
                        acc["target_lat"].append((glo + ghi) / 2 - (lo[j] + hi[j]) / 2)
                        acc["valid"].append(float(P.in_corridor(np.array([glo]), np.array([ghi]), pp.corridor_half_w)[0] and 0 < gz <= 80))
                    else:
                        acc["target"].append(np.nan); acc["target_lat"].append(np.nan); acc["valid"].append(0.0)
                    acc["split"].append(split_name); acc["seq"].append(log[:10]); acc["kind"].append(kind)
        print(f"  [{li}] {log} ({roles[log]}): {len(acc['target'])} samples so far", flush=True)
    np.savez_compressed(out / f"refiner_data__AV2dev_{si}of{sn}.npz", crops=np.array(acc["crops"], np.uint8).reshape(-1, 64, 64, 3),
                        feats=np.array(acc["feats"], np.float32).reshape(-1, 12), target=np.array(acc["target"], np.float32),
                        target_lat=np.array(acc["target_lat"], np.float32), valid=np.array(acc["valid"], np.float32),
                        split=np.array(acc["split"]), seq=np.array(acc["seq"]), kind=np.array(acc["kind"]))
    print("K23 REFDATA DONE", len(acc["target"]), flush=True)


if __name__ == "__main__":
    main()
