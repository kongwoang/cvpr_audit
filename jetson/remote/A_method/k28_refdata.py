#!/usr/bin/env python
"""K28 family B — refiner training data with a stronger verifier <det> (method_freeze_draft §2j; development /
training data only, no dev or test unit):
  --src av2   AV2 REF-TRAIN / REF-VAL logs (results/k23/av2_split.csv), the k23_refdata recipe with <det> in the crops
              and in u960 (the draft stays YOLOv8s at 320; the frozen v4-lite loop speculative.run, v_close 0, guard;
              every frame run, every 2nd frame sampled; samples within 80 m and 2 m of the ego corridor)
              -> data/refiner_data__AV2ref-<det>_<i>of<n>.npz
  --src nusc  the nuScenesAll TRAIN scenes of v2 / v3a (the scene list read from data/refiner_data__nuScenesAll.npz,
              split == train), the refiner_data_v2 recipe: whole frame 384x640 with <det>, detections (conf >= 0.25)
              matched to a projected annotation (IoU >= 0.3), 64x64 crop, 12 box features, target log(z_true / z_lift)
              -> data/refiner_data__nuScenesTrain-<det>.npz
usage: k28_refdata.py --det yolov8m|yolov8l|yolo11m --src av2|nusc [--shard i/n]   (av2 through k10_masked_run.py)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--src", required=True, choices=["av2", "nusc"])
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    _s = importlib.util.spec_from_file_location("ie", B / "scripts" / "A_method" / "instr_eval.py")
    IE = importlib.util.module_from_spec(_s); _s.loader.exec_module(IE)
    from rap import planner as P
    from rap.detect import Mode, TwoFidelityDetector
    from rap.mono import box_iou, predicted_geometry
    from vtf import speculative as SP
    sys.path.insert(0, str(B / "scripts" / "common"))
    from nusc_refiner_crop import crop64
    si, sn = (int(x) for x in args.shard.split("/"))
    pp = P.PlannerParams()
    vdet = TwoFidelityDetector(str(B / "models" / f"{args.det}.pt"), device="cuda:0", backend="torch")
    out = REAL / "data"

    def keep(d):
        k = d["conf"] >= IE.THR
        return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}
    if args.src == "nusc":
        from rap.cache import Camera
        from rap.nusc import NuScenesDB, make_adapter
        from rap.paths import NUSCENES_TRAINVAL
        z0 = np.load(REAL / "data" / "refiner_data__nuScenesAll.npz", allow_pickle=True)
        scenes = sorted(set(z0["seq"][z0["split"] == "train"].tolist()))
        del z0
        db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
        ad = make_adapter(db)
        by = {db.scene_name(t): t for t in db.scenes}
        F = Mode("f384x640", 384, 640)
        crops, feats, target, seq, frame = [], [], [], [], []
        for s in scenes:
            cam = Camera("nuScenes", s)
            R, t = cam.transform("ego")
            geom = ad.geometry(s)
            toks = db.samples(by[s])
            for fr in range(len(toks)):
                g = geom[geom["frame"] == fr]
                g = g[(g["y2"] - g["y1"]) >= 10]
                if not len(g):
                    continue
                img = cv2.imread(str(db.image_path(toks[fr])))
                if img is None:
                    continue
                H, W = img.shape[:2]
                d = keep(vdet.detect(img, F)[0])
                if not len(d["conf"]):
                    continue
                iou = box_iou(d["xyxy"], np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float))
                hit = np.flatnonzero(iou.max(1) >= 0.3)
                if not len(hit):
                    continue
                zz = predicted_geometry({"xyxy": d["xyxy"], "coarse": d["coarse"]}, None, cam.calib, cam.cam_h, float(cam.dt),
                                        cam_to_ego=(R, t))
                for o in hit:
                    zt = float(np.asarray(g["long_near"])[int(np.argmax(iou[o]))])
                    z = float(zz["z"][o])
                    if zt <= 0.5 or z <= 0.5:
                        continue
                    x1, y1, x2, y2 = d["xyxy"][o]
                    zg, zh = float(zz["z_ground"][o]), float(zz["z_height"][o])
                    crops.append(crop64(img, d["xyxy"][o]))
                    feats.append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(d["conf"][o]),
                                  float(CLS[str(d["coarse"][o])]), z, zg, zh, float(np.log(max(zg, .5) / max(zh, .5)))])
                    target.append(float(np.log(zt / z))); seq.append(s); frame.append(fr)
            print(f"  {s}: {len(target)} samples so far", flush=True)
        np.savez_compressed(out / f"refiner_data__nuScenesTrain-{args.det}.npz", crops=np.array(crops, np.uint8).reshape(-1, 64, 64, 3),
                            feats=np.array(feats, np.float32).reshape(-1, 12), target=np.array(target, np.float32),
                            split=np.full(len(target), "train"), seq=np.array(seq), frame=np.array(frame))
        print("K28 REFDATA NUSC DONE", len(target), flush=True)
        return
    from vtf import av2 as AV
    sp = pd.read_csv(REAL / "results" / "k23" / "av2_split.csv")
    roles = dict(zip(sp.log, sp.role))
    logs = sorted(l for l, r in roles.items() if r in ("REF-TRAIN", "REF-VAL"))
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    CH, F, SCALE = Mode("av2_c320", 320, 256), Mode("av2_f960", 960, 768), 960 / 2048
    root = B / "datasets" / "av2" / "sensor" / "train"
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

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * SCALE / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * SCALE / 32)) * 32)
                d = keep(vdet.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))[0])
                d["xyxy"] = d["xyxy"] + np.array([x1, y1, x1, y1], float)
                return d
            ch = keep(det.detect(img, CH)[0]); ch["mono"] = lift(ch, prev_c)
            prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            sc, crops, _ = SP.run({k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}, v, L, W, H, detect_crop,
                                  lambda full: lift(full, hist.get("scene")), pp, v_close=0.0, guard=True)
            hist["scene"] = {"xyxy": sc["xyxy"]}
            d9 = keep(vdet.detect(img, F)[0]); t9 = lift(d9, hist.get("u960")); hist["u960"] = d9
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
    np.savez_compressed(out / f"refiner_data__AV2ref-{args.det}_{si}of{sn}.npz", crops=np.array(acc["crops"], np.uint8).reshape(-1, 64, 64, 3),
                        feats=np.array(acc["feats"], np.float32).reshape(-1, 12), target=np.array(acc["target"], np.float32),
                        target_lat=np.array(acc["target_lat"], np.float32), valid=np.array(acc["valid"], np.float32),
                        split=np.array(acc["split"]), seq=np.array(acc["seq"]), kind=np.array(acc["kind"]))
    print("K28 REFDATA AV2 DONE", len(acc["target"]), flush=True)


if __name__ == "__main__":
    main()
