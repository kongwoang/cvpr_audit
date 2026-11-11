#!/usr/bin/env python
"""Per-frame wall-clock of every baseline on the Jetson AGX Xavier (KITTI frames 1242x375, image decode excluded,
CUDA synchronised, first 20 frames warm-up). Two backends, reported separately:
  TensorRT FP16 (the deployed engines): cheap 96x320, u960 288x960, v3 (crops in the 96x288 engine, one crop per call),
      crop heuristics (same machinery, v3's budget), FOVEA KDE_I (warp on GPU + 288x960 engine + unwarp)
  torch FP16 (ultralytics; baselines whose computation has no fixed-shape engine): u960 (for the ratio), SAHI (640
      slices), UFP (official packing + one pass on the mosaic, S = 640 / 960), LZU (official zoom/unzoom around
      YOLOv8s at 288x960), the bgt-ada skipping router (MobileNetV2-Lite 128 px) alone
ESOD needs its own YOLOv5 code base (VUW env): its compute is reported from its measure task (GFLOPs, A5000 FPS).
Writes results/raw/<run>/latency_baselines.csv.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".pydeps"))
os.environ.setdefault("B", str(ROOT))                   # lzu_yolo / ufp look for third_party under $B
_s = importlib.util.spec_from_file_location("kbc", ROOT / "scripts" / "50_kitti_baseline_cells.py")
K = importlib.util.module_from_spec(_s)
_s.loader.exec_module(K)
E, P, SP, runmeta = K.E, K.P, K.SP, K.runmeta
from rap.cache import Camera                                                     # noqa: E402
from rap.decision import KittiAdapter                                            # noqa: E402
from rap.detect import MODES, Mode, TwoFidelityDetector                          # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from rap.paths import KITTI_IMAGES, MODELS                                       # noqa: E402
from vtf.fastpost import FastDetector                                            # noqa: E402

N_FRAMES, WARM = 220, 20


def sync():
    torch.cuda.synchronize()


def timed(fn, *a):
    sync(); t0 = time.perf_counter()
    out = fn(*a)
    sync()
    return out, 1e3 * (time.perf_counter() - t0)


def main():
    run = runmeta.new_run("baseline_latency", {"frames": N_FRAMES, "warm": WARM})
    w = str(Path(MODELS) / "yolov8s.pt")
    base = FastDetector(w, backend="trt", engine_dir=str(Path(MODELS) / "engines"), engine_stem="yolov8s")
    crop = FastDetector(w, backend="trt", engine_dir=str(runmeta.ROOT / "engines"), engine_stem="yolov8s_crop")
    m_cheap, m_960, m_canvas = MODES["cheap_320"], Mode("288x960", 288, 960), Mode("96x288", 96, 288)
    tdet = TwoFidelityDetector(w, device="cuda:0", backend="torch")          # torch FP16 reference
    sys.path.insert(0, str(ROOT / "remote" / "common"))
    import lzu_yolo
    ldet = TwoFidelityDetector(w, device="cuda:0", backend="torch")
    lzu_yolo.patch(ldet.net, torch.load(ROOT / "third_party" / "lzu" / "saliency_27x48.pt"))
    sys.path.insert(0, str(ROOT / "third_party" / "UFPMP-Det-Tools"))
    # the packing glue of remote/A_method/ufp_baseline.py (display_merge_result) is inlined in ufp() below
    from ufp import UnifiedForegroundPacking
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    sahi = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=w, confidence_threshold=0.25,
                                              device="cuda:0", image_size=640)
    import torchvision
    import torch.nn as nn
    mb = torchvision.models.mobilenet_v2()
    mb.features = mb.features[:14]
    with torch.no_grad():
        oc = mb.features(torch.zeros(1, 3, 128, 128)).shape[1]
    router = nn.Sequential(mb.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(oc, 1)).cuda().half().eval()
    gg = K.fovea_gg()
    pp = P.PlannerParams()
    s = "0000"
    geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s)
    cam = Camera("KITTI", s)
    R, t = cam.transform("ego")
    frames = sorted(int(p.stem) for p in (Path(KITTI_IMAGES) / s).glob("*.png"))[:N_FRAMES]
    gen = None
    T = {}
    prev_scene = None
    for i, fr in enumerate(frames):
        img = cv2.imread(str(Path(KITTI_IMAGES) / s / f"{fr:06d}.png"))
        H, W = img.shape[:2]
        v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
        if gen is None:
            gen = gg.PlainKDEGrid(output_shape=(288, 960), grid_shape=(31, 51), separable=True, attraction_fwhm=4,
                                  anti_crop=True, bandwidth_scale=64, amplitude_scale=1)
        rec = {}

        def lift(xyxy, coarse, pv):
            gk = predicted_geometry({"xyxy": xyxy, "coarse": coarse}, pv, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
            return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

        def cheap_pass():
            d, _ = base.detect(img, m_cheap)
            k = d["conf"] >= 0.25
            dc = {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k],
                  "entropy": d["entropy"][k]}
            dc["mono"] = lift(dc["xyxy"], dc["coarse"], None)
            return dc
        dc, rec["cheap_trt"] = timed(cheap_pass)

        def u960():
            d = crop.detect(img, m_960)[0]
            return lift(d["xyxy"].astype(float), d["coarse"], None)
        _, rec["u960_trt"] = timed(u960)

        def detect_crop(cr):
            x1, y1 = int(cr[0]), int(cr[1])
            x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
            dk = crop.detect(img[y1:y2, x1:x2], m_canvas)[0]
            kk = dk["conf"] >= 0.25
            return {"xyxy": dk["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float), "conf": dk["conf"][kk].astype(float),
                    "coarse": dk["coarse"][kk]}

        def v3():
            c = cheap_pass()
            return SP.run({k: c[k] for k in ("xyxy", "conf", "coarse", "mono")}, v, cam, W, H, detect_crop,
                          lambda full: lift(full["xyxy"], full["coarse"], prev_scene), pp, v_close=10.0, max_rounds=3, guard=True)
        (scene, crops, _), rec["v3_trt"] = timed(v3)
        prev_scene = {"xyxy": scene["xyxy"]}
        budget = sum(K.area(c) for c in crops)

        def uncertainty():
            c = cheap_pass()
            order = np.lexsort((-c["entropy"], c["conf"]))
            cr_ = K.fill_strict([SP.box_crop(c["xyxy"][k], W, H) for k in order], budget)
            return K.merge_once(c, cr_, SP.guarded(detect_crop, W, H), lambda full: lift(full["xyxy"], full["coarse"], None))
        _, rec["uncertainty_crops_trt"] = timed(uncertainty)

        def fovea():
            c = cheap_pass()
            x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().cuda()
            with torch.no_grad():
                grid = gen(x, [{"pad_shape": (H, W, 3)}], [torch.tensor(c["xyxy"], dtype=torch.float32, device="cuda").reshape(-1, 4)])
                wimg = F.grid_sample(x, grid, align_corners=True)[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()[:, :, ::-1].copy()
            d = crop.detect(wimg, m_960)[0]
            bb = gg.unwarp_bboxes(torch.tensor(d["xyxy"], dtype=torch.float32, device="cuda"), grid[0], (H, W)).cpu().numpy() \
                if len(d["conf"]) else np.zeros((0, 4))
            return lift(bb.astype(float), d["coarse"], None)
        _, rec["fovea_trt"] = timed(fovea)
        _, rec["u960_torch"] = timed(lambda: tdet.detect(img, m_960))
        _, rec["sahi_torch"] = timed(lambda: get_sliced_prediction(img[:, :, ::-1].copy(), sahi, slice_height=640, slice_width=640,
                                                                   overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                                                                   perform_standard_pred=True, verbose=0))

        def lzu():
            x = cv2.resize(img, (960, 288), interpolation=cv2.INTER_LINEAR)
            xt = torch.from_numpy(np.ascontiguousarray(x[:, :, ::-1].transpose(2, 0, 1))).cuda().half()[None] / 255.0
            with torch.no_grad():
                return ldet.postprocess(ldet.forward(xt), Mode("s288x960", 288, 960), 1.0, (0, 0), (288, 960))
        _, rec["lzu_torch"] = timed(lzu)

        def ufp(S):
            c = cheap_pass()
            allb = base.detect(img, m_cheap)[0]["xyxy"].astype(float)
            if not len(allb):
                return None
            rec_, w_, h_ = UnifiedForegroundPacking(allb.copy(), 1.5, input_shape=[W, H])
            if not rec_:
                return None
            mosaic = np.zeros((math.ceil(h_), math.ceil(w_), 3), np.uint8)
            for r_ in rec_:
                x1, y1, cw, ch, nx, ny, sf = [math.floor(q) for q in r_]
                if cw and ch:
                    mosaic[ny:ny + ch * sf, nx:nx + cw * sf] = cv2.resize(img[y1:y1 + ch, x1:x1 + cw], (cw * sf, ch * sf))
            mh, mw = mosaic.shape[:2]
            r = S / max(mh, mw)
            mode = Mode("m", int(math.ceil(mh * r / 32) * 32), int(math.ceil(mw * r / 32) * 32))
            return tdet.detect(mosaic, mode)
        _, rec["ufp640_torch"] = timed(ufp, 640)
        _, rec["ufp960_torch"] = timed(ufp, 960)

        def route():
            x = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
            xt = torch.from_numpy(np.ascontiguousarray(x[:, :, ::-1].transpose(2, 0, 1))).cuda().half()[None] / 255.0
            with torch.no_grad():
                return router(xt)
        _, rec["router_mbv2lite_torch"] = timed(route)
        rec["v3_crops"], rec["v3_px"] = len(crops), budget / (W * H)
        if i >= WARM:
            for k, v_ in rec.items():
                T.setdefault(k, []).append(v_)
    out = pd.DataFrame([{"pipeline": k, "mean_ms": float(np.mean(v_)), "p95_ms": float(np.percentile(v_, 95)),
                         "p99_ms": float(np.percentile(v_, 99))} for k, v_ in T.items()])
    out.to_csv(run / "latency_baselines.csv", index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
