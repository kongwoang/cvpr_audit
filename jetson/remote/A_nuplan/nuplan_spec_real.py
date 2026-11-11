#!/usr/bin/env python
"""REAL-CROP variant of nuplan_spec_match.py: the spec branch runs YOLOv8s on each crop of the CAM_F0 image
(scale 1/3 = the FULL mode's 640-px resolution) with the truncation guard, instead of selecting cached FULL detections.
Output goes to $SPEC_OUT (default nuplan_real_spec_crops).

CVPR exploration: DEEP's 114 with one extra branch, `spec` (speculative perception).

Original docstring follows.
Task 5 B2-B4: nearest CAM_F0 image per state iteration, matched projection, branch tracks, checks.

Pre-registered in RESEARCH_LOG.md (2026-09-14 14:44).  For every Track B state and each of its four
history-buffer iterations:

  * the nearest fetched CAM_F0 image by timestamp (|dt| reported, > 50 ms flagged);
  * every tracked object's lidar box projected into that image: global -> ego at the image's ego pose ->
    camera by the inverse extrinsic (the devkit's boxes_lidar_to_img chain), corners clipped at
    z = 0.1 m, normalised coordinates clamped to 1.25x the undistorted image border, then the DB
    distortion (k1, k2, p1, p2, k3) and intrinsic;
  * per matching configuration, Hungarian assignment on IoU between class-compatible detections and
    projected boxes of eligible in-camera tracks (vehicle, pedestrian, bicycle);
  * per branch, the eligible in-camera tracks to remove and the false-positive agents to add (flat
    ground lift from the box bottom centre, class median size from train+val tracks, zero velocity).

Writes the branch specification for 115, a per-object table for the recall checks, the per-state
summary, the projection checks and 20 overlays.  Run with scripts/pynuplan.
"""
from __future__ import annotations

import os, argparse, json, pickle, sqlite3, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(os.environ["RAP_REPO"])
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "nuplan_devkit"))

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (   # noqa: E402
    NuPlanScenarioBuilder)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter        # noqa: E402
from nuplan.planning.utils.multithreading.worker_sequential import Sequential      # noqa: E402

from rap.cache import DetCache                                                     # noqa: E402
from rap.detector_model import NUPLAN_TO_COARSE, DetectorMissModel                 # noqa: E402
from rap.nuplan_perception import PerceptionFilter                                 # noqa: E402
from rap.paths import (CACHE, NUPLAN_CAM_F0, NUPLAN_DB, NUPLAN_MAPS,
                       NUPLAN_ROOT, RESULTS)                                      # noqa: E402

DB_DIR = NUPLAN_DB
IMG_ROOT = NUPLAN_CAM_F0
WORK = ROOT / "results" / "raw" / "nuplan_task5"
DET = Path(CACHE) / "nuplan_det"
OUTC = Path(CACHE) / os.environ.get("SPEC_OUT", "nuplan_real_spec_crops")
OVERLAYS = ROOT / "results" / "final" / "nuplan_real_perception_overlays"
BUFFER = 4
W, H = 1920, 1080
NEAR, CLAMP = 0.1, 1.25
FP_MIN_AHEAD, FP_MAX_RANGE = 0.5, 80.0
DT_FLAG_US = 50_000
ELIGIBLE = ("vehicle", "pedestrian", "bicycle")
COMPAT = {"vehicle": {"vehicle"}, "pedestrian": {"person"}, "bicycle": {"cyclist", "person"}}
DET_TO_CATEGORY = {"vehicle": "vehicle", "person": "pedestrian", "cyclist": "bicycle"}
MODE_CACHE = {"cheap": "ns_cheap_320", "full": "ns_full_640"}
# (mode, threshold key, IoU) -> matching configuration; branches are built from these
COMBOS = {"spec_025_iou30": ("spec", "0.25", 0.3), "cheap_025_iou30": ("cheap", "0.25", 0.3), "full_025_iou30": ("full", "0.25", 0.3),
          "full_s1_iou30": ("full", "s1", 0.3), "cheap_025_iou50": ("cheap", "0.25", 0.5),
          "full_025_iou50": ("full", "0.25", 0.5)}
BRANCHES = {"spec": ("spec_025_iou30", True), "cheap": ("cheap_025_iou30", True), "full": ("full_025_iou30", True),
            "cheap_nofp": ("cheap_025_iou30", False), "full_nofp": ("full_025_iou30", False),
            "full_s1": ("full_s1_iou30", True),
            "cheap_iou50": ("cheap_025_iou50", True), "full_iou50": ("full_025_iou50", True)}
# baseline branches (CVPR SOTA table): EXTRA_MODES=sahi,sahi_s033,fovea adds, per mode, a matching configuration and a
# branch built exactly like cheap / full (unmatched eligible tracks removed, unmatched detections lifted as FP agents)
EXTRA_MODES = [m for m in os.environ.get("EXTRA_MODES", "").split(",") if m]
for _m in EXTRA_MODES:
    COMBOS[f"{_m}_025_iou30"] = (_m, "0.25", 0.3)
    BRANCHES[_m] = (f"{_m}_025_iou30", True)
if EXTRA_MODES:                                   # keep the original run's overlays untouched
    OVERLAYS = OUTC / "overlays"
CORNER_SIGNS = np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)], float)
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8)
         if int(np.sum(CORNER_SIGNS[i] != CORNER_SIGNS[j])) == 1]


def quat_R(w, x, y, z):
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_yaw(w, x, y, z):
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Camera:
    """CAM_F0 of one log: extrinsic (camera -> ego), intrinsic, Caltech/OpenCV distortion."""

    def __init__(self, log: str):
        con = sqlite3.connect(f"file:{DB_DIR / (log + '.db')}?mode=ro", uri=True)
        rows = con.execute("select translation, rotation, intrinsic, distortion, width, height from camera "
                           "where channel = 'CAM_F0'").fetchall()
        con.close()
        assert len(rows) == 1, f"{log}: {len(rows)} CAM_F0 rows"
        t, q, K, D, w, h = rows[0]
        assert (w, h) == (W, H)
        t, q, K, D = (pickle.loads(t), pickle.loads(q), pickle.loads(K), pickle.loads(D))
        self.t = np.array([float(v) for v in t])
        self.R = quat_R(*[float(v) for v in q])
        self.K = np.array(K, float)
        self.D = np.array(D, float)
        border = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1], [0, H / 2], [W - 1, H / 2],
                           [W / 2, 0], [W / 2, H - 1]], np.float64)
        und = cv2.undistortPoints(border.reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)
        self.bx, self.by = CLAMP * float(np.abs(und[:, 0]).max()), CLAMP * float(np.abs(und[:, 1]).max())

    def monotone(self):
        k1, k2, _, _, k3 = self.D
        r = np.linspace(0, np.hypot(self.bx, self.by), 2000)
        deriv = 1 + 3 * k1 * r ** 2 + 5 * k2 * r ** 4 + 7 * k3 * r ** 6
        return bool((deriv > 0).all()), float(r[-1]), float(deriv.min())

    def distort(self, xn, yn):
        k1, k2, p1, p2, k3 = self.D
        r2 = xn * xn + yn * yn
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        return self.K[0, 0] * xd + self.K[0, 2], self.K[1, 1] * yd + self.K[1, 2]

    def project_box(self, pc):
        """pc: (8, 3) corners in the camera frame -> clipped (x1, y1, x2, y2) or None."""
        pts = [p for p in pc if p[2] > NEAR]
        for i, j in EDGES:
            zi, zj = pc[i, 2], pc[j, 2]
            if (zi - NEAR) * (zj - NEAR) < 0:
                a = (NEAR - zi) / (zj - zi)
                pts.append(pc[i] + a * (pc[j] - pc[i]))
        if not pts:
            return None
        P = np.asarray(pts)
        u, v = self.distort(np.clip(P[:, 0] / P[:, 2], -self.bx, self.bx),
                            np.clip(P[:, 1] / P[:, 2], -self.by, self.by))
        x1, y1, x2, y2 = max(u.min(), 0.0), max(v.min(), 0.0), min(u.max(), W - 1.0), min(v.max(), H - 1.0)
        if x2 <= x1 or y2 <= y1:
            return None
        return (float(x1), float(y1), float(x2), float(y2))

    def ground_point(self, u, v, ground_z):
        """Pixel -> ray -> ego-frame ground plane z = ground_z; returns ego (x, y) and the horizontal ray, or None.

        The nuPlan ego frame's origin is the rear axle, about 0.3 m above the road, so the plane is not z = 0
        (RESEARCH_LOG, Task 5 amendment): ground_z is the train+val median bottom of vehicle boxes.
        """
        xn, yn = cv2.undistortPoints(np.array([[[u, v]]], np.float64), self.K, self.D).reshape(2)
        d = self.R @ np.array([xn, yn, 1.0])
        if d[2] >= -1e-9:
            return None
        s = (ground_z - self.t[2]) / d[2]
        if s <= 0:
            return None
        g = self.t + s * d
        return g[:2], d[:2] / max(np.hypot(d[0], d[1]), 1e-9)


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def hungarian(obj_boxes, obj_cat, det_boxes, det_cls, thr):
    """Max-IoU assignment over class-compatible pairs with IoU >= thr. Returns {obj_i: (det_j, iou)}."""
    iou = iou_matrix(obj_boxes, det_boxes)
    if not iou.size:
        return {}
    ok = np.array([[d in COMPAT[c] for d in det_cls] for c in obj_cat], bool)
    w = np.where(ok & (iou >= thr), iou, 0.0)
    r, c = linear_sum_assignment(-w)
    return {int(i): (int(j), float(w[i, j])) for i, j in zip(r, c) if w[i, j] > 0}


def size_priors(logs):
    rows = []
    for log in logs:
        con = sqlite3.connect(f"file:{DB_DIR / (log + '.db')}?mode=ro", uri=True)
        rows += con.execute("select c.name, t.length, t.width, t.height from track t "
                            "join category c on t.category_token = c.token").fetchall()
        con.close()
    d = pd.DataFrame(rows, columns=["category", "length", "width", "height"])
    return {c: [float(d[d.category == c][k].median()) for k in ("length", "width", "height")] + [int((d.category == c).sum())]
            for c in ELIGIBLE}


def ground_height(logs):
    """Median bottom height of vehicle boxes 3-30 m ahead (|lateral| < 10 m) in the ego frame, over every
    20th lidar sweep of the benchmark scenario windows of `logs` (train+val only)."""
    meta = pd.read_csv(ROOT / "configs" / "benchmark_nuplan_scenarios.csv")
    zb = []
    for log in logs:
        con = sqlite3.connect(f"file:{DB_DIR / (log + '.db')}?mode=ro", uri=True)
        for _, s in meta[meta.log == log].iterrows():
            pcs = con.execute("select token, ego_pose_token from lidar_pc where timestamp between ? and ? "
                              "order by timestamp", (int(s.t0 * 1e6), int(s.t1 * 1e6))).fetchall()[::20]
            for tok, ept in pcs:
                ep = con.execute("select x, y, z, qw, qx, qy, qz from ego_pose where token = ?", (ept,)).fetchone()
                Re, te = quat_R(*ep[3:]), np.array(ep[:3])
                for x, y, z, h in con.execute(
                        "select lb.x, lb.y, lb.z, lb.height from lidar_box lb join track t on lb.track_token = t.token "
                        "join category c on t.category_token = c.token where lb.lidar_pc_token = ? and c.name = 'vehicle'",
                        (tok,)):
                    pe = (np.array([x, y, z]) - te) @ Re
                    if 3 < pe[0] < 30 and abs(pe[1]) < 10:
                        zb.append(pe[2] - h / 2)
        con.close()
    zb = np.asarray(zb)
    return {"ground_z_ego_m": float(np.median(zb)), "p25_m": float(np.percentile(zb, 25)),
            "p75_m": float(np.percentile(zb, 75)), "n_boxes": int(len(zb)), "logs": len(logs)}


def dashed_rect(img, p1, p2, color, th=2, dash=12):
    (x1, y1), (x2, y2) = p1, p2
    for (a, b) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(int(np.hypot(b[0] - a[0], b[1] - a[1]) // dash), 1)
        for k in range(0, n, 2):
            s = (int(a[0] + (b[0] - a[0]) * k / n), int(a[1] + (b[1] - a[1]) * k / n))
            e = (int(a[0] + (b[0] - a[0]) * min(k + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(k + 1, n) / n))
            cv2.line(img, s, e, color, th)


COLORS = {"vehicle": (60, 200, 60), "pedestrian": (0, 165, 255), "bicycle": (255, 140, 0)}


def overlay(path, img_path, objs, dets, matches, title):
    img = cv2.imread(str(img_path))
    for o in objs:
        if o["box"] is None:
            continue
        p1, p2 = (int(o["box"][0]), int(o["box"][1])), (int(o["box"][2]), int(o["box"][3]))
        if o["eligible"]:
            cv2.rectangle(img, p1, p2, COLORS[o["category"]], 3)
        else:
            dashed_rect(img, p1, p2, (170, 170, 170), 2)
    for mode, color in (("cheap", (40, 40, 230)), ("full", (230, 40, 230))):
        boxes, cls = dets[mode]
        for b, c in zip(boxes, cls):
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2 if mode == "cheap" else 1)
        for oi, (dj, iou) in matches[mode].items():
            ob, db = objs[oi]["box"], boxes[dj]
            cv2.line(img, (int((ob[0] + ob[2]) / 2), int((ob[1] + ob[3]) / 2)),
                     (int((db[0] + db[2]) / 2), int((db[1] + db[3]) / 2)), color, 2)
    cv2.rectangle(img, (0, 0), (W, 70), (0, 0, 0), -1)
    cv2.putText(img, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, "projected: vehicle green, pedestrian orange, bicycle blue, static grey dashed | "
                     "detections@0.25: 320 red (thick), 640 magenta (thin), lines = matches (IoU>=0.3)",
                (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(str(path), cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------------------------------------------------------------
# speculative perception on one CAM_F0 image (proxy: the cached 640 detections stand in for the full model on a crop)
from rap import planner as RP                                                        # noqa: E402
from vtf import dsf as VD, speculative as VS                                          # noqa: E402
PP = RP.PlannerParams()
V_CLOSE, R_GRID, Y_GRID, PROBE_W, PROBE_L, PROBE_H = 10.0, VS.R_GRID, VS.Y_GRID, 1.8, 4.5, 1.6


def lift_boxes(cam, boxes, ground_z):
    """Flat-ground monocular lift of 2D boxes to (z forward, lat_min, lat_max, ttc) in the ego frame."""
    z, lo, hi = [], [], []
    for b in boxes:
        gp = cam.ground_point(0.5 * (b[0] + b[2]), b[3], ground_z)
        if gp is None:
            z.append(np.inf); lo.append(0.0); hi.append(0.0); continue
        (gx, gy), _ = gp
        halfw = 0.5 * (b[2] - b[0]) * max(gx, 0.5) / cam.K[0, 0]
        z.append(gx); lo.append(gy - halfw); hi.append(gy + halfw)
    n = len(z)
    return (np.asarray(z, float), np.asarray(lo, float), np.asarray(hi, float), np.full(n, np.inf))


def cell_box(cam, r, y, ground_z):
    c = np.array([r + PROBE_L / 2, y, ground_z + PROBE_H / 2])
    corners = c + CORNER_SIGNS * [PROBE_L / 2, PROBE_W / 2, PROBE_H / 2]
    return cam.project_box((corners - cam.t) @ cam.R)


_DET = None
CROP_SCALE = float(os.environ.get("SPEC_CROP_SCALE", 640 / 1920))   # frozen v3 definition: 960 / long side = 0.5


def _detector():
    global _DET
    if _DET is None:
        import torch
        from rap.detect import TwoFidelityDetector
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        _DET = TwoFidelityDetector(str(Path(os.environ["B"]) / "models" / "yolov8s.pt"), device=dev, backend="torch")
    return _DET


def real_crop_detector(img):
    from rap.detect import Mode
    import math

    def detect_crop(cr):
        x1, y1 = int(cr[0]), int(cr[1])
        x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
        sub = img[y1:y2, x1:x2]
        nw = max(32, int(math.ceil(sub.shape[1] * CROP_SCALE / 32)) * 32)
        nh = max(32, int(math.ceil(sub.shape[0] * CROP_SCALE / 32)) * 32)
        d, _ = _detector().detect(sub, Mode(f"c{nh}x{nw}", nh, nw))
        kk = d["conf"] >= 0.25
        return {"xyxy": d["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                "conf": d["conf"][kk].astype(float), "coarse": d["coarse"][kk]}
    return detect_crop


_EXTRA = {}
SAHI_COARSE = {0: "person", 1: "cyclist", 2: "vehicle", 3: "cyclist", 5: "vehicle", 7: "vehicle"}


def _fovea_gg():
    """FOVEA's official grid generator (third_party/fovea, unchanged; registry imports stubbed as in fovea_baseline.py)."""
    import importlib.util, types
    if not hasattr(np, "float"):
        np.float = float
    B_ = Path(os.environ["B"])

    class _Reg:
        def __init__(self, name): self.name = name
        def register_module(self, *a, **k): return (lambda c: c)
    mm, mu = types.ModuleType("mmcv"), types.ModuleType("mmcv.utils")
    mu.Registry = _Reg
    md, mdm, mdb = (types.ModuleType(n) for n in ("mmdet", "mmdet.models", "mmdet.models.builder"))
    mdb.build = lambda *a, **k: None
    pkg, ut, vis = types.ModuleType("fovea_official"), types.ModuleType("fovea_official.utils"), types.ModuleType("fovea_official.utils.vis")
    pkg.__path__ = [str(B_ / "third_party" / "fovea" / "fovea")]
    vis.vis_batched_imgs = lambda *a, **k: None
    for n, m_ in {"mmcv": mm, "mmcv.utils": mu, "mmdet": md, "mmdet.models": mdm, "mmdet.models.builder": mdb,
                  "fovea_official": pkg, "fovea_official.utils": ut, "fovea_official.utils.vis": vis,
                  "fovea_official.models": types.ModuleType("fovea_official.models")}.items():
        sys.modules.setdefault(n, m_)
    spec_ = importlib.util.spec_from_file_location("fovea_official.models.grid_generator",
                                                   B_ / "third_party" / "fovea" / "fovea" / "models" / "grid_generator.py")
    gg = importlib.util.module_from_spec(spec_)
    gg.__package__ = "fovea_official.models"
    spec_.loader.exec_module(gg)
    return gg


def extra_dets(mode, img, cheap, spec_info=None):
    """Baseline detections on the CAM_F0 image (same COCO YOLOv8s as every branch). sahi: 640x640 slices, overlap 0.2,
    + standard prediction, slices at 640 px; sahi_s033: slices at 224 px (the FULL mode's 1/3 magnification);
    fovea: FOVEA KDE_I from the cheap draft (conf >= 0.25), warped to 384x640 (the FULL mode's compute), unwarped."""
    import torch
    if mode.startswith("sahi"):
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        if mode not in _EXTRA:
            _EXTRA[mode] = AutoDetectionModel.from_pretrained(
                model_type="ultralytics", model_path=str(Path(os.environ["B"]) / "models" / "yolov8s.pt"),
                confidence_threshold=0.25, device="cuda:0" if torch.cuda.is_available() else "cpu",
                image_size=640 if mode == "sahi" else 224)
        res = get_sliced_prediction(img[:, :, ::-1].copy(), _EXTRA[mode], slice_height=640, slice_width=640,
                                    overlap_height_ratio=0.2, overlap_width_ratio=0.2, perform_standard_pred=True, verbose=0)
        ob = [o for o in res.object_prediction_list if o.category.id in SAHI_COARSE]
        return {"xyxy": np.array([o.bbox.to_xyxy() for o in ob], float).reshape(-1, 4),
                "conf": np.array([o.score.value for o in ob], float),
                "coarse": np.array([SAHI_COARSE[o.category.id] for o in ob]) if ob else np.zeros(0, "<U8")}
    if mode == "fovea":
        import torch.nn.functional as F
        from rap.detect import Mode
        if "gg" not in _EXTRA:
            gg = _fovea_gg()
            _EXTRA["gg"] = gg
            _EXTRA["gen"] = gg.PlainKDEGrid(output_shape=(384, 640), grid_shape=(31, 51), separable=True, attraction_fwhm=4,
                                            anti_crop=True, bandwidth_scale=64, amplitude_scale=1)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().to(dev)
        k = cheap["conf"] >= 0.25
        with torch.no_grad():
            grid = _EXTRA["gen"](x, [{"pad_shape": (H, W, 3)}],
                                 [torch.tensor(cheap["xyxy"][k], dtype=torch.float32, device=dev).reshape(-1, 4)])
            wimg = F.grid_sample(x, grid, align_corners=True)[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()[:, :, ::-1].copy()
        d, _ = _detector().detect(wimg, Mode("fov384x640", 384, 640))
        bb = _EXTRA["gg"].unwarp_bboxes(torch.tensor(d["xyxy"], dtype=torch.float32, device=dev), grid[0], (H, W)).cpu().numpy() \
            if len(d["conf"]) else np.zeros((0, 4))
        return {"xyxy": bb.astype(float), "conf": d["conf"].astype(float), "coarse": d["coarse"]}
    if mode == "full960":                                  # uniform whole frame at 576x960 (u960)
        from rap.detect import Mode
        return _detector().detect(img, Mode("ns_full_960", 576, 960))[0]
    if mode == "uncertainty":
        # crop baseline: cheap boxes by (confidence ascending, entropy descending), crops taken strictly within the spec
        # branch's pixels on this image, same crop detector / guard / merge rule as the spec branch
        from rap.mono import box_iou
        k = cheap["conf"] >= 0.25
        xy, cf = np.asarray(cheap["xyxy"], float)[k], np.asarray(cheap["conf"], float)[k]
        ent = np.asarray(cheap["entropy"], float)[k] if "entropy" in cheap else np.zeros(len(cf))
        budget, used, crops = spec_info["pixels"] * W * H, 0.0, []
        for j in np.lexsort((-ent, cf)):
            c_ = VS.box_crop(xy[j], W, H)
            a_ = (c_[2] - c_[0]) * (c_[3] - c_[1])
            if used + a_ <= budget:
                crops.append(c_); used += a_
        gdet = VS.guarded(real_crop_detector(img), W, H)
        full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
        grown = []
        for c_ in crops:
            d_, ex_ = gdet(c_)
            grown += ex_
            full = {kk: np.concatenate([full[kk], np.asarray(d_[kk])]) for kk in full}
        if len(full["conf"]) > 1:
            o_, iou_ = np.argsort(-full["conf"], kind="stable"), box_iou(full["xyxy"], full["xyxy"])
            kp = []
            for i_ in o_:
                if not kp or iou_[i_, kp].max() < 0.5:
                    kp.append(i_)
            full = {kk: full[kk][np.sort(np.array(kp))] for kk in full}
        out = VS.centres_in(xy, crops) if crops else np.zeros(len(xy), bool)
        if grown and len(full["conf"]) and len(xy):
            out |= box_iou(xy, full["xyxy"]).max(1) >= 0.5
        co = np.asarray(cheap["coarse"])[k]
        return {"xyxy": np.r_[xy[~out], full["xyxy"]].reshape(-1, 4), "conf": np.r_[cf[~out], full["conf"]],
                "coarse": np.r_[co[~out], full["coarse"]]}
    raise ValueError(mode)


def speculative_dets(dd, cam, ego, ground_z, img=None, keep_conf=None):
    """keep_conf (dev variants; None = frozen v3): a cheap box inside a crop whose crop found no box overlapping it
    (IoU >= 0.3) is KEPT when its confidence >= keep_conf, instead of being dropped."""
    v = float(ego.dynamic_car_state.speed)
    keep = lambda d: d["conf"] >= 0.25                                            # noqa: E731
    ch = {k_: dd["cheap"][k_][keep(dd["cheap"])] for k_ in ("xyxy", "conf", "coarse")}
    fu = {k_: dd["full"][k_][keep(dd["full"])] for k_ in ("xyxy", "conf", "coarse")}
    ch["xyxy"], fu["xyxy"] = ch["xyxy"].astype(float), fu["xyxy"].astype(float)
    ch["mono"] = lift_boxes(cam, ch["xyxy"], ground_z)

    class _Cam:                                   # adapter: the loop only needs the field's image footprint
        pass

    gdet = VS.guarded(real_crop_detector(img), W, H)
    grown_all = []

    crops, scene, rounds = [], dict(ch), 0
    full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
    for _ in range(3):
        new = []
        sens = VD.insertion_grid(scene["mono"], v, R_GRID, Y_GRID, PROBE_W, PP, V_CLOSE)
        if sens.any():
            bs = [cell_box(cam, r, y, ground_z) for (a, r) in enumerate(R_GRID) for (b, y) in enumerate(Y_GRID) if sens[a, b]]
            bs = [b for b in bs if b is not None]
            if bs:
                bs = np.asarray(bs)
                fc = np.array([bs[:, 0].min(), bs[:, 1].min(), bs[:, 2].max(), bs[:, 3].max()])
                if not VS.covered(fc, crops):
                    new.append(fc)
        for k_ in VS.support(scene["mono"], v, PP):
            bc = VS.box_crop(scene["xyxy"][k_], W, H)
            if not VS.covered(bc, crops + grown_all + new):
                new.append(bc)
        if not new:
            break
        rounds += 1
        crops += new
        for c_ in new:
            d_, ex_ = gdet(c_)
            grown_all += ex_
            full = {k_: np.concatenate([full[k_], np.asarray(d_[k_])]) for k_ in ("xyxy", "conf", "coarse")}
        if len(full["conf"]) > 1:                     # de-duplicate across overlapping crops (IoU >= 0.5)
            from rap.mono import box_iou
            o_, iou_ = np.argsort(-full["conf"], kind="stable"), box_iou(full["xyxy"], full["xyxy"])
            kp = []
            for i_ in o_:
                if not kp or iou_[i_, kp].max() < 0.5:
                    kp.append(i_)
            kp = np.sort(np.array(kp))
            full = {k_: full[k_][kp] for k_ in full}
        out = VS.centres_in(ch["xyxy"], crops)
        if grown_all and len(full["conf"]) and len(ch["xyxy"]):
            from rap.mono import box_iou
            out |= box_iou(ch["xyxy"], full["xyxy"]).max(1) >= 0.5
        if keep_conf is not None and len(ch["xyxy"]):
            from rap.mono import box_iou
            redet = (box_iou(ch["xyxy"], full["xyxy"]).max(1) >= 0.3) if len(full["conf"]) else np.zeros(len(ch["xyxy"]), bool)
            out &= ~((np.asarray(ch["conf"]) >= keep_conf) & ~redet)
        scene = {k_: np.concatenate([ch[k_][~out], full[k_]]) for k_ in ("xyxy", "conf", "coarse")}
        scene["mono"] = lift_boxes(cam, scene["xyxy"], ground_z)
    area = sum((c_[2] - c_[0]) * (c_[3] - c_[1]) for c_ in crops + grown_all) / (W * H)
    det = {"xyxy": scene["xyxy"], "conf": scene["conf"], "coarse": scene["coarse"]}
    return det, {"n_crops": len(crops) + len(grown_all), "rounds": rounds, "pixels": area, "v": v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(NUPLAN_ROOT))
    ap.add_argument("--map_root", default=str(NUPLAN_MAPS))
    ap.add_argument("--n_overlays", type=int, default=20)
    args = ap.parse_args()
    t_start = time.time()

    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())["nuplan"]
    split_of = {l: k for k in ("train", "val", "test") for l in splits[k]}
    det_summary = json.loads((WORK / "detect_summary.json").read_text())
    thresholds = {"0.25": 0.25, "s1": float(det_summary["s1_full_threshold"])}
    priors = size_priors(sorted(splits["train"] + splits["val"]))
    ground = ground_height(sorted(splits["train"] + splits["val"]))
    ground_z = ground["ground_z_ego_m"]
    print(f"  ground plane z in the ego frame (train+val): {ground}", flush=True)
    print(f"  S1 FULL threshold {thresholds['s1']:.4f}; size priors {priors}", flush=True)

    idx = pd.read_csv(WORK / "image_index.csv.gz")
    raw = pd.read_csv(Path(RESULTS) / "final" / "phase0g_external_idm_raw.csv")
    states = raw[["scenario", "log", "iteration"]].drop_duplicates()
    assert len(states) == 1440

    builder = NuPlanScenarioBuilder(
        data_root=f"{args.data_root}/nuplan-v1.1/splits/mini", map_root=args.map_root,
        sensor_root=None, db_files=None, map_version="nuplan-maps-v1.0")
    flt = ScenarioFilter(scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
                         num_scenarios_per_type=None, limit_total_scenarios=60,
                         timestamp_threshold_s=None, ego_displacement_minimum_m=None,
                         expand_scenarios=False, remove_invalid_goals=True, shuffle=False)
    scen = {s.token: s for s in builder.get_scenarios(flt, Sequential())}
    assert set(scen) == set(states.scenario), "scenario set differs from Track B"

    z = np.load(Path(CACHE) / "detector_miss_model.npz", allow_pickle=False)
    model = DetectorMissModel(z["w_cheap"], z["w_rescue"], z["w_lose"])
    geomf = PerceptionFilter(model)
    rng = np.random.default_rng(0)
    pick = set(map(tuple, states[["scenario", "iteration"]].to_numpy()[
        rng.choice(len(states), args.n_overlays, replace=False)].tolist()))
    OVERLAYS.mkdir(parents=True, exist_ok=True)
    OUTC.mkdir(parents=True, exist_ok=True)

    cams, dets_cache, all_img = {}, {}, {}
    spec, obj_rows, state_rows, dt_rows, iou_rows = {}, [], [], [], []
    spec_rows = []
    fp_dropped = {b: {"no_ground_hit": 0, "not_ahead": 0, "beyond_range": 0} for b in BRANCHES}
    ground_check, overlay_files, lift_err = [], [], []
    for n_sc, (tok, g) in enumerate(states.groupby("scenario", sort=True), 1):
        sc = scen[tok]
        log = sc.log_name
        if log not in cams:
            cams[log] = Camera(log)
            dets_cache[log] = {m: DetCache(DET / MODE_CACHE[m] / f"{log}.npz") for m in MODE_CACHE}
            con = sqlite3.connect(f"file:{DB_DIR / (log + '.db')}?mode=ro", uri=True)
            all_img[log] = np.array(sorted(r[0] for r in con.execute(
                "select i.timestamp from image i join camera c on i.camera_token = c.token "
                "where c.channel = 'CAM_F0'")), np.int64)
            con.close()
        cam, li = cams[log], idx[idx.log == log].sort_values("timestamp")
        con = sqlite3.connect(f"file:{DB_DIR / (log + '.db')}?mode=ro", uri=True)
        its = sorted(int(i) for i in g.iteration)
        need_k = sorted({k for it in its for k in range(it - BUFFER + 1, it + 1)})
        for k in need_k:
            t_k = sc.get_time_point(k).time_us
            ts = li.timestamp.to_numpy()
            j = int(np.argmin(np.abs(ts - t_k)))
            im = li.iloc[j]
            j_all = int(np.argmin(np.abs(all_img[log] - t_k)))
            dt_rows.append({"scenario": tok, "iteration": k, "decision": k in its, "dt_us": int(im.timestamp - t_k),
                            "nearest_in_db_fetched": bool(all_img[log][j_all] == im.timestamp)})
            ep = con.execute("select x, y, z, qw, qx, qy, qz from ego_pose where token = ?",
                             (bytes.fromhex(im.ego_pose_token),)).fetchone()
            Re, te, yaw_e = quat_R(*ep[3:]), np.array(ep[:3]), quat_yaw(*ep[3:])
            lb = {r[0].hex(): r[1:] for r in con.execute(
                "select token, x, y, z, yaw, width, length, height from lidar_box where lidar_pc_token = ?",
                (bytes.fromhex(sc._lidarpc_tokens[k]),))}
            objs = list(sc.get_tracked_objects_at_iteration(k).tracked_objects)
            ego = sc.get_ego_state_at_iteration(k)
            dist, lat, coarse, fwd = geomf._geometry(objs, ego) if objs else (np.zeros(0),) * 4
            recs = []
            for oi, o in enumerate(objs):
                cat = str(o.tracked_object_type.name).lower()
                x, y, zc, yaw, wd, ln, ht = lb[o.metadata.token]
                cg = np.array([x, y, zc]) + (CORNER_SIGNS * [ln / 2, wd / 2, ht / 2]) @ np.array(
                    [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]).T
                pe = (cg - te) @ Re                       # ego frame (rows: R^T p)
                pcam = (pe - cam.t) @ cam.R               # camera frame
                centre_cam = ((np.array([x, y, zc]) - te) @ Re - cam.t) @ cam.R
                box = cam.project_box(pcam) if centre_cam[2] > 0 else None
                recs.append({"token": o.metadata.token, "category": cat, "eligible": cat in ELIGIBLE,
                             "box": box, "in_cam": box is not None,
                             "height_px": (box[3] - box[1]) if box else 0.0,
                             "dist": float(dist[oi]), "lat": float(lat[oi]), "coarse": str(coarse[oi])})
                if cat == "vehicle" and box is not None and float(dist[oi]) < 30:
                    ground_check.append(float(pe[:, 2].min()))
                if k in its and cat == "vehicle" and box is not None and box[3] < H - 2 and float(dist[oi]) < 60:
                    # the FP lift applied to a TRUE projected box: how far from the true centre it lands
                    gp = cam.ground_point(0.5 * (box[0] + box[2]), box[3], ground_z)
                    if gp is not None:
                        (gx, gy), hray = gp
                        est = np.array([gx, gy]) + hray * priors["vehicle"][0] / 2
                        tru = ((np.array([x, y, zc]) - te) @ Re)[:2]
                        lift_err.append((float(np.hypot(*(est - tru))), float(np.hypot(*est) - np.hypot(*tru)),
                                         float(np.hypot(*tru))))
            frame = int(im.frame)
            dd = {m: dets_cache[log][m].det(dets_cache[log][m].index[frame]) for m in MODE_CACHE}
            img_k = cv2.imread(str(IMG_ROOT / im.filename_jpg))
            assert img_k is not None, f"missing image {im.filename_jpg}"
            dd["spec"], spec_info = speculative_dets(dd, cam, ego, ground_z, img_k)
            for m_ in EXTRA_MODES:
                dd[m_] = extra_dets(m_, img_k, dd["cheap"], spec_info)
            spec_rows.append({"scenario": tok, "iteration": k, **spec_info})
            elig = [i for i, r in enumerate(recs) if r["eligible"] and r["in_cam"]]
            matches, unmatched = {}, {}
            for cname, (mode, tkey, iou_thr) in COMBOS.items():
                d = dd[mode]
                keep = d["conf"] >= thresholds[tkey]
                kidx = np.flatnonzero(keep)
                boxes, cls = d["xyxy"][keep], d["coarse"][keep]
                mt = hungarian([recs[i]["box"] for i in elig], [recs[i]["category"] for i in elig], boxes, cls, iou_thr)
                matches[cname] = {elig[i]: v for i, v in mt.items()}
                used = {v[0] for v in mt.values()}
                unmatched[cname] = [(boxes[jj], cls[jj], int(kidx[jj])) for jj in range(len(boxes)) if jj not in used]
                if k in its:
                    iou_rows += [{"combo": cname, "iou": v[1]} for v in mt.values()]
            for bname, (cname, with_fp) in BRANCHES.items():
                removed = frozenset(recs[i]["token"] for i in elig if i not in matches[cname])
                fps = []
                if with_fp:
                    fmode = COMBOS[cname][0]
                    for b, c, kd in unmatched[cname]:
                        gp = cam.ground_point(0.5 * (b[0] + b[2]), b[3], ground_z)
                        if gp is None:
                            fp_dropped[bname]["no_ground_hit"] += 1
                            continue
                        (gx, gy), hray = gp
                        if gx < FP_MIN_AHEAD:
                            fp_dropped[bname]["not_ahead"] += 1
                            continue
                        if np.hypot(gx, gy) > FP_MAX_RANGE:
                            fp_dropped[bname]["beyond_range"] += 1
                            continue
                        cat = DET_TO_CATEGORY[str(c)]
                        L, Wd, Ht, _ = priors[cat]
                        ce = np.array([gx, gy, 0.0]) + np.array([hray[0], hray[1], 0.0]) * L / 2
                        cgl = Re @ ce + te
                        # token per (mode, image, cached detection): the same detection is the same agent in
                        # every branch, so identical branches can share one planner call
                        fps.append((f"fp:{fmode}:{im.image_token}:{kd}", cat, float(cgl[0]), float(cgl[1]),
                                    yaw_e, float(L), float(Wd), float(Ht)))
                spec[(tok, k, bname)] = (removed, fps)
            spec[(tok, k, "identity")] = (frozenset(), [])
            if k in its:
                row = {"scenario": tok, "log": log, "split": split_of[log], "iteration": k,
                       "image_token": im.image_token, "dt_ms": (im.timestamp - t_k) / 1e3,
                       "n_tracks": len(recs), "n_eligible_in_cam": len(elig),
                       "n_static_in_cam": sum(1 for r in recs if r["in_cam"] and not r["eligible"])}
                for bname in BRANCHES:
                    rem, fps = spec[(tok, k, bname)]
                    row[f"kept_{bname}"] = len(elig) - len(rem)
                    row[f"fp_{bname}"] = len(fps)
                state_rows.append(row)
                for i in elig:
                    r = recs[i]
                    obj_rows.append({"scenario": tok, "log": log, "iteration": k, "token": r["token"],
                                     "category": r["category"], "dist": r["dist"], "lat": r["lat"],
                                     "height_px": r["height_px"],
                                     **{f"hit_{c}": i in matches[c] for c in COMBOS},
                                     **{f"iou_{c}": matches[c][i][1] if i in matches[c] else 0.0 for c in COMBOS}})
                if (tok, k) in pick:
                    f = OVERLAYS / f"{len(overlay_files):02d}_{log}_{tok}_{k}.jpg"
                    ov_d = {m: (dd[m]["xyxy"][dd[m]["conf"] >= 0.25], dd[m]["coarse"][dd[m]["conf"] >= 0.25])
                            for m in MODE_CACHE}
                    overlay(f, IMG_ROOT / im.filename_jpg, recs, ov_d,
                            {"cheap": matches["cheap_025_iou30"], "full": matches["full_025_iou30"]},
                            f"{log} | scenario {tok} | iteration {k} | dt {(im.timestamp - t_k) / 1e3:+.1f} ms | "
                            f"eligible in camera {len(elig)}, matched 320 {len(matches['cheap_025_iou30'])}, "
                            f"640 {len(matches['full_025_iou30'])}")
                    overlay_files.append(f.name)
        con.close()
        print(f"  [{n_sc:2d}/60] {tok} {log}: {len(its)} states, {len(need_k)} iterations, "
              f"{time.time() - t_start:6.0f}s", flush=True)

    pd.DataFrame(spec_rows).to_csv(OUTC / "spec_crops.csv.gz", index=False)
    with open(OUTC / "branches.pkl", "wb") as f:
        pickle.dump({"spec": spec, "branches": list(BRANCHES) + ["identity"], "thresholds": thresholds,
                     "priors": priors}, f)
    objs = pd.DataFrame(obj_rows)
    objs.to_csv(OUTC / "objects_decision.csv.gz", index=False)          # the edge env has no pyarrow
    st = pd.DataFrame(state_rows)
    st.to_csv(OUTC / "branch_state_summary.csv.gz", index=False)
    dt = pd.DataFrame(dt_rows)
    ious = pd.DataFrame(iou_rows)

    mono = {log: cams[log].monotone() for log in cams}
    q = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    checks = {
        "states": len(st), "iterations": len(dt), "scenarios": int(st.scenario.nunique()), "logs": int(st.log.nunique()),
        "s1_full_threshold": thresholds["s1"], "size_priors_train_val_median_LWH_n": priors,
        "dt": {"abs_ms_quantiles": dict(zip(map(str, q), (np.quantile(np.abs(dt.dt_us), q) / 1e3).round(2).tolist())),
               "n_iterations_over_50ms": int((np.abs(dt.dt_us) > DT_FLAG_US).sum()),
               "n_decision_iterations_over_50ms": int(((np.abs(dt.dt_us) > DT_FLAG_US) & dt.decision).sum()),
               "n_scenarios_any_image_over_50ms": int(dt.assign(f=np.abs(dt.dt_us) > DT_FLAG_US)
                                                      .groupby("scenario").f.sum().gt(0).sum()),
               "nearest_db_image_not_fetched": int((~dt.nearest_in_db_fetched).sum())},
        "distortion_monotone_to_clamp": {l: {"monotone": m[0], "r_max": round(m[1], 4), "min_derivative": round(m[2], 4)}
                                         for l, m in mono.items()},
        "all_distortion_monotone": bool(all(m[0] for m in mono.values())),
        "ground_plane_vehicle_bottom_z_ego_within_30m": {
            "n": len(ground_check), "median_m": float(np.median(ground_check)) if ground_check else None,
            "p25_m": float(np.percentile(ground_check, 25)) if ground_check else None,
            "p75_m": float(np.percentile(ground_check, 75)) if ground_check else None},
        "fp_lift_ground_plane_train_val": {**ground, "camera_height_above_ground_m":
                                           float(np.mean([c.t[2] for c in cams.values()]) - ground_z)},
        "fp_lift_on_true_vehicle_boxes_decision_iterations": (lambda e: {
            "n": len(e), "xy_error_m_quantiles": dict(zip(map(str, q), np.quantile(e[:, 0], q).round(2).tolist())),
            "range_error_m_quantiles": dict(zip(map(str, q), np.quantile(e[:, 1], q).round(2).tolist())),
            "median_abs_range_error_by_band": {f"{a}-{b}m": float(np.median(np.abs(e[(e[:, 2] >= a) & (e[:, 2] < b), 1])))
                                               for a, b in ((0, 15), (15, 30), (30, 60))
                                               if ((e[:, 2] >= a) & (e[:, 2] < b)).any()}})(np.asarray(lift_err)),
        "match_iou_quantiles": {c: dict(zip(map(str, q), np.quantile(ious[ious.combo == c].iou, q).round(3).tolist()))
                                for c in COMBOS if (ious.combo == c).any()},
        "match_counts": ious.combo.value_counts().to_dict(),
        "fp_dropped_by_reason_all_iterations": fp_dropped,
        "per_state_means": st.drop(columns=["scenario", "log", "split", "image_token"]).mean().round(3).to_dict(),
        "overlays": overlay_files,
        "seconds": time.time() - t_start,
    }
    (OUTC / "checks_projection.json").write_text(json.dumps(checks, indent=1, default=float))
    print(json.dumps({k: v for k, v in checks.items() if k not in ("distortion_monotone_to_clamp",)}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
