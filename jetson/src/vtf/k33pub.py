"""K33 A / B on the Jetson (docs/paper/handoff/K33/table1_design.md §2 G): the parts of the published methods that the
VUW evaluation takes from their official libraries, re-implemented without those libraries' dependencies so that the
TensorRT timing script can run them on the frame already on the GPU.
  sahi_slices / greedy_nmm   sahi 0.12.7 slicing.get_slice_bboxes and postprocess GREEDYNMM (IOS, 0.5, per class)
  fovea_gg                   FOVEA's official grid generator (third_party/fovea, mmcv / mmdet registry stubbed)
  ufp_*                      UFPMP-Det's display_merge_result / compute_iof / py_cpu_nms (ufpmp_det_eval.py)
  xgb_json                   the decision-value router's XGBoost regressor (cond_V) evaluated from its JSON dump
VUW parity: k33_eval.py part pub uses the official sahi; the Jetson rows store their actions for the parity check.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import cv2
import numpy as np


def sahi_slices(H, W, side, ov):
    """sahi.slicing.get_slice_bboxes(H, W, side, side, overlap_height_ratio=ov, overlap_width_ratio=ov) (0.12.7)"""
    out, y_min, y_max = [], 0, 0
    yo, xo = int(ov * side), int(ov * side)
    while y_max < H:
        x_min = x_max = 0
        y_max = y_min + side
        while x_max < W:
            x_max = x_min + side
            if y_max > H or x_max > W:
                xm, ym = min(W, x_max), min(H, y_max)
                out.append([max(0, xm - side), max(0, ym - side), xm, ym])
            else:
                out.append([x_min, y_min, x_max, y_max])
            x_min = x_max - xo
        y_min = y_max - yo
    return out


def _ios(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    sa, sb = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return ix * iy / max(min(sa, sb), 1e-9)


def greedy_nmm(xyxy, conf, cls, thr=0.5):
    """SAHI's GREEDYNMM postprocess (match metric IOS, threshold 0.5, per category): greedy grouping by score, then each
    kept box absorbs (union box, max score) the members of its group that still match it"""
    xyxy, conf, cls = np.asarray(xyxy, float).reshape(-1, 4), np.asarray(conf, float), np.asarray(cls)
    ox, oc, ok = [], [], []
    for c in np.unique(cls):
        idx = list(np.flatnonzero(cls == c)[np.argsort(-conf[cls == c], kind="stable")])
        while idx:
            k = idx.pop(0)
            grp = [j for j in idx if _ios(xyxy[k], xyxy[j]) >= thr]
            idx = [j for j in idx if j not in grp]
            b, s = xyxy[k].copy(), conf[k]
            for j in grp:
                if _ios(b, xyxy[j]) >= thr:
                    b = np.array([min(b[0], xyxy[j][0]), min(b[1], xyxy[j][1]), max(b[2], xyxy[j][2]), max(b[3], xyxy[j][3])])
                    s = max(s, conf[j])
            ox.append(b); oc.append(s); ok.append(c)
    return np.asarray(ox, float).reshape(-1, 4), np.asarray(oc, float), np.asarray(ok)


def fovea_gg(root):
    """the official grid_generator module of FOVEA (root = third_party/fovea), as fovea_baseline.py imports it"""
    if not hasattr(np, "float"):
        np.float = float
    mm, mmu = types.ModuleType("mmcv"), types.ModuleType("mmcv.utils")

    class _Registry:
        def __init__(self, name): self.name = name
        def register_module(self, *a, **k): return (lambda c: c)
    mmu.Registry = _Registry
    md, mdm, mdb = (types.ModuleType(n) for n in ("mmdet", "mmdet.models", "mmdet.models.builder"))
    mdb.build = lambda *a, **k: None
    sys.modules.update({"mmcv": mm, "mmcv.utils": mmu, "mmdet": md, "mmdet.models": mdm, "mmdet.models.builder": mdb})
    pkg = types.ModuleType("fovea_official"); pkg.__path__ = [str(Path(root) / "fovea")]
    ut, vis = types.ModuleType("fovea_official.utils"), types.ModuleType("fovea_official.utils.vis")
    vis.vis_batched_imgs = lambda *a, **k: None
    sys.modules.update({"fovea_official": pkg, "fovea_official.utils": ut, "fovea_official.utils.vis": vis,
                        "fovea_official.models": types.ModuleType("fovea_official.models")})
    sp = importlib.util.spec_from_file_location("fovea_official.models.grid_generator", Path(root) / "fovea" / "models" / "grid_generator.py")
    gg = importlib.util.module_from_spec(sp); gg.__package__ = "fovea_official.models"
    sp.loader.exec_module(gg)
    return gg


def ufp_mosaic(rec, img, w, h):
    """UFPMP-Det display_merge_result on an image array"""
    w, h = math.ceil(w), math.ceil(h)
    out = np.zeros((h, w, 3))
    for r in rec:
        x1, y1, cw, ch, nx, ny, sf = [math.floor(_) for _ in r]
        if cw == 0 or ch == 0:
            continue
        out[ny:ny + ch * sf, nx:nx + cw * sf, :] = cv2.resize(img[y1:y1 + ch, x1:x1 + cw, :], (cw * sf, ch * sf))
    return out


def ufp_iof(p1, p2):
    l1, t1, r1, d1 = p1
    l2, t2, r2, d2 = p2
    a1, a2 = (r1 - l1) * (d1 - t1), (r2 - l2) * (d2 - t2)
    le, ri, to, bo = max(l1, l2), min(r1, r2), max(t1, t2), min(d1, d2)
    if le >= ri or to >= bo:
        return 0
    return (ri - le) * (bo - to) / min(a1, a2)


def ufp_nms(dets, thresh):
    dets = np.array(dets)
    x1, y1, x2, y2, sc = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    ar = (x2 - x1 + 1) * (y2 - y1 + 1)
    order, keep = sc.argsort()[::-1], []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        order = order[np.where(inter / (ar[i] + ar[order[1:]] - inter) <= thresh)[0] + 1]
    return keep


def xgb_json(path):
    """predict(x) of an XGBoost gbtree regressor (reg:squarederror, numeric splits, one tree per round) from its JSON
    model file (the Jetson has no xgboost): a sample goes left iff x < split_condition (float32), a missing value follows
    default_left; prediction = base_score + the sum of the leaves (a leaf's value is its split_condition)"""
    import json
    m = json.load(open(path))["learner"]
    assert m["gradient_booster"]["name"] == "gbtree" and m["objective"]["name"] == "reg:squarederror"
    base = float(str(m["learner_model_param"]["base_score"]).strip("[]"))
    trees = [(t["left_children"], t["right_children"], t["split_indices"], np.asarray(t["split_conditions"], np.float32),
              t["default_left"]) for t in m["gradient_booster"]["model"]["trees"]]

    def predict(x):
        x = np.asarray(x, np.float32).reshape(-1)
        s = np.float32(0.0)
        for L, R, I, C, D in trees:
            n = 0
            while L[n] != -1:
                xv = x[I[n]]
                n = (L[n] if D[n] else R[n]) if np.isnan(xv) else (L[n] if xv < C[n] else R[n])
            s += C[n]
        return float(np.float32(base) + s)
    return predict
