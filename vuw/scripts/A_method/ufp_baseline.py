#!/usr/bin/env python
"""Baseline: UFP, Unified Foreground Packing of UFPMP-Det (Huang et al., AAAI 2022), official module, our detector pair.

The official `UFPMP-Det-Tools/ufp` package (PuAnysh/UFPMP-Det @ 6809b4f) is imported unchanged; the glue follows
their `eval_script/ufpmp_det_eval.py` line by line:
  coarse boxes  -> UnifiedForegroundPacking(boxes, 1.5, input_shape=[W, H])   (merge scaled boxes into foreground
                   regions, zoom factor 4 / 2 / 1 by mean object area < 32^2 / < 96^2 / else, strip packing)
  mosaic        -> display_merge_result (each region resized by its factor, pasted at its packed position)
  fine pass     -> one detector pass on the mosaic, resized keeping ratio to fit S x S (their MP-Det test scale is
                   1333) and padded to a multiple of 32
  map back      -> a detection belongs to a chip if IoF > 0.9, mapped by the chip's offset and factor; per-class NMS 0.6
                   (their py_cpu_nms); only fine-pass detections are kept (as in their script)
Our pair instead of theirs: coarse = the cached cheap YOLOv8s-320 draft (all cached boxes, conf >= 0.10: the cache
floor; their coarse score_thr is 0.05), fine = the same COCO YOLOv8s as every other arm (no MP-Det training: the
packing module is what is compared, as for SAHI).  Frames without coarse boxes produce no fine detections.
Variants S in {640, 960, 1333}.  Then DEEP's mono lift (own previous frame), braking, cheap action history; compared
with the confirmatory v3 numbers; detector tensor pixels per frame reported.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import _apply_range_source
from rap.detect import Mode, TwoFidelityDetector, letterbox
from rap.mono import predicted_geometry
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL

B = Path(os.environ["B"])
sys.path.insert(0, str(B / "third_party" / "UFPMP-Det" / "UFPMP-Det-Tools"))
from ufp import UnifiedForegroundPacking                                          # noqa: E402

W0, H0 = 1600, 900
THR = 0.25
SIZES = {"ufp_640": 640, "ufp_960": 960, "ufp_1333": 1333}


def compute_iof(pos1, pos2):                                  # ufpmp_det_eval.py
    left1, top1, right1, down1 = pos1
    left2, top2, right2, down2 = pos2
    area1 = (right1 - left1) * (down1 - top1)
    area2 = (right2 - left2) * (down2 - top2)
    left, right, top, bottom = max(left1, left2), min(right1, right2), max(top1, top2), min(down1, down2)
    if left >= right or top >= bottom:
        return 0
    return (right - left) * (bottom - top) / min(area1, area2)


def py_cpu_nms(dets, thresh):                                  # ufpmp_det_eval.py
    dets = np.array(dets)
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


def display_merge_result(results, img_data, w, h):             # ufpmp_det_eval.py, image array instead of a path
    w, h = math.ceil(w), math.ceil(h)
    new_img = np.zeros((h, w, 3))
    for result in results:
        x1, y1, w, h, n_x, n_y, scale_factor = [math.floor(_) for _ in result]
        if w == 0 or h == 0:
            continue
        new_img[n_y:n_y + h * scale_factor, n_x:n_x + w * scale_factor, :] = cv2.resize(
            img_data[y1:y1 + h, x1:x1 + w, :], (w * scale_factor, h * scale_factor))
    return new_img


def ufp_detect(det, img, coarse_xyxy, S):
    empty = {"xyxy": np.zeros((0, 4)), "coarse": np.zeros(0, "<U8"), "conf": np.zeros(0)}, 0, 0
    if len(coarse_xyxy) == 0:
        return empty
    rec, w, h = UnifiedForegroundPacking(np.asarray(coarse_xyxy, float).copy(), 1.5, input_shape=[W0, H0])
    if not rec or w < 1 or h < 1:
        return empty
    mosaic = display_merge_result(rec, img, w, h).astype(np.uint8)
    mh, mw = mosaic.shape[:2]
    r = S / max(mh, mw)
    nh, nw = int(math.ceil(mh * r / 32) * 32), int(math.ceil(mw * r / 32) * 32)
    mode = Mode(f"ufp{nh}x{nw}", nh, nw)
    x, r_, pad, _ = det.preprocess(mosaic, mode)
    d = det.postprocess(det.forward(x, mode), mode, r_, pad, mosaic.shape)
    k = d["conf"] >= THR
    out = {}
    for chips in rec:
        o_x1, o_y1, cw, ch, n_x, n_y, sf = [math.floor(_) for _ in chips]
        chip = [n_x, n_y, n_x + cw * sf, n_y + ch * sf]
        for (x1, y1, x2, y2), sc, cl in zip(d["xyxy"][k], d["conf"][k], d["coarse"][k]):
            if compute_iof([x1, y1, x2, y2], chip) > 0.9:
                nx, ny = (x1 - n_x) / sf + o_x1, (y1 - n_y) / sf + o_y1
                out.setdefault(str(cl), []).append([nx, ny, nx + (x2 - x1) / sf, ny + (y2 - y1) / sf, float(sc)])
    xyxy, cls, conf = [], [], []
    for cl, lst in out.items():
        arr = np.array(lst)
        for i in py_cpu_nms(arr, 0.6):
            xyxy.append(arr[i, :4]); cls.append(cl); conf.append(arr[i, 4])
    src_px = sum(c[2] * c[3] for c in rec)
    return ({"xyxy": np.array(xyxy, float).reshape(-1, 4), "coarse": np.array(cls) if cls else np.zeros(0, "<U8"),
             "conf": np.array(conf, float)},
            nh * nw, src_px)


def main():
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    scenes = sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq)
    max_s = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    scenes = scenes[: max_s or None]
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    rows = []
    with torch.no_grad():
        for s in scenes:
            geom, speeds = ad.geometry(s), ad.speeds(s)
            cam = Camera("nuScenes", s)
            R, t = cam.transform("ego")
            c = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
            toks = db.samples(by[s])
            prev_act, prev = None, {k: None for k in SIZES}
            for i, fr in enumerate(c.frames):
                fr = int(fr)
                v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
                g = geom[geom["frame"] == fr]
                g = g[(g["y2"] - g["y1"]) >= 10]
                a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
                act = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)             # noqa: E731
                cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]                   # noqa: E731
                dc = c.det(i)
                kc = dc["conf"] >= THR
                geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
                cons_c = tuple(np.asarray(a, float) for a in _apply_range_source(dc["xyxy"][kc].astype(float), geo_c, g, "mono", rng, 0.0))
                a_c = act(cons_c)
                rec = {"seq": s, "frame": fr, "J_c": cost(a_c), "n_coarse": int(len(dc["conf"]))}
                img = cv2.imread(str(db.image_path(toks[fr])))
                for k, S in SIZES.items():
                    dd, tpx, spx = ufp_detect(det, img, dc["xyxy"], S)
                    gg = predicted_geometry(dd, prev[k], cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                    prev[k] = dd
                    rec[f"J_{k}"] = cost(act(tuple(np.asarray(gg[q], float) for q in ("z", "lat_min", "lat_max", "ttc"))))
                    rec[f"tpx_{k}"], rec[f"spx_{k}"] = tpx / (W0 * H0), spx / (W0 * H0)
                rows.append(rec)
                prev_act = a_c
            print(f"  {s}", flush=True)
    d = pd.DataFrame(rows)
    out = B / "results" / "ufp"
    out.mkdir(parents=True, exist_ok=True)
    d.to_pickle(out / "eval_newtest.pkl")
    ref = pd.read_pickle(B / "results" / "nusc_full" / "eval__new_test_v3_confirm.pkl")
    m = d.merge(ref[["seq", "frame", "J_c", "J_u960", "J_spec", "J_spec_ref_resnet18", "px"]], on=["seq", "frame"], suffixes=("", "_ref"))
    assert np.allclose(m.J_c, m.J_c_ref)
    L = m.J_c.to_numpy()
    uq = m.seq.unique(); idx = [np.flatnonzero(m.seq.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    V = {k: (m.J_c - m[f"J_{k}"]).to_numpy() for k in SIZES}
    V.update({"u960": (m.J_c - m.J_u960).to_numpy(), "v3": (m.J_c - m.J_spec).to_numpy(),
              "v3_ref": (m.J_c - m.J_spec_ref_resnet18).to_numpy()})
    tpx = {k: float(m[f"tpx_{k}"].mean()) for k in SIZES}
    tpx.update({"u960": 576 * 960 / (W0 * H0), "v3": float((m.px * 0.36).mean()), "v3_ref": float((m.px * 0.36).mean())})
    res = []
    for k, v_ in V.items():
        r = {"method": k, "G": v_.sum() / L.sum(), "harmed": float(np.mean(v_ < 0)), "tensor_px_share": tpx[k]}
        for ref_k in ("v3", "v3_ref"):
            dr = [(V[ref_k][t_].sum() - v_[t_].sum()) / L[t_].sum() for t_ in boots]
            r[f"{ref_k}_minus"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        res.append(r)
    res = pd.DataFrame(res)
    res.to_csv(out / f"summary_newtest{'_smoke' if max_s else ''}.csv", index=False)
    print(res.to_string(index=False))
    print("UFP DONE")


if __name__ == "__main__":
    main()
