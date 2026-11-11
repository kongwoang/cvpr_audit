#!/usr/bin/env python
"""Fidelity check (B): standard detection metrics of every method on the nuScenes new TEST keyframes (every --step-th).

Each baseline's own paper claims a detection gain (SAHI / UFP / FOVEA / LZU: small objects; fine-tuning: domain);
this checks that our ports show it before decision value is reported.  Same GT as the decision benchmark: DEEP's
projected boxes, height >= 10 px, coarse classes vehicle / person / cyclist.  Operating threshold conf >= 0.25 for
every method (the one the decisions use).
  AP50        per coarse class, all-point VOC AP over the kept detections (so AP is truncated at 0.25), mean of 3
  R_all       recall at IoU 0.5 (class-aware)
  R_small     recall of GT boxes < 32 px tall
  R_rel       recall of decision-relevant GT: objects whose own required deceleration reaches DECELERATE
  FP/frame    unmatched detections per frame
Methods: cheap 320, u640 (cache), u960, SAHI s06 / s10, UFP 640 / 960 / 1333, FOVEA KDE_I / KDE_S 960 (frozen),
FOVEA-ft 960, u960-ft (plain), LZU frozen 960, LZU-ft 960, plainrect-ft 960 (those whose weights exist).
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

B = Path(os.environ["B"])
sys.path.insert(0, str(B / "scripts" / "A_method"))
from rap import planner as P                                                     # noqa: E402
from rap.cache import DetCache                                                   # noqa: E402
from rap.cache import Camera                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                    # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL                                   # noqa: E402

W0, H0, THR = 1600, 900, 0.25
CLASSES = ("vehicle", "person", "cyclist")
SAHI_COARSE = {0: "person", 1: "cyclist", 2: "vehicle", 3: "cyclist", 5: "vehicle", 7: "vehicle"}


def keep(d):
    k = np.asarray(d["conf"]) >= THR
    return {"xyxy": np.asarray(d["xyxy"], float).reshape(-1, 4)[k], "conf": np.asarray(d["conf"], float)[k],
            "coarse": np.asarray(d["coarse"])[k]}


def build_methods(only):
    import lzu_yolo
    import ufp_baseline as U
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    FB = runpy.run_path(str(B / "scripts" / "A_method" / "fovea_baseline.py"), run_name="fovea_lib")
    GG = FB["GG"]
    coco = str(B / "models" / "yolov8s.pt")
    det = TwoFidelityDetector(coco, device="cuda:0", backend="torch")
    M = {}
    M["u960"] = lambda img, cheap: det.detect(img, Mode("f576x960", 576, 960))[0]
    for k, sz in {"sahi_s06": 384, "sahi_s10": 640}.items():
        sm = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=coco, confidence_threshold=0.25,
                                                device="cuda:0", image_size=sz)

        def f(img, cheap, sm=sm):
            res = get_sliced_prediction(img[:, :, ::-1].copy(), sm, slice_height=640, slice_width=640, overlap_height_ratio=0.2,
                                        overlap_width_ratio=0.2, perform_standard_pred=True, verbose=0)
            ob = [o for o in res.object_prediction_list if o.category.id in SAHI_COARSE]
            return {"xyxy": np.array([o.bbox.to_xyxy() for o in ob], float).reshape(-1, 4),
                    "conf": np.array([o.score.value for o in ob], float),
                    "coarse": np.array([SAHI_COARSE[o.category.id] for o in ob]) if ob else np.zeros(0, "<U8")}
        M[k] = f
    for k, S in U.SIZES.items():
        M[k] = lambda img, cheap, S=S: U.ufp_detect(det, img, cheap["xyxy_all"], S)[0]
    gen = GG.PlainKDEGrid(output_shape=(576, 960), grid_shape=(31, 51), separable=True, attraction_fwhm=4, anti_crop=True,
                          bandwidth_scale=64, amplitude_scale=1)
    sal_s = torch.load(B / "results" / "fovea" / "dataset_saliency.pt").cuda()
    meta = [{"pad_shape": (H0, W0, 3)}]
    det_fw = B / "out" / "fovea_ft" / "warped" / "weights" / "best.pt"
    det_w = TwoFidelityDetector(str(det_fw), device="cuda:0", backend="torch") if det_fw.exists() else None

    def fovea(img, cheap, var, d_):
        x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().cuda()
        with torch.no_grad():
            if var == "kdei":
                grid = gen(x, meta, [torch.tensor(cheap["xyxy"], dtype=torch.float32, device="cuda").reshape(-1, 4)])
            else:
                grid = gen.separable_saliency_to_grid(x, meta, sal_s.sum(dim=2), sal_s.sum(dim=3), x.device)
            w = F.grid_sample(x, grid, align_corners=True)
        wimg = w[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()[:, :, ::-1].copy()
        d, _ = d_.detect(wimg, Mode("fov576x960", 576, 960))
        if len(d["conf"]):
            bb = GG.unwarp_bboxes(torch.tensor(d["xyxy"], dtype=torch.float32, device="cuda"), grid[0], (H0, W0)).cpu().numpy()
        else:
            bb = np.zeros((0, 4))
        return {"xyxy": bb, "conf": d["conf"], "coarse": d["coarse"]}
    M["fovea_kdei960"] = lambda img, cheap: fovea(img, cheap, "kdei", det)
    M["fovea_kdes960"] = lambda img, cheap: fovea(img, cheap, "kdes", det)
    if det_w is not None:
        M["fovea_ft960"] = lambda img, cheap: fovea(img, cheap, "kdei", det_w)
    pf = B / "out" / "fovea_ft" / "plain" / "weights" / "best.pt"
    if pf.exists():
        det_p = TwoFidelityDetector(str(pf), device="cuda:0", backend="torch")
        M["u960_ft"] = lambda img, cheap: det_p.detect(img, Mode("f576x960", 576, 960))[0]

    def stretch(d_, img):
        x = cv2.resize(img, (960, 576), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(np.ascontiguousarray(x[:, :, ::-1].transpose(2, 0, 1))).to(d_.device)
        x = ((x.half() if d_.half else x.float()) / 255.0)[None]
        with torch.no_grad():
            dd = d_.postprocess(d_.forward(x), Mode("s576x960", 576, 960), 1.0, (0, 0), (576, 960))
        dd["xyxy"] = dd["xyxy"].astype(float) * np.array([W0 / 960, H0 / 576, W0 / 960, H0 / 576])
        return dd
    sal = torch.load(B / "results" / "lzu" / "saliency_27x48.pt")
    d_l = TwoFidelityDetector(coco, device="cuda:0", backend="torch")
    lzu_yolo.patch(d_l.net, sal)
    M["lzu_frozen960"] = lambda img, cheap: stretch(d_l, img)
    for arm in ("lzu", "plainrect"):
        w = B / "out" / "lzu" / arm / "weights" / "last.pt"
        if w.exists():
            d_a = TwoFidelityDetector(str(w), device="cuda:0", backend="torch")
            M[f"{arm}_ft960"] = lambda img, cheap, d_a=d_a: stretch(d_a, img)
    uf = B / "out" / "ufp_ft" / "mosaic" / "weights" / "last.pt"
    if uf.exists():
        det_u = TwoFidelityDetector(str(uf), device="cuda:0", backend="torch")
        for k, S in (("ufp_ft_640", 640), ("ufp_ft_960", 960)):
            M[k] = lambda img, cheap, S=S: U.ufp_detect(det_u, img, cheap["xyxy_all"], S)[0]
    ed = B / "out" / "esod" / "test_new" / "labels"
    if ed.exists():                                   # ESOD's own saved predictions (their test.py --save-txt --save-conf)
        ec = {0: "vehicle", 1: "person", 2: "cyclist"}

        def esod(img, cheap):
            f = ed / f"{cheap['name']}.txt"
            a = np.loadtxt(f, ndmin=2) if f.exists() and f.stat().st_size else np.zeros((0, 6))
            xyxy = np.stack([(a[:, 1] - a[:, 3] / 2) * W0, (a[:, 2] - a[:, 4] / 2) * H0, (a[:, 1] + a[:, 3] / 2) * W0,
                             (a[:, 2] + a[:, 4] / 2) * H0], 1) if len(a) else np.zeros((0, 4))
            return {"xyxy": xyxy, "conf": a[:, 5] if len(a) else np.zeros(0),
                    "coarse": np.array([ec[int(k)] for k in a[:, 0]]) if len(a) else np.zeros(0, "<U8")}
        M["esod"] = esod
    return {k: v for k, v in M.items() if not only or k in only}


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])                          # noqa: E731
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def match(d, gt_xyxy, gt_cls):
    """Greedy by confidence, class-aware, IoU >= 0.5. Returns tp flags per detection and matched flags per GT."""
    tp = np.zeros(len(d["conf"]), bool)
    hit = np.zeros(len(gt_cls), bool)
    if len(d["conf"]) and len(gt_cls):
        M = iou(d["xyxy"], gt_xyxy)
        M[np.asarray(d["coarse"])[:, None] != np.asarray(gt_cls)[None, :]] = 0
        for i in np.argsort(-d["conf"], kind="stable"):
            j = np.argmax(np.where(hit, -1, M[i]))
            if M[i, j] >= 0.5 and not hit[j]:
                tp[i], hit[j] = True, True
    return tp, hit


def ap_allpoint(conf, tp, n_gt):
    if n_gt == 0:
        return np.nan
    o = np.argsort(-conf, kind="stable")
    tpc = np.cumsum(tp[o]); fpc = np.cumsum(~tp[o])
    rec = tpc / n_gt; prec = tpc / np.maximum(tpc + fpc, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    scenes = sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq)[: args.max_scenes or None]
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    pp = P.PlannerParams()
    methods = build_methods(args.only)
    names = ["cheap", "u640"] + list(methods)
    acc = {m: {"conf": {c: [] for c in CLASSES}, "tp": {c: [] for c in CLASSES}, "hit_all": [], "hit_small": [],
               "hit_rel": [], "fp": 0, "rel_det": 0, "rel_tp": 0} for m in names}
    ngt = {c: 0 for c in CLASSES}
    nfr = 0
    for s in scenes:
        geom, speeds = ad.geometry(s), ad.speeds(s)
        c3 = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
        c6 = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_full_640" / f"{s}.npz")
        toks = db.samples(by[s])
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        for i, fr in enumerate(c3.frames):
            if i % args.step:
                continue
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            gx = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
            gc = np.asarray(g["coarse"]).astype(str) if len(g) else np.zeros(0, "<U8")
            small = (gx[:, 3] - gx[:, 1]) < 32
            rel = np.array([P.discrete_action(P.required_decel(g["long_near"][j:j + 1], g["lat_min"][j:j + 1],
                                                               g["lat_max"][j:j + 1], g["ttc"][j:j + 1], v, pp)[0], pp) > 0
                            for j in range(len(g))], bool)
            for c in CLASSES:
                ngt[c] += int((gc == c).sum())
            nfr += 1
            d3 = c3.det(i)
            cheap = {"xyxy": d3["xyxy"][d3["conf"] >= THR].astype(float), "xyxy_all": d3["xyxy"].astype(float),
                     "name": f"{s}_{fr:03d}"}
            img = cv2.imread(str(db.image_path(toks[fr])))
            outs = {"cheap": d3, "u640": c6.det(c6.index[fr])}
            for m, fn in methods.items():
                outs[m] = fn(img, cheap)
            for m, d in outs.items():
                d = keep(d)
                tp, hit = match(d, gx, gc)
                A = acc[m]
                for c in CLASSES:
                    kk = d["coarse"] == c
                    A["conf"][c].append(d["conf"][kk]); A["tp"][c].append(tp[kk])
                A["hit_all"].append(hit); A["hit_small"].append(hit[small]); A["hit_rel"].append(hit[rel])
                A["fp"] += int((~tp).sum())
                # relevant precision: detections whose own single-frame monocular state demands braking (no track
                # history for any method here: ttc from the lift without a previous frame)
                if len(d["conf"]):
                    gk = predicted_geometry({"xyxy": d["xyxy"], "coarse": d["coarse"]}, None, cam.calib, cam.cam_h, cam.dt,
                                            cam_to_ego=(R, t))
                    mono = [np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc")]
                    own = np.array([P.discrete_action(P.required_decel(*(a[j:j + 1] for a in mono), v, pp)[0], pp) > 0
                                    for j in range(len(d["conf"]))], bool)
                    A["rel_det"] += int(own.sum()); A["rel_tp"] += int((own & tp).sum())
        print(f"  {s} ({nfr} frames)", flush=True)
    rows = []
    for m in names:
        A = acc[m]
        aps = [ap_allpoint(np.concatenate(A["conf"][c]), np.concatenate(A["tp"][c]), ngt[c]) for c in CLASSES]
        cat = lambda k: np.concatenate(A[k]) if A[k] else np.zeros(0)                    # noqa: E731
        rows.append({"method": m, "AP50": np.nanmean(aps), **{f"AP50_{c}": a for c, a in zip(CLASSES, aps)},
                     "R_all": cat("hit_all").mean(), "R_small": cat("hit_small").mean(), "R_rel": cat("hit_rel").mean(),
                     "n_rel": len(cat("hit_rel")), "FP_per_frame": A["fp"] / nfr,
                     "P_rel_nohist": A["rel_tp"] / max(A["rel_det"], 1),
                     "rel_FP_per_1k_frames_nohist": 1000 * (A["rel_det"] - A["rel_tp"]) / nfr})
    res = pd.DataFrame(rows)
    out = B / "results" / "det_metrics"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"newtest_step{args.step}{args.tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"{nfr} frames, GT per class {ngt}")
    print(res.round(3).to_string(index=False))
    print("DET METRICS DONE")


if __name__ == "__main__":
    main()
