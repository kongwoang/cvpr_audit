#!/usr/bin/env python
"""Baseline: LZU, Learning to Zoom and Unzoom (Thavamani et al., CVPR 2023), fixed-saliency LZU around YOLOv8s.

Official zoom / unzoom code (tchittesh/lzu @ 361afb2) via scripts/common/lzu_yolo.py; settings of their nuScenes
configs: separable grid 27 x 48, attraction_fwhm 10, anti_crop, dataset-wide saliency (FOVEA's KDE_SD recipe:
bbox2sal of all new-TRAIN ground-truth boxes, bandwidth_scale 64, amplitude_scale 1).

  sal     the saliency (1, 1, 27, 48) on the 576x960 training canvas (the nuScenes frame resized to it, as the
          fine-tuning images of data/fovea_ft/plain)
  train   --arm lzu | plainrect: YOLOv8s (COCO init) fine-tuned on data/fovea_ft/plain (the new-TRAIN keyframes, val on
          new VAL), 15 epochs, imgsz 960, batch 32 (the FOVEA-ft recipe), with LZU's own augmentation regime: rect
          576x960 inputs (no mosaic), no scale / translate jitter, horizontal flip.  plainrect = the same recipe
          without zoom/unzoom (the matching uniform control).  last.pt is evaluated for both (the in-training val uses
          608x960 padded rect batches, so best.pt selection would see a shifted saliency layout).
  eval    new TEST scenes, braking, monocular lift with each arm's own previous detections, cheap action history:
            lzu_frozen_640 / lzu_frozen_960   COCO YOLOv8s + LZU, no fine-tuning (LZU's zero-shot use), 384x640 / 576x960
            lzu_ft_960                        fine-tuned LZU
            plainrect_ft_960                  fine-tuned control
          compared with the confirmatory v3 / v3+ref / u960 and the FOVEA-ft run (u960_ft, v3 with the fine-tuned
          verifier); paired scene bootstrap, 2,000 draws.  Frames are resized (not letterboxed) to the canvas, as in
          training; boxes are mapped back per axis.
"""
from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

B = Path(os.environ["B"])
OUT = B / "results" / "lzu"
W0, H0 = 1600, 900
THR = 0.25
CANV = {"640": (384, 640), "960": (576, 960)}
SEED = int(os.environ.get("SEED", 0))                 # further seeds: out/lzu/<arm>_s<seed>, results tagged _s<seed>
SFX = f"_s{SEED}" if SEED else ""


def load_sal():
    return torch.load(OUT / "saliency_27x48.pt")


def stage_sal():
    from rap.nusc import NuScenesDB, make_adapter
    from rap.paths import NUSCENES_TRAINVAL
    FB = runpy.run_path(str(B / "scripts" / "A_method" / "fovea_baseline.py"), run_name="fovea_lib")
    gen = FB["GG"].PlainKDEGrid(output_shape=CANV["960"], grid_shape=(27, 48), separable=True, attraction_fwhm=10,
                                anti_crop=True, bandwidth_scale=64, amplitude_scale=1)
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    ad = make_adapter(NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval"))
    H, W = CANV["960"]
    sc = torch.tensor([W / W0, H / H0, W / W0, H / H0], device="cuda")
    meta = [{"pad_shape": (H, W, 3)}]
    acc = None
    for s in sorted(sp[(sp["set"] == "new") & (sp.split == "train")].seq):
        g = ad.geometry(s)
        for fr in np.unique(g["frame"]):
            q = g[(g["frame"] == fr) & ((g["y2"] - g["y1"]) >= 10)]
            if not len(q):
                continue
            bx = torch.tensor(np.stack([q["x1"], q["y1"], q["x2"], q["y2"]], 1), dtype=torch.float32, device="cuda") * sc
            sal = gen.bbox2sal([bx.clone()], meta)
            acc = sal if acc is None else acc + sal
    sal = (acc / acc.sum()).reshape(1, 1, 27, 48).cpu()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(sal, OUT / "saliency_27x48.pt")
    print("saliency rows (sum over x):", np.round(sal.sum(dim=3).flatten().numpy() * 27, 2))
    print("saliency cols (sum over y):", np.round(sal.sum(dim=2).flatten().numpy() * 48, 2))


def stage_train(arm):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    import lzu_yolo

    sal = load_sal()

    class LZUTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return lzu_yolo.patch(super().get_model(cfg, weights, verbose), sal)

    m = YOLO(str(B / "models" / "yolov8s.pt"))
    kw = dict(data=str(B / "data" / "fovea_ft" / "plain" / "data.yaml"), imgsz=960, epochs=15, batch=32, workers=8, device=0,
              amp=False, plots=False, project=str(B / "out" / "lzu"), name=arm + SFX, exist_ok=True, seed=SEED, deterministic=True,
              verbose=False, rect=True, scale=0.0, translate=0.0, fliplr=0.5)
    m.train(trainer=LZUTrainer if arm == "lzu" else DetectionTrainer, **kw)


def stage_eval(args):
    import cv2
    import lzu_yolo
    from rap import planner as P
    from rap.cache import Camera, DetCache
    from rap.decision import _apply_range_source
    from rap.detect import Mode, TwoFidelityDetector
    from rap.mono import predicted_geometry
    from rap.nusc import NuScenesDB, make_adapter
    from rap.paths import CACHE, NUSCENES_TRAINVAL

    sal = load_sal()
    coco = str(B / "models" / "yolov8s.pt")
    arms = {}                                              # name -> (detector, canvas)
    for c in ("640", "960"):
        d = TwoFidelityDetector(coco, device="cuda:0", backend="torch")
        lzu_yolo.patch(d.net, sal)
        arms[f"lzu_frozen_{c}"] = (d, CANV[c])
    for arm in ("lzu", "plainrect"):
        w = B / "out" / "lzu" / (arm + SFX) / "weights" / "last.pt"
        if w.exists():
            arms[f"{arm}_ft_960"] = (TwoFidelityDetector(str(w), device="cuda:0", backend="torch"), CANV["960"])
    if args.stretch_check:
        arms["u960_stretch"] = (TwoFidelityDetector(coco, device="cuda:0", backend="torch"), CANV["960"])
    print("arms:", list(arms), flush=True)

    def detect(det, img, hw):
        H, W = hw
        x = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(np.ascontiguousarray(x[:, :, ::-1].transpose(2, 0, 1))).to(det.device)
        x = ((x.half() if det.half else x.float()) / 255.0)[None]
        dd = det.postprocess(det.forward(x), Mode(f"s{H}x{W}", H, W), 1.0, (0, 0), (H, W))
        k = dd["conf"] >= THR
        xyxy = dd["xyxy"][k].astype(float) * np.array([W0 / W, H0 / H, W0 / W, H0 / H])
        return {"xyxy": xyxy, "coarse": dd["coarse"][k]}

    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    scenes = sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq)[: args.max_scenes or None]
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
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
            prev_act, prev = None, {k: None for k in arms}
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
                rec = {"seq": s, "frame": fr, "J_c": cost(a_c)}
                img = cv2.imread(str(db.image_path(toks[fr])))
                for k, (det, hw) in arms.items():
                    dd = detect(det, img, hw)
                    gg = predicted_geometry(dd, prev[k], cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                    prev[k] = dd
                    rec[f"J_{k}"] = cost(act(tuple(np.asarray(gg[q], float) for q in ("z", "lat_min", "lat_max", "ttc"))))
                rows.append(rec)
                prev_act = a_c
            print(f"  {s}", flush=True)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_pickle(OUT / f"eval_newtest{args.tag}{SFX}.pkl")
    ref = pd.read_pickle(B / "results" / "nusc_full" / "eval__new_test_v3_confirm.pkl")
    ft = pd.read_pickle(B / "results" / "fovea" / "eval_ft_newtest.pkl")
    m = d.merge(ref[["seq", "frame", "J_c", "J_u960", "J_spec", "J_spec_ref_resnet18"]], on=["seq", "frame"], suffixes=("", "_ref"))
    m = m.merge(ft[["seq", "frame", "J_u960_ft", "J_v3_ftverifier"]], on=["seq", "frame"])
    assert np.allclose(m.J_c, m.J_c_ref)
    L = m.J_c.to_numpy()
    uq = m.seq.unique(); idx = [np.flatnonzero(m.seq.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    V = {k: (m.J_c - m[f"J_{k}"]).to_numpy() for k in arms}
    V.update({"u960": (m.J_c - m.J_u960).to_numpy(), "v3": (m.J_c - m.J_spec).to_numpy(),
              "v3_ref": (m.J_c - m.J_spec_ref_resnet18).to_numpy(), "u960_ft": (m.J_c - m.J_u960_ft).to_numpy(),
              "v3_ftverifier": (m.J_c - m.J_v3_ftverifier).to_numpy()})
    res = []
    for k, v_ in V.items():
        r = {"method": k, "G": v_.sum() / L.sum(), "harmed": float(np.mean(v_ < 0))}
        for ref_k in ("v3", "v3_ref", "v3_ftverifier"):
            dr = [(V[ref_k][t_].sum() - v_[t_].sum()) / L[t_].sum() for t_ in boots]
            r[f"{ref_k}_minus"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        res.append(r)
    res = pd.DataFrame(res)
    res.to_csv(OUT / f"summary_newtest{args.tag}{SFX}.csv", index=False)
    print(res.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["sal", "train", "eval"])
    ap.add_argument("--arm", default="lzu", choices=["lzu", "plainrect"])
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--stretch_check", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.stage == "sal":
        stage_sal()
    elif args.stage == "train":
        stage_train(args.arm)
    else:
        stage_eval(args)
    print(f"LZU {args.stage.upper()} DONE", flush=True)


if __name__ == "__main__":
    main()
