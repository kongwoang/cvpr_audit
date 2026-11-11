#!/usr/bin/env python
"""UFP with its fine detector trained on packed mosaics (UFPMP-Det's recipe: `build_dataset/UFP_VisDrone2COCO.py`
builds the MP-Det training set from the coarse detector's packing; MP-Det is then trained on it).

  build   every new-TRAIN / new-VAL keyframe: the cheap draft's boxes (conf >= 0.10, the cache floor) ->
          UnifiedForegroundPacking(1.5) -> mosaic (their draw_ufp_result); a GT box (height >= 10 px) is labelled in
          every chip it overlaps with IoF > 0.9, shifted by the chip offset and scaled by its factor (their rule,
          unclipped); YOLO labels in COCO ids; frames without coarse boxes produce no mosaic and are skipped.
  train   by the chain: scripts/A_ext/d68_train.py mosaic ufp_ft 15 yolov8s 960 32 (the FOVEA-ft / u960-ft recipe)
  eval    new TEST: UFP with the fine-tuned fine detector at S = 640 / 960, decisions as in ufp_baseline.py; compared
          with u960-ft (the plain fine-tuned detector on the whole frame) and v3 with the fine-tuned verifier.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

B = Path(os.environ["B"])
sys.path.insert(0, str(B / "scripts" / "A_method"))
import ufp_baseline as U                                                          # noqa: E402
from rap import planner as P                                                     # noqa: E402
from rap.cache import Camera, DetCache                                           # noqa: E402
from rap.decision import _apply_range_source                                     # noqa: E402
from rap.detect import TwoFidelityDetector                                       # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                    # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL                                   # noqa: E402

DATA = B / "data" / "ufp_ft" / "mosaic"
COCO = {"vehicle": 2, "person": 0, "cyclist": 1}
SIZES = {"ufp_ft_640": 640, "ufp_ft_960": 960}
SFX = os.environ.get("FT_SUFFIX", "")                  # e.g. _s1: the detector trained with another seed


def stage_build(db, ad, by, sp):
    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
    n_img = n_lab = 0
    for split in ("train", "val"):
        for s in sorted(sp[(sp["set"] == "new") & (sp.split == split)].seq):
            g_all = ad.geometry(s)
            c = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
            toks = db.samples(by[s])
            for i, fr in enumerate(c.frames):
                fr = int(fr)
                coarse = c.det(i)["xyxy"].astype(float)
                if not len(coarse):
                    continue
                rec, w, h = U.UnifiedForegroundPacking(coarse.copy(), 1.5, input_shape=[U.W0, U.H0])
                if not rec or w < 1 or h < 1:
                    continue
                img = cv2.imread(str(db.image_path(toks[fr])))
                mosaic = U.display_merge_result(rec, img, w, h).astype(np.uint8)
                mh, mw = mosaic.shape[:2]
                g = g_all[(g_all["frame"] == fr) & ((g_all["y2"] - g_all["y1"]) >= 10)]
                lines = []
                for chip in rec:
                    x1, y1, cw, ch, n_x, n_y, sf = [math.floor(_) for _ in chip]
                    t = [x1, y1, x1 + cw, y1 + ch]
                    for o in g:
                        gx1, gy1, gx2, gy2 = (float(o[k]) for k in ("x1", "y1", "x2", "y2"))
                        if U.compute_iof(t, [gx1, gy1, gx2, gy2]) > 0.9:
                            nx, ny = max(0, n_x + (gx1 - x1) * sf), max(0, n_y + (gy1 - y1) * sf)
                            bw, bh = (gx2 - gx1) * sf, (gy2 - gy1) * sf
                            bx2, by2 = min(mw, nx + bw), min(mh, ny + bh)      # YOLO labels must lie in the image
                            if bx2 - nx > 2 and by2 - ny > 2:
                                lines.append("%d %.6f %.6f %.6f %.6f" % (COCO[str(o["coarse"])], (nx + bx2) / 2 / mw,
                                                                         (ny + by2) / 2 / mh, (bx2 - nx) / mw, (by2 - ny) / mh))
                name = f"{s}_{fr:03d}"
                cv2.imwrite(str(DATA / f"images/{split}" / f"{name}.jpg"), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (DATA / f"labels/{split}" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                n_img += 1; n_lab += len(lines)
            print(f"  {split} {s}: {n_img} mosaics, {n_lab} labels", flush=True)
    names = {i: f"c{i}" for i in range(80)}
    names.update({0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"})
    (DATA / "data.yaml").write_text(f"path: {DATA}\ntrain: images/train\nval: images/val\nnames:\n" +
                                    "".join(f"  {i}: {n}\n" for i, n in names.items()))


def stage_eval(db, ad, by, sp):
    det = TwoFidelityDetector(str(B / "out" / "ufp_ft" / f"mosaic{SFX}" / "weights" / "last.pt"), device="cuda:0", backend="torch")
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    rows = []
    with torch.no_grad():
        for s in sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq):
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
                kc = dc["conf"] >= U.THR
                geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
                cons_c = tuple(np.asarray(a, float) for a in _apply_range_source(dc["xyxy"][kc].astype(float), geo_c, g, "mono", rng, 0.0))
                a_c = act(cons_c)
                rec = {"seq": s, "frame": fr, "J_c": cost(a_c)}
                img = cv2.imread(str(db.image_path(toks[fr])))
                for k, S in SIZES.items():
                    dd, tpx, _ = U.ufp_detect(det, img, dc["xyxy"], S)
                    gg = predicted_geometry(dd, prev[k], cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                    prev[k] = dd
                    rec[f"J_{k}"] = cost(act(tuple(np.asarray(gg[q], float) for q in ("z", "lat_min", "lat_max", "ttc"))))
                    rec[f"tpx_{k}"] = tpx / (U.W0 * U.H0)
                rows.append(rec)
                prev_act = a_c
            print(f"  {s}", flush=True)
    d = pd.DataFrame(rows)
    out = B / "results" / "ufp"
    d.to_pickle(out / f"eval_ft_newtest{SFX}.pkl")
    ref = pd.read_pickle(B / "results" / "nusc_full" / "eval__new_test_v3_confirm.pkl")
    ft = pd.read_pickle(B / "results" / "fovea" / "eval_ft_newtest.pkl")
    m = d.merge(ref[["seq", "frame", "J_c", "J_spec", "J_spec_ref_resnet18"]], on=["seq", "frame"], suffixes=("", "_ref"))
    m = m.merge(ft[["seq", "frame", "J_u960_ft", "J_v3_ftverifier"]], on=["seq", "frame"])
    assert np.allclose(m.J_c, m.J_c_ref)
    L = m.J_c.to_numpy()
    uq = m.seq.unique(); idx = [np.flatnonzero(m.seq.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    V = {k: (m.J_c - m[f"J_{k}"]).to_numpy() for k in SIZES}
    V.update({"u960_ft": (m.J_c - m.J_u960_ft).to_numpy(), "v3_ftverifier": (m.J_c - m.J_v3_ftverifier).to_numpy(),
              "v3_ref": (m.J_c - m.J_spec_ref_resnet18).to_numpy()})
    res = []
    for k, v_ in V.items():
        r = {"method": k, "G": v_.sum() / L.sum(), "harmed": float(np.mean(v_ < 0)),
             "tensor_px_share": float(m[f"tpx_{k}"].mean()) if k in SIZES else np.nan}
        for ref_k in ("v3_ftverifier", "v3_ref"):
            dr = [(V[ref_k][t_].sum() - v_[t_].sum()) / L[t_].sum() for t_ in boots]
            r[f"{ref_k}_minus"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        res.append(r)
    res = pd.DataFrame(res)
    res.to_csv(out / f"summary_ft_newtest{SFX}.csv", index=False)
    print(res.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "eval"])
    args = ap.parse_args()
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    (stage_build if args.stage == "build" else stage_eval)(db, ad, by, sp)
    print(f"UFP FT {args.stage.upper()} DONE", flush=True)


if __name__ == "__main__":
    main()
