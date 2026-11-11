#!/usr/bin/env python
"""FOVEA with fine-tuning (their main setting) and the matching controls, nuScenes, canvas 576x960.

  build   warped training set: every new-TRAIN keyframe warped with the official PlainKDEGrid (KDE_I from the
          frame's cheap draft detections, as at test time) to 576x960; GT boxes mapped into the warped canvas
          through the inverse of the separable grid (exact per axis); YOLO labels in COCO ids. A plain (unwarped)
          control set = the same frames resized to 576x960 with the same labels.
  train   done by the chain with scripts/A_ext/d68_train.py (YOLOv8s, imgsz 960, 15 epochs) for both sets
  eval    new test scenes, braking, mono lift: FOVEA-ft (warped, fine-tuned-on-warped detector), uniform-ft 960 (plain
          fine-tuned detector on the whole frame), and v3 with the plain fine-tuned detector as the crop verifier
          (draft = COCO cheap, as everywhere), plus the COCO references from the confirmatory run.
"""
from __future__ import annotations

import argparse
import math
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
FB = runpy.run_path(str(B / "scripts" / "A_method" / "fovea_baseline.py"), run_name="fovea_lib")
GG = FB["GG"]
from rap import planner as P                                                     # noqa: E402
from rap.cache import Camera, DetCache                                           # noqa: E402
from rap.decision import _apply_range_source                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                    # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL                                   # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

H0, W0, HC, WC = 900, 1600, 576, 960
COCO = {"vehicle": 2, "person": 0, "cyclist": 1}
DATA = B / "data" / "fovea_ft"


def make_gen():
    return GG.PlainKDEGrid(output_shape=(HC, WC), grid_shape=(31, 51), separable=True, attraction_fwhm=4,
                           anti_crop=True, bandwidth_scale=64, amplitude_scale=1)


def warp(gen, img, boxes):
    x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().cuda()
    with torch.no_grad():
        grid = gen(x, [{"pad_shape": (H0, W0, 3)}], [torch.tensor(np.asarray(boxes, float).reshape(-1, 4), dtype=torch.float32, device="cuda")])
        w = F.grid_sample(x, grid, align_corners=True)
    return w[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()[:, :, ::-1].copy(), grid[0]


def forward_boxes(grid, boxes):
    """Input-image boxes -> warped-canvas boxes, by inverting the separable grid per axis."""
    g = grid.cpu().numpy()
    xs = (g[0, :, 0] + 1) / 2 * W0              # input x for each output column
    ys = (g[:, 0, 1] + 1) / 2 * H0              # input y for each output row
    cols, rows = np.arange(WC, dtype=float), np.arange(HC, dtype=float)
    out = np.array(boxes, float).reshape(-1, 4).copy()
    out[:, [0, 2]] = np.interp(out[:, [0, 2]], xs, cols)
    out[:, [1, 3]] = np.interp(out[:, [1, 3]], ys, rows)
    return out


def yline(c, b, w, h):
    x1, y1, x2, y2 = np.clip(b, [0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1])
    return f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def stage_build(db, ad, by, sp):
    gen = make_gen()
    for arm in ("warped", "plain"):
        for sub in ("images/train", "labels/train", "images/val", "labels/val"):
            (DATA / arm / sub).mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        for s in sorted(sp[(sp["set"] == "new") & (sp.split == split)].seq):
            g_all = ad.geometry(s)
            c = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
            for i, fr in enumerate(c.frames):
                fr = int(fr)
                d = c.det(i)
                cheap = d["xyxy"][d["conf"] >= 0.25]
                g = g_all[(g_all["frame"] == fr) & ((g_all["y2"] - g_all["y1"]) >= 10)]
                gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
                cls = [COCO[str(x)] for x in g["coarse"]] if len(g) else []
                img = cv2.imread(str(db.image_path(db.samples(by[s])[fr])))
                name = f"{s}_{fr:03d}"
                wimg, grid = warp(gen, img, cheap)
                wb = forward_boxes(grid, gt)
                cv2.imwrite(str(DATA / "warped" / f"images/{split}" / f"{name}.jpg"), wimg, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (DATA / "warped" / f"labels/{split}" / f"{name}.txt").write_text(
                    "\n".join(yline(k, b, WC, HC) for k, b in zip(cls, wb) if (b[2] - b[0]) > 2 and (b[3] - b[1]) > 2))
                cv2.imwrite(str(DATA / "plain" / f"images/{split}" / f"{name}.jpg"), cv2.resize(img, (WC, HC), interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                sc = np.array([WC / W0, HC / H0, WC / W0, HC / H0])
                (DATA / "plain" / f"labels/{split}" / f"{name}.txt").write_text(
                    "\n".join(yline(k, b * sc, WC, HC) for k, b in zip(cls, gt)))
            print(f"  {split} {s}", flush=True)
    names = {i: f"c{i}" for i in range(80)}
    names.update({0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"})
    for arm in ("warped", "plain"):
        (DATA / arm / "data.yaml").write_text(f"path: {DATA / arm}\ntrain: images/train\nval: images/val\nnames:\n" +
                                             "".join(f"  {i}: {n}\n" for i, n in names.items()))


def stage_eval(db, ad, by, sp):
    gen = make_gen()
    sfx = os.environ.get("FT_SUFFIX", "")                # e.g. "_s1": the second-seed detectors (warped_s1, plain_s1)
    det_w = TwoFidelityDetector(str(B / "out" / "fovea_ft" / f"warped{sfx}" / "weights" / "best.pt"), device="cuda:0", backend="torch")
    det_p = TwoFidelityDetector(str(B / "out" / "fovea_ft" / f"plain{sfx}" / "weights" / "best.pt"), device="cuda:0", backend="torch")
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    rows = []
    for s in sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq):
        geom, speeds = ad.geometry(s), ad.speeds(s)
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        c = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
        toks = db.samples(by[s])
        prev_act, prev, prev_scene = None, {}, None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            act = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)             # noqa: E731
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]                   # noqa: E731
            lift = lambda dd, pv: tuple(np.asarray(q, float) for q in (lambda gg: (gg["z"], gg["lat_min"], gg["lat_max"], gg["ttc"]))(  # noqa: E731
                predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))))
            dc = c.det(i)
            kc = dc["conf"] >= 0.25
            geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
            bxc = dc["xyxy"][kc].astype(float)
            cons_c = tuple(np.asarray(a, float) for a in _apply_range_source(bxc, geo_c, g, "mono", rng, 0.0))
            a_c = act(cons_c)
            img = cv2.imread(str(db.image_path(toks[fr])))
            rec = {"seq": s, "frame": fr, "J_c": cost(a_c)}
            # FOVEA-ft
            wimg, grid = warp(gen, img, bxc)
            d, _ = det_w.detect(wimg, Mode("fw", HC, WC))
            k = d["conf"] >= 0.25
            bb = GG.unwarp_bboxes(torch.tensor(d["xyxy"][k], dtype=torch.float32, device="cuda"), grid, (H0, W0)).cpu().numpy() \
                if k.any() else np.zeros((0, 4))
            dd = {"xyxy": bb.astype(float), "coarse": d["coarse"][k]}
            rec["J_fovea_ft"] = cost(act(lift(dd, prev.get("f"))))
            prev["f"] = dd
            # uniform fine-tuned 960
            d, _ = det_p.detect(img, Mode("fp", HC, WC))
            k = d["conf"] >= 0.25
            dd = {"xyxy": d["xyxy"][k].astype(float), "coarse": d["coarse"][k]}
            rec["J_u960_ft"] = cost(act(lift(dd, prev.get("u"))))
            prev["u"] = dd
            # v3 with the fine-tuned verifier on crops

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * 0.6 / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * 0.6 / 32)) * 32)
                dk, _ = det_p.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))
                kk = dk["conf"] >= 0.25
                return {"xyxy": dk["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                        "conf": dk["conf"][kk].astype(float), "coarse": dk["coarse"][kk]}
            cheap_d = {"xyxy": bxc, "conf": dc["conf"][kc].astype(float), "coarse": dc["coarse"][kc], "mono": cons_c}
            scene, crops, _ = SP.run(cheap_d, v, cam, W0, H0, detect_crop, lambda ff: lift(ff, prev_scene), pp, v_close=10.0, guard=True)
            rec["J_v3_ftverifier"] = cost(act(scene["mono"]))
            prev_scene = {"xyxy": scene["xyxy"]}
            rows.append(rec)
            prev_act = a_c
        print(f"  {s}", flush=True)
    d = pd.DataFrame(rows)
    out = B / "results" / "fovea"
    d.to_pickle(out / f"eval_ft_newtest{sfx}.pkl")
    ref = pd.read_pickle(B / "results" / "nusc_full" / "eval__new_test_v3_confirm.pkl")
    m = d.merge(ref[["seq", "frame", "J_c", "J_u960", "J_spec", "J_spec_ref_resnet18"]], on=["seq", "frame"], suffixes=("", "_ref"))
    assert np.allclose(m.J_c, m.J_c_ref)
    L = m.J_c.to_numpy()
    uq = m.seq.unique(); idx = [np.flatnonzero(m.seq.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    V = {k_: (m.J_c - m[c_]).to_numpy() for k_, c_ in (("fovea_ft", "J_fovea_ft"), ("u960_ft", "J_u960_ft"),
                                                      ("v3_ftverifier", "J_v3_ftverifier"), ("u960_coco", "J_u960"),
                                                      ("v3_coco", "J_spec"), ("v3_ref_coco", "J_spec_ref_resnet18"))}
    rows_ = []
    for k_, v_ in V.items():
        r = {"method": k_, "G": v_.sum() / L.sum()}
        for ref_k in ("v3_ftverifier", "v3_ref_coco"):
            dr = [(V[ref_k][t_].sum() - v_[t_].sum()) / L[t_].sum() for t_ in boots]
            r[f"{ref_k}_minus"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        rows_.append(r)
    res = pd.DataFrame(rows_)
    res.to_csv(out / f"summary_ft_newtest{sfx}.csv", index=False)
    print(res.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "eval"])
    args = ap.parse_args()
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    {"build": stage_build, "eval": stage_eval}[args.stage](db, ad, by, sp)


if __name__ == "__main__":
    main()
