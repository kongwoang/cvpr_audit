#!/usr/bin/env python
"""Baseline: FOVEA (Thavamani et al., ICCV 2021) — foveated image magnification — with the OFFICIAL grid generator.

`third_party/fovea/fovea/models/grid_generator.py` (commit c8707bd) is imported unchanged; its mmcv/mmdet registry
imports are satisfied by stubs (only the decorator is used). Settings = the paper's configs: separable grid
(31 x 51), attraction_fwhm 4, anti_crop, KDE bandwidth_scale 64, amplitude_scale 1.
  KDE_I  image-adaptive saliency from boxes: here the CURRENT cheap draft's detections (available to every pipeline)
  KDE_S  dataset-wide saliency: the KDE of all ground-truth boxes of the new TRAIN scenes (their compute_dataset_saliency
         recipe, re-run on nuScenes)
The full-resolution frame is warped to a canvas of the uniform modes' size (384x640 = u640 compute, 576x960 = u960
compute), YOLOv8s (COCO, not fine-tuned on warped images — FOVEA's no-finetuning setting) detects on it, boxes are
unwarped with FOVEA's `unwarp_bboxes`, then DEEP's mono lift, braking decision, cost with cheap history.
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

B = Path(os.environ["B"])
if not hasattr(np, "float"):
    np.float = float                                   # the official module (2022) still uses the removed alias
# --- stubs for the registry imports of the official module --------------------------------------------------------
_mmcv, _mmcv_utils = types.ModuleType("mmcv"), types.ModuleType("mmcv.utils")


class _Registry:
    def __init__(self, name): self.name = name
    def register_module(self, *a, **k): return (lambda c: c)


_mmcv_utils.Registry = _Registry
_mmdet, _mmdet_models, _mmdet_builder = (types.ModuleType(n) for n in ("mmdet", "mmdet.models", "mmdet.models.builder"))
_mmdet_builder.build = lambda *a, **k: None
sys.modules.update({"mmcv": _mmcv, "mmcv.utils": _mmcv_utils, "mmdet": _mmdet, "mmdet.models": _mmdet_models,
                    "mmdet.models.builder": _mmdet_builder})
_pkg = types.ModuleType("fovea_official")
_pkg.__path__ = [str(B / "third_party" / "fovea" / "fovea")]
_utils = types.ModuleType("fovea_official.utils")
_vis = types.ModuleType("fovea_official.utils.vis")
_vis.vis_batched_imgs = lambda *a, **k: None
sys.modules.update({"fovea_official": _pkg, "fovea_official.utils": _utils, "fovea_official.utils.vis": _vis})
import importlib.util                                                             # noqa: E402
_spec = importlib.util.spec_from_file_location("fovea_official.models.grid_generator",
                                               B / "third_party" / "fovea" / "fovea" / "models" / "grid_generator.py")
GG = importlib.util.module_from_spec(_spec)
GG.__package__ = "fovea_official.models"
sys.modules["fovea_official.models"] = types.ModuleType("fovea_official.models")
_spec.loader.exec_module(GG)

from rap import planner as P                                                     # noqa: E402
from rap.cache import Camera, DetCache                                           # noqa: E402
from rap.decision import _apply_range_source                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                    # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL                                   # noqa: E402

CANVASES = {"640": (384, 640), "960": (576, 960)}
W0, H0 = 1600, 900


def dataset_saliency(ad, sp, grid):
    """KDE_S: the average KDE saliency of all training GT boxes (FOVEA's compute_dataset_saliency, re-run)."""
    meta = [{"pad_shape": (H0, W0, 3)}]
    acc = None
    for s in sorted(sp[(sp["set"] == "new") & (sp.split == "train")].seq):
        g = ad.geometry(s)
        for fr in np.unique(g["frame"]):
            q = g[(g["frame"] == fr) & ((g["y2"] - g["y1"]) >= 10)]
            if not len(q):
                continue
            bx = torch.tensor(np.stack([q["x1"], q["y1"], q["x2"], q["y2"]], 1), dtype=torch.float32, device="cuda")
            sal = grid.bbox2sal([bx], meta)
            acc = sal if acc is None else acc + sal
    return acc / acc.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_scenes", type=int, default=0)
    args = ap.parse_args()
    out = B / "results" / "fovea"
    out.mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    scenes = sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq)[: args.max_scenes or None]
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    grids = {c: GG.PlainKDEGrid(output_shape=hw, grid_shape=(31, 51), separable=True, attraction_fwhm=4,
                                anti_crop=True, bandwidth_scale=64, amplitude_scale=1) for c, hw in CANVASES.items()}
    sal_s = dataset_saliency(ad, sp, grids["640"])
    torch.save(sal_s.cpu(), out / "dataset_saliency.pt")
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    rows = []
    meta = [{"pad_shape": (H0, W0, 3)}]
    for s in scenes:
        geom, speeds = ad.geometry(s), ad.speeds(s)
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        c = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
        toks = db.samples(by[s])
        prev_act = None
        prev = {}
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            act = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)             # noqa: E731
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]                   # noqa: E731
            dc = c.det(i)
            kc = dc["conf"] >= 0.25
            geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
            bxc = dc["xyxy"][kc].astype(float)
            cons_c = tuple(np.asarray(a, float) for a in _apply_range_source(bxc, geo_c, g, "mono", rng, 0.0))
            a_c = act(cons_c)
            img = cv2.imread(str(db.image_path(toks[fr])))
            x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().cuda()
            rec = {"seq": s, "frame": fr, "J_c": cost(a_c)}
            for cname, (h, w) in CANVASES.items():
                gen = grids[cname]
                for var in ("kdei", "kdes"):
                    with torch.no_grad():
                        if var == "kdei":
                            grid = gen(x, meta, [torch.tensor(bxc, dtype=torch.float32, device="cuda")])
                        else:
                            sal = sal_s.cuda()
                            grid = gen.separable_saliency_to_grid(x, meta, sal.sum(dim=2), sal.sum(dim=3), x.device)
                        warped = F.grid_sample(x, grid, align_corners=True)
                    wimg = warped[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()[:, :, ::-1].copy()
                    d, _ = det.detect(wimg, Mode(f"fov{h}x{w}", h, w))
                    k = d["conf"] >= 0.25
                    if k.any():
                        bb = GG.unwarp_bboxes(torch.tensor(d["xyxy"][k], dtype=torch.float32, device="cuda"), grid[0], (H0, W0))
                        bb = bb.cpu().numpy().astype(float)
                    else:
                        bb = np.zeros((0, 4))
                    dd = {"xyxy": bb, "coarse": d["coarse"][k]}
                    key = f"{var}{cname}"
                    gg = predicted_geometry(dd, prev.get(key), cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                    prev[key] = dd
                    rec[f"J_fovea_{key}"] = cost(act(tuple(np.asarray(gg[q], float) for q in ("z", "lat_min", "lat_max", "ttc"))))
            rows.append(rec)
            prev_act = a_c
        print(f"  {s}", flush=True)
    d = pd.DataFrame(rows)
    d.to_pickle(out / "eval_newtest.pkl")
    ref = pd.read_pickle(B / "results" / "nusc_full" / "eval__new_test_v3_confirm.pkl")
    m = d.merge(ref[["seq", "frame", "J_c", "J_u960", "J_spec", "J_spec_ref_resnet18", "brake_V"]], on=["seq", "frame"],
                suffixes=("", "_ref"))
    assert np.allclose(m.J_c, m.J_c_ref)
    L = m.J_c.to_numpy()
    uq = m.seq.unique(); idx = [np.flatnonzero(m.seq.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    V = {"u640": m.brake_V.to_numpy(), "u960": (m.J_c - m.J_u960).to_numpy(), "v3": (m.J_c - m.J_spec).to_numpy(),
         "v3_ref": (m.J_c - m.J_spec_ref_resnet18).to_numpy()}
    V.update({k[8:]: (m.J_c - m[k]).to_numpy() for k in m.columns if k.startswith("J_fovea_")})
    out_rows = []
    for k_, v_ in V.items():
        r = {"method": k_, "G": v_.sum() / L.sum()}
        for ref_k in ("v3", "v3_ref"):
            dr = [(V[ref_k][t_].sum() - v_[t_].sum()) / L[t_].sum() for t_ in boots]
            r[f"{ref_k}_minus"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        out_rows.append(r)
    res = pd.DataFrame(out_rows)
    res.to_csv(out / "summary_newtest.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
