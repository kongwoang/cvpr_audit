#!/usr/bin/env python
"""DEEP's nuScenes planning cells (plan_ade / plan_fde) for every pipeline of an evaluation dump.

Consumer = PKL's released planner ("Planner C", weights third_party/tip/planner.pt), scored against the REAL future
ego trajectory exactly as DEEP scripts 70 (rasters, targets) and 74 (argmax path, ADE/FDE) do; their functions are
imported, not re-implemented. Pipelines come from `nusc_ablate.py --dump` (cheap, uniform 960, speculative v3,
v3 + refiner) plus DEEP's cached full 640 mode; each becomes a nuScenes detection submission through DEEP's
`build_submission` (variant mono or oracle). Decision value per cell = (sum J_cheap - sum J_pipeline) / sum J_cheap.

usage: pkl_cells.py --dump results/nusc_full/scenes__new_test_dump.pkl --variant oracle [--chunk i --nchunks n]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

B = Path(os.environ["B"])
REPO = B / "code" / "risk-aware-perception"
sys.path.insert(0, str(REPO / "third_party" / "pkl"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S70 = _load("s70", REPO / "scripts" / "70_planner_d_data.py")
S74 = _load("s74", REPO / "scripts" / "74_plannerC_vs_truth.py")
from nuscenes.eval.common import loaders as L                                   # noqa: E402
from nuscenes.eval.detection.config import config_factory                      # noqa: E402
from nuscenes.eval.detection.data_classes import DetectionBox                  # noqa: E402
from nuscenes.map_expansion.map_api import NuScenesMap                         # noqa: E402
from nuscenes.nuscenes import NuScenes                                         # noqa: E402
from planning_centric_metrics.models import compile_model                      # noqa: E402
from rap import cache as _rc                                                    # noqa: E402
from rap.cache import DetCache                                                   # noqa: E402
from rap.ego_traj import future_waypoints                                        # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                    # noqa: E402
from rap.nusc_submission import build_submission, write_submission               # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL                                   # noqa: E402
from rap.planner_d import HORIZONS, ade_fde                                      # noqa: E402
from rap.risk import RiskConfig                                                  # noqa: E402


class MemCache:
    """The slice of rap.cache.DetCache that build_submission reads, over dumped scenes."""

    def __init__(self, entries, key):
        self.frames = [e["frame"] for e in entries]
        self._e = [e[key] for e in entries]

    def det(self, i):
        e = self._e[i]
        return {"xyxy": e["xyxy"], "conf": e["conf"], "coarse": e["coarse"]}

    def geo(self, i):
        z, lo, hi, tt = self._e[i]["mono"]
        return {"z": z, "lat_min": lo, "lat_max": hi, "ttc": tt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--variant", default="oracle", choices=["mono", "oracle"])
    ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dump = pd.read_pickle(B / args.dump)
    scenes = sorted({e["seq"] for e in dump})
    k, r = divmod(len(scenes), args.nchunks)
    lo = args.chunk * k + min(args.chunk, r)
    scenes = scenes[lo:lo + k + (1 if args.chunk < r else 0)]
    by_scene = {s: [e for e in dump if e["seq"] == s] for s in scenes}
    pipes = [p for p in dump[0] if p not in ("seq", "frame")]
    out = B / "results" / "pkl_cells"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.dump).stem}_{args.variant}_n{args.nchunks}c{args.chunk:02d}{args.tag}"

    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    tok_of = {db.scene_name(t_): t_ for t_ in db.scenes}
    for s in scenes:                        # builds the track map annotations_for needs (oracle variant); the
        ad.geometry(s)                      # adapter may serve geometry from its cache without building it, so
        db.sequence_geometry(tok_of[s])     # the database call is made explicitly
        db._track_map.setdefault(s, {})     # a scene with no eligible GT object gets no map in DEEP's code: empty
    cfg = RiskConfig()
    subs = {}
    for p in pipes:
        dc = {s: MemCache(by_scene[s], p) for s in scenes}
        subs[p] = write_submission(build_submission(db, ad, dc, scenes, cfg, args.variant, "cheap"), out / f"{stem}__{p}.json")
    full = {s: DetCache(Path(CACHE) / "nusc_det_tv" / "ns_full_640" / f"{s}.npz") for s in scenes}
    subs["full640"] = write_submission(build_submission(db, ad, full, scenes, cfg, args.variant, "full"), out / f"{stem}__full640.json")

    nusc = NuScenes(version="v1.0-trainval", dataroot=str(NUSCENES_TRAINVAL), verbose=False)
    loc_of = {sc["name"]: nusc.get("log", sc["log_token"])["location"] for sc in nusc.scene if sc["name"] in set(scenes)}
    _orig = L.create_splits_scenes
    L.create_splits_scenes = lambda: {**_orig(), "val": sorted(scenes)}
    dcfg = config_factory("detection_cvpr_2019")
    gt = L.load_gt(nusc, "val", DetectionBox, verbose=False)
    L.add_center_dist(nusc, gt)
    gt = L.filter_eval_boxes(nusc, gt, dcfg.class_range, verbose=False)
    preds = {}
    for p, path in subs.items():
        pr, _ = L.load_prediction(str(path), dcfg.max_boxes_per_sample, DetectionBox, verbose=False)
        pr = S70.subset(pr, list(gt.boxes))
        L.add_center_dist(nusc, pr)
        preds[p] = L.filter_eval_boxes(nusc, pr, dcfg.class_range, verbose=False)
    maps = {m: NuScenesMap(dataroot=str(NUSCENES_TRAINVAL), map_name=m) for m in S70.MAPS if m in set(loc_of.values())}

    dev = torch.device(args.device)
    model = compile_model(cin=5, cout=16, with_skip=True, dropout_p=0.0).to(dev)
    model.load_state_dict(torch.load(REPO / "third_party" / "tip" / "planner.pt", map_location="cpu"))
    model.eval()
    masks = (torch.Tensor(json.loads((REPO / "third_party" / "tip" / "masks_trainval.json").read_text())) == 1).to(dev)

    name_to_scene = {sc["name"]: sc for sc in nusc.scene}
    kinds = ["gt"] + list(preds)
    packed = {k_: [] for k_ in kinds}
    toks, tgts, scn = [], [], []
    for name in scenes:
        tokens, tok = [], name_to_scene[name]["first_sample_token"]
        while tok:
            tokens.append(tok)
            tok = nusc.get("sample", tok)["next"]
        ts, xy, yaw = S70.scene_poses(nusc, tokens)
        for i, tk in enumerate(tokens):
            if tk not in gt.boxes:
                continue
            wp, _, clamped = future_waypoints(ts, xy, yaw, i, HORIZONS)
            if clamped > 0:
                continue
            samp = nusc.get("sample", tk)
            packed["gt"].append(np.packbits(S70.render(samp, nusc, gt[tk], maps)[0].astype(bool)))
            for p in preds:
                packed[p].append(np.packbits(S70.render(samp, nusc, preds[p][tk], maps)[0].astype(bool)))
            toks.append(tk); tgts.append(wp.astype(np.float32)); scn.append(name)
        print(f"  {name}: {len(toks)} samples", flush=True)
    true = np.stack(tgts)
    rows = {"sample_token": toks, "scene": scn}
    for k_ in kinds:
        path = S74.argmax_paths(model, masks, packed[k_], dev)
        ade, fde = ade_fde(path, true)
        rows[f"ade_{k_}"], rows[f"fde_{k_}"] = ade, fde
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{stem}.csv", index=False)
    print(f"  wrote {out / (stem + '.csv')}: {len(df)} samples")


if __name__ == "__main__":
    main()
