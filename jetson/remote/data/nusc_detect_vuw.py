#!/usr/bin/env python
"""nuScenes-full: both fidelities over CAM_FRONT keyframes of every downloaded trainval scene, on a GPU server.

Same computation as DEEP's scripts/40_nusc_detect.py (same weights, modes, lift, cache format and camera-height
constant), with the PyTorch backend instead of TensorRT, resumable per scene, and the cam->ego entry of every scene
appended to cam_to_ego.json (as DEEP's 143_cam_to_ego_table.py computes it).  Run from the DEEP code copy on the server.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(sys.argv[0]).resolve().parents[1] / "code" / "risk-aware-perception"
sys.path.insert(0, str(REPO / "src"))
from rap import cache, features, mono                   # noqa: E402
from rap.detect import MODES, TwoFidelityDetector       # noqa: E402
from rap.nusc import FRAME_DT, NuScenesDB               # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL          # noqa: E402

CAM_HEIGHT = 1.51


def lst(a):
    return [[float(v) for v in row] for row in np.asarray(a, float)] if np.ndim(a) == 2 else [float(v) for v in a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default=str(Path(CACHE) / "nusc_det_tv"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(2)
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    det = TwoFidelityDetector(args.weights, device=args.device, backend="torch")
    modes = ["ns_cheap_320", "ns_full_640"]
    outdir = Path(args.out)
    c2e_path = Path(CACHE) / "cam_to_ego.json"
    c2e = json.loads(c2e_path.read_text())
    done = n_new = 0
    for sc in db.scenes:
        name = db.scene_name(sc)
        toks = db.samples(sc)
        if all((outdir / m / f"{name}.npz").exists() for m in modes):
            done += 1
            continue
        if not all(db.image_path(t).exists() for t in toks):
            continue
        store = {m: {"dets": [], "geos": [], "scalars": {}} for m in modes}
        prev_det = {m: None for m in modes}
        prev_small = None
        calib = db.calib(toks[0])
        t0 = time.time()
        for tk in toks:
            img = cv2.imread(str(db.image_path(tk)))
            small = cv2.cvtColor(cv2.resize(img, (MODES["ns_cheap_320"].net_w, MODES["ns_cheap_320"].net_h)),
                                 cv2.COLOR_BGR2GRAY)
            for m in modes:
                d, _ = det.detect(img, MODES[m])
                g = mono.predicted_geometry(d, prev_det[m], calib, CAM_HEIGHT, FRAME_DT)
                store[m]["dets"].append(d)
                store[m]["geos"].append(g)
                for k in ("n_cand", "n_cand_raw", "n_post"):
                    store[m]["scalars"].setdefault(k, []).append(d[k])
                prev_det[m] = d
            for k, v in features.image_stats(small, prev_small).items():
                store["ns_cheap_320"]["scalars"].setdefault(k, []).append(v)
            store["ns_cheap_320"]["scalars"].setdefault("img_h", []).append(img.shape[0])
            store["ns_cheap_320"]["scalars"].setdefault("img_w", []).append(img.shape[1])
            prev_small = small
        for m in modes:
            cache.save(outdir / m / f"{name}.npz", list(range(len(toks))), store[m]["dets"], store[m]["geos"],
                       store[m]["scalars"])
        R, t = db._cam_to_ego(toks[0])
        c2e["nuScenes"][name] = {"R": lst(R), "t": lst(t), "t_task22": lst(t), "K": lst(calib.P2[:3, :3]),
                                 "cam_h": CAM_HEIGHT, "dt": FRAME_DT}
        c2e_path.write_text(json.dumps(c2e, indent=1))
        n_new += 1
        print(f"[{name}] {len(toks)} frames {time.time() - t0:.1f}s", flush=True)
    print(f"done: {done} scenes already cached, {n_new} new", flush=True)


if __name__ == "__main__":
    main()
