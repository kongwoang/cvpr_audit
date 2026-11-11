#!/usr/bin/env python
"""Stage F (CPU): the decision-sensitivity field and support set of PDM-Closed itself — "probing the planner" with the
planner NAVSIM scores (the braking field of v3 misses what PDM-Closed reacts to: adjacent-lane and crossing objects).

Per token, on the CHEAP draft scene only (the scene the planner would get from the cheap detector under the evaluation
convention; no other ground truth enters):
  base      PDM-Closed trajectory on the draft (exactly as navsim_pdms.plan)
  field     an unseen static vehicle probe (class-median size, ego heading) inserted at each grid cell
            x in X_GRID (forward, m), y in Y_GRID (lateral, lanes of 3.5 m) whose box projects into CAM_F0; the cell is
            sensitive if the planned trajectory moves by more than MOVE_M (max position difference over the 4 s poses)
  support   each draft object inside CAM_F0 removed in turn; it supports the plan if the trajectory moves > MOVE_M
Writes <NAVSIM_FIELD>/<log>.pkl: {token: {"field_boxes": [image boxes of sensitive cells], "support": [indices into
the cheap detections, via their matched annotation], "n_calls": int}}.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import os
import pickle
from pathlib import Path

import numpy as np

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("np_", B / "scripts/A_navsim/navsim_pdms.py")
NP = importlib.util.module_from_spec(_s); _s.loader.exec_module(NP)
from navsim.common.dataclasses import Annotations, SceneFilter, SensorConfig    # noqa: E402
from navsim.common.dataloader import SceneLoader                                 # noqa: E402
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor  # noqa: E402

X_GRID = np.array([8.0, 12.0, 17.0, 23.0, 30.0, 40.0, 55.0])
Y_GRID = np.array([-10.5, -7.0, -3.5, 0.0, 3.5, 7.0, 10.5])
MOVE_M = 0.5
OUT = Path(os.environ.get("NAVSIM_FIELD", NP.EXP / "field"))


def with_ann(scene, idx, ann):
    fr = dataclasses.replace(scene.frames[idx], annotations=ann)
    frames = list(scene.frames); frames[idx] = fr
    return dataclasses.replace(scene, frames=frames)


def add_box(ann, box, tok):
    return Annotations(boxes=np.vstack([np.asarray(ann.boxes, np.float32).reshape(-1, 7), np.asarray(box, np.float32)[None]]),
                       names=list(ann.names) + ["vehicle"],
                       velocity_3d=np.vstack([np.asarray(ann.velocity_3d, np.float32).reshape(-1, 3), np.zeros((1, 3), np.float32)]),
                       instance_tokens=list(ann.instance_tokens) + [tok], track_tokens=list(ann.track_tokens) + [tok])


def drop(ann, i):
    k = [j for j in range(len(ann.names)) if j != i]
    return Annotations(boxes=np.asarray(ann.boxes, np.float32)[k].reshape(-1, 7), names=[ann.names[j] for j in k],
                       velocity_3d=np.asarray(ann.velocity_3d, np.float32)[k].reshape(-1, 3),
                       instance_tokens=[ann.instance_tokens[j] for j in k], track_tokens=[ann.track_tokens[j] for j in k])


def moved(t0, t1):
    return float(np.max(np.hypot(t0.poses[:, 0] - t1.poses[:, 0], t0.poses[:, 1] - t1.poses[:, 1]))) > MOVE_M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--every", type=int, default=1, help="dev speed-up: every k-th token of a log")
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))
    split = os.environ.get("NAVSIM_SPLIT", "test")
    cache = Path(os.environ.get("NAVSIM_CACHE", NP.EXP / "metric_cache"))
    gz = float((NP.DETS / "ground_z.txt").read_text())
    priors = pickle.load(open(NP.EXP / "dets" / "size_priors.pkl", "rb"))
    L, Wd, Ht = priors["vehicle"]
    OUT.mkdir(parents=True, exist_ok=True)
    logs = sorted(p.stem for p in NP.DETS.glob("*.pkl"))
    cached = {p.parent.name for p in cache.glob("*/*/*/metric_cache.pkl")}
    proc = MetricCacheProcessor(cache_path=None, force_feature_computation=False)
    for li, log in enumerate(logs):
        if li % sn != si or (OUT / f"{log}.pkl").exists():
            continue
        dets = pickle.load(open(NP.DETS / f"{log}.pkl", "rb"))
        toks = [t for t in sorted(dets) if t in cached and "cam" in dets[t]][::args.every]
        if not toks:
            continue
        sl = SceneLoader(NP.DATA / "navsim_logs" / split, NP.DATA / "sensor_blobs" / split,
                         SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                                     log_names=[log], tokens=toks), SensorConfig.build_no_sensors())
        out = {}
        for tok in sl.tokens:
            try:
                scene = sl.get_scene_from_token(tok)
                idx = scene.scene_metadata.num_history_frames - 1
                cam = NP.Cam(*dets[tok]["cam"])
                draft = NP.branch_annotations(scene.frames[idx].annotations, dets[tok]["cheap"], cam, gz, priors, "cheap")
                t0 = NP.plan(proc, with_ann(scene, idx, draft))
                n, boxes = 1, []
                for x in X_GRID:
                    for y in Y_GRID:
                        box = np.array([x, y, gz + Ht / 2, L, Wd, Ht, 0.0])
                        ib = cam.project(box)
                        if ib is None:
                            continue
                        n += 1
                        if moved(t0, NP.plan(proc, with_ann(scene, idx, add_box(draft, box, f"probe:{x}:{y}")))):
                            boxes.append(ib)
                support = []
                for i, name in enumerate(draft.names):
                    if name not in NP.ELIGIBLE or cam.project(draft.boxes[i]) is None:
                        continue
                    n += 1
                    if moved(t0, NP.plan(proc, with_ann(scene, idx, drop(draft, i)))):
                        # the crop comes from the CHEAP DETECTION of that object (no ground-truth geometry): the FP index
                        # for lifted detections, else the detection best overlapping the object's image footprint
                        tk = draft.track_tokens[i]
                        dx = np.asarray(dets[tok]["cheap"]["xyxy"], float).reshape(-1, 4)
                        if tk.startswith("fp:"):
                            support.append(dx[int(tk.split(":")[-1])])
                        elif len(dx):
                            support.append(dx[int(np.argmax(NP.iou([cam.project(draft.boxes[i])], dx)[0]))])
                out[tok] = {"field_boxes": boxes, "support_boxes": support, "n_calls": n}
            except Exception as e:                                                    # noqa: BLE001
                out[tok] = {"err": type(e).__name__}
        with open(OUT / f"{log}.pkl", "wb") as fh:
            pickle.dump(out, fh)
        print(f"  [{li + 1}/{len(logs)}] {log}: {len(out)} tokens, "
              f"{np.mean([o.get('n_calls', 0) for o in out.values()]):.0f} planner calls/token", flush=True)
    print("NAVSIM FIELD DONE")


if __name__ == "__main__":
    main()
