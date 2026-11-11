#!/usr/bin/env python
"""K41-2 P02 (post-hoc; PREREG K41-2): bounded-order interaction diagnostics of PDM-Closed on the saved K20 seed-0 subset
of 1,000 navtest tokens (k20_pdm.subset), CPU only. Built on k20_pdm.py --mode pairs (same draft scene from the cheap
detections, same 7 x 7 static vehicle probe grid restricted to cells projecting into CAM_F0, same frozen field, same
eligible in-camera objects, moved = trajectory change > 0.5 m over 4 s; every query from the original draft scene).
Per token:
  removal triples    of objects whose single removal does not move the plan (free objects), removed together
  insertion pairs    of probe cells that are individually not sensitive, inserted together
  insertion triples  the same with three cells
                     each family: all combinations if <= 100, otherwise 100 sampled without replacement with
                     numpy.random.default_rng(seed), seed = int(SHA256("navtest|token|kind|0")[:16], 16)
  k2 field           every pair of individually quiet projecting cells at Manhattan distance one on the 7 x 7 grid,
                     inserted together (the k = 2 field adds the cells of the moving pairs)
Every joint effect (moving combination) keeps its trajectory for the PDMS scoring pass. Per-call wall times recorded.
Output: results/k41/p02/<log>.pkl (one row per token).
P03 (2026-11-02, user-authorized beyond the cost check): --subset remaining = every other field_test navtest token
(12,146 - 1,000), --fresh_per_token builds a new PDM-Closed instance per token (NAVSIM v1.1 PDMObservation never clears
_collided_track_ids, so red-light ids leak across scenes; within a token every query still shares one instance, as in P02).
usage (env.sh; scripts/common/env_navsim.sh; CUDA hidden): k41_p02_interactions.py [--shard i/n] [--limit N]
       [--subset k20_1000|remaining] [--fresh_per_token] [--out DIR]"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np

B = Path(os.environ["B"])
_k = importlib.util.spec_from_file_location("k20pdm", B / "scripts/A_navsim/k20_pdm.py")
K20 = importlib.util.module_from_spec(_k); _k.loader.exec_module(K20)
NF, NP = K20.NF, K20.NP
from navsim.common.dataclasses import SceneFilter, SensorConfig                  # noqa: E402
from navsim.common.dataloader import SceneLoader                                 # noqa: E402
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor  # noqa: E402

OUT = B / "results" / "k41" / "p02"
CAP = 100


def seed(token, kind):
    return int(hashlib.sha256(f"navtest|{token}|{kind}|0".encode()).hexdigest()[:16], 16)


def combos(items, k, token, kind):
    """all k-combinations if <= CAP, else CAP of them sampled without replacement (fixed seed)"""
    n = len(items)
    if n < k:
        return [], 0
    total = math.comb(n, k)
    allc = list(itertools.combinations(range(n), k)) if total <= 20000 else None
    if total <= CAP:
        pick = allc
    elif allc is not None:
        rng = np.random.default_rng(seed(token, kind))
        pick = [allc[i] for i in sorted(rng.choice(total, size=CAP, replace=False))]
    else:                                                                     # rejection sampling of distinct sets
        rng = np.random.default_rng(seed(token, kind)); s = set()
        while len(s) < CAP:
            s.add(tuple(sorted(rng.choice(n, size=k, replace=False).tolist())))
        pick = sorted(s)
    return [tuple(items[i] for i in c) for c in pick], total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0, help="first N subset tokens (smoke)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--subset", default="k20_1000", choices=["k20_1000", "remaining"])
    ap.add_argument("--fresh_per_token", action="store_true")
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    split = os.environ.get("NAVSIM_SPLIT", "test")
    DETS, FIELD = K20.DETS, K20.FIELD
    gz = float((DETS / "ground_z.txt").read_text()) if (DETS / "ground_z.txt").exists() else float((NP.DETS / "ground_z.txt").read_text())
    priors = pickle.load(open(K20.X / "dets" / "size_priors.pkl", "rb"))
    L, Wd, Ht = priors["vehicle"]
    proc = MetricCacheProcessor(cache_path=None, force_feature_computation=False)
    T = K20.Timer()
    todo = K20.subset()
    if args.subset == "remaining":
        first = set(todo); todo = []
        for f in sorted(FIELD.glob("*.pkl")):
            d = pickle.load(open(f, "rb"))
            todo += [(f.stem, t) for t, v in sorted(d.items()) if "err" not in v and (f.stem, t) not in first]
    if args.limit:
        todo = todo[:args.limit]
    logs = sorted({lg for lg, _ in todo})
    for li, log in enumerate(logs):
        if li % sn != si:
            continue
        fo = out / f"{log}.pkl"
        if fo.exists():
            continue
        toks = [t for lg, t in todo if lg == log]
        dets = pickle.load(open(DETS / f"{log}.pkl", "rb"))
        frozen = pickle.load(open(FIELD / f"{log}.pkl", "rb"))
        sl = SceneLoader(NP.DATA / "navsim_logs" / split, NP.DATA / "sensor_blobs" / split,
                         SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                                     log_names=[log], tokens=toks), SensorConfig.build_no_sensors())
        rows = []
        for tok in sl.tokens:
            try:
                if args.fresh_per_token:
                    proc = MetricCacheProcessor(cache_path=None, force_feature_computation=False)
                scene = sl.get_scene_from_token(tok)
                idx = scene.scene_metadata.num_history_frames - 1
                cam = NP.Cam(*dets[tok]["cam"])
                draft = NP.branch_annotations(scene.frames[idx].annotations, dets[tok]["cheap"], cam, gz, priors, "cheap")
                calls = []

                def call(kind, ann, meta):
                    T.t = {}
                    t0_ = time.perf_counter()
                    tr = K20.plan_timed(proc, NF.with_ann(scene, idx, ann), T)
                    calls.append({"kind": kind, "wall": time.perf_counter() - t0_, **meta})
                    return tr
                t_tok = time.perf_counter()
                t0 = call("base", draft, {})
                cells = []
                for x in NF.X_GRID:
                    for y in NF.Y_GRID:
                        box = np.array([x, y, gz + Ht / 2, L, Wd, Ht, 0.0])
                        ib = cam.project(box)
                        if ib is None:
                            continue
                        mv = any(np.allclose(ib, b) for b in frozen[tok].get("field_boxes", []))
                        cells.append({"x": float(x), "y": float(y), "box": box, "moved": mv})
                objs = []
                for i, name in enumerate(draft.names):
                    if name not in NP.ELIGIBLE or cam.project(draft.boxes[i]) is None:
                        continue
                    objs.append({"i": i, "moved": NF.moved(t0, call("support", NF.drop(draft, i), {"obj": i}))})
                free = [o["i"] for o in objs if not o["moved"]]
                quiet = [c for c in cells if not c["moved"]]
                joint = {}

                def test(kind, ann, meta):
                    tr = call(kind, ann, meta)
                    if NF.moved(t0, tr):
                        joint.setdefault(kind, []).append({**meta, "poses": tr.poses.copy()})

                tr3, n_tr3 = combos(free, 3, tok, "removal_triple")
                for c in tr3:
                    ann = draft
                    for i in sorted(c, reverse=True):
                        ann = NF.drop(ann, i)
                    test("removal_triple", ann, {"objs": list(c)})
                res = {}
                for k, kind in ((2, "insertion_pair"), (3, "insertion_triple")):
                    cs, tot = combos(list(range(len(quiet))), k, tok, kind)
                    res[kind] = (len(cs), tot)
                    for c in cs:
                        ann = draft
                        for j in c:
                            q = quiet[j]; ann = NF.add_box(ann, q["box"], f"probe:{q['x']}:{q['y']}")
                        test(kind, ann, {"cells": [(quiet[j]["x"], quiet[j]["y"]) for j in c]})
                xs, ys = sorted(set(NF.X_GRID)), sorted(set(NF.Y_GRID))
                pos = {(c["x"], c["y"]): (xs.index(c["x"]), ys.index(c["y"])) for c in quiet}
                adj = [(a, b) for a, b in itertools.combinations(range(len(quiet)), 2)
                       if sum(abs(u - w) for u, w in zip(pos[(quiet[a]["x"], quiet[a]["y"])], pos[(quiet[b]["x"], quiet[b]["y"])])) == 1]
                for a, b in adj:
                    ann = NF.add_box(NF.add_box(draft, quiet[a]["box"], f"probe:{quiet[a]['x']}:{quiet[a]['y']}"),
                                     quiet[b]["box"], f"probe:{quiet[b]['x']}:{quiet[b]['y']}")
                    test("k2_pair", ann, {"cells": [(quiet[a]["x"], quiet[a]["y"]), (quiet[b]["x"], quiet[b]["y"])]})
                rows.append({"log": log, "token": tok, "n_cells": len(cells), "n_sens_cells": sum(c["moved"] for c in cells),
                             "n_objs": len(objs), "n_support": sum(o["moved"] for o in objs), "n_free": len(free),
                             "removal_triples": (len(tr3), n_tr3), **{k_: v for k_, v in res.items()}, "k2_pairs": len(adj),
                             "joint": joint, "base_poses": t0.poses.copy(),
                             "calls": [{"kind": c["kind"], "wall": c["wall"]} for c in calls], "wall_token": time.perf_counter() - t_tok})
            except Exception as e:                                                    # noqa: BLE001
                rows.append({"log": log, "token": tok, "err": f"{type(e).__name__}: {e}"})
        tmp = fo.with_suffix(".partial")
        with open(tmp, "wb") as fh:
            pickle.dump(rows, fh)
        tmp.replace(fo)
        print(f"  [{li + 1}/{len(logs)}] {log}: {len(rows)} tokens", flush=True)
    print("K41 P02 DONE", flush=True)


if __name__ == "__main__":
    main()
