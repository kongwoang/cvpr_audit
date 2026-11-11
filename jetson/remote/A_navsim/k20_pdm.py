#!/usr/bin/env python
"""K20 — cost of probing PDM-Closed (the frozen spec_pdm field of navsim_field.py) and interacting objects (post-hoc).
CPU only, nuPlan env (env_navsim.sh), OMP / MKL threads = 1.  Subset: 1,000 navtest tokens drawn with
numpy.random.default_rng(0) from the tokens of the frozen field (results/navsim_exp/field_test), sorted.
  --mode cost    per token, the frozen field / support computation re-run call by call (navsim_field's rule: the draft
                 scene from the cheap detections, a static class-median vehicle probe at every cell of the 7 x 7 grid that
                 projects into CAM_F0, every eligible in-camera draft object removed in turn; moved = trajectory changes by
                 > 0.5 m), every call timed and split into scenario / planner-input construction, observation update,
                 proposal-manager (centerline, map), proposal generation (IDM unroll), simulation (ego roll-outs),
                 scoring, trajectory extraction; parity of the sensitive cells / support with the stored frozen field
  --mode pairs   on the same tokens: (a) every pair of eligible in-camera objects whose single removals do NOT move the
                 plan, removed together; (b) every probe cell that is not sensitive alone combined with the removal of
                 every object that does not support alone; frequencies of "moves" (> 0.5 m), the pairs found
  --mode base    the draft plan only, for every navtest token of the field (corridor-of-plan cells, K20(3))
usage: k20_pdm.py --mode cost|pairs|base [--shard i/n]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import time
from pathlib import Path

import numpy as np

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("nf", B / "scripts/A_navsim/navsim_field.py")
NF = importlib.util.module_from_spec(_s); _s.loader.exec_module(NF)
NP = NF.NP
from navsim.common.dataclasses import SceneFilter, SensorConfig                  # noqa: E402
from navsim.common.dataloader import SceneLoader                                 # noqa: E402
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor  # noqa: E402
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario      # noqa: E402
from nuplan.common.actor_state.state_representation import TimePoint             # noqa: E402
from nuplan.common.geometry.convert import absolute_to_relative_poses            # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402
from navsim.common.dataclasses import Trajectory                                 # noqa: E402

X = B / "results" / "navsim_exp"
FIELD = X / "field_test"
DETS = X / "dets_s05"
OUT = B / "results" / "k20"


class Timer:
    def __init__(self):
        self.t = {}

    def wrap(self, obj, name, key):
        f = getattr(obj, name)

        def g(*a, **k):
            t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                self.t[key] = self.t.get(key, 0.0) + time.perf_counter() - t0
        setattr(obj, name, g)


def plan_timed(proc, scene, T):
    """navsim_pdms.plan with stage timers (T.t is reset by the caller)."""
    t0 = time.perf_counter()
    scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")
    pi, init = proc._get_planner_inputs(scenario)
    proc._pdm_closed.initialize(init)
    t1 = time.perf_counter()
    traj = proc._pdm_closed.compute_planner_trajectory(pi)
    t2 = time.perf_counter()
    ego = scenario.initial_ego_state
    ts = TrajectorySampling(time_horizon=4, interval_length=0.5)
    states = [traj.get_state_at_time(TimePoint(int(ego.time_point.time_us + (k + 1) * 0.5e6))).rear_axle for k in range(ts.num_poses)]
    rel = absolute_to_relative_poses([ego.rear_axle] + states)[1:]
    t3 = time.perf_counter()
    T.t.update({"setup": t1 - t0, "planner": t2 - t1, "extract": t3 - t2})
    return Trajectory(np.array([[p.x, p.y, p.heading] for p in rel], np.float32), ts)


def subset(n=1000):
    toks = []
    for f in sorted(FIELD.glob("*.pkl")):
        d = pickle.load(open(f, "rb"))
        toks += [(f.stem, t) for t, v in sorted(d.items()) if "err" not in v]
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(toks), size=min(n, len(toks)), replace=False))
    return [toks[i] for i in pick]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["cost", "pairs", "base"])
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))
    split = os.environ.get("NAVSIM_SPLIT", "test")
    gz = float((DETS / "ground_z.txt").read_text()) if (DETS / "ground_z.txt").exists() else float((NP.DETS / "ground_z.txt").read_text())
    priors = pickle.load(open(X / "dets" / "size_priors.pkl", "rb"))
    L, Wd, Ht = priors["vehicle"]
    proc = MetricCacheProcessor(cache_path=None, force_feature_computation=False)
    T = Timer()
    pl = proc._pdm_closed
    T.wrap(pl._observation, "update", "observation")
    T.wrap(pl, "_update_proposal_manager", "proposal_manager")
    T.wrap(pl._generator, "generate_proposals", "generation")
    T.wrap(pl._simulator, "simulate_proposals", "simulation")
    T.wrap(pl._scorer, "score_proposals", "scoring")
    if args.mode == "base":
        todo = []
        for f in sorted(FIELD.glob("*.pkl")):
            d = pickle.load(open(f, "rb"))
            todo += [(f.stem, t) for t, v in sorted(d.items()) if "err" not in v]
    else:
        todo = subset()
    logs = sorted({lg for lg, _ in todo})
    OUT.mkdir(parents=True, exist_ok=True)
    for li, log in enumerate(logs):
        if li % sn != si:
            continue
        fo = OUT / f"{args.mode}__{log}.pkl"
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
                scene = sl.get_scene_from_token(tok)
                idx = scene.scene_metadata.num_history_frames - 1
                cam = NP.Cam(*dets[tok]["cam"])
                draft = NP.branch_annotations(scene.frames[idx].annotations, dets[tok]["cheap"], cam, gz, priors, "cheap")
                calls = []

                def call(kind, ann, meta):
                    T.t = {}
                    t0 = time.perf_counter()
                    tr = plan_timed(proc, NF.with_ann(scene, idx, ann), T)
                    rec = {"kind": kind, "wall": time.perf_counter() - t0, **T.t, **meta}
                    calls.append(rec)
                    return tr
                t_tok = time.perf_counter()
                t0 = call("base", draft, {})
                if args.mode == "base":
                    rows.append({"log": log, "token": tok, "poses": t0.poses.copy()})
                    continue
                cells, support_boxes, objs = [], [], []
                for x in NF.X_GRID:
                    for y in NF.Y_GRID:
                        box = np.array([x, y, gz + Ht / 2, L, Wd, Ht, 0.0])
                        ib = cam.project(box)
                        if ib is None:
                            continue
                        if args.mode == "cost":
                            mv = NF.moved(t0, call("probe", NF.add_box(draft, box, f"probe:{x}:{y}"), {"x": x, "y": y}))
                        else:
                            mv = any(np.allclose(ib, b) for b in frozen[tok].get("field_boxes", []))
                        cells.append({"x": x, "y": y, "box": box, "ib": ib, "moved": mv})
                for i, name in enumerate(draft.names):
                    if name not in NP.ELIGIBLE or cam.project(draft.boxes[i]) is None:
                        continue
                    mv = NF.moved(t0, call("support", NF.drop(draft, i), {"obj": i}))
                    objs.append({"i": i, "name": name, "moved": mv})
                rec = {"log": log, "token": tok, "n_probe": len(cells), "n_support_calls": len(objs),
                       "sens_cells": [(c["x"], c["y"]) for c in cells if c["moved"]],
                       "support_objs": [o["i"] for o in objs if o["moved"]], "wall_token": time.perf_counter() - t_tok}
                if args.mode == "cost":
                    fb = frozen[tok].get("field_boxes", [])
                    mine = [c["ib"] for c in cells if c["moved"]]
                    rec["parity_field"] = len(fb) == len(mine) and all(np.allclose(a, b) for a, b in zip(fb, mine))
                    rec["parity_support"] = len(frozen[tok].get("support_boxes", [])) == len(rec["support_objs"])
                    rec["calls"] = calls
                else:
                    single = {o["i"] for o in objs if o["moved"]}
                    free = [o["i"] for o in objs if not o["moved"]]
                    pairs, pp_hits = [], []
                    for a in range(len(free)):
                        for b in range(a + 1, len(free)):
                            i, j = free[a], free[b]
                            ann = NF.drop(NF.drop(draft, max(i, j)), min(i, j))
                            if NF.moved(t0, call("pair", ann, {"i": i, "j": j})):
                                pairs.append((i, j))
                    for c in cells:
                        if c["moved"]:
                            continue
                        for i in free:
                            ann = NF.add_box(NF.drop(draft, i), c["box"], f"probe:{c['x']}:{c['y']}")
                            if NF.moved(t0, call("probe_drop", ann, {"x": c["x"], "y": c["y"], "i": i})):
                                pp_hits.append((c["x"], c["y"], i))
                    dx = np.asarray(dets[tok]["cheap"]["xyxy"], float).reshape(-1, 4)

                    def det_box(i):
                        tk = draft.track_tokens[i]
                        if tk.startswith("fp:"):
                            return dx[int(tk.split(":")[-1])]
                        return dx[int(np.argmax(NP.iou([cam.project(draft.boxes[i])], dx)[0]))] if len(dx) else None
                    rec.update({"n_free": len(free), "n_pairs_tested": len(free) * (len(free) - 1) // 2, "pairs_moving": pairs,
                                "n_probe_drop_tested": sum(not c["moved"] for c in cells) * len(free), "probe_drop_moving": pp_hits,
                                "pair_det_boxes": [det_box(i) for i in sorted({k for p in pairs for k in p})],
                                "single_support": sorted(single), "n_calls": len(calls)})
                rows.append(rec)
            except Exception as e:                                                    # noqa: BLE001
                rows.append({"log": log, "token": tok, "err": f"{type(e).__name__}: {e}"})
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{li + 1}/{len(logs)}] {log}: {len(rows)} tokens", flush=True)
    print(f"K20 {args.mode.upper()} DONE", flush=True)


if __name__ == "__main__":
    main()
