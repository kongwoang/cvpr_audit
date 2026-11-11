#!/usr/bin/env python
"""K20(2) — cheaper implementations of the frozen PDM-Closed probing with IDENTICAL output (post-hoc; CPU; nuPlan env).
The frozen implementation (navsim_field / navsim_pdms.plan) rebuilds the NAVSIM scenario, the planner inputs, the route
dictionaries, the drivable-area map and the proposal paths for every probe / removal, although only the current
objects differ between the calls of one token (the planner's history buffer holds one state: the current ego state and
the current tracked objects). The fast implementation builds them once per token and, per call, only converts the
modified annotations into tracked objects (annotations_to_detection_tracks, as NavSimScenario does) and runs the
planner's per-observation stages (observation update, IDM proposal unroll, simulation, scoring); the simulated ego
roll-outs are reused when a call's proposals are bit-identical to an earlier call's of the same token.
Parity: per call the 8 output poses are compared with the frozen implementation (bit-identical required) on every
token of --parity_tokens; the sensitive cells / support with the stored frozen field on all tokens.
Parallel: the calls of a token are split over N worker processes (each builds the token's planner state once);
wall clock per token measured.
usage: k20_fast.py --mode fast [--parity_tokens 100] [--shard i/n] | --mode parallel --workers N [--tokens 200]
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
_s = importlib.util.spec_from_file_location("kp", B / "scripts/A_navsim/k20_pdm.py")
KP = importlib.util.module_from_spec(_s); _s.loader.exec_module(KP)
NF, NP = KP.NF, KP.NP
from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory      # noqa: E402
from navsim.common.dataloader import SceneLoader                                 # noqa: E402
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor  # noqa: E402
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario      # noqa: E402
from navsim.planning.scenario_builder.navsim_scenario_utils import annotations_to_detection_tracks  # noqa: E402
from nuplan.common.actor_state.state_representation import TimePoint             # noqa: E402
from nuplan.common.geometry.convert import absolute_to_relative_poses            # noqa: E402
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer  # noqa: E402
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput     # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402

TS = TrajectorySampling(time_horizon=4, interval_length=0.5)


class FastToken:
    """planner state built once for one token's draft scene; call(annotations) -> 8 poses (Trajectory)."""

    def __init__(self, proc, scene, idx, draft):
        self.proc, self.pl = proc, proc._pdm_closed
        scenario = NavSimScenario(NF.with_ann(scene, idx, draft), map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")
        self.pi, init = proc._get_planner_inputs(scenario)
        self.pl.initialize(init)
        self.ego = scenario.initial_ego_state
        self.t0 = self.ego.time_point.time_us
        self.sim_cache = {}
        sim = self.pl._simulator
        orig = sim.simulate_proposals.__func__ if hasattr(sim.simulate_proposals, "__func__") else None
        self._orig_sim = sim.simulate_proposals
        self.hits = 0

        def sim_cached(proposals_array, ego_state):
            key = proposals_array.tobytes()
            if key in self.sim_cache:
                self.hits += 1
                return self.sim_cache[key].copy()
            out = self._orig_sim(proposals_array, ego_state)
            self.sim_cache[key] = out.copy()
            return out
        sim.simulate_proposals = sim_cached
        self._orig = orig
        self.base = self.call(draft)                     # first call (iteration 0): route correction, map, proposal paths

    def restore(self):
        self.pl._simulator.simulate_proposals = self._orig_sim

    def call(self, ann):
        obs = annotations_to_detection_tracks(ann, self.ego)
        hist = SimulationHistoryBuffer.initialize_from_list(buffer_size=1, ego_states=[self.ego], observations=[obs])
        pi = PlannerInput(iteration=self.pi.iteration, history=hist, traffic_light_data=self.pi.traffic_light_data)
        traj = self.pl.compute_planner_trajectory(pi)
        states = [traj.get_state_at_time(TimePoint(int(self.t0 + (k + 1) * 0.5e6))).rear_axle for k in range(TS.num_poses)]
        rel = absolute_to_relative_poses([self.ego.rear_axle] + states)[1:]
        return Trajectory(np.array([[p.x, p.y, p.heading] for p in rel], np.float32), TS)


def token_calls(cam, draft, gz, L, Wd, Ht):
    specs = []
    for x in NF.X_GRID:
        for y in NF.Y_GRID:
            box = np.array([x, y, gz + Ht / 2, L, Wd, Ht, 0.0])
            if cam.project(box) is not None:
                specs.append(("probe", x, y, box))
    for i, name in enumerate(draft.names):
        if name in NP.ELIGIBLE and cam.project(draft.boxes[i]) is not None:
            specs.append(("support", i))
    return specs


def ann_of(draft, s):
    return NF.add_box(draft, s[3], f"probe:{s[1]}:{s[2]}") if s[0] == "probe" else NF.drop(draft, s[1])


CTX = {}


def _w_init():
    CTX["proc"] = MetricCacheProcessor(cache_path=None, force_feature_computation=False)


def _w_run(job):
    log, tok, idxs = job
    if CTX.get("log") != log:
        split = os.environ.get("NAVSIM_SPLIT", "test")
        CTX["sl"] = SceneLoader(NP.DATA / "navsim_logs" / split, NP.DATA / "sensor_blobs" / split,
                                SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                                            log_names=[log]), SensorConfig.build_no_sensors())
        CTX["log"] = log
        CTX["dets"] = pickle.load(open(KP.DETS / f"{log}.pkl", "rb"))
    t0 = time.perf_counter()
    scene = CTX["sl"].get_scene_from_token(tok)
    idx = scene.scene_metadata.num_history_frames - 1
    dets = CTX["dets"]
    cam = NP.Cam(*dets[tok]["cam"])
    gz, pri = CTX["gz"], CTX["pri"]
    draft = NP.branch_annotations(scene.frames[idx].annotations, dets[tok]["cheap"], cam, gz, pri, "cheap")
    ft = FastToken(CTX["proc"], scene, idx, draft)
    specs = token_calls(cam, draft, gz, *pri["vehicle"])
    out = [(i, NF.moved(ft.base, ft.call(ann_of(draft, specs[i])))) for i in idxs]
    ft.restore()
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fast", "parallel"])
    ap.add_argument("--parity_tokens", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))
    split = os.environ.get("NAVSIM_SPLIT", "test")
    gz = float((KP.DETS / "ground_z.txt").read_text())
    priors = pickle.load(open(KP.X / "dets" / "size_priors.pkl", "rb"))
    L, Wd, Ht = priors["vehicle"]
    todo = KP.subset()
    KP.OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "parallel":
        import multiprocessing as mp
        CTX["gz"], CTX["pri"] = gz, priors
        todo = todo[:args.tokens]
        rows = []
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers, initializer=_w_init) as pool:
            for log, tok in todo:
                dets = pickle.load(open(KP.DETS / f"{log}.pkl", "rb"))
                sl = SceneLoader(NP.DATA / "navsim_logs" / split, NP.DATA / "sensor_blobs" / split,
                                 SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                                             log_names=[log], tokens=[tok]), SensorConfig.build_no_sensors())
                scene = sl.get_scene_from_token(tok)
                idx = scene.scene_metadata.num_history_frames - 1
                cam = NP.Cam(*dets[tok]["cam"])
                draft = NP.branch_annotations(scene.frames[idx].annotations, dets[tok]["cheap"], cam, gz, priors, "cheap")
                n = len(token_calls(cam, draft, gz, L, Wd, Ht))
                parts = [list(range(k, n, args.workers)) for k in range(args.workers)]
                t0 = time.perf_counter()
                res = pool.map(_w_run, [(log, tok, p) for p in parts if p])
                wall = time.perf_counter() - t0
                moved = sorted(m for r, _ in res for m in r)
                rows.append({"log": log, "token": tok, "workers": args.workers, "n_calls": n + 1, "wall_token": wall,
                             "worker_max": max(t for _, t in res), "moved": moved})
        with open(KP.OUT / f"parallel_w{args.workers}.pkl", "wb") as fh:
            pickle.dump(rows, fh)
        print(f"K20 PARALLEL w{args.workers} DONE", np.mean([r["wall_token"] for r in rows]), flush=True)
        return
    proc = MetricCacheProcessor(cache_path=None, force_feature_computation=False)
    proc_ref = MetricCacheProcessor(cache_path=None, force_feature_computation=False)
    logs = sorted({lg for lg, _ in todo})
    parity_set = set(todo[:args.parity_tokens])
    for li, log in enumerate(logs):
        if li % sn != si:
            continue
        fo = KP.OUT / f"fast__{log}.pkl"
        if fo.exists():
            continue
        toks = [t for lg, t in todo if lg == log]
        dets = pickle.load(open(KP.DETS / f"{log}.pkl", "rb"))
        frozen = pickle.load(open(KP.FIELD / f"{log}.pkl", "rb"))
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
                specs = token_calls(cam, draft, gz, L, Wd, Ht)
                t0 = time.perf_counter()
                ft = FastToken(proc, scene, idx, draft)
                trajs, ct = [], []
                for s in specs:
                    tc = time.perf_counter()
                    trajs.append(ft.call(ann_of(draft, s)))
                    ct.append(time.perf_counter() - tc)
                wall = time.perf_counter() - t0
                ft.restore()
                moved = [NF.moved(ft.base, tr) for tr in trajs]
                sens = [cam.project(s[3]) for s, m in zip(specs, moved) if m and s[0] == "probe"]
                fb = frozen[tok].get("field_boxes", [])
                rec = {"log": log, "token": tok, "n_calls": len(specs) + 1, "wall_token": wall, "call_mean": float(np.mean(ct)),
                       "sim_reused": ft.hits, "parity_field": len(fb) == len(sens) and all(np.allclose(a, b) for a, b in zip(fb, sens)),
                       "parity_support": len(frozen[tok].get("support_boxes", [])) == sum(m for s, m in zip(specs, moved) if s[0] == "support")}
                if (log, tok) in parity_set:
                    d0 = NP.plan(proc_ref, NF.with_ann(scene, idx, draft))
                    diffs = [float(np.abs(d0.poses - ft.base.poses).max())]
                    for s, tr in zip(specs, trajs):
                        diffs.append(float(np.abs(NP.plan(proc_ref, NF.with_ann(scene, idx, ann_of(draft, s))).poses - tr.poses).max()))
                    rec["pose_maxdiff"] = max(diffs)
                rows.append(rec)
            except Exception as e:                                                    # noqa: BLE001
                rows.append({"log": log, "token": tok, "err": f"{type(e).__name__}: {e}"})
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{li + 1}/{len(logs)}] {log}: {len(rows)} tokens", flush=True)
    print("K20 FAST DONE", flush=True)


if __name__ == "__main__":
    main()
