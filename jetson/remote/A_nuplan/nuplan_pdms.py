#!/usr/bin/env python
"""Standard driving metric on nuPlan: NAVSIM-v1 PDM Score (PDMS) of each perception branch's planned trajectory.

Same 1,440 nuPlan-mini states, branches (reference = logged tracks, cheap, full, spec = v3 with real crops) and planners
(PDM-Closed, IDM) as nuplan_spec_counterfactual.py; only the scoring differs.  Scoring follows NAVSIM v1.1
(navsim/evaluate/pdm_score.py and metric_caching) with tuPlan Garage's PDMSimulator / PDMScorer:
  metric cache   PDM-Closed planned on the reference (true) observation -> its trajectory, centerline, route lanes and
                 drivable-area map; observation = the TRUE tracked objects of the log at 0.0 ... 5.0 s every 0.1 s
                 (non-reactive; NAVSIM's `update_detections_tracks`, re-implemented in GTObservation below)
  scoring        the branch trajectory and the metric-cache trajectory, both sampled 4 s at 0.1 s, LQR-simulated
                 (PDMSimulator), scored together (PDMScorer); for the branch (index 1):
                 NC, DAC, DDC (multiplicative), EP, TTC, C (weighted); PDMS_v1 = NC * DAC * (5 EP + 5 TTC + 2 C) / 12
                 with EP normalised against the metric-cache trajectory as in NAVSIM v1 (DDC not in PDMS_v1; the
                 tuPlan-Garage aggregate with DDC is kept as pdm_tuplan).
Run inside env_nuplan.sh with cwd = code/risk-aware-perception, sharded like the counterfactual.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("nsc", HERE / "nuplan_spec_counterfactual.py")
NC = importlib.util.module_from_spec(_s)
_s.loader.exec_module(NC)
cf82 = NC.cf82

from nuplan.common.actor_state.state_representation import TimePoint                                 # noqa: E402
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder  # noqa: E402
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter                           # noqa: E402
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer      # noqa: E402
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks                  # noqa: E402
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput   # noqa: E402
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration  # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling              # noqa: E402
from nuplan.planning.utils.multithreading.worker_sequential import Sequential                         # noqa: E402
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation  # noqa: E402
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap  # noqa: E402
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (                # noqa: E402
    PROGRESS_DISTANCE_THRESHOLD, PDMScorer)
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator  # noqa: E402
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (    # noqa: E402
    ego_states_to_state_array)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (                   # noqa: E402
    MultiMetricIndex, WeightedMetricIndex)

from rap.paths import NUPLAN_MAPS, NUPLAN_ROOT, RESULTS                                              # noqa: E402

SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)        # NAVSIM: 4 s at 10 Hz
BRANCHES = ("reference",) + NC.REAL_BRANCHES


class GTObservation(PDMObservation):
    """PDMObservation filled with the true future tracks (NAVSIM v1.1 `update_detections_tracks`)."""

    def update_detections_tracks(self, detection_tracks: List[DetectionsTracks]) -> None:
        occupancy_maps, unique = [], {}
        for dt_ in detection_tracks:
            tokens, polygons = [], []
            for o in dt_.tracked_objects:
                tokens.append(o.track_token)
                polygons.append(o.box.geometry)
                unique.setdefault(o.track_token, o)
            occupancy_maps.append(PDMOccupancyMap(tokens, polygons))
        assert len(occupancy_maps) == self._observation_samples + 1, (len(occupancy_maps), self._observation_samples)
        self._occupancy_maps = occupancy_maps
        self._collided_track_ids = []
        self._unique_objects = unique
        self._initialized = True

    @property
    def unique_objects(self):
        return self._unique_objects


def as_array(traj, start: TimePoint):
    """navsim.evaluate.pdm_score.get_trajectory_as_array"""
    times_s = np.arange(0.0, SAMPLING.time_horizon + SAMPLING.interval_length, SAMPLING.interval_length) + start.time_s
    times_us = np.clip([int(t * 1e6) for t in times_s], traj.start_time.time_us, traj.end_time.time_us)
    return ego_states_to_state_array(traj.get_state_at_times([TimePoint(int(t)) for t in times_us]))


def plan(kind, sc, it, hist, route_fix=True):
    p = cf82.make_planner(kind)
    p.initialize(PlannerInitialization(route_roadblock_ids=cf82.route_for(kind, sc, it, route_fix),
                                       mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
    traj = p.compute_trajectory(PlannerInput(
        iteration=SimulationIteration(sc.get_ego_state_at_iteration(it).time_point, it), history=hist,
        traffic_light_data=list(sc.get_traffic_light_status_at_iteration(it))))
    return p, traj


def pdms(cache, traj):
    ego = cache["ego"]
    states = np.concatenate([as_array(cache["traj"], ego.time_point)[None], as_array(traj, ego.time_point)[None]], 0)
    sim = PDMSimulator(SAMPLING).simulate_proposals(states, ego)
    scorer = PDMScorer(SAMPLING)
    tup = scorer.score_proposals(sim, ego, cache["obs"], cache["centerline"], cache["route"], cache["dam"], cache["map"])
    mm, wm = scorer._multi_metrics, scorer._weighted_metrics
    nc, dac, ddc = mm[MultiMetricIndex.NO_COLLISION], mm[MultiMetricIndex.DRIVABLE_AREA], mm[MultiMetricIndex.DRIVING_DIRECTION]
    raw = scorer._progress_raw * nc * dac                                   # NAVSIM v1: EP normalised over NC * DAC
    ep = raw / raw.max() if raw.max() > PROGRESS_DISTANCE_THRESHOLD else np.where(nc * dac > 0, 1.0, 0.0)
    ttc, c = wm[WeightedMetricIndex.TTC], wm[WeightedMetricIndex.COMFORTABLE]
    i = 1
    return {"NC": float(nc[i]), "DAC": float(dac[i]), "DDC": float(ddc[i]), "EP": float(ep[i]), "TTC": float(ttc[i]),
            "C": float(c[i]), "PDMS": float(nc[i] * dac[i] * (5 * ep[i] + 5 * ttc[i] + 2 * c[i]) / 12),
            "pdm_tuplan": float(tup[i])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--buffer", type=int, default=4)
    ap.add_argument("--planners", nargs="+", default=["pdm_closed", "idm"])
    args = ap.parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    t0 = time.time()
    import pickle
    with open(NC.OUTC / "branches.pkl", "rb") as f:
        filt = NC.RealPerceptionFilter(pickle.load(f)["spec"])
    states = pd.read_csv(Path(RESULTS) / "final" / "phase0g_external_pdm_closed_raw.csv")[["scenario", "log", "iteration"]]
    builder = NuPlanScenarioBuilder(data_root=f"{NUPLAN_ROOT}/nuplan-v1.1/splits/mini", map_root=str(NUPLAN_MAPS),
                                    sensor_root=None, db_files=None, map_version="nuplan-maps-v1.0")
    flt = ScenarioFilter(scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
                         num_scenarios_per_type=None, limit_total_scenarios=60, timestamp_threshold_s=None,
                         ego_displacement_minimum_m=None, expand_scenarios=False, remove_invalid_goals=True, shuffle=False)
    scen = {s.token: s for s in builder.get_scenarios(flt, Sequential())}
    ckpt = NC.OUTC / "rows_pdms"
    ckpt.mkdir(parents=True, exist_ok=True)
    for n_sc, (tok, g) in enumerate(states.groupby("scenario", sort=True), 1):
        out_f = ckpt / f"{tok}.json"
        if out_f.exists() or (n_sc - 1) % shard_n != shard_i:
            continue
        sc = scen[tok]
        dt = sc.database_interval
        n_it = sc.get_number_of_iterations()
        rows = []
        for it in sorted(int(i) for i in g.iteration):
            base = SimulationHistoryBuffer.initialize_from_scenario(args.buffer, sc, DetectionsTracks)
            for k in range(max(it - args.buffer + 1, 0), it + 1):
                base.append(sc.get_ego_state_at_iteration(k), sc.get_tracked_objects_at_iteration(k))
            hist = {b: cf82.filtered_history(base, filt, tok, it, b, sc) for b in BRANCHES}
            row = {"scenario": tok, "log": sc.log_name, "iteration": it}
            try:
                p_ref, traj_ref = plan("pdm_closed", sc, it, hist["reference"])
                obs = GTObservation(SAMPLING, SAMPLING, map_radius=100.0, observation_sample_res=1)
                steps = obs._observation_samples + 1
                obs.update_detections_tracks([sc.get_tracked_objects_at_iteration(min(it + int(round(j * 0.1 / dt)), n_it - 1))
                                              for j in range(steps)])
                cache = {"ego": sc.get_ego_state_at_iteration(it), "traj": traj_ref, "obs": obs, "map": sc.map_api,
                         "centerline": p_ref._centerline, "route": p_ref._route_lane_dict, "dam": p_ref._drivable_area_map}
            except Exception as e:                                                             # noqa: BLE001
                row["err_cache"] = type(e).__name__
                rows.append(row)
                continue
            for kind in args.planners:
                by_sig = {}
                for b in BRANCHES:
                    s = NC.obs_signature(hist[b])
                    if s not in by_sig:
                        try:
                            by_sig[s] = pdms(cache, traj_ref if (kind == "pdm_closed" and b == "reference")
                                             else plan(kind, sc, it, hist[b])[1])
                        except Exception as e:                                                 # noqa: BLE001
                            by_sig[s] = {"err": type(e).__name__}
                    row.update({f"{k}_{kind}_{b}": v for k, v in by_sig[s].items()})
            rows.append(row)
        out_f.write_text(json.dumps(rows))
        print(f"  [{n_sc:2d}/60] {tok} {len(rows)} states {time.time() - t0:6.0f}s", flush=True)
    if len(list(ckpt.glob("*.json"))) < 60:
        print(f"  shard {args.shard} done", flush=True)
        return
    d = pd.DataFrame([r for f in sorted(ckpt.glob("*.json")) for r in json.loads(f.read_text())])
    d.to_csv(NC.OUTC / "pdms_states.csv", index=False)
    uq = d.scenario.unique(); idx = [np.flatnonzero(d.scenario.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    out = []
    for kind in args.planners:
        for br in BRANCHES:
            col = f"PDMS_{kind}_{br}"
            if col not in d:
                continue
            r = {"planner": kind, "branch": br, "n": int(d[col].notna().sum())}
            r.update({m: float(d[f"{m}_{kind}_{br}"].mean()) for m in ("PDMS", "NC", "DAC", "DDC", "EP", "TTC", "C", "pdm_tuplan")})
            for other in [b_ for b_ in BRANCHES if b_ not in ("reference", "spec")]:
                if br == "spec":
                    x, y = d[col].to_numpy(), d[f"PDMS_{kind}_{other}"].to_numpy()
                    dd = [np.nanmean(x[t] - y[t]) for t in boots]
                    r[f"spec_minus_{other}"] = f"[{np.percentile(dd, 2.5):+.4f}, {np.percentile(dd, 97.5):+.4f}]"
            out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(NC.OUTC / "pdms_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))
    print("NUPLAN PDMS DONE")


if __name__ == "__main__":
    main()
