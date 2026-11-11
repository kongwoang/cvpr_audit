"""K30 V8a: the K20 / NAVSIM perception-in-the-loop PDM-Closed as a NAVSIM v2 agent, so that the installed v2.2 devkit's own
run_pdm_score_one_stage.py / run_pdm_score.py compute EPDMS and its sub-scores (and the two-stage aggregation on navhard).
Per token: the branch's stored CAM_F0 detections (K20 / final NAVSIM run: dets_final; K30 V6 i: dets_k30 adds hybrid_pdm and
native) replace the current frame's annotations by navsim_pdms.branch_annotations (K20's rule: in-camera eligible objects
kept only if matched, unmatched detections become static ground-plane obstacles, everything outside CAM_F0 untouched);
PDM-Closed is initialised on that scene exactly as navsim_pdms.plan (NavSimScenario, MetricCacheProcessor's planner) and its
8 poses at 0.5 s are returned. branch "reference" = the scene's own annotations (privileged PDM-Closed).
Hydra: agent._target_=k30_nav_v2agent.PerceptionPDMAgent +agent.branch=<b> +agent.dets_dirs=<dir1,dir2> (scripts/A_navsim on
PYTHONPATH; env OPENSCENE_DATA_ROOT, NAVSIM_EXP_ROOT, NUPLAN_MAPS_ROOT, B).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import pickle
from pathlib import Path

import numpy as np
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.geometry.convert import absolute_to_relative_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

_B = Path(os.environ["B"])


def _np_module():
    os.environ.setdefault("NAVSIM_DEVKIT_ROOT", str(_B / "third_party" / "navsim"))
    s = importlib.util.spec_from_file_location("k30_np", _B / "scripts" / "A_navsim" / "navsim_pdms.py")
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


class PerceptionPDMAgent(AbstractAgent):
    requires_scene = True

    def __init__(self, branch: str = "cheap", dets_dirs: str = "",
                 trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)):
        super().__init__(trajectory_sampling, requires_scene=True)
        self.branch = branch
        exp = Path(os.environ["NAVSIM_EXP_ROOT"])
        self.dirs = [Path(d) if "/" in d else exp / d for d in (dets_dirs.split(",") if dets_dirs else ["dets_k30", "dets_final"])]

    def name(self) -> str:
        return f"perception_pdm_{self.branch}"

    def initialize(self) -> None:
        self.NP = _np_module()
        self.proc = MetricCacheProcessor(cache_path=None, force_feature_computation=False,
                                         proposal_sampling=TrajectorySampling(time_horizon=4, interval_length=0.1))
        exp = Path(os.environ["NAVSIM_EXP_ROOT"])
        self.gz = float((exp / "dets_final" / "ground_z.txt").read_text())
        self.priors = pickle.load(open(exp / "dets" / "size_priors.pkl", "rb"))
        self.cache = {}

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def _dets(self, log, tok):
        if log not in self.cache:
            self.cache = {log: {}}
            for d in reversed(self.dirs):                       # later dirs first, earlier ones override
                f = d / f"{log}.pkl"
                if f.exists():
                    for t, r in pickle.load(open(f, "rb")).items():
                        self.cache[log].setdefault(t, {}).update(r)
        return self.cache[log].get(tok)

    def compute_trajectory(self, agent_input: AgentInput, scene=None) -> Trajectory:
        idx = scene.scene_metadata.num_history_frames - 1
        if self.branch != "reference":
            rec = self._dets(scene.scene_metadata.log_name, scene.scene_metadata.initial_token)
            if rec is None or self.branch not in rec or "cam" not in rec:
                raise KeyError(f"no {self.branch} detections for {scene.scene_metadata.initial_token}")
            cam = self.NP.Cam(*rec["cam"])
            fr = dataclasses.replace(scene.frames[idx], annotations=self.NP.branch_annotations(
                scene.frames[idx].annotations, rec[self.branch], cam, self.gz, self.priors, self.branch))
            frames = list(scene.frames); frames[idx] = fr
            scene = dataclasses.replace(scene, frames=frames)
        scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")
        pi, init = self.proc._get_planner_inputs(scenario)
        self.proc._pdm_closed.initialize(init)
        traj = self.proc._pdm_closed.compute_planner_trajectory(pi)
        ego = scenario.initial_ego_state
        ts = self._trajectory_sampling
        states = [traj.get_state_at_time(TimePoint(int(ego.time_point.time_us + (k + 1) * ts.interval_length * 1e6))).rear_axle
                  for k in range(ts.num_poses)]
        rel = absolute_to_relative_poses([ego.rear_axle] + states)[1:]
        return Trajectory(np.array([[p.x, p.y, p.heading] for p in rel], np.float32), ts)
