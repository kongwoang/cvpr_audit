#!/usr/bin/env python
"""NAVSIM navtest, stage B (CPU): PDMS of PDM-Closed planning on each perception branch's objects.

Per navtest token and branch:
  objects   reference = the scene's annotations (the privileged PDM-Closed baseline). Other branches follow the nuPlan
            experiment's rule (DEEP 114): eligible annotations (vehicle, pedestrian, bicycle) projected into CAM_F0
            (NAVSIM's convention: annotation boxes are in the LiDAR frame of the camera extrinsics); Hungarian matching
            against the branch's detections (class-compatible, IoU >= 0.3); unmatched in-camera eligible objects
            removed; unmatched detections added as static objects lifted on the ground plane (box bottom centre, class
            median sizes of the navtest annotations, heading along the viewing ray, zero velocity); everything outside
            CAM_F0 untouched.
  planner   PDM-Closed exactly as NAVSIM's MetricCacheProcessor builds it (same sampling, IDM policies, offsets,
            radius), initialised on the modified scene's NavSimScenario, current objects = the branch's objects.
  score     NAVSIM's pdm_score against the token's metric cache (true future objects), simulator / scorer from
            NAVSIM's default_scoring_parameters.yaml; the planned trajectory handed over as NAVSIM's Trajectory
            (8 poses at 0.5 s, ego-local), like any agent.
Writes results/navsim_exp/pdms/<log>.pkl (per-token rows) and, with --summary, pdms_summary.csv (+ paired log
bootstrap of spec − each branch).
"""
from __future__ import annotations

import argparse
import dataclasses
import lzma
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment

from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.geometry.convert import absolute_to_relative_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Annotations, SceneFilter, SensorConfig, Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

B = Path(os.environ["B"])
DATA = Path(os.environ["OPENSCENE_DATA_ROOT"])
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
DEV = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
DETS = Path(os.environ.get("NAVSIM_DETS", EXP / "dets"))             # smoke runs point both elsewhere
OUT = Path(os.environ.get("NAVSIM_PDMS_OUT", EXP / "pdms"))
W, H = 1920, 1080
BRANCHES = tuple(os.environ.get("NAVSIM_BRANCHES", "reference,cheap,full,spec,uncertainty,sahi,sahi_s033,fovea").split(","))
ELIGIBLE = ("vehicle", "pedestrian", "bicycle")
COMPAT = {"vehicle": {"vehicle"}, "pedestrian": {"person"}, "bicycle": {"cyclist", "person"}}
DET_TO_NAME = {"vehicle": "vehicle", "person": "pedestrian", "cyclist": "bicycle"}
SIGNS = np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)], float)
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if int(np.sum(SIGNS[i] != SIGNS[j])) == 1]
NEAR, CLAMP, IOU_THR, FP_MIN_AHEAD, FP_MAX_RANGE = 0.1, 1.25, 0.3, 0.5, 80.0


class Cam:
    """Projection / ground lift for OpenScene CAM_F0 (camera -> LiDAR frame R, t; OpenCV distortion)."""

    def __init__(self, K, D, R, t):
        import cv2
        self.K, self.D, self.R, self.t = (np.asarray(a, float) for a in (K, D, R, t))
        border = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1], [0, H / 2], [W - 1, H / 2], [W / 2, 0],
                           [W / 2, H - 1]], np.float64)
        und = cv2.undistortPoints(border.reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)
        self.bx, self.by = CLAMP * float(np.abs(und[:, 0]).max()), CLAMP * float(np.abs(und[:, 1]).max())

    def distort(self, xn, yn):
        k1, k2, p1, p2, k3 = self.D
        r2 = xn * xn + yn * yn
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        return self.K[0, 0] * xd + self.K[0, 2], self.K[1, 1] * yd + self.K[1, 2]

    def project(self, box):
        """box (x, y, z, l, w, h, heading) in the LiDAR frame -> clipped image box or None (as nuplan_spec_real)."""
        x, y, z, l, w, h, yaw = box
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        pc = (np.array([x, y, z]) + (SIGNS * [l / 2, w / 2, h / 2]) @ Rz.T - self.t) @ self.R
        if (np.array([x, y, z]) - self.t) @ self.R[:, 2] <= 0:
            return None
        pts = [p for p in pc if p[2] > NEAR]
        for i, j in EDGES:
            zi, zj = pc[i, 2], pc[j, 2]
            if (zi - NEAR) * (zj - NEAR) < 0:
                a = (NEAR - zi) / (zj - zi)
                pts.append(pc[i] + a * (pc[j] - pc[i]))
        if not pts:
            return None
        P = np.asarray(pts)
        u, v = self.distort(np.clip(P[:, 0] / P[:, 2], -self.bx, self.bx), np.clip(P[:, 1] / P[:, 2], -self.by, self.by))
        x1, y1, x2, y2 = max(u.min(), 0.0), max(v.min(), 0.0), min(u.max(), W - 1.0), min(v.max(), H - 1.0)
        return None if x2 <= x1 or y2 <= y1 else (x1, y1, x2, y2)

    def ground_point(self, u, v, gz):
        import cv2
        xn, yn = cv2.undistortPoints(np.array([[[u, v]]], np.float64), self.K, self.D).reshape(2)
        d = self.R @ np.array([xn, yn, 1.0])
        if d[2] >= -1e-9:
            return None
        s = (gz - self.t[2]) / d[2]
        if s <= 0:
            return None
        g = self.t + s * d
        return g[:2], d[:2] / max(np.hypot(d[0], d[1]), 1e-9)


def iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])                    # noqa: E731
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def branch_annotations(ann: Annotations, det, cam: Cam, gz, priors, tag):
    """The branch's objects as NAVSIM Annotations (rule in the module docstring)."""
    boxes, names = np.asarray(ann.boxes, float), list(ann.names)
    elig = [i for i, n in enumerate(names) if n in ELIGIBLE]
    proj = {i: cam.project(boxes[i]) for i in elig}
    inc = [i for i in elig if proj[i] is not None]
    dx, dc = np.asarray(det["xyxy"], float).reshape(-1, 4), np.asarray(det["coarse"]).astype(str)
    matched, used = set(), set()
    if inc and len(dx):
        M = iou([proj[i] for i in inc], dx)
        for a, i in enumerate(inc):
            M[a, [j for j in range(len(dx)) if dc[j] not in COMPAT[names[i]]]] = 0
        r, c = linear_sum_assignment(-M)
        for a, j in zip(r, c):
            if M[a, j] >= IOU_THR:
                matched.add(inc[a]); used.add(j)
    keep = [i for i in range(len(names)) if not (i in inc and i not in matched)]
    nb, nn = [boxes[i] for i in keep], [names[i] for i in keep]
    nv = [np.asarray(ann.velocity_3d)[i] for i in keep]
    nt = [ann.instance_tokens[i] for i in keep]; nk = [ann.track_tokens[i] for i in keep]
    for j in range(len(dx)):
        if j in used:
            continue
        b = dx[j]
        gp = cam.ground_point(0.5 * (b[0] + b[2]), b[3], gz)
        if gp is None:
            continue
        (gx, gy), ray = gp
        if gx < FP_MIN_AHEAD or np.hypot(gx, gy) > FP_MAX_RANGE:
            continue
        name = DET_TO_NAME[dc[j]]
        L, Wd, Ht = priors[name]
        cx, cy = gx + ray[0] * L / 2, gy + ray[1] * L / 2
        nb.append(np.array([cx, cy, gz + Ht / 2, L, Wd, Ht, float(np.arctan2(ray[1], ray[0]))]))
        nn.append(name); nv.append(np.zeros(3)); tok = f"fp:{tag}:{j}"; nt.append(tok); nk.append(tok)
    return Annotations(boxes=np.asarray(nb, np.float32).reshape(-1, 7), names=nn,
                       velocity_3d=np.asarray(nv, np.float32).reshape(-1, 3), instance_tokens=nt, track_tokens=nk)


def plan(proc: MetricCacheProcessor, scene):
    scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")
    pi, init = proc._get_planner_inputs(scenario)
    proc._pdm_closed.initialize(init)
    traj = proc._pdm_closed.compute_planner_trajectory(pi)
    ego = scenario.initial_ego_state
    ts = TrajectorySampling(time_horizon=4, interval_length=0.5)
    states = [traj.get_state_at_time(TimePoint(int(ego.time_point.time_us + (k + 1) * 0.5e6))).rear_axle
              for k in range(ts.num_poses)]
    rel = absolute_to_relative_poses([ego.rear_axle] + states)[1:]
    return Trajectory(np.array([[p.x, p.y, p.heading] for p in rel], np.float32), ts)


def run(args):
    si, sn = (int(x) for x in args.shard.split("/"))
    cache = Path(os.environ.get("NAVSIM_CACHE", EXP / "metric_cache"))
    split = os.environ.get("NAVSIM_SPLIT", "test")
    if os.environ.get("NAVSIM_CACHE"):
        ps = list(cache.glob("*/*/*/metric_cache.pkl"))
        f = {"tokens": [p.parent.name for p in ps], "log_names": sorted({p.parts[-4] for p in ps})}
    else:
        f = yaml.safe_load((DEV / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml").read_text())
    params = OmegaConf.load(DEV / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml")
    simulator, scorer = instantiate(params.simulator), instantiate(params.scorer)
    # metric-cache files by token (<log>/<scenario type>/<token>/metric_cache.pkl, MetricCacheLoader's layout)
    mcp = {p.parent.name: p for p in cache.glob("*/*/*/metric_cache.pkl")}
    gz = float((DETS / "ground_z.txt").read_text())
    priors = pickle.load(open(EXP / "dets" / "size_priors.pkl", "rb"))
    OUT.mkdir(parents=True, exist_ok=True)
    logs = sorted(set(f["log_names"]))
    for li, log in enumerate(logs):
        if li % sn != si or (OUT / f"{log}.pkl").exists() or not (DETS / f"{log}.pkl").exists():
            continue
        dets = pickle.load(open(DETS / f"{log}.pkl", "rb"))
        flt = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                          log_names=[log], tokens=[t for t in f["tokens"] if t in dets])
        sl = SceneLoader(DATA / "navsim_logs" / split, DATA / "sensor_blobs" / split, flt, SensorConfig.build_no_sensors())
        proc = MetricCacheProcessor(cache_path=None, force_feature_computation=False)
        rows = []
        for tok in sl.tokens:
            if tok not in mcp or "cam" not in dets[tok]:
                continue
            with lzma.open(mcp[tok], "rb") as fh:
                mc = pickle.load(fh)
            scene = sl.get_scene_from_token(tok)
            idx = scene.scene_metadata.num_history_frames - 1
            cam = Cam(*dets[tok]["cam"])
            row = {"log": log, "token": tok, **{f"info_{k}": v for k, v in dets[tok]["info"].items()}}
            for br in BRANCHES:
                try:
                    if br != "reference":
                        fr = dataclasses.replace(scene.frames[idx], annotations=branch_annotations(
                            scene.frames[idx].annotations, dets[tok][br], cam, gz, priors, br))
                        frames = list(scene.frames); frames[idx] = fr
                        sc = dataclasses.replace(scene, frames=frames)
                    else:
                        sc = scene
                    res = pdm_score(mc, plan(proc, sc), simulator.proposal_sampling, simulator, scorer)
                    row.update({f"{k}_{br}": float(v) for k, v in dataclasses.asdict(res).items()})
                except Exception as e:                                                       # noqa: BLE001
                    row[f"err_{br}"] = type(e).__name__
            rows.append(row)
        with open(OUT / f"{log}.pkl", "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{li + 1}/{len(logs)}] {log}: {len(rows)} tokens", flush=True)
    print("NAVSIM PDMS SHARD DONE")


def summary():
    d = pd.DataFrame([r for f in sorted(OUT.glob("*.pkl")) for r in pickle.load(open(f, "rb"))])
    d.to_csv(OUT.parent / f"pdms_states{os.environ.get('NAVSIM_TAG', '')}.csv", index=False)
    uq = d.log.unique(); idx = [np.flatnonzero(d.log.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    out = []
    metrics = ("score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
               "time_to_collision_within_bound", "comfort")
    for br in BRANCHES:
        if f"score_{br}" not in d:
            continue
        r = {"branch": br, "n": int(d[f"score_{br}"].notna().sum())}
        r.update({m: float(d[f"{m}_{br}"].mean()) for m in metrics if f"{m}_{br}" in d})
        main = os.environ.get("NAVSIM_MAIN", "spec")           # the method's branch the CIs are taken against
        if br != main and f"score_{main}" in d:
            x, y = d[f"score_{main}"].to_numpy(), d[f"score_{br}"].to_numpy()
            dd = [np.nanmean(x[t] - y[t]) for t in boots]
            r[f"{main}_minus"] = f"[{np.percentile(dd, 2.5):+.4f}, {np.percentile(dd, 97.5):+.4f}]"
        out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(OUT.parent / f"pdms_summary{os.environ.get('NAVSIM_TAG', '')}.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))
    print("NAVSIM PDMS SUMMARY DONE")


def priors_and_check():
    """Class-median box sizes of the navtest annotations (for the FP objects)."""
    f = yaml.safe_load((DEV / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml").read_text())
    acc = {n: [] for n in ELIGIBLE}
    for log in sorted(set(f["log_names"])):
        for fr in pickle.load(open(DATA / "navsim_logs" / "test" / f"{log}.pkl", "rb")):
            b, n = np.asarray(fr["anns"]["gt_boxes"], float), np.asarray(fr["anns"]["gt_names"])
            for name in ELIGIBLE:
                acc[name] += [tuple(x) for x in b[n == name][:, 3:6]]
    pri = {n: tuple(np.median(np.asarray(v), 0)) for n, v in acc.items()}
    DETS.mkdir(parents=True, exist_ok=True)
    pickle.dump(pri, open(DETS / "size_priors.pkl", "wb"))
    print("size priors (l, w, h):", {k: tuple(round(x, 2) for x in v) for k, v in pri.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["priors", "run", "summary"], required=True)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    {"priors": lambda: priors_and_check(), "run": lambda: run(args), "summary": summary}[args.stage]()


if __name__ == "__main__":
    main()
