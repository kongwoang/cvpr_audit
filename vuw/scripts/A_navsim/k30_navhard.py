#!/usr/bin/env python
"""K30 V8a navhard (post-hoc; pre-registered in handoff/K30/STATUS.md): the perception branches on NAVSIM v2.2's
navhard_two_stage (450 stage-one real scenes from navtest's test logs + 5,462 reactive synthetic stage-two scenes with
rendered sensors), so that run_pdm_score.py two-stage scores them with k30_nav_v2agent.PerceptionPDMAgent
(+agent.dets_dirs=navhard_dets). Same code and settings as the navtest final run / K30 V6 (i), applied to these scenes:
  detect1  GPU: per token, the current frame's CAM_F0 (cv2.imread of the frame's camera_path under its sensor root, as
           navsim_detect.py) ->
           cheap (YOLOv8s 192x320), full960 (576x960), native (1088x1920); conf >= 0.25; camera = the frame's CAM_F0
           intrinsics / distortion / sensor2lidar  -> <EXP>/navhard_d1/<log>.pkl
  field    CPU: per token, navsim_field.py's probing of PDM-Closed on the cheap draft (7x7 probe grid, support removals,
           MOVE_M 0.5; the frozen per-call implementation) and PDM-Closed's route centerline in the annotation frame
           (k30_nav_hybrid.py center) -> <EXP>/navhard_field/shard<i>.pkl
  detect2  GPU: spec_pdm (navsim_detect.spec_pdm_dets on the field, SPEC_CROP_SCALE 0.5) and hybrid_pdm
           (k30_nav_hybrid.hybrid_dets: support crops + the route-ahead tube at the token's spec_pdm crop area) -> <EXP>/navhard_d2/part<i>.pkl
  split    -> <EXP>/navhard_dets/<log>.pkl (per log, as the agent loads them)
Env: scripts/common/env_nuplan.sh with PYTHONPATH = scripts/A_navsim : third_party/navsim (v2.2), OPENSCENE_DATA_ROOT,
NAVSIM_EXP_ROOT, NUPLAN_MAPS_ROOT, B, SPEC_CROP_SCALE=0.5.
usage: k30_navhard.py --stage detect1|field|detect2|split [--shard i/n] [--max_tokens n]   (K30_NH_TAG: folder suffix)
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
from pathlib import Path

import numpy as np
import yaml

B = Path(os.environ["B"])
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
DATA = Path(os.environ["OPENSCENE_DATA_ROOT"])
DEV = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", B / "third_party" / "navsim"))
TAG = os.environ.get("K30_NH_TAG", "")                 # smoke runs: separate folders
D1, FLD, D2, OUT = (EXP / f"navhard_{x}{TAG}" for x in ("d1", "field", "d2", "dets"))


def load_mod(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def loader(sensors):
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader
    y = yaml.safe_load((DEV / "navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml").read_text())
    flt = SceneFilter(**{k: v for k, v in y.items() if not k.startswith("_")})
    sc = SensorConfig(cam_f0=[3], cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False,
                      lidar_pc=False) if sensors else SensorConfig.build_no_sensors()
    sl = SceneLoader(data_path=DATA / "navsim_logs" / "test", original_sensor_path=DATA / "sensor_blobs" / "test", scene_filter=flt,
                     synthetic_sensor_path=DATA / "navhard_two_stage" / "sensor_blobs",
                     synthetic_scenes_path=DATA / "navhard_two_stage" / "synthetic_scene_pickles", sensor_config=sc)
    s1, s2 = sorted(sl.tokens_stage_one), sorted(sl.reactive_tokens_stage_two)
    return sl, [(t, 1) for t in s1] + [(t, 2) for t in s2]


def image_of(scene, idx, stage):
    """(BGR image read with cv2.imread as navsim_detect.py, absolute path, camera); camera_path is relative to the
    sensor root: sensor_blobs/test for stage-one scenes, navhard_two_stage/sensor_blobs for the synthetic ones"""
    import cv2
    c = scene.frames[idx].cameras.cam_f0
    roots = [DATA / "sensor_blobs" / "test", DATA / "navhard_two_stage" / "sensor_blobs"]
    for r in (roots if stage == 1 else roots[::-1]):
        p = r / str(c.camera_path)
        if c.camera_path is not None and p.exists():
            return cv2.imread(str(p)), p, c
    return None, c.camera_path, c


def mine(todo, si, sn):
    return [x for i, x in enumerate(todo) if i % sn == si]


def detect1(args):
    ND = load_mod("nd_", B / "scripts" / "A_navsim" / "navsim_detect.py")
    NS, MODES = ND.NS, ND.MODES
    from rap.detect import Mode
    NATIVE = Mode("ns_native_1920", 1088, 1920)
    si, sn = (int(x) for x in args.shard.split("/"))
    fo = D1 / f"shard{si}.pkl"
    D1.mkdir(parents=True, exist_ok=True)
    if fo.exists():
        return
    sl, todo = loader(True)
    todo = mine(todo, si, sn)[: args.max_tokens or None]
    det = NS._detector()
    out = {}
    for n, (tok, stg) in enumerate(todo):
        scene = sl.get_scene_from_token(tok)
        idx = scene.scene_metadata.num_history_frames - 1
        img, path, c = image_of(scene, idx, stg)
        rec = {"log": scene.scene_metadata.log_name, "stage": stg}
        if img is None:
            out[tok] = dict(rec, missing_image=str(path)); continue
        assert img.shape[:2] == (NS.H, NS.W), img.shape
        cam = ND.NavCam({"cam_intrinsic": c.intrinsics, "distortion": c.distortion, "sensor2lidar_rotation": c.sensor2lidar_rotation,
                         "sensor2lidar_translation": c.sensor2lidar_translation})
        rec["cam"] = (cam.K, cam.D, cam.R, cam.t)
        rec["image"] = str(path)
        for b, md in (("cheap", MODES["ns_cheap_320"]), ("full960", ND.FULL960), ("native", NATIVE)):
            d = det.detect(img, md)[0]
            k = np.asarray(d["conf"]) >= 0.25
            rec[b] = {kk: np.asarray(d[kk])[k] for kk in ("xyxy", "conf", "coarse")}
        out[tok] = rec
        if n % 200 == 0:
            print(f"  {n}/{len(todo)}", flush=True)
    with open(fo, "wb") as fh:
        pickle.dump(out, fh)
    print("K30 NAVHARD DETECT1 SHARD DONE", flush=True)


def d1_all():
    out = {}
    for f in sorted(D1.glob("shard*.pkl")):
        out.update(pickle.load(open(f, "rb")))
    return out


def field(args):
    import dataclasses  # noqa: F401
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    NF = load_mod("nf_", B / "scripts" / "A_navsim" / "navsim_field.py")
    NP = NF.NP
    si, sn = (int(x) for x in args.shard.split("/"))
    fo = FLD / f"shard{si}.pkl"
    FLD.mkdir(parents=True, exist_ok=True)
    if fo.exists():
        return
    d1 = d1_all()
    gz = float((EXP / "dets_final" / "ground_z.txt").read_text())
    priors = pickle.load(open(EXP / "dets" / "size_priors.pkl", "rb"))
    L, Wd, Ht = priors["vehicle"]
    sl, todo = loader(False)
    todo = mine([x for x in todo if "cam" in d1.get(x[0], {})], si, sn)[: args.max_tokens or None]
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    # v2.2 takes the proposal sampling as an argument; 40 x 0.1 s (future 50 x 0.1 s) = v1.1's fixed values
    proc = MetricCacheProcessor(cache_path=None, force_feature_computation=False,
                                proposal_sampling=TrajectorySampling(time_horizon=4, interval_length=0.1))
    out = {}
    for n, (tok, _) in enumerate(todo):
        try:
            scene = sl.get_scene_from_token(tok)
            idx = scene.scene_metadata.num_history_frames - 1
            cam = NP.Cam(*d1[tok]["cam"])
            cheap = d1[tok]["cheap"]
            draft = NP.branch_annotations(scene.frames[idx].annotations, cheap, cam, gz, priors, "cheap")
            t0 = NP.plan(proc, NF.with_ann(scene, idx, draft))
            # route centerline (k30_nav_hybrid.center): planner state of the base call, annotation (= rear axle) frame
            from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
            cl = proc._pdm_closed._centerline
            xy = np.asarray(cl._states_se2_array, float)[:, :2] if hasattr(cl, "_states_se2_array") else \
                np.array([[s.x, s.y] for s in cl._discrete_path], float)
            ego = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0").initial_ego_state.rear_axle
            dx, dy = xy[:, 0] - ego.x, xy[:, 1] - ego.y
            ch_, sh_ = np.cos(ego.heading), np.sin(ego.heading)
            pe = np.stack([ch_ * dx + sh_ * dy, -sh_ * dx + ch_ * dy], 1)
            centre = pe[(pe[:, 0] >= -5) & (pe[:, 0] <= 100)]
            calls, boxes = 1, []
            for x in NF.X_GRID:
                for y in NF.Y_GRID:
                    box = np.array([x, y, gz + Ht / 2, L, Wd, Ht, 0.0])
                    ib = cam.project(box)
                    if ib is None:
                        continue
                    calls += 1
                    if NF.moved(t0, NP.plan(proc, NF.with_ann(scene, idx, NF.add_box(draft, box, f"probe:{x}:{y}")))):
                        boxes.append(ib)
            support = []
            for i, name in enumerate(draft.names):
                if name not in NP.ELIGIBLE or cam.project(draft.boxes[i]) is None:
                    continue
                calls += 1
                if NF.moved(t0, NP.plan(proc, NF.with_ann(scene, idx, NF.drop(draft, i)))):
                    tk = draft.track_tokens[i]
                    dxx = np.asarray(cheap["xyxy"], float).reshape(-1, 4)
                    if tk.startswith("fp:"):
                        support.append(dxx[int(tk.split(":")[-1])])
                    elif len(dxx):
                        support.append(dxx[int(np.argmax(NP.iou([cam.project(draft.boxes[i])], dxx)[0]))])
            out[tok] = {"field_boxes": boxes, "support_boxes": support, "n_calls": calls, "centerline": centre}
        except Exception as e:                                                        # noqa: BLE001
            out[tok] = {"err": f"{type(e).__name__}: {e}"[:200]}
        if n % 50 == 0:
            print(f"  {n}/{len(todo)}", flush=True)
    with open(fo, "wb") as fh:
        pickle.dump(out, fh)
    print("K30 NAVHARD FIELD SHARD DONE", flush=True)


def detect2(args):
    import cv2
    ND = load_mod("nd_", B / "scripts" / "A_navsim" / "navsim_detect.py")
    NP = load_mod("np2_", B / "scripts" / "A_navsim" / "navsim_pdms.py")
    NH = load_mod("nh_", B / "scripts" / "A_navsim" / "k30_nav_hybrid.py")
    NS = ND.NS
    si, sn = (int(x) for x in args.shard.split("/"))
    fo = D2 / f"part{si}.pkl"
    D2.mkdir(parents=True, exist_ok=True)
    if fo.exists():
        return
    d1 = d1_all()
    flds = {}
    for f in sorted(FLD.glob("shard*.pkl")):
        flds.update(pickle.load(open(f, "rb")))
    gz = float((EXP / "dets_final" / "ground_z.txt").read_text())
    toks = mine(sorted(t for t in d1 if "cam" in d1[t]), si, sn)[: args.max_tokens or None]
    out = {}
    for n, tok in enumerate(toks):
        rec = dict(d1[tok])
        fld = flds.get(tok, {})
        img = cv2.imread(rec["image"])
        if "err" in fld or img is None:
            rec["field_err"] = fld.get("err", "no image")
            out[tok] = rec
            continue
        rec["spec_pdm"], rec["px_spec_pdm"] = ND.spec_pdm_dets({"cheap": rec["cheap"]}, img, fld)
        k = np.asarray(rec["spec_pdm"]["conf"]) >= 0.25
        rec["spec_pdm"] = {kk: np.asarray(rec["spec_pdm"][kk])[k] for kk in ("xyxy", "conf", "coarse")}
        # hybrid_pdm: k30_nav_hybrid.hybrid_dets (support crops + the route-ahead tube at the spec_pdm crop area)
        rec["hybrid_pdm"], rec["px_hybrid_pdm"], rec["hybrid_info"] = NH.hybrid_dets(
            rec, img, fld.get("support_boxes", []), fld.get("centerline"), NP.Cam(*rec["cam"]), gz, NS, rec["px_spec_pdm"])
        rec["n_calls"] = fld.get("n_calls")
        out[tok] = rec
        if n % 200 == 0:
            print(f"  {n}/{len(toks)}", flush=True)
    with open(fo, "wb") as fh:
        pickle.dump(out, fh)
    print("K30 NAVHARD DETECT2 SHARD DONE", flush=True)


def split_logs():
    """navhard_d2/part*.pkl -> navhard_dets/<log>.pkl (the agent loads detections per log)"""
    allr = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(D2.glob("part*.pkl")):
        allr.update(pickle.load(open(f, "rb")))
    by = {}
    for t, r in allr.items():
        by.setdefault(r["log"], {})[t] = r
    for lg, d in by.items():
        with open(OUT / f"{lg}.pkl", "wb") as fh:
            pickle.dump(d, fh)
    print(f"{len(allr)} tokens in {len(by)} logs; K30 NAVHARD SPLIT DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["detect1", "field", "detect2", "split"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max_tokens", type=int, default=0)
    args = ap.parse_args()
    {"detect1": detect1, "field": field, "detect2": detect2, "split": lambda a: split_logs()}[args.stage](args)


if __name__ == "__main__":
    main()
