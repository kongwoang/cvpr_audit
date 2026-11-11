#!/usr/bin/env python
"""K30 V6 (i) / V8a (post-hoc): a hybrid for PDM-Closed on NAVSIM navtest = the support of PDM-Closed (the frozen probing
run's support boxes, field_test) + a fixed tube along PDM-Closed's route centerline, at the token's spec_pdm crop area;
and a native-resolution whole-frame branch. Stages (env: scripts/common/env_navsim.sh, NAVSIM v1.1 as K20 / the final run):
  center   CPU: per navtest token, PDM-Closed initialised exactly as navsim_pdms.plan on the token's scene (annotations;
           the centerline does not depend on the objects), its route centerline (planner._centerline) transformed into
           the annotation (= LiDAR = ego, lidar2ego is the identity) frame -> <EXP>/k30_center/<log>.pkl {token: (N, 2)}
  detect   GPU: per token, the tube = union of the CAM_F0 image boxes of 2 m long, 1.6 m high ground boxes centred on the
           centerline at x in the probing field's X_GRID (8 ... 55 m), heading along the centerline, half-width widened in
           0.25 m steps from 0.5 m while (support crops + tube) area <= the token's spec_pdm crop area (px_spec_pdm);
           crops = support crops (box_crop, as spec_pdm_dets) + tube, detected with spec_pdm_dets' crop / guard / merge
           machinery (SPEC_CROP_SCALE 0.5 as the final run); plus "native" = YOLOv8s on the whole 1920x1080 image (1088x1920).
           Writes <EXP>/dets_k30/<log>.pkl = dets_final's record + hybrid_pdm, px_hybrid_pdm, native.
  rehybrid GPU: hybrid_pdm again with the FIXED tube (2026-10-16): the tube follows the route AHEAD (forward_path: from the
           centerline point nearest the ego until the route turns back, x falling 0.5 m below its running maximum). The
           first version interpolated y over ALL centerline points in 0 <= x <= 100 m sorted by x, which mixes the parts of
           a turning route (39% of navtest tokens had a tube point > 1 m off the route ahead). Reads dets_k30, writes
           <EXP>/dets_k30b/<log>.pkl = the dets_k30 record with hybrid_pdm / px_hybrid_pdm / hybrid_info replaced.
PDMS then with navsim_pdms.py (NAVSIM_DETS=dets_k30, NAVSIM_BRANCHES=spec_pdm,hybrid_pdm,native, NAVSIM_MAIN=spec_pdm).
usage: k30_nav_hybrid.py --stage center|detect|rehybrid [--shard i/n] [--only_log L]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
from pathlib import Path

import numpy as np

B = Path(os.environ["B"])
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
DATA = Path(os.environ["OPENSCENE_DATA_ROOT"])
X_GRID = np.array([8.0, 12.0, 17.0, 23.0, 30.0, 40.0, 55.0])


def load_mod(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def center(args):
    import yaml
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    NP = load_mod("np_", B / "scripts" / "A_navsim" / "navsim_pdms.py")
    f = yaml.safe_load((NP.DEV / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml").read_text())
    si, sn = (int(x) for x in args.shard.split("/"))
    OUT = EXP / "k30_center"
    OUT.mkdir(parents=True, exist_ok=True)
    logs = sorted(set(f["log_names"])) if not args.only_log else [args.only_log]
    for li, log in enumerate(logs):
        if li % sn != si or (OUT / f"{log}.pkl").exists():
            continue
        flt = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True, log_names=[log],
                          tokens=[t for t in f["tokens"]])
        sl = SceneLoader(DATA / "navsim_logs" / "test", DATA / "sensor_blobs" / "test", flt, SensorConfig.build_no_sensors())
        proc = MetricCacheProcessor(cache_path=None, force_feature_computation=False)
        out = {}
        for tok in sl.tokens:
            try:
                scene = sl.get_scene_from_token(tok)
                scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")
                pi, init = proc._get_planner_inputs(scenario)
                proc._pdm_closed.initialize(init)
                proc._pdm_closed.compute_planner_trajectory(pi)
                cl = proc._pdm_closed._centerline
                xy = np.asarray(cl._states_se2_array, float)[:, :2] if hasattr(cl, "_states_se2_array") else \
                    np.array([[s.x, s.y] for s in cl._discrete_path], float)
                ego = scenario.initial_ego_state.rear_axle               # nuPlan's ego frame = rear axle = annotation frame
                dx, dy = xy[:, 0] - ego.x, xy[:, 1] - ego.y
                ch_, sh_ = np.cos(ego.heading), np.sin(ego.heading)
                pe = np.stack([ch_ * dx + sh_ * dy, -sh_ * dx + ch_ * dy], 1)
                out[tok] = pe[(pe[:, 0] >= -5) & (pe[:, 0] <= 100)]
            except Exception as ex:                                                     # noqa: BLE001
                out[tok] = type(ex).__name__
        with open(OUT / f"{log}.pkl", "wb") as fh:
            pickle.dump(out, fh)
        print(f"  [{li + 1}/{len(logs)}] {log}: {len(out)} tokens", flush=True)
    print("K30 NAV CENTER SHARD DONE", flush=True)


def forward_path(cl):
    """the route ahead: centerline points from the one nearest the ego until the route turns back (x below its running
    maximum by more than 0.5 m), keeping only points that advance x (strictly increasing x for interpolation)"""
    c = np.asarray(cl, float).reshape(-1, 2)
    r = c[int(np.argmin(np.hypot(c[:, 0], c[:, 1]))):]
    stop = np.flatnonzero(r[:, 0] < np.maximum.accumulate(r[:, 0]) - 0.5)
    r = r[: stop[0]] if len(stop) else r
    return r[np.r_[True, r[1:, 0] > np.maximum.accumulate(r[:-1, 0])]] if len(r) > 1 else r


def hybrid_dets(rec, img, support_boxes, cl, cam, gz, NS, budget_px):
    """hybrid_pdm on one token: support crops (box_crop) + a centerline tube widened while (support + tube) area <= the
    spec_pdm crop area, detected with spec_pdm_dets' crop / guard / merge machinery -> (dets, px, info)"""
    from rap.mono import box_iou
    W, H = NS.W, NS.H
    budget = float(budget_px) * W * H
    crops = []
    for b in support_boxes:
        bc = NS.VS.box_crop(np.asarray(b, float), W, H)
        if not NS.VS.covered(bc, crops):
            crops.append(bc)
    used = sum((c_[2] - c_[0]) * (c_[3] - c_[1]) for c_ in crops)
    tube, hw_used = None, np.nan
    fp = forward_path(cl) if isinstance(cl, np.ndarray) and len(cl) >= 2 else np.zeros((0, 2))
    if len(fp) >= 2 and budget - used > 0:
        xs, ys = fp[:, 0], fp[:, 1]
        pts = [(x, float(np.interp(x, xs, ys))) for x in X_GRID if xs.min() <= x <= xs.max()]
        for hw in np.arange(0.5, 12.01, 0.25):
            ib = []
            for k, (x, y) in enumerate(pts):
                x2, y2 = pts[min(k + 1, len(pts) - 1)] if k + 1 < len(pts) else pts[max(k - 1, 0)]
                yaw = float(np.arctan2(y2 - y, x2 - x)) if (x2, y2) != (x, y) else 0.0
                if k + 1 >= len(pts) and len(pts) > 1:
                    yaw = float(np.arctan2(y - y2, x - x2))
                q = cam.project(np.array([x, y, gz + 0.8, 2.0, 2 * hw, 1.6, yaw]))
                if q is not None:
                    ib.append(np.asarray(q, float))
            if not ib:
                break
            ib = np.asarray(ib)
            c_ = np.array([ib[:, 0].min(), ib[:, 1].min(), ib[:, 2].max(), ib[:, 3].max()])
            if used + (c_[2] - c_[0]) * (c_[3] - c_[1]) > budget:
                break
            tube, hw_used = c_, hw
    sel = crops + ([tube] if tube is not None else [])
    keep = np.asarray(rec["cheap"]["conf"]) >= 0.25
    ch = {k_: np.asarray(rec["cheap"][k_])[keep] for k_ in ("xyxy", "conf", "coarse")}
    ch["xyxy"] = ch["xyxy"].astype(float)
    full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
    grown = []
    if sel and img is not None:
        gdet = NS.VS.guarded(NS.real_crop_detector(img), W, H)
        for c_ in sel:
            d_, ex_ = gdet(c_)
            grown += ex_
            full = {k_: np.concatenate([full[k_], np.asarray(d_[k_])]) for k_ in full}
        if len(full["conf"]) > 1:
            o_, iou_ = np.argsort(-full["conf"], kind="stable"), box_iou(full["xyxy"], full["xyxy"])
            kp = []
            for i_ in o_:
                if not kp or iou_[i_, kp].max() < 0.5:
                    kp.append(i_)
            full = {k_: full[k_][np.sort(np.array(kp))] for k_ in full}
    outm = NS.VS.centres_in(ch["xyxy"], sel) if sel else np.zeros(len(ch["xyxy"]), bool)
    if grown and len(full["conf"]) and len(ch["xyxy"]):
        outm |= box_iou(ch["xyxy"], full["xyxy"]).max(1) >= 0.5
    dets = {k_: np.concatenate([ch[k_][~outm], full[k_]]) for k_ in ("xyxy", "conf", "coarse")}
    px = sum((c_[2] - c_[0]) * (c_[3] - c_[1]) for c_ in sel + grown) / (W * H)
    return dets, px, {"n_support_crops": len(crops), "tube": tube, "tube_half_width_m": hw_used, "tube_version": "forward_path"}


def rehybrid(args):
    import cv2
    ND = load_mod("nd_", B / "scripts" / "A_navsim" / "navsim_detect.py")
    NP = load_mod("np2_", B / "scripts" / "A_navsim" / "navsim_pdms.py")
    NS = ND.NS
    si, sn = (int(x) for x in args.shard.split("/"))
    SRC, OUT = EXP / "dets_k30", EXP / "dets_k30b"
    FLD, CEN = EXP / "field_test", EXP / "k30_center"
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "ground_z.txt").exists():
        (OUT / "ground_z.txt").write_text((SRC / "ground_z.txt").read_text())
    gz = float((EXP / "dets_final" / "ground_z.txt").read_text())
    logs = sorted(p.stem for p in SRC.glob("*.pkl")) if not args.only_log else [args.only_log]
    for li, log in enumerate(logs):
        if li % sn != si or (OUT / f"{log}.pkl").exists():
            continue
        src = pickle.load(open(SRC / f"{log}.pkl", "rb"))
        fields = pickle.load(open(FLD / f"{log}.pkl", "rb")) if (FLD / f"{log}.pkl").exists() else {}
        cen = pickle.load(open(CEN / f"{log}.pkl", "rb"))
        frames = {fr["token"]: fr for fr in pickle.load(open(DATA / "navsim_logs" / "test" / f"{log}.pkl", "rb"))}
        out = {}
        for tok, rec in src.items():
            rec = dict(rec)
            if "cam" in rec and tok in frames and "hybrid_pdm" in rec:
                img = cv2.imread(str(DATA / "sensor_blobs" / "test" / frames[tok]["cams"]["CAM_F0"]["data_path"]))
                rec["hybrid_pdm"], rec["px_hybrid_pdm"], rec["hybrid_info"] = hybrid_dets(
                    rec, img, fields.get(tok, {}).get("support_boxes", []), cen.get(tok), NP.Cam(*rec["cam"]), gz, NS,
                    rec.get("px_spec_pdm", 0.0))
            out[tok] = rec
        with open(OUT / f"{log}.pkl", "wb") as fh:
            pickle.dump(out, fh)
        print(f"  [{li + 1}/{len(logs)}] {log}: {len(out)} tokens", flush=True)
    print("K30 NAV REHYBRID SHARD DONE", flush=True)


def detect(args):
    import cv2
    ND = load_mod("nd_", B / "scripts" / "A_navsim" / "navsim_detect.py")
    NP = load_mod("np2_", B / "scripts" / "A_navsim" / "navsim_pdms.py")
    NS = ND.NS
    from rap.detect import Mode
    from rap.mono import box_iou
    W, H = NS.W, NS.H
    si, sn = (int(x) for x in args.shard.split("/"))
    SRC, OUT = EXP / "dets_final", EXP / "dets_k30"
    FLD, CEN = EXP / "field_test", EXP / "k30_center"
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "ground_z.txt").exists():
        (OUT / "ground_z.txt").write_text((SRC / "ground_z.txt").read_text())
    gz = float((SRC / "ground_z.txt").read_text())
    det = NS._detector()
    NATIVE = Mode("ns_native_1920", 1088, 1920)
    logs = sorted(p.stem for p in SRC.glob("*.pkl")) if not args.only_log else [args.only_log]
    for li, log in enumerate(logs):
        if li % sn != si or (OUT / f"{log}.pkl").exists():
            continue
        src = pickle.load(open(SRC / f"{log}.pkl", "rb"))
        fields = pickle.load(open(FLD / f"{log}.pkl", "rb")) if (FLD / f"{log}.pkl").exists() else {}
        cen = pickle.load(open(CEN / f"{log}.pkl", "rb"))
        frames = {fr["token"]: fr for fr in pickle.load(open(DATA / "navsim_logs" / "test" / f"{log}.pkl", "rb"))}
        out = {}
        for tok, rec in src.items():
            rec = dict(rec)
            if "cam" not in rec or tok not in frames:
                out[tok] = rec
                continue
            img = cv2.imread(str(DATA / "sensor_blobs" / "test" / frames[tok]["cams"]["CAM_F0"]["data_path"]))
            cam = NP.Cam(*rec["cam"])
            budget = float(rec.get("px_spec_pdm", 0.0)) * W * H
            crops = []
            for b in fields.get(tok, {}).get("support_boxes", []):
                bc = NS.VS.box_crop(np.asarray(b, float), W, H)
                if not NS.VS.covered(bc, crops):
                    crops.append(bc)
            used = sum((c_[2] - c_[0]) * (c_[3] - c_[1]) for c_ in crops)
            tube, hw_used = None, np.nan
            cl = cen.get(tok)
            if isinstance(cl, np.ndarray) and len(cl) >= 2 and budget - used > 0:
                o = np.argsort(cl[:, 0]); xs, ys = cl[o, 0], cl[o, 1]
                pts = [(x, float(np.interp(x, xs, ys))) for x in X_GRID if xs.min() <= x <= xs.max()]
                for hw in np.arange(0.5, 12.01, 0.25):
                    ib = []
                    for k, (x, y) in enumerate(pts):
                        x2, y2 = pts[min(k + 1, len(pts) - 1)] if k + 1 < len(pts) else pts[max(k - 1, 0)]
                        yaw = float(np.arctan2(y2 - y, x2 - x)) if (x2, y2) != (x, y) else 0.0
                        if k + 1 >= len(pts) and len(pts) > 1:
                            yaw = float(np.arctan2(y - y2, x - x2))
                        q = cam.project(np.array([x, y, gz + 0.8, 2.0, 2 * hw, 1.6, yaw]))
                        if q is not None:
                            ib.append(np.asarray(q, float))
                    if not ib:
                        break
                    ib = np.asarray(ib)
                    c_ = np.array([ib[:, 0].min(), ib[:, 1].min(), ib[:, 2].max(), ib[:, 3].max()])
                    if used + (c_[2] - c_[0]) * (c_[3] - c_[1]) > budget:
                        break
                    tube, hw_used = c_, hw
            sel = crops + ([tube] if tube is not None else [])
            # spec_pdm_dets' crop / guard / merge machinery on these crops
            keep = np.asarray(rec["cheap"]["conf"]) >= 0.25
            ch = {k_: np.asarray(rec["cheap"][k_])[keep] for k_ in ("xyxy", "conf", "coarse")}
            ch["xyxy"] = ch["xyxy"].astype(float)
            full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
            grown = []
            if sel and img is not None:
                gdet = NS.VS.guarded(NS.real_crop_detector(img), W, H)
                for c_ in sel:
                    d_, ex_ = gdet(c_)
                    grown += ex_
                    full = {k_: np.concatenate([full[k_], np.asarray(d_[k_])]) for k_ in full}
                if len(full["conf"]) > 1:
                    o_, iou_ = np.argsort(-full["conf"], kind="stable"), box_iou(full["xyxy"], full["xyxy"])
                    kp = []
                    for i_ in o_:
                        if not kp or iou_[i_, kp].max() < 0.5:
                            kp.append(i_)
                    full = {k_: full[k_][np.sort(np.array(kp))] for k_ in full}
            outm = NS.VS.centres_in(ch["xyxy"], sel) if sel else np.zeros(len(ch["xyxy"]), bool)
            if grown and len(full["conf"]) and len(ch["xyxy"]):
                outm |= box_iou(ch["xyxy"], full["xyxy"]).max(1) >= 0.5
            rec["hybrid_pdm"] = {k_: np.concatenate([ch[k_][~outm], full[k_]]) for k_ in ("xyxy", "conf", "coarse")}
            rec["px_hybrid_pdm"] = sum((c_[2] - c_[0]) * (c_[3] - c_[1]) for c_ in sel + grown) / (W * H)
            rec["hybrid_info"] = {"n_support_crops": len(crops), "tube": tube, "tube_half_width_m": hw_used}
            if img is not None:
                d = det.detect(img, NATIVE)[0]
                kk = np.asarray(d["conf"]) >= 0.25
                rec["native"] = {k_: np.asarray(d[k_])[kk] for k_ in ("xyxy", "conf", "coarse")}
            out[tok] = rec
        with open(OUT / f"{log}.pkl", "wb") as fh:
            pickle.dump(out, fh)
        print(f"  [{li + 1}/{len(logs)}] {log}: {len(out)} tokens", flush=True)
    print("K30 NAV DETECT SHARD DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["center", "detect", "rehybrid"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only_log", default="")
    args = ap.parse_args()
    {"center": center, "detect": detect, "rehybrid": rehybrid}[args.stage](args)


if __name__ == "__main__":
    main()
