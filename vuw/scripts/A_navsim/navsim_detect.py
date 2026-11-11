#!/usr/bin/env python
"""NAVSIM navtest, stage A (GPU): every perception branch's detections on the CAM_F0 image of each navtest token.

Same branches and code as the nuPlan real-crop experiment (scripts/A_nuplan/nuplan_spec_real.py, imported unchanged):
  cheap  YOLOv8s 192x320 · full  YOLOv8s 384x640 · spec  v3 loop with real crops at the full mode's 1/3 scale
  uncertainty  cheap boxes by uncertainty, crops strictly within spec's pixels · sahi / sahi_s033  SAHI 640-px slices at
  640 / 224 px · fovea  FOVEA KDE_I at 384x640.
The camera is OpenScene's CAM_F0 (same sensor as nuPlan's): intrinsics / distortion / sensor2lidar from the metadata;
the "ego" frame of the lift is the annotation (LiDAR) frame, the frame NAVSIM projects boxes from; the ground plane is
the median bottom of vehicle boxes within 30 m over the navtest annotations.
Writes results/navsim_exp/dets/<log>.pkl: {token: {branch: {xyxy, conf, coarse}, "cam": (K, D, R, t), "info": ...}}.
Run in env_navsim.sh (nuPlan env: ultralytics, sahi) with EXTRA_MODES unset.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

B = Path(os.environ["B"])
os.environ.pop("EXTRA_MODES", None)
_s = importlib.util.spec_from_file_location("nsr", B / "scripts" / "A_nuplan" / "nuplan_spec_real.py")
NS = importlib.util.module_from_spec(_s)
_s.loader.exec_module(NS)
from rap.detect import MODES, Mode                                               # noqa: E402

DATA = Path(os.environ["OPENSCENE_DATA_ROOT"])
OUT = Path(os.environ.get("NAVSIM_DETS", Path(os.environ["NAVSIM_EXP_ROOT"]) / "dets"))   # smoke runs: another dir
FILT = Path(os.environ["NAVSIM_DEVKIT_ROOT"]) / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
BRANCHES = ("cheap", "full", "spec", "uncertainty", "sahi", "sahi_s033", "fovea")
FULL960 = MODES.get("ns_full_960") or __import__("rap.detect", fromlist=["Mode"]).Mode("ns_full_960", 576, 960)


class NavCam(NS.Camera):
    """nuplan_spec_real.Camera with OpenScene's CAM_F0 parameters (camera -> annotation/LiDAR frame)."""

    def __init__(self, c):
        self.K = np.asarray(c["cam_intrinsic"], float)
        self.D = np.asarray(c["distortion"], float)
        self.R = np.asarray(c["sensor2lidar_rotation"], float)
        self.t = np.asarray(c["sensor2lidar_translation"], float)
        W, H = NS.W, NS.H
        border = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1], [0, H / 2], [W - 1, H / 2],
                           [W / 2, 0], [W / 2, H - 1]], np.float64)
        und = cv2.undistortPoints(border.reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)
        self.bx, self.by = NS.CLAMP * float(np.abs(und[:, 0]).max()), NS.CLAMP * float(np.abs(und[:, 1]).max())


SPLIT = os.environ.get("NAVSIM_SPLIT", "test")          # dev: "mini" with NAVSIM_CACHE = its metric cache


def spec_pdm_dets(dd, img, fld):
    """v3's crop / guard / merge machinery on the crops chosen by probing PDM-Closed (navsim_field.py): one crop per
    lateral lane column of sensitive probe cells (union of their image boxes) + one crop per supporting object."""
    from rap.mono import box_iou
    W, H = NS.W, NS.H
    keep = np.asarray(dd["cheap"]["conf"]) >= 0.25
    ch = {k_: np.asarray(dd["cheap"][k_])[keep] for k_ in ("xyxy", "conf", "coarse")}
    ch["xyxy"] = ch["xyxy"].astype(float)
    cols = {}
    for b in fld.get("field_boxes", []):
        cols.setdefault(round(0.5 * (b[0] + b[2]) / 160), []).append(b)       # group by horizontal image position
    crops = []
    for bs in cols.values():
        bs = np.asarray(bs)
        crops.append(np.array([bs[:, 0].min(), bs[:, 1].min(), bs[:, 2].max(), bs[:, 3].max()]))
    for b in fld.get("support_boxes", []):
        bc = NS.VS.box_crop(np.asarray(b, float), W, H)
        if not NS.VS.covered(bc, crops):
            crops.append(bc)
    full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
    grown = []
    if crops:
        gdet = NS.VS.guarded(NS.real_crop_detector(img), W, H)
        for c_ in crops:
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
    out = NS.VS.centres_in(ch["xyxy"], crops) if crops else np.zeros(len(ch["xyxy"]), bool)
    if grown and len(full["conf"]) and len(ch["xyxy"]):
        out |= box_iou(ch["xyxy"], full["xyxy"]).max(1) >= 0.5
    det = {k_: np.concatenate([ch[k_][~out], full[k_]]) for k_ in ("xyxy", "conf", "coarse")}
    area = sum((c_[2] - c_[0]) * (c_[3] - c_[1]) for c_ in crops + grown) / (W * H)
    return det, area


def navtest():
    """(tokens, logs): navtest from its scene filter, or every token of a metric cache (dev split)."""
    if os.environ.get("NAVSIM_CACHE"):
        ps = list(Path(os.environ["NAVSIM_CACHE"]).glob("*/*/*/metric_cache.pkl"))
        return {p.parent.name for p in ps}, sorted({p.parts[-4] for p in ps})
    f = yaml.safe_load(FILT.read_text())
    return set(f["tokens"]), sorted(set(f["log_names"]))


def ground_z(logs):
    zs = []
    for log in logs:
        for fr in pickle.load(open(DATA / "navsim_logs" / SPLIT / f"{log}.pkl", "rb")):
            b, n = np.asarray(fr["anns"]["gt_boxes"], float), np.asarray(fr["anns"]["gt_names"])
            if len(b):
                k = (n == "vehicle") & (np.hypot(b[:, 0], b[:, 1]) < 30)
                zs += list(b[k, 2] - b[k, 5] / 2)
    return float(np.median(zs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max_tokens", type=int, default=0)
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))
    tokens, logs = navtest()
    if os.environ.get("NAVSIM_ONLY_LOG"):                 # smoke: one log
        logs = [os.environ["NAVSIM_ONLY_LOG"]]
    OUT.mkdir(parents=True, exist_ok=True)
    gz_f = OUT / "ground_z.txt"
    if not gz_f.exists() and (Path(os.environ.get("NAVSIM_SRC_DETS", OUT)) / "ground_z.txt").exists():
        gz_f.write_text((Path(os.environ["NAVSIM_SRC_DETS"]) / "ground_z.txt").read_text())
    if not gz_f.exists():
        gz_f.write_text(f"{ground_z(navtest()[1]):.4f}\n")
    gz = float(gz_f.read_text())
    print(f"ground z (annotation frame) {gz:.3f} m; {len(tokens)} navtest tokens in {len(logs)} logs", flush=True)
    det = NS._detector()
    redo = [b for b in os.environ.get("NAVSIM_REDO", "").split(",") if b]   # recompute only these branches, keep the rest
    SRC = Path(os.environ.get("NAVSIM_SRC_DETS", OUT))
    n = 0
    for li, log in enumerate(logs):
        if li % sn != si or (OUT / f"{log}.pkl").exists():
            continue
        out = {}
        fp_ = Path(os.environ.get("NAVSIM_FIELD", "/nonexistent")) / f"{log}.pkl"
        fields = pickle.load(open(fp_, "rb")) if fp_.exists() else {}
        for fr in pickle.load(open(DATA / "navsim_logs" / SPLIT / f"{log}.pkl", "rb")):
            if fr["token"] not in tokens:
                continue
            c = fr["cams"]["CAM_F0"]
            img = cv2.imread(str(DATA / "sensor_blobs" / SPLIT / c["data_path"]))
            if img is None:
                out[fr["token"]] = {"missing_image": c["data_path"]}
                continue
            assert img.shape[:2] == (NS.H, NS.W), img.shape
            cam = NavCam(c)
            dd = {"cheap": det.detect(img, MODES["ns_cheap_320"])[0], "full": det.detect(img, MODES["ns_full_640"])[0]}
            v = float(np.hypot(*np.asarray(fr["ego_dynamic_state"], float)[:2]))
            ego = SimpleNamespace(dynamic_car_state=SimpleNamespace(speed=v))
            k = lambda d: np.asarray(d["conf"]) >= 0.25                          # noqa: E731
            if "spec_pdm" in redo and fr["token"] not in fields:   # probing field computed on a subset (dev: every 2nd)
                continue
            if redo:                                      # keep the previous run's other branches as they are
                prev = pickle.load(open(SRC / f"{log}.pkl", "rb"))[fr["token"]]
                rec = dict(prev)
                if "spec" in redo or "uncertainty" in redo:
                    dd["spec"], info = NS.speculative_dets(dd, cam, ego, gz, img)
                    rec["info"] = info
                for b in redo:
                    if b == "spec":
                        rec[b] = dd["spec"]
                    elif b == "spec_pdm":                     # crops from probing PDM-Closed (navsim_field.py)
                        rec[b], rec["px_spec_pdm"] = spec_pdm_dets(dd, img, fields.get(fr["token"], {}))
                    elif b.startswith("spec_"):               # dev variants: spec_[s075_][k025] = crop scale 0.75,
                        kc, sc0 = None, NS.CROP_SCALE            # keep cheap boxes with conf >= 0.25 not re-detected
                        for part in b.split("_")[1:]:
                            if part.startswith("k"):
                                kc = int(part[1:]) / 100
                            elif part.startswith("s"):
                                NS.CROP_SCALE = int(part[1:]) / 100
                        rec[b] = NS.speculative_dets(dd, cam, ego, gz, img, keep_conf=kc)[0]
                        NS.CROP_SCALE = sc0
                    elif b == "full960":
                        rec[b] = det.detect(img, FULL960)[0]
                    elif b in ("s480", "s800"):               # AdaScale's other scales (nuScenes adascale.mode_for)
                        sc_ = int(b[1:])
                        rec[b] = det.detect(img, Mode(f"ns_{b}", int(round(sc_ * 9 / 16 / 32)) * 32, sc_))[0]
                    else:
                        rec[b] = NS.extra_dets(b, img, dd["cheap"], rec["info"])
                    rec[b] = {kk: np.asarray(rec[b][kk])[k(rec[b])] for kk in ("xyxy", "conf", "coarse")}
                out[fr["token"]] = rec
            else:
                dd["spec"], info = NS.speculative_dets(dd, cam, ego, gz, img)
                for m in ("uncertainty", "sahi", "sahi_s033", "fovea"):
                    dd[m] = NS.extra_dets(m, img, dd["cheap"], info)
                out[fr["token"]] = {b: {kk: np.asarray(dd[b][kk])[k(dd[b])] for kk in ("xyxy", "conf", "coarse")} for b in BRANCHES}
                out[fr["token"]].update({"cam": (cam.K, cam.D, cam.R, cam.t), "info": info})
            n += 1
            if args.max_tokens and n >= args.max_tokens:
                break
        with open(OUT / f"{log}.pkl", "wb") as fh:
            pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [{li + 1}/{len(logs)}] {log}: {len(out)} tokens", flush=True)
        if args.max_tokens and n >= args.max_tokens:
            break
    print("NAVSIM DETECT DONE")


if __name__ == "__main__":
    main()
