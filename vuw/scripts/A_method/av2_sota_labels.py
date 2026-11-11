#!/usr/bin/env python
"""AV2 port of the frame-level SOTA baselines' data (AdaScale, Budget-Adaptive Routing): one GPU pass per split.

Splits (the AV2 dev/test rule): "train" / "val" = AV2 sensor TRAIN logs split by log (md5 of the log id, first hex digit
0-2 -> val, ~19%), every 2nd frame of DEEP's 5 Hz AV2 frames; "test" = AV2 sensor VAL (all frames, the same frames and
cheap costs as av2_eval_v4.py --split val). Per frame, exactly av2_eval_v4.py's conventions (front-centre camera, cheap
320 draft, mono lift with the arm's own history, braking, DEEP's cost, cheap action history):
  J_c, J_u{480,640,800,960}      whole frame at long side s (each scale with its own lift history)
  loss_{320,...,960}            AdaScale's detection loss 1 - F1@IoU0.5 against the projected GT (conf >= 0.25)
  boxes                         GT / weak (cheap) / strong (960) boxes for bgt-ada's ORIC and detection features
Writes results/av2_sota/{frames,boxes}__<split>.pkl.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rap import planner as P
from rap.detect import Mode, TwoFidelityDetector
from rap.mono import box_iou, predicted_geometry
from vtf import av2 as AV

B = Path(os.environ["B"])
OUT = B / "results" / "av2_sota"
THR = 0.25
SCALES = (320, 480, 640, 800, 960)


def mode_for(s):
    """long side s on AV2's portrait front camera; the 320 / 640 / 960 modes are av2_eval_v4's (w = 0.8 s)."""
    return Mode({320: "av2_c320", 640: "av2_f640", 960: "av2_f960"}.get(s, f"av2_s{s}"), s, int(round(0.8 * s / 32)) * 32)


def train_logs():
    return sorted(p.stem for p in (B / "datasets" / "av2" / "lists" / "train").glob("*.done"))


def split_of(log):
    return "val" if hashlib.md5(log.encode()).hexdigest()[0] in "012" else "train"


def logs_of(split):
    if split == "test":
        root = B / "datasets" / "av2" / "sensor" / "val"
        return root, sorted(p.name for p in root.iterdir() if (p / "annotations.feather").exists())
    return B / "datasets" / "av2" / "sensor" / "train", [l for l in train_logs() if split_of(l) == split]


def f1_loss(xyxy, gtb):
    if not len(gtb) and not len(xyxy):
        return 0.0
    if not len(gtb) or not len(xyxy):
        return 1.0
    tp = int((box_iou(xyxy, gtb).max(0) >= 0.5).sum())
    prec, rec = tp / len(xyxy), tp / len(gtb)
    return 1.0 - (2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["test", "val", "train"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_logs", type=int, default=0)
    args = ap.parse_args()
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device=args.device, backend="torch")
    pp, cp = P.PlannerParams(), P.CostParams()
    OUT.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        if (OUT / f"frames__{split}.pkl").exists():
            print(f"{split}: exists, skipped", flush=True)
            continue
        root, logs = logs_of(split)
        logs = logs[: args.max_logs or None]
        step = 1 if split == "test" else 2
        rows, boxes = [], {}
        for li, log in enumerate(logs):
            L = AV.AV2Log(root, log)
            geom, speeds = L.geometry(), L.speeds()
            R, t = L.transform("ego")
            prev = {s: None for s in SCALES}
            prev_act = None

            def lift(dd, pv):
                g = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, L.calib, L.cam_h, L.dt, cam_to_ego=(R, t))
                return tuple(np.asarray(g[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def keep(d):
                k = d["conf"] >= THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}

            for fr in range(len(L.frame_ts)):
                img = cv2.imread(str(L.image_path(fr)))
                if img is None:
                    continue
                # every frame runs the cheap / per-scale histories; only every `step`-th frame is a sample
                v = float(speeds[fr])
                g = geom[geom["frame"] == fr] if len(geom) else geom
                g = g[(g["y2"] - g["y1"]) >= 10] if len(g) else g
                a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
                act = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)            # noqa: E731
                cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]                  # noqa: E731
                gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
                raw, rec = {}, {"log": log, "frame": fr, "path": str(L.image_path(fr)), "v_ego": v}
                for s in SCALES:
                    raw[s] = det.detect(img, mode_for(s))[0]
                    dd = keep(raw[s])
                    a = act(lift(dd, prev[s]))
                    rec[f"loss_{s}"] = f1_loss(dd["xyxy"], gtb)
                    rec["J_c" if s == 320 else f"J_u{s}"] = cost(a)
                    if s == 320:
                        a_c = a
                    prev[s] = dd
                rec["a_c"] = float(a_c)
                if fr % step == 0:
                    rows.append(rec)
                    boxes[(log, fr)] = {"gt_xyxy": gtb, "gt_cls": np.asarray(g["coarse"]).astype("U8") if len(g) else np.zeros(0, "U8"),
                                        **{f"{tag}_{k}": np.asarray(raw[s][k]).astype(np.float64 if k != "coarse" else "U8")
                                           for tag, s in (("weak", 320), ("strong", 960)) for k in ("xyxy", "conf")},
                                        "weak_cls": np.asarray(raw[320]["coarse"]).astype("U8"),
                                        "strong_cls": np.asarray(raw[960]["coarse"]).astype("U8")}
                prev_act = a_c
            print(f"  {split} [{li + 1}/{len(logs)}] {log}", flush=True)
        d = pd.DataFrame(rows)
        if split == "test" and not args.max_logs:
            ref = pd.read_pickle(B / "results" / "av2" / "eval__val_v4_final.pkl")
            m = d.merge(ref[["log", "frame", "J_c", "J_u960"]], on=["log", "frame"], suffixes=("", "_ref"))
            assert len(m) == len(d) and np.allclose(m.J_c, m.J_c_ref), "cheap cost differs from av2_eval_v4"
            print(f"  test: J_u960 equal to av2_eval_v4 on {np.mean(np.isclose(m.J_u960, m.J_u960_ref)):.4f} of frames")
        d.to_pickle(OUT / f"frames__{split}.pkl")
        with open(OUT / f"boxes__{split}.pkl", "wb") as fh:
            pickle.dump(boxes, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{split}: {len(d)} frames from {len(logs)} logs; optimal-scale shares "
              f"{pd.Series(np.argmin(d[[f'loss_{s}' for s in SCALES]].to_numpy() + 1e-6 * np.arange(5), 1)).map(dict(enumerate(SCALES))).value_counts(normalize=True).round(3).to_dict()}",
              flush=True)
    print("AV2 SOTA LABELS DONE")


if __name__ == "__main__":
    main()
