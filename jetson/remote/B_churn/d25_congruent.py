#!/usr/bin/env python
"""D25 at inference time: a decision-congruent detector upgrade via verification in the decision's regions.

For an upgrade old -> new (cached full-frame detections of both, D5), the NEW model drafts the scene; the OLD model
is consulted only inside the new scene's braking field footprint and support crops (v_close 10), fused as
  replace     inside the crops the old model's objects replace the new model's (the plain speculative merge)
  union       inside the crops keep both models' objects (braking is max-composable: the more cautious wins)
  congruent   a support object of the new model survives only if the old model also has it (IoU >= 0.3); in the
              field footprint, old objects the new model lacks are added (asymmetric: never lose an old object that
              could raise the action, never act on a new object the old model does not confirm)
Compared with the plain upgrade on the new test scenes: frames whose decision changes vs the old model, harmed share,
net loss change relative to the old model. The old model's action history (DEEP convention).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from rap import cache as _rc
from rap import planner as P
from rap.cache import Camera, DetCache
from rap.mono import box_iou, predicted_geometry
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL
from vtf import speculative as SP

_rc.NUSC_CACHES = type(_rc.NUSC_CACHES)(set(_rc.NUSC_CACHES) | {"nusc_models"})
B = Path(os.environ["B"])
PAIRS = [("yolov8s", "yolov8l"), ("yolov8m", "yolov8l"), ("yolov8l", "yolov8x"), ("yolov8m", "yolo11m"),
         ("yolov8s", "rtdetr-l"), ("yolo11s", "yolo11m")]
THR = {"rtdetr-l": 0.4404296875}
W, H = 1600, 900


def main():
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    scenes = sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq)
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    pp, cp = P.PlannerParams(), P.CostParams()
    rows = []
    for s in scenes:
        geom, speeds = ad.geometry(s), ad.speeds(s)
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        models = {m for pr in PAIRS for m in pr}
        caches = {m: DetCache(Path(CACHE) / "nusc_models" / m / f"{s}.npz") for m in models}
        n = min(len(c.frames) for c in caches.values())
        dets = {m: [] for m in models}
        for m in models:
            prev = None
            for i in range(n):
                d = caches[m].det(i)
                k = d["conf"] >= THR.get(m, 0.25)
                dd = {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}
                gg = predicted_geometry(dd, prev, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                dd["mono"] = tuple(np.asarray(gg[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))
                dets[m].append(dd)
                prev = dd
        for old, new in PAIRS:
            prev_act = None
            for i in range(n):
                fr = int(caches[old].frames[i])
                v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
                g = geom[geom["frame"] == fr]
                g = g[(g["y2"] - g["y1"]) >= 10]
                a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
                act = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)        # noqa: E731
                cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]              # noqa: E731
                do, dn = dets[old][i], dets[new][i]
                rec = {"seq": s, "frame": fr, "pair": f"{old}->{new}", "J_old": cost(act(do["mono"])),
                       "J_new": cost(act(dn["mono"]))}
                # crops of the new scene: field footprint + support boxes (one round, as D63 showed suffices)
                crops = []
                fc = SP.field_crop(dn["mono"], v, cam, W, H, pp, 10.0)
                if fc is not None:
                    crops.append(fc)
                sup = SP.support(dn["mono"], v, pp)
                crops += [SP.box_crop(dn["xyxy"][k], W, H) for k in sup]
                o_in = SP.centres_in(do["xyxy"], crops)
                n_in = SP.centres_in(dn["xyxy"], crops)
                cat = lambda a, ma, b, mb: tuple(np.r_[x[ma], y[mb]] for x, y in zip(a["mono"], b["mono"]))  # noqa: E731
                t_rep = cat(dn, ~n_in, do, o_in)
                t_uni = cat(dn, np.ones(len(dn["conf"]), bool), do, o_in & ~(
                    box_iou(do["xyxy"], dn["xyxy"]).max(1) >= 0.3 if len(dn["xyxy"]) and len(do["xyxy"]) else
                    np.zeros(len(do["xyxy"]), bool)))
                # congruent: drop unconfirmed new support objects; add old objects in the field footprint the new lacks
                keep_n = np.ones(len(dn["conf"]), bool)
                if len(sup):
                    conf_by_old = (box_iou(dn["xyxy"][sup], do["xyxy"]).max(1) >= 0.3) if len(do["xyxy"]) else \
                        np.zeros(len(sup), bool)
                    keep_n[sup[~conf_by_old]] = False
                in_field = SP.centres_in(do["xyxy"], [fc]) if fc is not None else np.zeros(len(do["xyxy"]), bool)
                lacks = ~(box_iou(do["xyxy"], dn["xyxy"]).max(1) >= 0.3) if len(dn["xyxy"]) and len(do["xyxy"]) else \
                    np.ones(len(do["xyxy"]), bool)
                t_con = cat(dn, keep_n, do, in_field & lacks)
                rec.update({"J_replace": cost(act(t_rep)), "J_union": cost(act(t_uni)), "J_congruent": cost(act(t_con)),
                            "px": sum((c[2] - c[0]) * (c[3] - c[1]) for c in crops) / (W * H)})
                rows.append(rec)
                prev_act = act(do["mono"])
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_pickle(B / "results" / "d25_congruent.pkl")
    out = []
    for pair, q in d.groupby("pair", sort=False):
        L = q.J_old.sum()
        for arm in ("new", "replace", "union", "congruent"):
            V = (q.J_old - q[f"J_{arm}"]).to_numpy()
            aff = np.abs(V) > 1e-9
            out.append({"pair": pair, "arm": "plain upgrade" if arm == "new" else arm, "changed": aff.mean(),
                        "harmed": float((V < -1e-9).sum() / max(aff.sum(), 1)), "net_loss_change": -V.sum() / L,
                        "congruent_bound": -np.clip((q.J_old - q.J_new).to_numpy(), 0, None).sum() / L})
    res = pd.DataFrame(out)
    res.to_csv(B / "results" / "d25_congruent.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
