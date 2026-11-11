#!/usr/bin/env python
"""D68 evaluation: the braking decision cost of the CHEAP mode (192x320) for each fine-tuned arm, new test scenes.

Per model: YOLOv8s weights -> DEEP's detector at ns_cheap_320 -> DEEP's monocular lift with the model's own previous
frame -> braking action with the model's own action history -> DEEP's decision cost against the ground-truth action.
Reports total cost relative to the COCO cheap model (the reference, whose costs must equal DEEP's table), recall of
decision-relevant ground-truth objects (in corridor, <= 50 m), recall of all objects, false positives per frame, and
paired scene-bootstrap intervals of each arm's cost difference with 'plain' and with COCO.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from rap import planner as P
from rap.cache import Camera
from rap.decision import _apply_range_source
from rap.detect import MODES, TwoFidelityDetector
from rap.mono import box_iou, predicted_geometry
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import NUSCENES_TRAINVAL

B = Path(os.environ["B"])
THR = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path/to/weights.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--night", action="store_true", help="only the night scenes of the new test split")
    args = ap.parse_args()
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    scenes = sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq)[: args.max_scenes or None]
    if args.night:
        import json
        sc = json.load(open(Path(NUSCENES_TRAINVAL) / "v1.0-trainval" / "scene.json"))
        night = {x["name"] for x in sc if "night" in x["description"].lower()}
        scenes = [x for x in scenes if x in night]
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    pp, cp = P.PlannerParams(), P.CostParams()
    models = {n: TwoFidelityDetector(p, device=args.device, backend="torch")
              for n, p in (m.split("=", 1) for m in args.models)}
    mode = MODES["ns_cheap_320"]
    rng = np.random.default_rng(0)
    rows = []
    for s in scenes:
        g_all, speeds = ad.geometry(s), ad.speeds(s)
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        toks = db.samples(by[s])
        state = {n: {"prev_det": None, "prev_act": None, "prev_act_o": None} for n in models}
        for fr, tk in enumerate(toks):
            img = cv2.imread(str(db.image_path(tk)))
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = g_all[g_all["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
            rel = (P.in_corridor(g["lat_min"], g["lat_max"], pp.corridor_half_w) & (g["long_near"] <= 50)) if len(g) \
                else np.zeros(0, bool)
            for n, det in models.items():
                d, _ = det.detect(img, mode)
                k = d["conf"] >= THR
                dd = {"xyxy": d["xyxy"][k].astype(float), "coarse": d["coarse"][k]}
                geo = predicted_geometry(dd, state[n]["prev_det"], cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                cons = tuple(np.asarray(a, float) for a in _apply_range_source(dd["xyxy"], geo, g, "mono", rng, 0.0))
                a = P.discrete_action(P.required_decel(*cons, v, pp)[0], pp)
                J = P.decision_cost(a, a_gt, state[n]["prev_act"], pp, cp)["J"]
                # oracle geometry: matched detections take the ground-truth state (existence errors only)
                cons_o = tuple(np.asarray(x, float) for x in _apply_range_source(dd["xyxy"], geo, g, "gt", rng, 0.0))
                a_o = P.discrete_action(P.required_decel(*cons_o, v, pp)[0], pp)
                J_o = P.decision_cost(a_o, a_gt, state[n]["prev_act_o"], pp, cp)["J"]
                iou = box_iou(gtb, dd["xyxy"]) if len(gtb) and len(dd["xyxy"]) else np.zeros((len(gtb), len(dd["xyxy"])))
                hit = iou.max(1) >= 0.5 if iou.size else np.zeros(len(gtb), bool)
                fp = int((iou.max(0) < 0.3).sum()) if iou.size else len(dd["xyxy"])
                rows.append({"seq": s, "frame": fr, "model": n, "a": a, "a_gt": a_gt, "J": J, "J_oracle": J_o, "n_gt": len(gtb),
                             "hit": int(hit.sum()), "n_rel": int(rel.sum()), "hit_rel": int((hit & rel).sum()),
                             "fp": fp})
                state[n] = {"prev_det": dd, "prev_act": a, "prev_act_o": a_o}
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_pickle(B / "results" / f"d68_eval{args.tag}.pkl")
    names = list(models)
    res_all = []
    for col in ("J", "J_oracle"):
        res_all.append(summarise(d, col, names, args))
    res = pd.concat(res_all)
    res.to_csv(B / "results" / f"d68_eval{args.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False))


def summarise(d, col, names, args):
    w = d.pivot_table(index=["seq", "frame"], columns="model", values=col).reset_index()
    ref = "coco" if "coco" in names else names[0]
    uq = w.seq.unique()
    idx = [np.flatnonzero(w.seq.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    out = []
    for n in names:
        gsum = d[d.model == n]
        r = {"geometry": "mono" if col == "J" else "oracle", "model": n, "J_total": float(gsum[col].sum()),
             "J_rel_coco": float(gsum[col].sum() / w[ref].sum()),
             "recall_all": gsum.hit.sum() / max(gsum.n_gt.sum(), 1), "recall_decision_relevant":
             gsum.hit_rel.sum() / max(gsum.n_rel.sum(), 1), "fp_per_frame": gsum.fp.mean(),
             "action_agree_gt": float((gsum.a == gsum.a_gt).mean())}
        for base in (ref, "plain"):
            if base in names and base != n:
                dr = []
                for _ in range(2000):
                    tt = np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))])
                    dr.append((w[n].to_numpy()[tt].sum() - w[base].to_numpy()[tt].sum()) / w[ref].to_numpy()[tt].sum())
                r[f"dJ_vs_{base}"] = float((w[n].sum() - w[base].sum()) / w[ref].sum())
                r[f"dJ_vs_{base}_ci"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        out.append(r)
    print(f"{col}: scenes {len(uq)}, frames {len(w)}")
    return pd.DataFrame(out)


if __name__ == "__main__":
    main()
