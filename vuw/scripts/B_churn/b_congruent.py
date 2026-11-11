#!/usr/bin/env python
"""Direction B: decision-congruent fine-tuning of an upgraded detector (focal-distillation-style frame weighting).

Stages:
  detect  --weights W --name N --split S : run a detector at 384x640 on the new-scene split S (train/test), cache in
          data/cache/nusc_b/<N>/<scene>.npz
  flips   : braking decisions of old (b_old) and new (b_new_plain) on the TRAIN scenes vs ground truth; frames where the
          new model's decision cost exceeds the old model's = negative flips; writes data/b_congruent/negflips.csv
  build   : dataset d68/plain images + labels, negative-flip frames repeated 4x -> data/b_cong/congruent
  eval    : churn old -> new_plain and old -> new_congruent on the TEST scenes (same metrics as D5), plus recall
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rap import cache as _rc
from rap import cache
from rap import planner as P
from rap.cache import Camera, DetCache
from rap.detect import Mode, TwoFidelityDetector
from rap.mono import box_iou
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL

_rc.NUSC_CACHES = type(_rc.NUSC_CACHES)(set(_rc.NUSC_CACHES) | {"nusc_b"})
B = Path(os.environ["B"])
OUTC = Path(CACHE) / "nusc_b"


def scenes_of(split):
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    return sorted(sp[(sp["set"] == "new") & (sp.split == split)].seq)


def stage_detect(args, db):
    from rap import mono
    det = TwoFidelityDetector(args.weights, device=args.device, backend="torch")
    by = {db.scene_name(t): t for t in db.scenes}
    out = OUTC / args.name
    for s in scenes_of(args.split):
        if (out / f"{s}.npz").exists():
            continue
        toks = db.samples(by[s])
        calib = db.calib(toks[0])
        dets, geos, prev, scal = [], [], None, {}
        for tk in toks:
            d, _ = det.detect(cv2.imread(str(db.image_path(tk))), Mode("m640", 384, 640))
            geos.append(mono.predicted_geometry(d, prev, calib, 1.51, 0.5))
            dets.append(d)
            for k in ("n_cand", "n_cand_raw", "n_post"):
                scal.setdefault(k, []).append(d[k])
            prev = d
        cache.save(out / f"{s}.npz", list(range(len(toks))), dets, geos, scal)
    print(f"detect {args.name} {args.split} done", flush=True)


def decisions(db, ad, names, split):
    """Per frame and model: braking action and cost with the OLD model's history, and GT hits."""
    from rap.mono import predicted_geometry
    pp, cp = P.PlannerParams(), P.CostParams()
    rows = []
    for s in scenes_of(split):
        geom, speeds = ad.geometry(s), ad.speeds(s)
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        cs = {n: DetCache(OUTC / n / f"{s}.npz") for n in names}
        prev = {n: None for n in names}
        prev_act = None
        for i in range(min(len(c.frames) for c in cs.values())):
            fr = int(cs[names[0]].frames[i])
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
            rec = {"seq": s, "frame": fr, "n_gt": len(gtb)}
            acts = {}
            for n in names:
                d = cs[n].det(i)
                k = d["conf"] >= 0.25
                dd = {"xyxy": d["xyxy"][k].astype(float), "coarse": d["coarse"][k]}
                gg = predicted_geometry(dd, prev[n], cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                prev[n] = dd
                acts[n] = P.discrete_action(P.required_decel(*(np.asarray(gg[x], float) for x in ("z", "lat_min", "lat_max", "ttc")),
                                                              v, pp)[0], pp)
                rec[f"J_{n}"] = P.decision_cost(acts[n], a_gt, prev_act, pp, cp)["J"]
                iou = box_iou(gtb, dd["xyxy"]) if len(gtb) and len(dd["xyxy"]) else np.zeros((len(gtb), len(dd["xyxy"])))
                rec[f"hit_{n}"] = int((iou.max(1) >= 0.5).sum()) if iou.size else 0
            rows.append(rec)
            prev_act = acts[names[0]]
    return pd.DataFrame(rows)


def stage_flips(args, db):
    ad = make_adapter(db)
    d = decisions(db, ad, ["b_old", "b_new_plain"], "train")
    neg = d[d.J_b_new_plain > d.J_b_old + 1e-9]
    (B / "data" / "b_cong").mkdir(parents=True, exist_ok=True)
    neg[["seq", "frame"]].to_csv(B / "data" / "b_cong" / "negflips.csv", index=False)
    print(f"train frames {len(d)}, negative flips {len(neg)}, positive {int((d.J_b_new_plain < d.J_b_old - 1e-9).sum())}")


def stage_build(args, db):
    src = B / "data" / "d68" / "plain"
    dst = B / "data" / "b_cong" / "congruent"
    neg = pd.read_csv(B / "data" / "b_cong" / "negflips.csv")
    negset = {f"{s}_{int(f):03d}" for s, f in zip(neg.seq, neg.frame)}
    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("train", "val"):
        for lab in (src / "labels" / sub).glob("*.txt"):
            reps = 4 if (sub == "train" and lab.stem in negset) else 1
            img = (src / "images" / sub / f"{lab.stem}.jpg").resolve()
            for r in range(reps):
                nm = f"{lab.stem}_r{r}"
                if not (dst / "images" / sub / f"{nm}.jpg").exists():
                    (dst / "images" / sub / f"{nm}.jpg").symlink_to(img)
                shutil.copy(lab, dst / "labels" / sub / f"{nm}.txt")
    (dst / "data.yaml").write_text((src / "data.yaml").read_text().replace(str(src), str(dst)))
    print(f"built congruent set; negative-flip frames repeated: {len(negset)}")


def stage_eval(args, db):
    ad = make_adapter(db)
    d = decisions(db, ad, ["b_old", "b_new_plain", "b_new_cong"], "test")
    d.to_pickle(B / "results" / "b_congruent_eval.pkl")
    L = d.J_b_old.sum()
    uq = d.seq.unique(); idx = [np.flatnonzero(d.seq.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    out = []
    for n in ("b_new_plain", "b_new_cong"):
        V = (d.J_b_old - d[f"J_{n}"]).to_numpy()
        aff = np.abs(V) > 1e-9
        neg = (V < -1e-9)
        r = {"new": n, "recall_old": d.hit_b_old.sum() / d.n_gt.sum(), "recall_new": d[f"hit_{n}"].sum() / d.n_gt.sum(),
             "changed": aff.mean(), "harmed_share": neg.sum() / max(aff.sum(), 1), "negative_flip_rate": neg.mean(),
             "net_loss_change": -V.sum() / L, "congruent_bound": -np.clip(V, 0, None).sum() / L}
        out.append(r)
    Vp = (d.J_b_old - d.J_b_new_plain).to_numpy(); Vc = (d.J_b_old - d.J_b_new_cong).to_numpy()
    Lo = d.J_b_old.to_numpy()
    dr = []
    for _ in range(2000):
        tt = np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))])
        dr.append(((Vc[tt] < -1e-9).mean() - (Vp[tt] < -1e-9).mean(), (Vc[tt].sum() - Vp[tt].sum()) / Lo[tt].sum()))
    dr = np.array(dr)
    res = pd.DataFrame(out)
    res.to_csv(B / "results" / "b_congruent.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    print(f"congruent - plain: negative-flip rate {np.percentile(dr[:, 0], [2.5, 97.5]).round(4)}, "
          f"net value {np.percentile(dr[:, 1], [2.5, 97.5]).round(4)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["detect", "flips", "build", "eval"])
    ap.add_argument("--weights")
    ap.add_argument("--name")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    {"detect": stage_detect, "flips": stage_flips, "build": stage_build, "eval": stage_eval}[args.stage](args, db)


if __name__ == "__main__":
    main()
