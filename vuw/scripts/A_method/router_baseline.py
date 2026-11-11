#!/usr/bin/env python
"""Baseline: learned frame routers (Budget-Adaptive Routing, Geng et al.; official bgt-ada @ 6669ab0) on nuScenes new.

Weak = the cheap YOLOv8s-320 draft (cached, runs on every frame); strong = YOLOv8s on the whole frame at 576x960
(the u960 arm of the confirmatory run).  A routed frame's decision uses the strong detections, lifted with a 960
history (favourable to the router: in deployment the previous frame would often be cheap).  Action history = the cheap
actions (DEEP's convention, as for every other arm).

Estimators, all fit on the new TRAIN scenes only (val scenes pick the arbiter; test scenes are scored once):
  skip_offbin   bgt-ada's headline skipping estimator: MobileNetV2-Lite (128 px, truncated), OffloadBin = 1[ORIC > 0]
                with dataset-wide ORIC (official `compute_dataset_wide_oric`, context_size 0, dets conf >= 0.3,
                all-point AP), focal loss, 100 epochs, lr 5e-4, batch 512 (their `approaches.py`)
  cond_moric    bgt-ada's conditioned estimator: XGBoost (100 trees, depth 5, lr 0.1) on MORIC (train-ECDF of ORIC)
                from their 21 detection statistics of the weak detections
  skip_V        the same MobileNetV2-Lite / focal on 1[V > 0], V = J_cheap - J_u960 (the decision-value target)
  cond_V        the same XGBoost on V (regression) from the same features + ego speed
  adaptive      Geng's budget-adaptive arbiter: per budget, skip_offbin or cond_moric, whichever is better on val
Budget: online ECDF thresholding against the estimator's train predictions (their `online_ecdf_calibrated`), target
offload ratios rho in RHOS, plus the rho that matches v3's network pixels.  Also random and oracle (true V, top-k).

Stages (each its own process):  --stage labels (GPU)  |  oric  |  train (GPU)  |  score
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
GENG = B / "third_party" / "bgt-ada"
OUT = B / "results" / "router"
SPLITS = ("train", "val", "test")
THR, G_CONF = 0.25, 0.30
RHOS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
PX_CHEAP = 320 * 192 / (1600 * 900)            # letterboxed 320 draft
PX_STRONG = 576 * 960 / (1600 * 900)
EST = ("skip_offbin", "cond_moric", "skip_V", "cond_V")
SEED = int(os.environ.get("SEED", 42))                 # bgt-ada's seed 42 = the main run; others tagged _s<seed>
TAG = "" if SEED == 42 else f"_s{SEED}"


def scenes_of(split):
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    return sorted(sp[(sp["set"] == "new") & (sp.split == split)].seq)


# ------------------------------------------------------------------------------------------------ labels (GPU)

def stage_labels(args):
    import cv2
    import torch
    from rap import planner as P
    from rap.cache import Camera, DetCache
    from rap.decision import _apply_range_source
    from rap.detect import Mode, TwoFidelityDetector
    from rap.geometry import coarse_classes
    from rap.mono import predicted_geometry
    from rap.nusc import NuScenesDB, make_adapter
    from rap.paths import CACHE, NUSCENES_TRAINVAL
    torch.set_num_threads(args.threads)
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device=args.device, backend="torch")
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    pp, cp = P.PlannerParams(), P.CostParams()
    OUT.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        if (OUT / f"frames__{split}.pkl").exists():
            print(f"{split}: exists, skipped", flush=True)
            continue
        rng = np.random.default_rng(0)
        rows, boxes = [], {}
        for s in scenes_of(split):
            geom, speeds = ad.geometry(s), ad.speeds(s)
            cam = Camera("nuScenes", s)
            R, t = cam.transform("ego")
            c = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
            toks = db.samples(by[s])
            prev_act, prev960 = None, None
            for i, fr in enumerate(c.frames):
                fr = int(fr)
                v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
                g = geom[geom["frame"] == fr]
                g = g[(g["y2"] - g["y1"]) >= 10]
                a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
                cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]              # noqa: E731
                decide = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)    # noqa: E731
                dc = c.det(i)
                kc = dc["conf"] >= THR
                geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
                cons_c = tuple(np.asarray(a, float) for a in
                               _apply_range_source(dc["xyxy"][kc].astype(float), geo_c, g, "mono", rng, 0.0))
                a_c = decide(cons_c)
                path = str(db.image_path(toks[fr]))
                d9, _ = det.detect(cv2.imread(path), Mode("f576x960", 576, 960))
                k9 = d9["conf"] >= THR
                dd9 = {"xyxy": d9["xyxy"][k9].astype(float), "coarse": d9["coarse"][k9]}
                g9 = predicted_geometry(dd9, prev960, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                prev960 = dd9
                rows.append({"seq": s, "frame": fr, "path": path, "v_ego": v, "a_c": float(a_c), "J_c": cost(a_c),
                             "J_u960": cost(decide(tuple(np.asarray(g9[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))))})
                boxes[(s, fr)] = {"gt_xyxy": np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(np.float64) if len(g)
                                  else np.zeros((0, 4)), "gt_cls": coarse_classes(g) if len(g) else np.zeros(0, "U8"),
                                  "weak_xyxy": dc["xyxy"].astype(np.float64), "weak_conf": dc["conf"].astype(np.float64),
                                  "weak_cls": dc["coarse"].astype("U8"), "strong_xyxy": d9["xyxy"].astype(np.float64),
                                  "strong_conf": d9["conf"].astype(np.float64), "strong_cls": d9["coarse"].astype("U8")}
                prev_act = a_c
            print(f"  {split} {s}", flush=True)
        d = pd.DataFrame(rows)
        if split == "test":             # the same numbers as the confirmatory run
            ref = pd.read_pickle(B / "results" / "nusc_full" / "eval__new_test_v3_confirm.pkl")
            m = d.merge(ref[["seq", "frame", "J_c", "J_u960"]], on=["seq", "frame"], suffixes=("", "_ref"))
            assert len(m) == len(d) and np.allclose(m.J_c, m.J_c_ref), "cheap cost differs from the confirmatory run"
            print(f"  test: J_u960 equal to confirmatory on {np.mean(np.isclose(m.J_u960, m.J_u960_ref)):.4f} of frames")
        d.to_pickle(OUT / f"frames__{split}.pkl")
        with open(OUT / f"boxes__{split}.pkl", "wb") as fh:
            pickle.dump(boxes, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{split}: {len(d)} frames", flush=True)


# ------------------------------------------------------------------------------------------------ ORIC labels

def _ltwh(b):
    return [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]


def geng_raw(boxes, keys):
    """bgt-ada's per-frame dicts (as in DEEP's scripts/148): (left, top, w, h, conf, class), conf >= 0.3."""
    raw = []
    for seq, fr in keys:
        r = boxes[(seq, fr)]
        det = {}
        for tag in ("weak", "strong"):
            m = r[f"{tag}_conf"] >= G_CONF
            det[tag] = [tuple(_ltwh(b)) + (float(c), str(k))
                        for b, c, k in zip(r[f"{tag}_xyxy"][m], r[f"{tag}_conf"][m], r[f"{tag}_cls"][m])]
        raw.append({"frame_id": int(fr), "video_name": str(seq), "edge_dets": det["weak"], "cloud_dets": det["strong"],
                    "gt_objs": [{"bbox": _ltwh(b), "class": str(c)} for b, c in zip(r["gt_xyxy"], r["gt_cls"])]})
    return raw


def stage_oric(args):
    sys.path.insert(0, str(GENG))
    from src import proxy_metrics
    for split in SPLITS:
        d = pd.read_pickle(OUT / f"frames__{split}.pkl")
        with open(OUT / f"boxes__{split}.pkl", "rb") as fh:
            boxes = pickle.load(fh)
        keys = list(zip(d.seq, d.frame.astype(int)))
        res = proxy_metrics.compute_dataset_wide_oric(geng_raw(boxes, keys), quiet=True, context_size=args.context,
                                                      context_draws=1)
        d["oric"] = [res.get((str(s), int(f)), {}).get("oric_allpoint", 0.0) for s, f in keys]
        d.to_pickle(OUT / f"frames__{split}.pkl")
        V = d.J_c - d.J_u960
        print(f"{split}: ORIC>0 {np.mean(d.oric > 0):.3f}  ORIC<0 {np.mean(d.oric < 0):.3f}  V>0 {np.mean(V > 0):.3f}  "
              f"V<0 {np.mean(V < 0):.3f}  spearman(ORIC,V) {d.oric.corr(V, method='spearman'):+.3f}", flush=True)


# ------------------------------------------------------------------------------------------------ train (GPU)

def feats(boxes, d):
    from src.features import BASE_FEATURES, extract_detection_features
    X = []
    for s, f in zip(d.seq, d.frame.astype(int)):
        r = boxes[(s, f)]
        m = r["weak_conf"] >= THR
        dets = [tuple(_ltwh(b)) + (float(c), str(k)) for b, c, k in zip(r["weak_xyxy"][m], r["weak_conf"][m], r["weak_cls"][m])]
        e = extract_detection_features(dets)
        X.append([e[k] for k in BASE_FEATURES])
    return np.asarray(X, float)


def stage_train(args):
    sys.path.insert(0, str(GENG))
    import torch
    from src.models.classical import XGBoostEstimator
    from src.models.cnn import MobileNetV2LiteEstimator
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    D, BX = {}, {}
    for split in SPLITS:
        D[split] = pd.read_pickle(OUT / f"frames__{split}.pkl")
        with open(OUT / f"boxes__{split}.pkl", "rb") as fh:
            BX[split] = pickle.load(fh)
    tr = D["train"]
    ref = np.sort(tr.oric.to_numpy(np.float64))                    # bgt-ada prepare_transforms._fit_moric_reference
    moric = lambda o: np.searchsorted(ref, np.asarray(o, np.float64), side="right") / len(ref)   # noqa: E731  (_apply_moric)
    V = (tr.J_c - tr.J_u960).to_numpy(float)
    scores = {}
    for name in args.est:
        if name.startswith("skip"):
            y = (tr.oric.to_numpy() > 0) if name == "skip_offbin" else (V > 0)
            est = MobileNetV2LiteEstimator(image_size=128, device=args.device)
            est.fit(list(tr.path), y.astype(np.float32), epochs=100, lr=5e-4, batch_size=512, loss="focal",
                    num_workers=args.workers, compile=False)
            pred = {sp: est.predict(list(D[sp].path), batch_size=256, num_workers=args.workers) for sp in SPLITS}
            torch.save(est.model.state_dict(), OUT / f"{name}{TAG}.pt")
        else:
            X = {sp: feats(BX[sp], D[sp]) for sp in SPLITS}
            if name == "cond_V":
                X = {sp: np.column_stack([X[sp], D[sp].v_ego.to_numpy(float)]) for sp in SPLITS}
            y = moric(tr.oric) if name == "cond_moric" else V
            est = XGBoostEstimator(n_estimators=100, max_depth=5, learning_rate=0.1, task_type="regression")
            est.fit(X["train"], y)
            pred = {sp: np.asarray(est.predict(X[sp]), float) for sp in SPLITS}
        scores[name] = pred
        yt = (D["test"].J_c - D["test"].J_u960).to_numpy(float)
        print(f"{name}: test spearman(score, V) {pd.Series(pred['test']).corr(pd.Series(yt), method='spearman'):+.3f}",
              flush=True)
        with open(OUT / f"scores__{name}{TAG}.pkl", "wb") as fh:
            pickle.dump(pred, fh)


# ------------------------------------------------------------------------------------------------ score

def stage_score(args):
    sys.path.insert(0, str(GENG))
    from src.offloader import Offloader
    D = {sp: pd.read_pickle(OUT / f"frames__{sp}.pkl") for sp in ("val", "test")}
    S = {}
    for name in EST:
        p = OUT / f"scores__{name}{TAG}.pkl"
        if p.exists():
            with open(p, "rb") as fh:
                S[name] = pickle.load(fh)
    ref = pd.read_pickle(B / "results" / "nusc_full" / "eval__new_test_v3_confirm.pkl")
    te = D["test"].merge(ref[["seq", "frame", "J_u960", "J_spec", "J_spec_ref_resnet18", "px"]], on=["seq", "frame"],
                         suffixes=("_gpu", ""))
    assert len(te) == len(D["test"])        # test scored with the confirmatory u960 (equal on 99.4% of frames; FP16 GPU)
    v3_px = PX_CHEAP + float((te.px * 0.36).mean())
    rho_match = {"cond": (v3_px - PX_CHEAP) / PX_STRONG, "skip": (v3_px - PX_CHEAP) / (PX_STRONG - PX_CHEAP)}
    print(f"v3 network pixels {v3_px:.4f} of native; matched rho: conditioned {rho_match['cond']:.3f}, "
          f"skipping {rho_match['skip']:.3f}")

    def route(name, split, rho):
        pr = S[name][split]
        dec = Offloader("offload_binary").decide_calibrated(pr, rho, calibration_predictions=S[name]["train"])
        return dec.mask

    def G_of(d, mask):
        return float(((d.J_c - d.J_u960).to_numpy() * mask).sum() / d.J_c.sum())

    rng = np.random.default_rng(0)
    units = te.seq.to_numpy(); uq = np.unique(units); idx = [np.flatnonzero(units == u) for u in uq]
    boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    L = te.J_c.to_numpy()
    Vt = (te.J_c - te.J_u960).to_numpy()
    v3 = {"v3": (te.J_c - te.J_spec).to_numpy(), "v3_ref": (te.J_c - te.J_spec_ref_resnet18).to_numpy()}
    rows = []

    def add(method, rho_t, mask, px):
        r = {"method": method, "rho_target": rho_t, "rho_actual": float(mask.mean()), "px": px,
             "G": float((Vt * mask).sum() / L.sum()), "harmed": float(np.mean((Vt * mask) < 0))}
        for k, vv in v3.items():
            dr = [((vv[b] - Vt[b] * mask[b]).sum()) / L[b].sum() for b in boots]
            r[f"{k}_minus"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        rows.append(r)

    for k, vv in v3.items():
        rows.append({"method": k, "px": v3_px, "G": float(vv.sum() / L.sum()), "harmed": float(np.mean(vv < 0))})
    rows.append({"method": "u960", "rho_actual": 1.0, "px": PX_STRONG, "G": float(Vt.sum() / L.sum())})
    grid = sorted(set(RHOS) | {round(rho_match["cond"], 4), round(rho_match["skip"], 4)})
    for rho in grid:
        n = int(round(rho * len(te)))
        orc = np.zeros(len(te), bool); orc[np.argsort(-Vt, kind="stable")[:n]] = True
        add("oracle_V", rho, orc, PX_CHEAP + rho * PX_STRONG)
        rnd = np.zeros(len(te), bool); rnd[np.random.default_rng(1).permutation(len(te))[:n]] = True
        add("random", rho, rnd, PX_CHEAP + rho * PX_STRONG)
        for name in S:
            m = route(name, "test", rho)
            px = (PX_CHEAP + m.mean() * PX_STRONG) if name.startswith("cond") else ((1 - m.mean()) * PX_CHEAP + m.mean() * PX_STRONG)
            add(name, rho, m, px)
        if {"skip_offbin", "cond_moric"} <= set(S):       # Geng's arbiter: the better of the two on val at this rho
            pick = max(("skip_offbin", "cond_moric"), key=lambda nm: G_of(D["val"], route(nm, "val", rho)))
            m = route(pick, "test", rho)
            px = (PX_CHEAP + m.mean() * PX_STRONG) if pick.startswith("cond") else ((1 - m.mean()) * PX_CHEAP + m.mean() * PX_STRONG)
            add(f"adaptive({pick})", rho, m, px)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"summary_newtest{TAG}.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["labels", "oric", "train", "score"])
    ap.add_argument("--splits", nargs="*", default=["test", "val", "train"])
    ap.add_argument("--est", nargs="*", default=list(EST))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--context", type=int, default=0)
    args = ap.parse_args()
    {"labels": stage_labels, "oric": stage_oric, "train": stage_train, "score": stage_score}[args.stage](args)
    print(f"ROUTER {args.stage.upper()} DONE", flush=True)


if __name__ == "__main__":
    main()
