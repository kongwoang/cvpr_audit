#!/usr/bin/env python
"""NAVSIM ports of the frame-level SOTA baselines: AdaScale (Chin et al.) and Budget-Adaptive Routing (Geng et al.,
official bgt-ada), with the nuScenes recipes (scripts/A_method/adascale.py, router_baseline.py) unchanged where possible.

Data rule: everything is fit on the NAVSIM DEV split (OpenScene mini = the dev tokens of metric_cache_dev), split by
log (md5 of the log name, first hex digit 0-2 -> "val", else "train"); navtest ("test") is only scored.
PDMS is per token (NAVSIM scores every token independently and PDM-Closed has no perception history), so a frame-level
method's navtest PDMS = the mean over tokens of the PDMS of the branch it picks for that token:
  AdaScale  ResNet-18 on the 320x180 CAM_F0 image -> long side in {320, 480, 640, 800, 960} (labels: argmin of
            1 - F1@IoU0.5 of YOLOv8s at that scale against the projected eligible GT, conf >= 0.25)
            -> branch cheap / s480 / full / s800 / full960
  routers   weak = cheap, strong = full960; estimators skip_offbin / cond_moric (ORIC from bgt-ada's
            compute_dataset_wide_oric) and skip_V / cond_V (V = PDMS(full960) - PDMS(cheap) of the dev tokens);
            Geng's adaptive arbiter picks per budget on dev-val; budgets RHOS + the rho matching our NAVSIM
            configuration's network pixels.
Stages: labels (CPU) | oric (CPU) | train (GPU: AdaScale net + MobileNetV2-Lite estimators, predictions on all splits)
| score (CPU).  Env: env_navsim.sh. Inputs: NAVSIM_DETS_DEV (dev dets with cheap, full, full960, s480, s800),
NAVSIM_DETS_TEST (navtest dets with the same branches), pdms_states_dev.csv, pdms_states_final.csv, pdms_states_sota.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
DATA = Path(os.environ["OPENSCENE_DATA_ROOT"])
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
OUT = EXP / "sota"
GENG = B / "third_party" / "bgt-ada"
DETS = {"dev": Path(os.environ.get("NAVSIM_DETS_DEV", EXP / "dets_dev_sota")),
        "test": Path(os.environ.get("NAVSIM_DETS_TEST", EXP / "dets_sota"))}
SPLITS = ("train", "val", "test")
SCALES = (320, 480, 640, 800, 960)
BR = {320: "cheap", 480: "s480", 640: "full", 800: "s800", 960: "full960"}
W0, H0 = 1920, 1080
PX = {s: (int(round(s * 9 / 16 / 32)) * 32) * s / (W0 * H0) for s in SCALES}
PX[320], PX[640], PX[960] = 192 * 320 / (W0 * H0), 384 * 640 / (W0 * H0), 576 * 960 / (W0 * H0)
RHOS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
EST = ("skip_offbin", "cond_moric", "skip_V", "cond_V")
THR, G_CONF = 0.25, 0.30
NAME2COARSE = {"vehicle": "vehicle", "pedestrian": "person", "bicycle": "cyclist"}
SEED = int(os.environ.get("SEED", 42))
TAG = "" if SEED == 42 else f"_s{SEED}"
sys.path.insert(0, str(B / "scripts" / "A_method"))


def save_df(d, p):
    """pandas-version-independent (labels run in the nuPlan env, the other stages in the main venv)."""
    with open(p, "wb") as fh:
        pickle.dump({c: d[c].tolist() for c in d.columns}, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_df(p):
    return pd.DataFrame(pickle.load(open(p, "rb")))


def split_of(log):
    return "val" if hashlib.md5(log.encode()).hexdigest()[0] in "012" else "train"


def f1_loss(xyxy, gtb):
    from rap.mono import box_iou
    if not len(gtb) and not len(xyxy):
        return 0.0
    if not len(gtb) or not len(xyxy):
        return 1.0
    tp = int((box_iou(xyxy, gtb).max(0) >= 0.5).sum())
    prec, rec = tp / len(xyxy), tp / len(gtb)
    return 1.0 - (2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)


# ------------------------------------------------------------------------------------------------ labels (CPU)

def stage_labels(args):
    import importlib.util
    _s = importlib.util.spec_from_file_location("np_", B / "scripts" / "A_navsim" / "navsim_pdms.py")
    NP = importlib.util.module_from_spec(_s); _s.loader.exec_module(NP)
    OUT.mkdir(parents=True, exist_ok=True)
    sd = pd.read_csv(EXP / "pdms_states_dev.csv").set_index("token")
    st = pd.read_csv(EXP / "pdms_states_final.csv").set_index("token")
    ss = pd.read_csv(EXP / "pdms_states_sota.csv").set_index("token")
    dev_tok = {p.parent.name for p in (EXP / "metric_cache_dev").glob("*/*/*/metric_cache.pkl")}
    for which, split_dir, states in (("dev", "mini", sd), ("test", "test", st.join(ss[[f"score_{BR[s]}" for s in (480, 800)]]))):
        rows, boxes = [], {}
        for f in sorted(DETS[which].glob("*.pkl")):
            log = f.stem
            if log == "size_priors":
                continue
            dets = pickle.load(open(f, "rb"))
            frames = {fr["token"]: fr for fr in pickle.load(open(DATA / "navsim_logs" / split_dir / f"{log}.pkl", "rb"))}
            for tok, r in dets.items():
                if "cam" not in r or tok not in states.index or (which == "dev" and tok not in dev_tok):
                    continue
                fr = frames[tok]
                sp = split_of(log) if which == "dev" else "test"
                row = {"split": sp, "log": log, "token": tok,
                       "path": str(DATA / "sensor_blobs" / split_dir / fr["cams"]["CAM_F0"]["data_path"]),
                       "v_ego": float(np.hypot(*np.asarray(fr["ego_dynamic_state"], float)[:2]))}
                for s in SCALES:
                    if f"score_{BR[s]}" in states.columns:
                        row[f"pdms_{BR[s]}"] = float(states.at[tok, f"score_{BR[s]}"])
                if which == "test":
                    for b in ("spec_pdm", "spec", "sahi", "fovea", "uncertainty", "reference"):
                        row[f"pdms_{b}"] = float(states.at[tok, f"score_{b}"])
                    row["px_spec_pdm"] = float(r.get("px_spec_pdm", np.nan))
                else:
                    cam = NP.Cam(*r["cam"])
                    gb, gn = np.asarray(fr["anns"]["gt_boxes"], float).reshape(-1, 7), np.asarray(fr["anns"]["gt_names"])
                    gtx, gtc = [], []
                    for b, n in zip(gb, gn):
                        if str(n) in NAME2COARSE:
                            pr = cam.project(b)
                            if pr is not None and pr[3] - pr[1] >= 10:
                                gtx.append(pr); gtc.append(NAME2COARSE[str(n)])
                    gtb = np.asarray(gtx, float).reshape(-1, 4)
                    for s in SCALES:
                        row[f"loss_{s}"] = f1_loss(np.asarray(r[BR[s]]["xyxy"], float).reshape(-1, 4), gtb)
                    boxes[tok] = {"gt_xyxy": gtb, "gt_cls": np.asarray(gtc, "U8"),
                                  **{f"{t}_{k}": np.asarray(r[b][kk]).astype(np.float64)
                                     for t, b in (("weak", "cheap"), ("strong", "full960")) for k, kk in (("xyxy", "xyxy"), ("conf", "conf"))},
                                  "weak_cls": np.asarray(r["cheap"]["coarse"]).astype("U8"),
                                  "strong_cls": np.asarray(r["full960"]["coarse"]).astype("U8")}
                rows.append(row)
        d = pd.DataFrame(rows)
        for sp in sorted(d.split.unique()):
            ds = d[d.split == sp].reset_index(drop=True)
            save_df(ds, OUT / f"frames__{sp}.pkl")
            if sp != "test":
                with open(OUT / f"boxes__{sp}.pkl", "wb") as fh:
                    pickle.dump({t: boxes[t] for t in ds.token}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            msg = f"{sp}: {len(ds)} tokens, {ds.log.nunique()} logs"
            if sp != "test":
                opt = np.argmin(ds[[f"loss_{s}" for s in SCALES]].to_numpy() + 1e-6 * np.arange(5), 1)
                msg += f"; optimal-scale shares {pd.Series(opt).map(dict(enumerate(SCALES))).value_counts(normalize=True).round(3).to_dict()}"
            print(msg, flush=True)


# ------------------------------------------------------------------------------------------------ ORIC

def geng_raw(boxes, toks):
    raw = []
    lt = lambda b: [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]   # noqa: E731
    for i, t in enumerate(toks):
        r = boxes[t]
        det = {}
        for tag in ("weak", "strong"):
            m = r[f"{tag}_conf"] >= G_CONF
            det[tag] = [tuple(lt(b)) + (float(c), str(k)) for b, c, k in zip(r[f"{tag}_xyxy"][m], r[f"{tag}_conf"][m], r[f"{tag}_cls"][m])]
        raw.append({"frame_id": i, "video_name": "navsim", "edge_dets": det["weak"], "cloud_dets": det["strong"],
                    "gt_objs": [{"bbox": lt(b), "class": str(c)} for b, c in zip(r["gt_xyxy"], r["gt_cls"])]})
    return raw


def stage_oric(args):
    sys.path.insert(0, str(GENG))
    from src import proxy_metrics
    for split in ("train", "val"):
        d = load_df(OUT / f"frames__{split}.pkl")
        boxes = pickle.load(open(OUT / f"boxes__{split}.pkl", "rb"))
        res = proxy_metrics.compute_dataset_wide_oric(geng_raw(boxes, list(d.token)), quiet=True, context_size=0, context_draws=1)
        d["oric"] = [res.get(("navsim", i), {}).get("oric_allpoint", 0.0) for i in range(len(d))]
        save_df(d, OUT / f"frames__{split}.pkl")
        V = d.pdms_full960 - d.pdms_cheap
        print(f"{split}: ORIC>0 {np.mean(d.oric > 0):.3f}  V>0 {np.mean(V > 0):.3f}  V<0 {np.mean(V < 0):.3f}  "
              f"spearman(ORIC,V) {d.oric.corr(V, method='spearman'):+.3f}", flush=True)


# ------------------------------------------------------------------------------------------------ train (GPU)

def feats(boxes, d):
    from src.features import BASE_FEATURES, extract_detection_features
    lt = lambda b: (float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1]))   # noqa: E731
    X = []
    for t in d.token:
        r = boxes[t]
        m = r["weak_conf"] >= THR
        e = extract_detection_features([lt(b) + (float(c), str(k)) for b, c, k in zip(r["weak_xyxy"][m], r["weak_conf"][m], r["weak_cls"][m])])
        X.append([e[k] for k in BASE_FEATURES])
    return np.asarray(X, float)


def test_weak(d):
    """weak-detection features for navtest tokens (the cheap detections of NAVSIM_DETS_TEST)."""
    out = {}
    for log in d.log.unique():
        dets = pickle.load(open(DETS["test"] / f"{log}.pkl", "rb"))
        for t in d[d.log == log].token:
            c = dets[t]["cheap"]
            out[t] = {"weak_xyxy": np.asarray(c["xyxy"], float).reshape(-1, 4), "weak_conf": np.asarray(c["conf"], float),
                      "weak_cls": np.asarray(c["coarse"]).astype("U8")}
    return out


def stage_train(args):
    import cv2
    import torch
    import torch.nn as nn
    sys.path.insert(0, str(GENG))
    from src.models.classical import XGBoostEstimator
    from src.models.cnn import MobileNetV2LiteEstimator
    import runpy
    AS = runpy.run_path(str(B / "scripts" / "A_method" / "adascale.py"), run_name="adascale_lib")
    torch.manual_seed(SEED); np.random.seed(SEED)
    D = {sp: load_df(OUT / f"frames__{sp}.pkl") for sp in SPLITS}
    BX = {sp: pickle.load(open(OUT / f"boxes__{sp}.pkl", "rb")) for sp in ("train", "val")}
    BX["test"] = test_weak(D["test"])
    dev = torch.device(args.device)
    pred = {}
    # --- AdaScale (adascale.py's Net / recipe: 15 epochs, AdamW 3e-4, class-balanced CE, val = detection loss)
    if "adascale" in args.est:
        X = {sp: np.stack([AS["small"](cv2.imread(p)) for p in D[sp].path]) for sp in SPLITS}
        Y = {sp: np.argmin(D[sp][[f"loss_{s}" for s in SCALES]].to_numpy() + 1e-6 * np.arange(5), 1) for sp in ("train", "val")}
        net = AS["Net"]().to(dev)
        opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
        w = torch.tensor(1.0 / np.maximum(np.bincount(Y["train"], minlength=len(SCALES)), 1), dtype=torch.float32)
        lossf = nn.CrossEntropyLoss(weight=(w / w.sum() * len(SCALES)).to(dev))
        toT = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).float().div(255).to(dev)       # noqa: E731
        rng = np.random.default_rng(0)
        best = (1e9, None)

        def predict(x):
            net.eval()
            with torch.no_grad():
                return np.concatenate([net(toT(x[b])).argmax(1).cpu().numpy() for b in np.array_split(np.arange(len(x)), max(1, len(x) // 256))])
        for ep in range(15):
            net.train()
            for b in np.array_split(rng.permutation(len(Y["train"])), max(1, len(Y["train"]) // 64)):
                loss = lossf(net(toT(X["train"][b])), torch.from_numpy(Y["train"][b]).to(dev))
                opt.zero_grad(); loss.backward(); opt.step()
            pv = predict(X["val"])
            vl = D["val"][[f"loss_{s}" for s in SCALES]].to_numpy()[np.arange(len(pv)), pv].mean()
            print(f"  adascale epoch {ep}: val detection loss {vl:.4f}, acc {(pv == Y['val']).mean():.3f}", flush=True)
            if vl < best[0]:
                best = (vl, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        net.load_state_dict(best[1])
        torch.save(best[1], OUT / f"adascale_net{TAG}.pt")
        pred["adascale"] = {sp: predict(X[sp]) for sp in SPLITS}
        with open(OUT / f"scores__adascale{TAG}.pkl", "wb") as fh:
            pickle.dump(pred["adascale"], fh)
    # --- routers (router_baseline.py's estimators)
    tr = D["train"]
    if any(e in args.est for e in EST):
        ref = np.sort(tr.oric.to_numpy(np.float64))
        moric = lambda o: np.searchsorted(ref, np.asarray(o, np.float64), side="right") / len(ref)   # noqa: E731
        V = (tr.pdms_full960 - tr.pdms_cheap).to_numpy(float)
    for name in [e for e in args.est if e in EST]:
        if name.startswith("skip"):
            y = (tr.oric.to_numpy() > 0) if name == "skip_offbin" else (V > 0)
            est = MobileNetV2LiteEstimator(image_size=128, device=args.device)
            est.fit(list(tr.path), y.astype(np.float32), epochs=100, lr=5e-4, batch_size=512, loss="focal",
                    num_workers=args.workers, compile=False)
            p = {sp: est.predict(list(D[sp].path), batch_size=256, num_workers=args.workers) for sp in SPLITS}
            torch.save(est.model.state_dict(), OUT / f"{name}{TAG}.pt")
        else:
            X = {sp: feats(BX[sp], D[sp]) for sp in SPLITS}
            if name == "cond_V":
                X = {sp: np.column_stack([X[sp], D[sp].v_ego.to_numpy(float)]) for sp in SPLITS}
            est = XGBoostEstimator(n_estimators=100, max_depth=5, learning_rate=0.1, task_type="regression")
            est.fit(X["train"], moric(tr.oric) if name == "cond_moric" else V)
            p = {sp: np.asarray(est.predict(X[sp]), float) for sp in SPLITS}
        vv = (D["val"].pdms_full960 - D["val"].pdms_cheap)
        print(f"{name}: val spearman(score, V) {pd.Series(p['val']).corr(vv.reset_index(drop=True), method='spearman'):+.3f}", flush=True)
        with open(OUT / f"scores__{name}{TAG}.pkl", "wb") as fh:
            pickle.dump(p, fh)


# ------------------------------------------------------------------------------------------------ score (CPU)

def stage_score(args):
    sys.path.insert(0, str(GENG))
    from src.offloader import Offloader
    D = {sp: load_df(OUT / f"frames__{sp}.pkl") for sp in SPLITS}
    S = {}
    for name in ("adascale",) + EST:
        p = OUT / f"scores__{name}{TAG}.pkl"
        if p.exists():
            S[name] = pickle.load(open(p, "rb"))
    te = D["test"]
    uq = te.log.unique(); idx = [np.flatnonzero(te.log.to_numpy() == u) for u in uq]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    ours = te.pdms_spec_pdm.to_numpy()
    ours_px = PX[320] + float(np.nanmean(te.px_spec_pdm)) * 0.25        # cheap + crops at scale 0.5
    rho_m = {"cond": (ours_px - PX[320]) / PX[960], "skip": (ours_px - PX[320]) / (PX[960] - PX[320])}
    print(f"ours (NAVSIM config) network pixels {ours_px:.4f} of native; matched rho: conditioned {rho_m['cond']:.3f}, skipping {rho_m['skip']:.3f}")
    rows = []

    def add(method, score, px, rho_t=np.nan, rho_a=np.nan):
        dr = [ours[b].mean() - score[b].mean() for b in boots]
        rows.append({"method": method, "rho_target": rho_t, "rho_actual": rho_a, "px": px, "PDMS": float(score.mean()),
                     "ours_minus": f"[{np.percentile(dr, 2.5):+.4f}, {np.percentile(dr, 97.5):+.4f}]"})

    rows.append({"method": "ours (spec_pdm)", "px": ours_px, "PDMS": float(ours.mean())})
    for s in SCALES:
        add(f"uniform {s}", te[f"pdms_{BR[s]}"].to_numpy(), PX[s])
    for b in ("sahi", "fovea", "uncertainty", "spec"):
        add(b, te[f"pdms_{b}"].to_numpy(), np.nan)
    if "adascale" in S:
        pi = np.asarray(S["adascale"]["test"])
        sc = te[[f"pdms_{BR[s]}" for s in SCALES]].to_numpy()[np.arange(len(te)), pi]
        add("AdaScale", sc, float(np.mean([PX[SCALES[i]] for i in pi])))
        print("AdaScale scale shares", pd.Series(pi).map(dict(enumerate(SCALES))).value_counts(normalize=True).round(3).to_dict())
    lo, hi = te.pdms_cheap.to_numpy(), te.pdms_full960.to_numpy()

    def route(name, split, rho):
        return Offloader("offload_binary").decide_calibrated(S[name][split], rho, calibration_predictions=S[name]["train"]).mask

    def G_val(mask):
        v = D["val"]
        return float(np.where(mask, v.pdms_full960, v.pdms_cheap).mean())

    for rho in sorted(set(RHOS) | {round(rho_m["cond"], 4), round(rho_m["skip"], 4)}):
        n = int(round(rho * len(te)))
        orc = np.zeros(len(te), bool); orc[np.argsort(-(hi - lo), kind="stable")[:n]] = True
        add("oracle_V", np.where(orc, hi, lo), PX[320] + rho * PX[960], rho, orc.mean())
        rnd = np.zeros(len(te), bool); rnd[np.random.default_rng(1).permutation(len(te))[:n]] = True
        add("random", np.where(rnd, hi, lo), PX[320] + rho * PX[960], rho, rnd.mean())
        for name in [e for e in EST if e in S]:
            m = route(name, "test", rho)
            px = (PX[320] + m.mean() * PX[960]) if name.startswith("cond") else ((1 - m.mean()) * PX[320] + m.mean() * PX[960])
            add(name, np.where(m, hi, lo), px, rho, m.mean())
        if {"skip_offbin", "cond_moric"} <= set(S):
            pick = max(("skip_offbin", "cond_moric"), key=lambda nm: G_val(route(nm, "val", rho)))
            m = route(pick, "test", rho)
            px = (PX[320] + m.mean() * PX[960]) if pick.startswith("cond") else ((1 - m.mean()) * PX[320] + m.mean() * PX[960])
            add(f"adaptive({pick})", np.where(m, hi, lo), px, rho, m.mean())
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"summary_navtest{TAG}.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["labels", "oric", "train", "score"])
    ap.add_argument("--est", nargs="*", default=["adascale"] + list(EST))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    {"labels": stage_labels, "oric": stage_oric, "train": stage_train, "score": stage_score}[args.stage](args)
    print(f"NAVSIM SOTA {args.stage.upper()} DONE", flush=True)


if __name__ == "__main__":
    main()
