#!/usr/bin/env python
"""AV2 ports of AdaScale (Chin et al.) and Budget-Adaptive Routing (Geng et al., official bgt-ada), recipes of
adascale.py / router_baseline.py unchanged; fit on AV2 TRAIN logs only (av2_sota_labels.py: "train" / "val" by log),
scored once on AV2 VAL ("test") against our AV2 configuration (v4-lite + refiner v2, av2_eval_v4 --tag _v4_final).

Stages:  oric (CPU) | train (GPU: AdaScale ResNet-18 + MobileNetV2-Lite estimators, XGBoost; predictions on all splits)
         | ada_eval (GPU: AdaScale on the test frames, the chosen scale's detections with AdaScale's own lift history,
         as adascale.py stage_eval) | score (CPU).
"""
from __future__ import annotations

import argparse
import os
import pickle
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
OUT = B / "results" / "av2_sota"
GENG = B / "third_party" / "bgt-ada"
SPLITS = ("train", "val", "test")
SCALES = (320, 480, 640, 800, 960)
W0, H0 = 1550, 2048
LAB = runpy.run_path(str(B / "scripts" / "A_method" / "av2_sota_labels.py"), run_name="av2_labels_lib")
PX = {s: LAB["mode_for"](s).net_h * LAB["mode_for"](s).net_w / (W0 * H0) for s in SCALES}
RHOS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
EST = ("skip_offbin", "cond_moric", "skip_V", "cond_V")
THR, G_CONF = 0.25, 0.30
SEED = int(os.environ.get("SEED", 42))
TAG = "" if SEED == 42 else f"_s{SEED}"
RB = runpy.run_path(str(B / "scripts" / "A_method" / "router_baseline.py"), run_name="router_lib")


def frames(sp):
    return pd.read_pickle(OUT / f"frames__{sp}.pkl").rename(columns={"log": "seq"})


def boxes(sp):
    return pickle.load(open(OUT / f"boxes__{sp}.pkl", "rb"))


def stage_oric(args):
    sys.path.insert(0, str(GENG))
    from src import proxy_metrics
    for split in ("train", "val"):
        d = frames(split)
        res = proxy_metrics.compute_dataset_wide_oric(RB["geng_raw"](boxes(split), list(zip(d.seq, d.frame.astype(int)))),
                                                      quiet=True, context_size=0, context_draws=1)
        d["oric"] = [res.get((str(s), int(f)), {}).get("oric_allpoint", 0.0) for s, f in zip(d.seq, d.frame)]
        d.rename(columns={"seq": "log"}).to_pickle(OUT / f"frames__{split}.pkl")
        V = d.J_c - d.J_u960
        print(f"{split}: ORIC>0 {np.mean(d.oric > 0):.3f}  V>0 {np.mean(V > 0):.3f}  V<0 {np.mean(V < 0):.3f}  "
              f"spearman(ORIC,V) {d.oric.corr(V, method='spearman'):+.3f}", flush=True)


def stage_train(args):
    import cv2
    import torch
    import torch.nn as nn
    sys.path.insert(0, str(GENG))
    from src.models.classical import XGBoostEstimator
    from src.models.cnn import MobileNetV2LiteEstimator
    AS = runpy.run_path(str(B / "scripts" / "A_method" / "adascale.py"), run_name="adascale_lib")
    torch.manual_seed(SEED); np.random.seed(SEED)
    D = {sp: frames(sp) for sp in SPLITS}
    BX = {sp: boxes(sp) for sp in SPLITS}
    dev = torch.device(args.device)
    if "adascale" in args.est:
        small = lambda p: cv2.resize(cv2.imread(p), (180, 240), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()   # noqa: E731  portrait
        X = {sp: np.stack([small(p) for p in D[sp].path]) for sp in SPLITS}
        Y = {sp: np.argmin(D[sp][[f"loss_{s}" for s in SCALES]].to_numpy() + 1e-6 * np.arange(5), 1) for sp in ("train", "val")}
        net = AS["Net"]().to(dev)
        opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
        w = torch.tensor(1.0 / np.maximum(np.bincount(Y["train"], minlength=len(SCALES)), 1), dtype=torch.float32)
        lossf = nn.CrossEntropyLoss(weight=(w / w.sum() * len(SCALES)).to(dev))
        toT = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).float().div(255).to(dev)       # noqa: E731
        rng, best = np.random.default_rng(0), (1e9, None)

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
        torch.save(best[1], OUT / f"adascale_net{TAG}.pt")
    tr = D["train"]
    if any(e in args.est for e in EST):
        ref = np.sort(tr.oric.to_numpy(np.float64))
        moric = lambda o: np.searchsorted(ref, np.asarray(o, np.float64), side="right") / len(ref)   # noqa: E731
        V = (tr.J_c - tr.J_u960).to_numpy(float)
    for name in [e for e in args.est if e in EST]:
        if name.startswith("skip"):
            y = (tr.oric.to_numpy() > 0) if name == "skip_offbin" else (V > 0)
            est = MobileNetV2LiteEstimator(image_size=128, device=args.device)
            est.fit(list(tr.path), y.astype(np.float32), epochs=100, lr=5e-4, batch_size=512, loss="focal",
                    num_workers=args.workers, compile=False)
            p = {sp: est.predict(list(D[sp].path), batch_size=256, num_workers=args.workers) for sp in SPLITS}
            torch.save(est.model.state_dict(), OUT / f"{name}{TAG}.pt")
        else:
            X = {sp: RB["feats"](BX[sp], D[sp]) for sp in SPLITS}
            if name == "cond_V":
                X = {sp: np.column_stack([X[sp], D[sp].v_ego.to_numpy(float)]) for sp in SPLITS}
            est = XGBoostEstimator(n_estimators=100, max_depth=5, learning_rate=0.1, task_type="regression")
            est.fit(X["train"], moric(tr.oric) if name == "cond_moric" else V)
            p = {sp: np.asarray(est.predict(X[sp]), float) for sp in SPLITS}
        vv = (D["val"].J_c - D["val"].J_u960).reset_index(drop=True)
        print(f"{name}: val spearman(score, V) {pd.Series(p['val']).corr(vv, method='spearman'):+.3f}", flush=True)
        with open(OUT / f"scores__{name}{TAG}.pkl", "wb") as fh:
            pickle.dump(p, fh)


def stage_ada_eval(args):
    """AdaScale on AV2 val, exactly as adascale.py stage_eval: predicted scale per frame, YOLOv8s at that scale on the
    whole frame, mono lift with AdaScale's own previous detections, braking, cheap action history."""
    import cv2
    import torch
    from rap import planner as P
    from rap.detect import TwoFidelityDetector
    from rap.mono import predicted_geometry
    from vtf import av2 as AV
    AS = runpy.run_path(str(B / "scripts" / "A_method" / "adascale.py"), run_name="adascale_lib")
    dev = torch.device(args.device)
    net = AS["Net"]().to(dev)
    net.load_state_dict(torch.load(OUT / f"adascale_net{TAG}.pt", map_location=dev))
    net.eval()
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device=args.device, backend="torch")
    pp, cp = P.PlannerParams(), P.CostParams()
    root, logs = LAB["logs_of"]("test")
    rows = []
    for li, log in enumerate(logs):
        L = AV.AV2Log(root, log)
        geom, speeds = L.geometry(), L.speeds()
        R, t = L.transform("ego")
        prev_c, prev, prev_act = None, None, None

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
            v = float(speeds[fr])
            g = geom[geom["frame"] == fr] if len(geom) else geom
            g = g[(g["y2"] - g["y1"]) >= 10] if len(g) else g
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
            act = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)            # noqa: E731
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]                  # noqa: E731
            ch = keep(det.detect(img, LAB["mode_for"](320))[0])
            a_c = act(lift(ch, prev_c))
            x = cv2.resize(img, (180, 240), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
            with torch.no_grad():
                si = int(net(torch.from_numpy(x[None]).permute(0, 3, 1, 2).float().div(255).to(dev)).argmax(1).item())
            sc = SCALES[si]
            dd = ch if sc == 320 else keep(det.detect(img, LAB["mode_for"](sc))[0])
            rows.append({"log": log, "frame": fr, "J_c": cost(a_c), "scale": sc, "net_px": PX[sc], "J_ada": cost(act(lift(dd, prev)))})
            prev, prev_c, prev_act = dd, ch, a_c
        print(f"  [{li + 1}/{len(logs)}] {log}", flush=True)
    pd.DataFrame(rows).to_pickle(OUT / f"ada_eval_test{TAG}.pkl")


def stage_score(args):
    sys.path.insert(0, str(GENG))
    from src.offloader import Offloader
    D = {sp: frames(sp) for sp in ("train", "val", "test")}
    S = {n: pickle.load(open(OUT / f"scores__{n}{TAG}.pkl", "rb")) for n in EST if (OUT / f"scores__{n}{TAG}.pkl").exists()}
    ref = pd.read_pickle(B / "results" / "av2" / "eval__val_v4_final.pkl").rename(columns={"log": "seq"})
    bl = pd.read_pickle(B / "results" / "av2" / "baselines__val.pkl").rename(columns={"log": "seq"})
    te = D["test"].merge(ref[["seq", "frame", "J_spec_ref_v4lite", "J_spec_ref_v3", "netpx_v4lite"]], on=["seq", "frame"])
    te = te.merge(bl[["seq", "frame", "J_fovea_kdei960", "J_sahi_s10", "J_uncertainty"]], on=["seq", "frame"])
    assert len(te) == len(D["test"])
    ap_ = OUT / f"ada_eval_test{TAG}.pkl"
    if ap_.exists():
        ad = pd.read_pickle(ap_).rename(columns={"log": "seq"})
        te = te.merge(ad[["seq", "frame", "J_ada", "scale", "net_px"]], on=["seq", "frame"], how="left")
    L = te.J_c.to_numpy()
    units = te.seq.to_numpy(); uq = np.unique(units); idx = [np.flatnonzero(units == u) for u in uq]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    ours = (te.J_c - te.J_spec_ref_v4lite).to_numpy()
    ours_px = PX[320] + float(te.netpx_v4lite.mean())
    rho_m = {"cond": (ours_px - PX[320]) / PX[960], "skip": (ours_px - PX[320]) / (PX[960] - PX[320])}
    print(f"ours (v4-lite + ref) network pixels {ours_px:.4f}; matched rho: conditioned {rho_m['cond']:.3f}, skipping {rho_m['skip']:.3f}")
    rows = [{"method": "ours (v4-lite + ref)", "px": ours_px, "G": float(ours.sum() / L.sum())}]

    def add(method, v, px, rho_t=np.nan, rho_a=np.nan):
        dr = [(ours[b].sum() - v[b].sum()) / L[b].sum() for b in boots]
        rows.append({"method": method, "rho_target": rho_t, "rho_actual": rho_a, "px": px, "G": float(v.sum() / L.sum()),
                     "harmed": float(np.mean(v < -1e-9) / max(np.mean(np.abs(v) > 1e-9), 1e-9)),
                     "ours_minus": f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"})
    for s in SCALES[1:]:
        add(f"uniform {s}", (te.J_c - te[f"J_u{s}"]).to_numpy(), PX[s])
    add("v3 + ref", (te.J_c - te.J_spec_ref_v3).to_numpy(), np.nan)
    for c, n in (("J_fovea_kdei960", "FOVEA"), ("J_sahi_s10", "SAHI"), ("J_uncertainty", "uncertainty crops")):
        add(n, (te.J_c - te[c]).to_numpy(), np.nan)
    if "J_ada" in te:
        add("AdaScale", (te.J_c - te.J_ada).to_numpy(), float(te.net_px.mean()))
        print("AdaScale scale shares", te.scale.value_counts(normalize=True).round(3).to_dict())
    Vt = (te.J_c - te.J_u960).to_numpy()

    def route(name, split, rho):
        return Offloader("offload_binary").decide_calibrated(S[name][split], rho, calibration_predictions=S[name]["train"]).mask

    def G_val(mask):
        v = D["val"]
        return float(((v.J_c - v.J_u960).to_numpy() * mask).sum() / v.J_c.sum())

    for rho in sorted(set(RHOS) | {round(rho_m["cond"], 4), round(rho_m["skip"], 4)}):
        n = int(round(rho * len(te)))
        orc = np.zeros(len(te), bool); orc[np.argsort(-Vt, kind="stable")[:n]] = True
        add("oracle_V", Vt * orc, PX[320] + rho * PX[960], rho, orc.mean())
        rnd = np.zeros(len(te), bool); rnd[np.random.default_rng(1).permutation(len(te))[:n]] = True
        add("random", Vt * rnd, PX[320] + rho * PX[960], rho, rnd.mean())
        for name in S:
            m = route(name, "test", rho)
            px = (PX[320] + m.mean() * PX[960]) if name.startswith("cond") else ((1 - m.mean()) * PX[320] + m.mean() * PX[960])
            add(name, Vt * m, px, rho, m.mean())
        if {"skip_offbin", "cond_moric"} <= set(S):
            pick = max(("skip_offbin", "cond_moric"), key=lambda nm: G_val(route(nm, "val", rho)))
            m = route(pick, "test", rho)
            px = (PX[320] + m.mean() * PX[960]) if pick.startswith("cond") else ((1 - m.mean()) * PX[320] + m.mean() * PX[960])
            add(f"adaptive({pick})", Vt * m, px, rho, m.mean())
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"summary_av2val{TAG}.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["oric", "train", "ada_eval", "score"])
    ap.add_argument("--est", nargs="*", default=["adascale"] + list(EST))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    {"oric": stage_oric, "train": stage_train, "ada_eval": stage_ada_eval, "score": stage_score}[args.stage](args)
    print(f"AV2 SOTA {args.stage.upper()} DONE", flush=True)


if __name__ == "__main__":
    main()
