#!/usr/bin/env python
"""Fine-tuned detectors on AV2 (Table 2 on a second dataset): AV2 VAL (= the AV2 test split), every DEEP frame, the
conventions of av2_eval_v4.py (front-centre camera, COCO cheap 320 draft for every pipeline, mono lift with the arm's
own history, braking, DEEP's cost, cheap action history).

  --stage dets (GPU)   u960-ft = YOLOv8s fine-tuned on the AV2 train set (av2_ft_prepare.py; out/av2_ft/<det>) on the
                       whole frame at 960; ours-ft = our AV2 configuration (v4-lite: v_close 0, <= 3 rounds, crops at
                       960 / long side, guard, merged-scene history) with the fine-tuned detector as crop verifier, with
                       and without refiner v2; also stores the cheap action per frame (for scoring prediction files)
  --stage preds (CPU)  a baseline's saved whole-frame predictions (YOLO txt: class cx cy w h conf, normalised, 0 vehicle /
                       1 person / 2 cyclist; ESOD test.py / zoomdet_infer.py) -> J per frame
  --stage score (CPU)  G and paired log bootstraps against ours-ft (+ the COCO references of av2_eval_v4 --tag _v4_final)
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

B = Path(os.environ["B"])
OUT = B / "results" / "av2_ft"
THR = 0.25
COARSE = {0: "vehicle", 1: "person", 2: "cyclist"}
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}


def logs():
    root = B / "datasets" / "av2" / "sensor" / "val"
    return root, sorted(p.name for p in root.iterdir() if (p / "annotations.feather").exists())


def stage_dets(args):
    import torch
    from rap import planner as P
    from rap.detect import Mode, TwoFidelityDetector
    from rap.mono import predicted_geometry
    from vtf import av2 as AV
    from vtf import speculative as SP
    import train_refiner_remote as TR
    from nusc_refiner_crop import crop64
    dev = torch.device(args.device)
    cheap_det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device=args.device, backend="torch")
    ft = TwoFidelityDetector(str(B / "out" / "av2_ft" / args.det / "weights" / "best.pt"), device=args.device, backend="torch")
    ck = torch.load(B / "out" / "resnet18_v2" / "image_refiner.pt", map_location=dev, weights_only=False)
    rnet = TR.Res18(len(ck["fmu"])); rnet.load_state_dict(ck["state"]); rnet = rnet.to(dev).eval()
    fmu, fsd = ck["fmu"], ck["fsd"]
    M_CHEAP, M_960 = Mode("av2_c320", 320, 256), Mode("av2_f960", 960, 768)
    pp, cp = P.PlannerParams(), P.CostParams()
    root, lg = logs()
    rows = []
    for li, log in enumerate(lg[: args.max_logs or None]):
        L = AV.AV2Log(root, log)
        geom, speeds = L.geometry(), L.speeds()
        R, t = L.transform("ego")
        W, H = L.W, L.H
        prev_c, prev_u, prev_scene, prev_act = None, None, None, None

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
            ch = keep(cheap_det.detect(img, M_CHEAP)[0])
            ch["mono"] = lift(ch, prev_c)
            a_c = act(ch["mono"])
            rec = {"log": log, "frame": fr, "J_c": cost(a_c), "a_c": float(a_c), "prev_a": np.nan if prev_act is None else float(prev_act)}
            du = keep(ft.detect(img, M_960)[0])
            rec["J_u960_ft"] = cost(act(lift(du, prev_u)))
            prev_u = du
            scale = 960 / max(W, H)

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * scale / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * scale / 32)) * 32)
                d = keep(ft.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))[0])
                d["xyxy"] = d["xyxy"] + np.array([x1, y1, x1, y1], float)
                return d
            scene, crops, _ = SP.run(ch, v, L, W, H, detect_crop, lambda ff: lift(ff, prev_scene), pp, v_close=0.0, guard=True)
            rec["J_ours_ft"] = cost(act(scene["mono"]))
            rec["netpx_ours_ft"] = sum((q[2] - q[0]) * (q[3] - q[1]) for q in crops) * scale ** 2 / (W * H)
            refined = [np.asarray(a, float).copy() for a in scene["mono"]]
            for p_ in SP.support(scene["mono"], v, pp):
                if not scene["from_full"][p_]:
                    continue
                x1, y1, x2, y2 = scene["xyxy"][p_]
                zz = predicted_geometry({"xyxy": scene["xyxy"][p_:p_ + 1], "coarse": scene["coarse"][p_:p_ + 1]}, None,
                                        L.calib, L.cam_h, L.dt, cam_to_ego=(R, t))
                zg, zh = float(zz["z_ground"][0]), float(zz["z_height"][0])
                f = np.array([[x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(scene["conf"][p_]),
                               float(CLS[str(scene["coarse"][p_])]), float(scene["mono"][0][p_]), zg, zh,
                               float(np.log(max(zg, .5) / max(zh, .5))), 1.0]], np.float32)
                f = ((f - fmu) / fsd).astype(np.float32)
                x = torch.from_numpy(crop64(img, scene["xyxy"][p_])).permute(2, 0, 1)[None].float().to(dev) / 255.0
                with torch.no_grad():
                    corr = float(np.exp(rnet(x, torch.from_numpy(f).to(dev)).item()))
                for a in refined[:3]:
                    a[p_] *= corr
            rec["J_ours_ft_ref"] = cost(act(tuple(refined)))
            prev_scene = {"xyxy": scene["xyxy"]}
            rows.append(rec)
            prev_c, prev_act = ch, a_c
        print(f"  [{li + 1}/{len(lg)}] {log}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_pickle(OUT / f"dets_{args.det}{args.tag}.pkl")


def stage_preds(args):
    """J of a whole-frame baseline from its prediction files; the cheap action history comes from the dets stage."""
    from rap import planner as P
    from rap.mono import predicted_geometry
    from vtf import av2 as AV
    base = pd.read_pickle(OUT / f"dets_{args.det}{args.tag}.pkl")
    pa = {(r.log, r.frame): (r.prev_a, r.a_c) for r in base.itertuples()}
    pred = Path(args.pred)
    pp, cp = P.PlannerParams(), P.CostParams()
    root, lg = logs()
    rows = []
    for log in lg:
        if not (base.log == log).any():
            continue
        L = AV.AV2Log(root, log)
        geom, speeds = L.geometry(), L.speeds()
        R, t = L.transform("ego")
        prev = None
        for fr in range(len(L.frame_ts)):
            if (log, fr) not in pa:
                continue
            prev_a, _ = pa[(log, fr)]
            v = float(speeds[fr])
            g = geom[geom["frame"] == fr] if len(geom) else geom
            g = g[(g["y2"] - g["y1"]) >= 10] if len(g) else g
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
            f = pred / f"{log}_{fr:03d}.txt"
            arr = np.loadtxt(f, ndmin=2) if f.exists() and f.stat().st_size else np.zeros((0, 6))
            arr = arr[arr[:, 5] >= THR] if len(arr) else arr
            W, H = L.W, L.H
            xyxy = np.stack([(arr[:, 1] - arr[:, 3] / 2) * W, (arr[:, 2] - arr[:, 4] / 2) * H,
                             (arr[:, 1] + arr[:, 3] / 2) * W, (arr[:, 2] + arr[:, 4] / 2) * H], 1) if len(arr) else np.zeros((0, 4))
            dd = {"xyxy": xyxy, "coarse": np.array([COARSE[int(k)] for k in arr[:, 0]]) if len(arr) else np.zeros(0, "<U8")}
            gg = predicted_geometry(dd, prev, L.calib, L.cam_h, L.dt, cam_to_ego=(R, t))
            prev = dd
            tt = tuple(np.asarray(gg[q], float) for q in ("z", "lat_min", "lat_max", "ttc"))
            a = P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)
            rows.append({"log": log, "frame": fr, f"J_{args.method}": P.decision_cost(a, a_gt, None if np.isnan(prev_a) else int(prev_a), pp, cp)["J"]})
    n_files = len(list(pred.glob("*.txt")))
    d = pd.DataFrame(rows)
    d.to_pickle(OUT / f"preds_{args.method}{args.tag}.pkl")
    print(f"{args.method}: {len(d)} frames scored, {n_files} prediction files")


def stage_score(args):
    d = pd.read_pickle(OUT / f"dets_{args.det}{args.tag}.pkl")
    ref = pd.read_pickle(B / "results" / "av2" / "eval__val_v4_final.pkl")
    d = d.merge(ref[["log", "frame", "J_c", "J_u960", "J_spec_ref_v4lite"]], on=["log", "frame"], suffixes=("", "_ref"))
    assert np.allclose(d.J_c, d.J_c_ref), "cheap cost differs from av2_eval_v4"
    for p in sorted(OUT.glob(f"preds_*{args.tag}.pkl")):
        d = d.merge(pd.read_pickle(p), on=["log", "frame"], how="left")
    L = d.J_c.to_numpy()
    uq = d.log.unique(); idx = [np.flatnonzero(d.log.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    cols = {"ours-ft + ref": "J_ours_ft_ref", "ours-ft": "J_ours_ft", "u960-ft": "J_u960_ft", "u960 (COCO)": "J_u960",
            "ours (COCO, v4-lite + ref)": "J_spec_ref_v4lite"}
    cols.update({c[2:]: c for c in d.columns if c.startswith("J_") and c[2:] in ("esod", "zoomdet")})
    V = {k: (d.J_c - d[c]).to_numpy() for k, c in cols.items() if c in d and d[c].notna().all()}
    res = []
    for k, v_ in V.items():
        r = {"method": k, "G": v_.sum() / L.sum()}
        for rk in ("ours-ft + ref", "ours-ft"):
            if k != rk:
                dr = [(V[rk][t_].sum() - v_[t_].sum()) / L[t_].sum() for t_ in boots]
                r[f"{rk} minus"] = f"[{np.percentile(dr, 2.5):+.3f}, {np.percentile(dr, 97.5):+.3f}]"
        res.append(r)
    res = pd.DataFrame(res)
    res.to_csv(OUT / f"summary{args.tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"frames {len(d)}, logs {len(uq)}; ours-ft network px {d.netpx_ours_ft.mean():.4f}")
    print(res.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["dets", "preds", "score"])
    ap.add_argument("--det", default="plain960")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pred", default="")
    ap.add_argument("--method", default="esod")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_logs", type=int, default=0)
    args = ap.parse_args()
    {"dets": stage_dets, "preds": stage_preds, "score": stage_score}[args.stage](args)
    print(f"AV2 FT {args.stage.upper()} DONE", flush=True)


if __name__ == "__main__":
    main()
