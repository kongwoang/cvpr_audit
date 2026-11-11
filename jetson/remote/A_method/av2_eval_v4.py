#!/usr/bin/env python
"""CONFIRMATORY 3 (pre-registered in docs/method_freeze_draft.md §2d): tuned v4 / v4-lite and frozen v3 on Argoverse 2
sensor TRAIN logs (copy of av2_eval.py with several method variants per pass; av2_eval.py stays the confirm-2 script).

Original description:
CONFIRMATORY 2 (pre-registered in docs/method_freeze_draft.md §2c): frozen v3 on Argoverse 2 sensor val.

Per frame (5 Hz, front-centre camera): cheap 320x256 draft, uniform 640x512, uniform 960x768, speculative v3 (proved
loop, field + support, v_close 10, truncation guard, merged-scene ttc history, crops at 960 / long side = 0.469) and
v3 + the frozen ResNet-18 refiner on the support objects from crops. Braking, monocular ego-frame lift, DEEP's cost,
cheap action history. Paired bootstrap over logs (2,000).
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from rap import planner as P
from rap.detect import Mode, TwoFidelityDetector
from rap.mono import predicted_geometry
from vtf import av2 as AV
from vtf import speculative as SP
import train_refiner_remote as TR
from nusc_refiner_crop import crop64

B = Path(os.environ["B"])
VARIANTS = {"v3": (10.0, 960, "resnet18"), "v4": (0.0, 1280, "resnet18_v2"), "v4lite": (0.0, 960, "resnet18_v2")}
THR, V_CLOSE = 0.25, 10.0
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
M_CHEAP, M_640, M_960 = Mode("av2_c320", 320, 256), Mode("av2_f640", 640, 512), Mode("av2_f960", 960, 768)


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_logs", type=int, default=0)
    ap.add_argument("--tag", default="_confirm3")
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard", default="0/1", help="i/n: logs[i::n] only, saved as a shard pickle (no summary)")
    ap.add_argument("--merge", type=int, default=0, help="n: concatenate the n shard pickles and summarise")
    args = ap.parse_args()
    out = B / "results" / "av2"
    if args.merge:
        d = pd.concat([pd.read_pickle(out / f"eval__{args.split}{args.tag}_s{i}of{args.merge}.pkl") for i in range(args.merge)],
                      ignore_index=True)
        d.to_pickle(out / f"eval__{args.split}{args.tag}.pkl")
        return summarise(d, out, args)
    ROOT = B / "datasets" / "av2" / "sensor" / args.split
    print("md5 speculative.py", md5(SP.__file__), "dsf.py", md5(Path(SP.__file__).parent / "dsf.py"),
          "refiner", md5(B / "out" / "resnet18" / "image_refiner.pt"), flush=True)
    dev = torch.device(args.device)
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device=args.device, backend="torch")
    refs = {}
    for rname in {v[2] for v in VARIANTS.values()}:
        ck = torch.load(B / "out" / rname / "image_refiner.pt", map_location=dev, weights_only=False)
        net_ = TR.Res18(len(ck["fmu"]))
        net_.load_state_dict(ck["state"])
        refs[rname] = (net_.to(dev).eval(), ck["fmu"], ck["fsd"])
        print("refiner", rname, md5(B / "out" / rname / "image_refiner.pt"), flush=True)
    pp, cp = P.PlannerParams(), P.CostParams()
    done = B / "datasets" / "av2" / "lists" / args.split
    logs = sorted(p.stem for p in done.glob("*.done")) if args.split != "val" else \
        sorted(p.name for p in ROOT.iterdir() if (p / "annotations.feather").exists())
    logs = logs[: args.max_logs or None]
    si, sn = (int(x) for x in args.shard.split("/"))
    logs = logs[si::sn]
    print(f"{len(logs)} fully downloaded logs", flush=True)
    rows = []
    for li, log in enumerate(logs):
        L = AV.AV2Log(ROOT, log)
        geom, speeds = L.geometry(), L.speeds()
        R, t = L.transform("ego")
        W, H = L.W, L.H
        prev = {"c": None, "640": None, "960": None}
        prev_scene, prev_act = {k_: None for k_ in VARIANTS}, None

        def lift(dd, pv):
            g = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, L.calib, L.cam_h, L.dt,
                                   cam_to_ego=(R, t))
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
            ch = keep(det.detect(img, M_CHEAP)[0])
            ch["mono"] = lift(ch, prev["c"])
            a_c = act(ch["mono"])
            rec = {"log": log, "frame": fr, "v": v, "J_c": cost(a_c)}
            for name, mode in (("640", M_640), ("960", M_960)):
                dd = keep(det.detect(img, mode)[0])
                rec[f"J_u{name}"] = cost(act(lift(dd, prev[name])))
                prev[name] = dd

            for vname, (vclose, long_px, rname) in VARIANTS.items():
                scale = long_px / max(W, H)

                def detect_crop(cr, scale=scale):
                    x1, y1 = int(cr[0]), int(cr[1])
                    x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                    sub = img[y1:y2, x1:x2]
                    nw = max(32, int(math.ceil(sub.shape[1] * scale / 32)) * 32)
                    nh = max(32, int(math.ceil(sub.shape[0] * scale / 32)) * 32)
                    d = keep(det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))[0])
                    d["xyxy"] = d["xyxy"] + np.array([x1, y1, x1, y1], float)
                    return d

                scene, crops, rounds = SP.run(ch, v, L, W, H, detect_crop, lambda ff, vn=vname: lift(ff, prev_scene[vn]), pp,
                                              v_close=vclose, guard=True)
                rec.update({f"J_spec_{vname}": cost(act(scene["mono"])), f"n_crops_{vname}": len(crops),
                            f"px_{vname}": sum((q[2] - q[0]) * (q[3] - q[1]) for q in crops) / (W * H),
                            f"netpx_{vname}": sum((q[2] - q[0]) * (q[3] - q[1]) for q in crops) * scale ** 2 / (W * H)})
                net, fmu, fsd = refs[rname]
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
                        corr = float(np.exp(net(x, torch.from_numpy(f).to(dev)).item()))
                    for a in refined[:3]:
                        a[p_] *= corr
                rec[f"J_spec_ref_{vname}"] = cost(act(tuple(refined)))
                prev_scene[vname] = {"xyxy": scene["xyxy"]}
            rows.append(rec)
            prev["c"], prev_act = ch, a_c
        print(f"  [{li + 1}/{len(logs)}] {log} {len(L.frame_ts)} frames", flush=True)
    d = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    if sn > 1:
        d.to_pickle(out / f"eval__{args.split}{args.tag}_s{si}of{sn}.pkl")
        print(f"SHARD {si}/{sn} DONE", flush=True)
        return
    d.to_pickle(out / f"eval__{args.split}{args.tag}.pkl")
    summarise(d, out, args)


def summarise(d, out, args):
    Lc = d.J_c.to_numpy()
    G = {"uniform 640": (d.J_c - d.J_u640).to_numpy(), "uniform 960": (d.J_c - d.J_u960).to_numpy()}
    for vname in VARIANTS:
        G[f"{vname}"] = (d.J_c - d[f"J_spec_{vname}"]).to_numpy()
        G[f"{vname} + refiner"] = (d.J_c - d[f"J_spec_ref_{vname}"]).to_numpy()
    uq = d.log.unique(); idx = [np.flatnonzero(d.log.to_numpy() == u) for u in uq]
    b = np.random.default_rng(0)
    draws = []
    for _ in range(2000):
        tt = np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))])
        if Lc[tt].sum() > 0:
            draws.append({k: g_[tt].sum() / Lc[tt].sum() for k, g_ in G.items()})
    D = pd.DataFrame(draws)
    res = []
    for k, g_ in G.items():
        aff = np.abs(g_) > 1e-9
        res.append({"method": k, "G": g_.sum() / Lc.sum(), "affected": int(aff.sum()),
                    "harmed": float((g_ < -1e-9).sum() / max(aff.sum(), 1)),
                    "minus_u640": f"[{np.percentile(D[k] - D['uniform 640'], 2.5):+.3f}, {np.percentile(D[k] - D['uniform 640'], 97.5):+.3f}]",
                    "minus_u960": f"[{np.percentile(D[k] - D['uniform 960'], 2.5):+.3f}, {np.percentile(D[k] - D['uniform 960'], 97.5):+.3f}]"})
    res = pd.DataFrame(res)
    dr = [((G["v4 + refiner"][t_] - G["v3 + refiner"][t_]).sum()) / Lc[t_].sum() for t_ in
          (np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))]) for _ in range(2000))]
    res.to_csv(out / f"eval_summary__{args.split}{args.tag}.csv", index=False)
    print(f"logs {len(uq)}, frames {len(d)}; " + "; ".join(
        f"{k_}: crops {d[f'n_crops_{k_}'].mean():.2f}, px {d[f'px_{k_}'].mean():.4f}, network px {d[f'netpx_{k_}'].mean():.4f}"
        for k_ in VARIANTS))
    print(f"v4+ref - v3+ref: {np.percentile(dr, [2.5, 97.5]).round(3)}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
