#!/usr/bin/env python
"""ABLATION copy of nusc_full_eval.py (adds --max_rounds, --v_close); the frozen script stays untouched.

Confirmatory evaluation on nuScenes-full: new scenes, test split, braking consumer, monocular lift.

Per frame (cheap draft from the cache, DEEP's shared action history):
  uniform 640   the full-640 branch of the decision table
  uniform 960   YOLOv8s on the whole image at 576x960 (0.6 scale), lifted — the fair baseline for 960-scale crops
  speculative   closed-form field + pivots (vtf.dsf), YOLOv8s on each crop at the 0.6 scale, lift, fuse
  + refiner     the fused decision's pivots from crops rescaled by an image-level range refiner (each checkpoint given)
Paired scene bootstrap (2,000 draws) of every method minus uniform 640 and minus uniform 960.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import _apply_range_source
from rap.detect import Mode, TwoFidelityDetector
from rap.mono import predicted_geometry
from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL
from vtf import dsf
from vtf import speculative as SP

from rap import cache as _rc                                # new cache roots are nuScenes CAM_FRONT caches too
_rc.NUSC_CACHES = type(_rc.NUSC_CACHES)(set(_rc.NUSC_CACHES) | {"nusc_models", "nusc_det_ft"})

B = Path(os.environ["B"])
sys.path.insert(0, str(B))
import train_refiner_remote as TR                      # noqa: E402
from nusc_refiner_crop import crop64                   # noqa: E402

R_GRID = np.array([12.5, 17.5, 22.5, 27.5, 35.0, 45.0, 55.0, 70.0])
Y_GRID = np.array([-6.0, -4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0])
OBJ_W, OBJ_H, CTX, THR, SCALE, V_CLOSE = 1.8, 1.6, 2.0, 0.25, 0.6, 10.0
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}


def load_refiner(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = (TR.Small if ck["arch"] == "small" else TR.Res18)(len(ck["fmu"]))
    net.load_state_dict(ck["state"])
    return net.to(dev).eval(), ck["fmu"], ck["fsd"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refiners", nargs="*", default=[])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--set", default="new")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--guard", default="all", choices=["all", "corridor", "none"], help="truncation guard (none = v2)")
    ap.add_argument("--hist", default="scene", choices=["scene", "cheap"],
                    help="ttc history of crop detections: previous merged scene (v3) or previous cheap boxes (v2)")
    ap.add_argument("--obj_h", type=float, default=SP.OBJ_H)
    ap.add_argument("--max_rounds", type=int, default=3)
    ap.add_argument("--geometry", default="mono", choices=["mono", "oracle"],
                    help="oracle: every pipeline's detections matched to GT (IoU >= 0.5) take the GT state (DEEP's oracle cell)")
    ap.add_argument("--dump", action="store_true",
                    help="also save every pipeline's final scene per frame (for the PKL-planner cells)")
    ap.add_argument("--cheap_dir", default="nusc_det_tv", help="cache dir of the draft (a fine-tuned draft: no DEEP check)")
    ap.add_argument("--v_close", type=float, default=V_CLOSE)
    args = ap.parse_args()
    global SCALE
    SCALE = args.scale
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    out = B / "results" / "nusc_full"
    tab = pd.read_pickle(out / f"flips__{args.geometry}.pkl")
    tab = tab[(tab["set"] == args.set) & tab.split.isin(args.split.split("+"))]
    scenes = sorted(tab.seq.unique())[: args.max_scenes or None]
    refiners = {Path(p).parent.name: load_refiner(p, dev) for p in args.refiners}
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device=args.device, backend="torch")
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    adapter = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    pp, cp = P.PlannerParams(), P.CostParams()
    rng = np.random.default_rng(0)
    rows, dumps = [], []
    for s in scenes:
        geom, speeds = adapter.geometry(s), adapter.speeds(s)
        cam = Camera("nuScenes", s)
        R, t = cam.transform("ego")
        P2 = cam.calib.P2
        fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
        c = DetCache(Path(CACHE) / args.cheap_dir / "ns_cheap_320" / f"{s}.npz")
        toks = db.samples(by[s])
        prev_act, prev_det, prev960, prev_scene = None, None, None, None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]              # noqa: E731
            decide_m = lambda tt: P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)  # noqa: E731

            def decide(tt, xyxy=None):
                if args.geometry == "oracle" and xyxy is not None:
                    geo = dict(zip(("z", "lat_min", "lat_max", "ttc"), (np.asarray(a, float) for a in tt)))
                    tt = tuple(np.asarray(a, float) for a in _apply_range_source(np.asarray(xyxy, float).reshape(-1, 4),
                                                                                 geo, g, "gt", rng, 0.0))
                return decide_m(tt)
            dc = c.det(i)
            kc = dc["conf"] >= THR
            geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
            bxc = dc["xyxy"][kc].astype(float)
            cons_c = tuple(np.asarray(a, float) for a in _apply_range_source(bxc, geo_c, g, args.geometry.replace("oracle", "gt"),
                                                                             rng, 0.0))
            a_c = decide(cons_c)
            img = cv2.imread(str(db.image_path(toks[fr])))
            H, W = img.shape[:2]
            rec = {"seq": s, "frame": fr, "J_c": cost(a_c)}
            # uniform 960
            d9, _ = det.detect(img, Mode("f576x960", 576, 960))
            k9 = d9["conf"] >= THR
            dd9 = {"xyxy": d9["xyxy"][k9].astype(float), "coarse": d9["coarse"][k9]}
            g9 = predicted_geometry(dd9, prev960, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
            rec["J_u960"] = cost(decide(tuple(np.asarray(g9[x], float) for x in ("z", "lat_min", "lat_max", "ttc")), dd9["xyxy"]))
            prev960 = dd9
            # speculative (the proved loop: support + field crops, iterated; probe closing at V_CLOSE)
            cheap_d = {"xyxy": bxc, "conf": dc["conf"][kc].astype(float), "coarse": dc["coarse"][kc], "mono": cons_c}

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * SCALE / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * SCALE / 32)) * 32)
                d, _ = det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))
                kk = d["conf"] >= THR
                return {"xyxy": d["xyxy"][kk].astype(float) + np.array([x1, y1, x1, y1], float),
                        "conf": d["conf"][kk].astype(float), "coarse": d["coarse"][kk]}

            def lift(full):
                gk = predicted_geometry({"xyxy": full["xyxy"], "coarse": full["coarse"]},
                                        prev_scene if args.hist == "scene" else prev_det, cam.calib, cam.cam_h,
                                        cam.dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            scene, crops, rounds = SP.run(cheap_d, v, cam, W, H, detect_crop, lift, pp, v_close=args.v_close, max_rounds=args.max_rounds,
                                           guard={"all": True, "none": False}.get(args.guard, args.guard), obj_h=args.obj_h)
            rec["n_crops"], rec["rounds"] = len(crops), rounds
            rec["px"] = sum((c_[2] - c_[0]) * (c_[3] - c_[1]) for c_ in crops) / (W * H)
            J_spec = cost(decide(scene["mono"], scene["xyxy"]))
            J_ref, refined_by = {}, {}
            if crops and len(scene["conf"]):
                for name, (net, fmu, fsd) in refiners.items():
                    refined = [np.asarray(a, float).copy() for a in scene["mono"]]
                    for p_ in SP.support(scene["mono"], v, pp):
                        if not scene["from_full"][p_]:
                            continue
                        x1, y1, x2, y2 = scene["xyxy"][p_]
                        z = float(scene["mono"][0][p_])
                        zz = predicted_geometry({"xyxy": scene["xyxy"][p_:p_ + 1], "coarse": scene["coarse"][p_:p_ + 1]},
                                                None, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                        zg, zh = float(zz["z_ground"][0]), float(zz["z_height"][0])
                        f = np.array([[x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(scene["conf"][p_]),
                                       float(CLS[str(scene["coarse"][p_])]), z, zg, zh,
                                       float(np.log(max(zg, .5) / max(zh, .5))), 1.0]], np.float32)
                        f = ((f - fmu) / fsd).astype(np.float32)
                        x = torch.from_numpy(crop64(img, scene["xyxy"][p_])).permute(2, 0, 1)[None].float().to(dev) / 255.0
                        with torch.no_grad():
                            corr = float(np.exp(net(x, torch.from_numpy(f).to(dev)).item()))
                        for a in refined[:3]:
                            a[p_] *= corr
                    J_ref[name] = cost(decide(tuple(refined), scene["xyxy"]))
                    refined_by[name] = tuple(refined)
            for name in refiners:
                J_ref.setdefault(name, J_spec)
            rec["J_spec"] = J_spec
            rec.update({f"J_spec_ref_{n}": j for n, j in J_ref.items()})
            rows.append(rec)
            if args.dump:
                t9 = tuple(np.asarray(g9[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))
                pk = lambda xy, cf, co, mo: {"xyxy": np.asarray(xy, float).reshape(-1, 4), "conf": np.asarray(cf, float),  # noqa: E731
                                             "coarse": np.asarray(co), "mono": tuple(np.asarray(a, float) for a in mo)}
                entry = {"seq": s, "frame": fr, "cheap": pk(bxc, dc["conf"][kc], dc["coarse"][kc], cons_c),
                         "u960": pk(dd9["xyxy"], d9["conf"][k9], dd9["coarse"], t9),
                         "spec": pk(scene["xyxy"], scene["conf"], scene["coarse"], scene["mono"])}
                for name, rt in refined_by.items():
                    entry[f"spec_ref_{name}"] = pk(scene["xyxy"], scene["conf"], scene["coarse"], rt)
                for name in refiners:
                    entry.setdefault(f"spec_ref_{name}", entry["spec"])
                dumps.append(entry)
            prev_act, prev_det, prev_scene = a_c, {"xyxy": dc["xyxy"][kc]}, {"xyxy": scene["xyxy"]}
        print(f"  {s} done", flush=True)
    d = pd.DataFrame(rows).merge(tab[["seq", "frame", "brake_J_c", "brake_J_f", "brake_V"]], on=["seq", "frame"])
    if args.cheap_dir == "nusc_det_tv":
        assert np.allclose(d.J_c, d.brake_J_c, atol=1e-9)
    d.to_pickle(out / f"eval__{args.set}_{args.split}{args.tag}.pkl")
    if args.dump:
        pd.to_pickle(dumps, out / f"scenes__{args.set}_{args.split}{args.tag}.pkl")
    units = d.seq.to_numpy(); uq = np.unique(units); idx = [np.flatnonzero(units == u) for u in uq]
    L = d.J_c.to_numpy()
    G = {"uniform 640": d.brake_V.to_numpy(), "uniform 960": (d.J_c - d.J_u960).to_numpy(),
         "speculative": (d.J_c - d.J_spec).to_numpy()}
    G.update({f"speculative + refiner {n}": (d.J_c - d[f"J_spec_ref_{n}"]).to_numpy() for n in refiners})
    b = np.random.default_rng(0)
    draws = []
    for _ in range(2000):
        tt = np.concatenate([idx[i] for i in b.integers(0, len(uq), len(uq))])
        if L[tt].sum() > 0:
            draws.append({k: g[tt].sum() / L[tt].sum() for k, g in G.items()})
    D = pd.DataFrame(draws)
    res = pd.DataFrame([{"method": k, "G": g.sum() / L.sum(),
                         "minus_u640_lo": np.percentile(D[k] - D["uniform 640"], 2.5),
                         "minus_u640_hi": np.percentile(D[k] - D["uniform 640"], 97.5),
                         "minus_u960_lo": np.percentile(D[k] - D["uniform 960"], 2.5),
                         "minus_u960_hi": np.percentile(D[k] - D["uniform 960"], 97.5)} for k, g in G.items()])
    res.to_csv(out / f"eval_summary__{args.set}_{args.split}{args.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"scenes {len(uq)}, frames {len(d)}, crops/frame {d.n_crops.mean():.2f}, pixels {d.px.mean():.4f}, rounds {d.rounds.mean():.2f}")
    print(res.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
