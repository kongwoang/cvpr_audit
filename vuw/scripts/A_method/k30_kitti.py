#!/usr/bin/env python
"""K30 V8c + the K28 KITTI flag re-run (post-hoc; VUW, torch YOLOv8s; pre-registered in handoff/K30/STATUS.md).
DEEP's KITTI test split (6 tracking sequences, the frames of DEEP's decision table), K28's shared v6-S configuration as
61_k28_v6_latency.py --dataset kitti: draft YOLOv8s 96x320; v6-S = vtf.speculative.run (field + support, v_close 0, <= 3
rounds, guard) with crops at 960 / 1242 of native, C1 lift on its own same-scale history; refiner v3a on the braking
support; whole frame u960 (288x960) and native (384x1248), each plain, + T4 + refiner, + C1 + refiner. Every refiner is
applied twice: with KITTI's camera flag 0.0 (as v3a was trained) and with 1.0 (what K28's Jetson run fed, the bug of
handoff/K28/phase3 section 16). Per frame and pipeline: DEEP's four KITTI cells (brake mono / brake oracle / Planner B mono /
oracle, as 61's cells(): cheap action / plan histories) and the outputs for KITTI 2D AP (boxes with conf >= 0.1: draft,
whole frames, and v6-S as the K7 definition).
results/k30/kitti/<seq>.pkl = {"rows": [...], "dets": {(frame, pipeline): {xyxy, conf, coarse}}}
usage: k30_kitti.py [--max_units n] [--units 0007,...]
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", B))
_th = importlib.util.spec_from_file_location("k28t", B / "scripts" / "A_method" / "k28_ttrain.py")
KTT = importlib.util.module_from_spec(_th); _th.loader.exec_module(KTT)
_td = importlib.util.spec_from_file_location("k28d", B / "scripts" / "A_method" / "k28_tdata.py")
KTD = importlib.util.module_from_spec(_td); _td.loader.exec_module(KTD)
_k = importlib.util.spec_from_file_location("k27", B / "scripts" / "A_method" / "k27_dev.py")
K27 = importlib.util.module_from_spec(_k); _k.loader.exec_module(K27)
IE, KT = K27.IE, K27.KT
ttc_ss, parse = K27.ttc_ss, K27.parse
from rap import planner as P                                                     # noqa: E402
from rap import planner_b as PB                                                  # noqa: E402
from rap.cache import Camera                                                     # noqa: E402
from rap.decision import KittiAdapter, _apply_range_source                       # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import box_iou, predicted_geometry                                 # noqa: E402
from rap.paths import KITTI_IMAGES                                               # noqa: E402
from vtf import speculative as SP                                                # noqa: E402

CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
LOW, THR = 0.1, IE.THR
TPOL = "T4+T2_1"
CH = Mode("k_c96x320", 96, 320)
WF = {"u960": (Mode("k_u960", 288, 960), 960 / 1242), "native": (Mode("k_native", 384, 1248), 1.0)}
SCALE = 960 / 1242
FLAGS = (0.0, 1.0)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_units", type=int, default=0)
    ap.add_argument("--units", default="")
    args = ap.parse_args()
    sys.path.insert(0, str(B / "scripts" / "common"))
    from nusc_refiner_crop import crop64
    dev = torch.device("cuda:0")
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    ck = torch.load(REAL / "out" / "resnet18_v3a" / "image_refiner.pt", map_location=dev, weights_only=False)
    net = KT.Res18x3(len(ck["fmu"])); net.load_state_dict(ck["state"]); net = net.to(dev).eval(); fmu, fsd = ck["fmu"], ck["fsd"]
    ckh = torch.load(REAL / "out" / "k28_temporal" / "temporal_head.pt", map_location=dev, weights_only=False)
    head = KTT.TempHead(ckh["n_feat"]); head.load_state_dict(ckh["state"]); head = head.to(dev).eval()
    hmu, hsd = ckh["fmu"], ckh["fsd"]
    pp, cp = P.PlannerParams(), P.CostParams()
    pbp, pbc = PB.PARAMS_B["static_obstacles"], PB.COSTS_B["default"]
    dv = pd.read_csv(Path(os.environ["RAP_REPO"]) / "results" / "final" / "benchmark_decision_values.csv.gz",
                     dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[(dv.track == "KITTI") & (dv.geometry == "mono") & (dv.system == "brake") & (dv.split == "test")].copy()
    dv["frame"] = dv.frame.astype(int)
    od = REAL / "results" / "k30" / "kitti"
    od.mkdir(parents=True, exist_ok=True)
    units = [u for u in args.units.split(",") if u]
    for ui, s in enumerate(sorted(set(dv.seq))):
        if (args.max_units and ui >= args.max_units) or (units and s not in units) or (od / f"{s}.pkl").exists():
            continue
        geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s)
        cam = Camera("KITTI", s)
        R, t = cam.transform("ego")
        dt = float(cam.dt)
        rng = np.random.default_rng(0)
        st, rows, dets, imgs = {}, [], {}, []
        prev_act, prev_act_o, prev_plan, prev_plan_o, prev_c = None, None, None, None, None
        for fr in sorted(int(f) for f in dv[dv.seq == s].frame):
            img = cv2.imread(str(Path(KITTI_IMAGES) / s / f"{fr:06d}.png"))
            if img is None:
                continue
            H, W = img.shape[:2]
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            gt_t = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc")) if len(g) else tuple(np.zeros(0) for _ in range(4))
            a_gt, _ = P.required_decel(*gt_t, v, pp) if len(g) else (0.0, None)
            rec = {"unit": s, "frame": fr, "v": v, "act_gt": P.discrete_action(a_gt, pp)}

            def geomf(dd, pv=None):
                gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def keep(d, thr=THR):
                k = d["conf"] >= thr
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}

            def act(tt):
                return P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)

            def cells(nm, xyxy, tt):
                """61.cells: brake mono (J), brake oracle (Jbo), Planner B mono (Jtm) / oracle (Jto), cheap histories"""
                a = act(tt)
                rec[f"a_{nm}"], rec[f"J_{nm}"] = a, P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]
                geo = dict(zip(("z", "lat_min", "lat_max", "ttc"), (np.asarray(q, float) for q in tt)))
                to = tuple(np.asarray(q, float) for q in _apply_range_source(np.asarray(xyxy, float).reshape(-1, 4), geo, g, "gt", rng, 0.0))
                rec[f"Jbo_{nm}"] = P.decision_cost(act(to), a_gt, prev_act_o, pp, cp)["J"]
                for tag, t_, pv in (("Jtm", tt, prev_plan), ("Jto", to, prev_plan_o)):
                    rec[f"{tag}_{nm}"] = PB.executed_cost(PB.plan(*t_, v, pv, pbp, pbc), *gt_t, v, pv, pbp, pbc)["J"]
                return to

            def push(name, xyxy):
                h = st.setdefault(name, {})
                h["s2"] = h.get("s1")
                h["s1"] = np.asarray(xyxy, float).reshape(-1, 4)

            def lift_t4(name, sc):
                h = st.get(name, {})
                pol = parse(TPOL)

                def f(full):
                    z, lo, hi, _ = geomf(full)
                    tt, _ = ttc_ss(full["xyxy"], h.get("s1"), h.get("s2"), sc, dt, pol["gate"], pol["t4"], None)
                    return z, lo, hi, tt
                return f

            def lift_c1(name, sc):
                h = st.get(name, {})

                def f(full):
                    X = np.asarray(full["xyxy"], float).reshape(-1, 4)
                    z, lo, hi, _ = geomf(full)
                    ttc = np.full(len(X), np.inf)
                    chn = KTD.chains(X, h.get("s1"), h.get("s2")) if len(imgs) == 2 else []
                    if not chn:
                        return z, lo, hi, ttc
                    idx = np.array([c[0] for c in chn])
                    zz = predicted_geometry({"xyxy": X[idx], "coarse": np.asarray(full["coarse"])[idx]}, None, cam.calib, cam.cam_h, dt,
                                            cam_to_ego=(R, t))
                    cr, fe = [], []
                    for n, (i, j1, j2) in enumerate(chn):
                        b1, b2 = h["s1"][j1], h["s2"][j2]
                        x1, y1, x2, y2 = X[i]
                        zg, zh = float(zz["z_ground"][n]), float(zz["z_height"][n])
                        hh = (y2 - y1, b1[3] - b1[1], b2[3] - b2[1])
                        cr.append(np.stack([crop64(img, X[i]), crop64(imgs[-1], b1), crop64(imgs[-2], b2)]))
                        fe.append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(full["conf"][i]),
                                   float(CLS[str(full["coarse"][i])]), float(z[i]), zg, zh, float(np.log(max(zg, .5) / max(zh, .5))),
                                   float(np.log(max(hh[0] / max(hh[1], 1e-3), 1e-3))), float(np.log(max(hh[1] / max(hh[2], 1e-3), 1e-3))), dt, 1.0])
                    xin = torch.from_numpy(np.stack(cr)).permute(0, 1, 4, 2, 3).float().to(dev) / 255.0
                    fin = torch.from_numpy(((np.array(fe, np.float32) - hmu) / hsd).astype(np.float32)).to(dev)
                    with torch.no_grad():
                        rh = np.clip(head(xin, fin).float().cpu().numpy(), 0, 2)
                    for n, (i, _, _) in enumerate(chn):
                        ttc[i] = min(1 / rh[n], 1e3) if rh[n] > 1e-3 / dt else np.inf
                    return z, lo, hi, ttc
                return f

            def refine(scene, flag):
                tt = [np.asarray(a, float).copy() for a in scene["mono"]]
                sup = [p_ for p_ in SP.support(scene["mono"], v, pp) if scene["from_full"][p_]]
                if not sup:
                    return tuple(tt)
                zz = predicted_geometry({"xyxy": scene["xyxy"][sup], "coarse": scene["coarse"][sup]}, None, cam.calib, cam.cam_h,
                                        dt, cam_to_ego=(R, t))
                Fm = []
                for n, p_ in enumerate(sup):
                    x1, y1, x2, y2 = scene["xyxy"][p_]
                    zg, zh = float(zz["z_ground"][n]), float(zz["z_height"][n])
                    Fm.append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(scene["conf"][p_]),
                               float(CLS[str(scene["coarse"][p_])]), float(scene["mono"][0][p_]), zg, zh,
                               float(np.log(max(zg, .5) / max(zh, .5))), flag])
                Fm = ((np.array(Fm, np.float32) - fmu) / fsd).astype(np.float32)
                x = torch.from_numpy(np.stack([crop64(img, scene["xyxy"][p_]) for p_ in sup])).permute(0, 3, 1, 2).float().to(dev) / 255.0
                with torch.no_grad():
                    o = net(x, torch.from_numpy(Fm).to(dev)).float().cpu().numpy().reshape(len(sup), -1)
                for n, p_ in enumerate(sup):
                    c = float(np.exp(o[n, 0]))
                    for a in tt[:3]:
                        a[p_] *= c
                return tuple(tt)
            # ---- draft (online; K28's Jetson run detected the draft online too)
            raw_c = det.detect(img, CH)[0]
            ch = keep(raw_c); ch["mono"] = geomf(ch, prev_c)
            dets[(fr, "cheap")] = keep(raw_c, LOW)
            to_c = cells("cheap", ch["xyxy"], ch["mono"])
            # ---- whole frames
            for nm, (md, ms) in WF.items():
                raw = det.detect(img, md)[0]
                d = keep(raw)
                dets[(fr, nm)] = keep(raw, LOW)
                cells(nm, d["xyxy"], geomf(d, st.get(nm, {}).get("prev")))
                sc4 = {**d, "mono": lift_t4(nm, ms)(d), "from_full": np.ones(len(d["conf"]), bool)}
                sc1 = {**d, "mono": lift_c1(nm, ms)(d), "from_full": np.ones(len(d["conf"]), bool)}
                for f_ in FLAGS:
                    cells(f"{nm}+T4+ref_f{f_:g}", d["xyxy"], refine(sc4, f_))
                    cells(f"{nm}+C1+ref_f{f_:g}", d["xyxy"], refine(sc1, f_))
                cells(f"{nm}+C1", d["xyxy"], sc1["mono"])
                st.setdefault(nm, {})["prev"] = {"xyxy": d["xyxy"], "coarse": d["coarse"]}
                push(nm, d["xyxy"])
            # ---- v6-S (61: SP.run, v_close 0, <= 3 rounds, guard; crops at 960 / 1242; C1 lift)
            lowc, klog, tpx = {}, [], [0.0]

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                nw = max(32, int(math.ceil((min(x2, W) - x1) * SCALE / 32)) * 32)
                nh = max(32, int(math.ceil((min(y2, H) - y1) * SCALE / 32)) * 32)
                tpx[0] += nh * nw
                raw = det.detect(img[y1:y2, x1:x2], Mode(f"c{nh}x{nw}", nh, nw))[0]
                lo_ = keep(raw, LOW); lo_["xyxy"] = lo_["xyxy"] + np.array([x1, y1, x1, y1], float)
                klog.append((np.array([x1, y1, x2, y2], float), lo_))
                d = keep(raw)
                return {"xyxy": d["xyxy"] + np.array([x1, y1, x1, y1], float), "conf": d["conf"], "coarse": d["coarse"]}
            base = {k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}
            scene, crops, rounds = SP.run(base, v, cam, W, H, detect_crop, lift_c1("v6S", SCALE), pp, v_close=0.0, max_rounds=3, guard=True)
            cells("v6S_noref", scene["xyxy"], scene["mono"])
            for f_ in FLAGS:
                cells(f"v6S_f{f_:g}", scene["xyxy"], refine(scene, f_))
            rec.update({"px_v6S": (CH.net_h * CH.net_w + tpx[0]) / (W * H), "ncrops_v6S": len(crops), "rounds_v6S": rounds})
            lc = dets[(fr, "cheap")]
            regs = [r_ for r_, _ in klog]
            outside = ~SP.centres_in(lc["xyxy"], regs) if regs else np.ones(len(lc["conf"]), bool)
            full = {k_: np.concatenate([np.asarray(l_[k_]) for _, l_ in klog]) if klog else np.zeros((0, 4) if k_ == "xyxy" else 0)
                    for k_ in ("xyxy", "conf", "coarse")}
            if len(full["conf"]) > 1:
                o_ = np.argsort(-full["conf"], kind="stable"); iou_ = box_iou(full["xyxy"], full["xyxy"]); kp = []
                for i_ in o_:
                    if not kp or iou_[i_, kp].max() < 0.5:
                        kp.append(i_)
                full = {k_: full[k_][np.sort(np.array(kp))] for k_ in full}
            dets[(fr, "v6S")] = {k_: np.concatenate([np.asarray(lc[k_])[outside], np.asarray(full[k_])]) for k_ in ("xyxy", "conf", "coarse")}
            ff = np.asarray(scene["from_full"], bool)
            push("v6S", np.asarray(scene["xyxy"], float).reshape(-1, 4)[ff])
            # ---- histories (61: cheap action / plan histories)
            prev_act = rec["a_cheap"]
            prev_act_o = act(to_c)
            prev_plan = PB.plan(*ch["mono"], v, prev_plan, pbp, pbc)
            prev_plan_o = PB.plan(*to_c, v, prev_plan_o, pbp, pbc)
            prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            imgs = (imgs + [img])[-2:]
            rows.append(rec)
        with open(od / f"{s}.pkl", "wb") as fh:
            pickle.dump({"rows": rows, "dets": dets}, fh)
        print(f"  [{ui}] {s}: {len(rows)} frames", flush=True)
    print("K30 KITTI DONE", flush=True)


if __name__ == "__main__":
    main()
