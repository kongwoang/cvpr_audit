#!/usr/bin/env python
"""K28 PHASE 3 (post-hoc, reporting only; v6 frozen in method_freeze_draft §2n, nothing here selects anything).
Code = k28_test.py (md5 acb53791, unchanged) copied, with:
  --part instr      instrumented re-run of v6-S on the test units (K2 agreement, K4 missed objects, K9 second consumer,
                    K16 attribution chain, K17 regime flag): per frame the draft, YOLOv8s whole frame 640 / 960 / native
                    (plain and + C1, own history), v5.1-S (budget of the S heuristics), v6-S, support only / field only
                    of v6-S, the 9 S heuristics + C1; base setting also the three K3 rows of phase 0 (1) for v6-S:
                    pivots (pv6S), one round (or6S), no guard (ng6S); for every pipeline action / cost (braking and TTC warning consumer)
                    and which decision-relevant annotated objects its scene matches (instr_eval.matched). Base setting
                    also: ours driven by the TTC-warning consumer (spec_general, + C1) and uncertainty crops at its
                    tensor px; the K16 counterfactual chain (crop detector replaced by F = u960 + C1) and coverage.
  --setting del_all (instr only) K4 B: the draft's matched detections of decision-relevant annotated objects are
                    deleted before every policy runs (from the draft and from its history), as instr_eval.
  --ver W           detector of the crops, whole frame and heuristics (models/W.pt or a path under B; default yolov8s)
  --draft W         draft detector (default yolov8s; anything else runs the draft online on nuScenes too)
  --only_s          part ours: v5.1-S and v6-S only (v6-S's crop detector = --ver)
With the defaults, parts ours / basewf / baseheur are k28_test.py's (parity smoke).
results/k28/p3/<ds>_<split>_<part>[-<setting>][_<tag>]/<unit>.pkl
usage: k28_p3.py --dataset nusc|av2|ahd --part ours|basewf|baseheur|instr [--setting base|del_all] [--ver W] [--draft W]
                 [--only_s] [--split test|dev] [--det yolo11m|yolov8m|""] [--wf ...] [--configs ...] [--tag t]
                 [--shard i/n] [--max_units n]
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pickle
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
_th = importlib.util.spec_from_file_location("k28t", B / "scripts" / "A_method" / "k28_ttrain.py")
KTT = importlib.util.module_from_spec(_th); _th.loader.exec_module(KTT)
_td = importlib.util.spec_from_file_location("k28d", B / "scripts" / "A_method" / "k28_tdata.py")
KTD = importlib.util.module_from_spec(_td); _td.loader.exec_module(KTD)
_k = importlib.util.spec_from_file_location("k27", B / "scripts" / "A_method" / "k27_dev.py")
K27 = importlib.util.module_from_spec(_k); _k.loader.exec_module(K27)
_k24 = importlib.util.spec_from_file_location("k24t", B / "scripts" / "A_method" / "k24_test.py")
K24 = importlib.util.module_from_spec(_k24); _k24.loader.exec_module(K24)
IE, KL, KT = K27.IE, K27.KL, K27.KT
ttc_ss, parse = K27.ttc_ss, K27.parse
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
from vtf import dsf                                                              # noqa: E402
from vtf import spec_general as SG                                               # noqa: E402
from vtf import speculative as SP                                                # noqa: E402
from rap.mono import box_iou                                                     # noqa: E402

CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
TPOL = "T4+T2_1"
CAPS = {"S": 1.0, "M": 1.5, "L": 2.5}
# family A: name -> (lift arm, v_close mode, value, ctx, crop long side, max_calls)
A = {"v51S": (TPOL, "c", 0.0, 2.0, 960, None),
     "vc5": (TPOL, "c", 5.0, 2.0, 960, None), "vc10": (TPOL, "c", 10.0, 2.0, 960, None),
     "va0.25": (TPOL, "a", 0.25, 2.0, 960, None), "va0.5": (TPOL, "a", 0.5, 2.0, 960, None),
     "va1": (TPOL, "a", 1.0, 2.0, 960, None), "T3": ("T3+T4+T2_1", "c", 0.0, 2.0, 960, None),
     "ctx2.5": (TPOL, "c", 0.0, 2.5, 960, None), "ctx3": (TPOL, "c", 0.0, 3.0, 960, None),
     "s1280": (TPOL, "c", 0.0, 2.0, 1280, None), "N2": (TPOL, "c", 0.0, 2.0, 960, 2), "N3": (TPOL, "c", 0.0, 2.0, 960, 3)}
UNC = [(c, o) for c in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0) for o in ("conf", "ent")]
TCR = [(g, n) for g in (0.0, 0.25, 0.5, 1.0) for n in (0, 3, 1)]                 # n = 0: all previous objects
TUBE = [(near, far, w) for near in (5.0, 10.0) for far in (40.0, 70.0, 100.0) for w in ("fill", "lane")]
WF = {"nusc": (448, 512, 576, 640, 768, 896, 960, 1088, 1280, 1408, 1536, 1600),
      "av2": (512, 640, 768, 896, 960, 1088, 1280, 1408, 1536, 1664, 1792, 2048)}
VER_WF = {"nusc": (448, 640, 768, 960, 1280, 1600), "av2": (512, 640, 896, 1280, 1536, 2048),
          "ahd": (512, 640, 896, 1280, 1536, 1920)}
NWS = {"nusc": 1600, "av2": 2048, "ahd": 1920}
# the frozen v6 configurations (§2m / §2n): crop detector, crop long side, temporal cue, refiner
V6 = {"v6S": ("yolov8s", 960, "C1", "v3a"), "v6M": ("yolov8m", 768, "C1", "own-yolov8m"), "v6L": ("yolo11m", 960, "C1", "own-yolo11m")}
# phase 3: the 9 S heuristics of §2n, F passes of the instrumented run, the K16 chain (k16_attr.py)
CFG_S = ("unc2@S", "unc4@S", "tcr0@S", "tcr3@S", "cor0@S", "cor8@S", "tsp2@S", "tsp8@S", "rnd@S")
F_SIDES = (640, 960, "native")
R_EXT = np.array([8.0, 12.5, 17.5, 22.5, 27.5, 35.0, 45.0, 55.0, 70.0, 80.0])
Y_EXT = np.arange(-9.0, 9.01, 1.5)
CHAIN = (("a", {}, False), ("b", {}, True), ("c1", dict(v_closes=(10.0,)), True), ("c2", dict(v_closes=(20.0,)), True),
         ("d", dict(v_closes=(20.0,), r_grid=R_EXT, y_grid=Y_EXT), True),
         ("e", dict(v_closes=(20.0,), r_grid=R_EXT, y_grid=Y_EXT, probes=(KL.PED, KL.CAR, KL.TRUCK)), True),
         ("f", dict(v_closes=(20.0,), r_grid=R_EXT, y_grid=Y_EXT, probes=(KL.PED, KL.CAR, KL.TRUCK), max_rounds=10), True))


def wpath(w):
    return B / w if "/" in w else B / "models" / f"{w}.pt"


class _Proxy:
    """vtf.speculative with support() replaced (call-scoped, k_loop's reference only)"""
    def __init__(self, mod, fn):
        self._m, self._fn = mod, fn

    def support(self, *a, **k):
        return self._fn(*a, **k)

    def __getattr__(self, n):
        return getattr(self._m, n)


def wf_mode(ds, L):
    """whole-frame mode at long side L (the K24 modes where they exist; else height / width rounded up to 32)"""
    if ds == "ahd":
        return Mode(f"ahd_u{L}", int(math.ceil(L * 1200 / 1920 / 32)) * 32, L), L / 1920
    if ds == "nusc":
        k24 = {640: (384, 640), 960: (576, 960), 1280: (736, 1280), 1536: (864, 1536), 1600: (928, 1600)}
        h, w = k24.get(L, (int(math.ceil(L * 900 / 1600 / 32)) * 32, L))
        return Mode(f"ng{h}x{w}", h, w), L / 1600
    k24 = {640: (640, 512), 960: (960, 768), 1280: (1280, 992), 1536: (1536, 1184), 2048: (2048, 1568)}
    h, w = k24.get(L, (L, int(math.ceil(L * 1550 / 2048 / 32)) * 32))
    return Mode(f"ag{h}x{w}", h, w), L / 2048


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2", "ahd"])
    ap.add_argument("--part", required=True, choices=["ours", "basewf", "baseheur", "instr"])
    ap.add_argument("--setting", default="base", choices=["base", "del_all"])
    ap.add_argument("--ver", default="yolov8s")
    ap.add_argument("--draft", default="yolov8s")
    ap.add_argument("--only_s", action="store_true")
    ap.add_argument("--capture", default="", help="instr: csv with unit, frame; only these units run, and these frames "
                                                   "also store the geometry for the panels (rec['cap'])")
    ap.add_argument("--det", default="yolo11m")
    ap.add_argument("--configs", default="")
    ap.add_argument("--hdet", default="yolov8s")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--wf", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max_units", type=int, default=0)
    ap.add_argument("--seeds", action="store_true", help="ours: also score with the seed-1 / seed-2 refiners of each v6 "
                    "configuration (v3a_s1 / _s2, own-<det>_s1 / _s2; family D of §2k-§2m), for S6 on the test units")
    args = ap.parse_args()
    ds, part = args.dataset, args.part
    si, sn = (int(x) for x in args.shard.split("/"))
    sys.path.insert(0, str(B / "scripts" / "common"))
    from nusc_refiner_crop import crop64
    dev = torch.device("cuda:0")
    det = TwoFidelityDetector(str(wpath(args.ver)), device="cuda:0", backend="torch")
    online = args.draft != "yolov8s"
    ddet = det if args.draft == args.ver else TwoFidelityDetector(str(wpath(args.draft)), device="cuda:0", backend="torch")
    vname = (args.det or "yolov8s") if part != "baseheur" else args.hdet
    vdet = TwoFidelityDetector(str(B / "models" / f"{vname}.pt"), device="cuda:0", backend="torch") if vname != "yolov8s" and part in ("basewf", "baseheur") else det
    VD = sorted({c[0] for c in V6.values()} - {"yolov8s"}) if part == "ours" and not args.only_s else []
    dets = {n: TwoFidelityDetector(str(B / "models" / f"{n}.pt"), device="cuda:0", backend="torch") for n in VD}
    nets = {}

    def load_net(key, path):
        ck = torch.load(path, map_location=dev, weights_only=False)
        n3 = KT.Res18x3(len(ck["fmu"])); n3.load_state_dict(ck["state"]); nets[key] = (n3.to(dev).eval(), ck["fmu"], ck["fsd"])
    load_net("v3a", REAL / "out" / "resnet18_v3a" / "image_refiner.pt")
    ckh = torch.load(REAL / "out" / "k28_temporal" / "temporal_head.pt", map_location=dev, weights_only=False)
    head = KTT.TempHead(ckh["n_feat"]); head.load_state_dict(ckh["state"]); head = head.to(dev).eval()
    hmu, hsd = ckh["fmu"], ckh["fsd"]
    if vname != "yolov8s" and part in ("basewf", "baseheur"):
        load_net("own", REAL / "out" / f"resnet18_v3a_{vname}" / "image_refiner.pt")
    for n in VD:
        load_net(f"own-{n}", REAL / "out" / f"resnet18_v3a_{n}" / "image_refiner.pt")
    if args.seeds and part == "ours":
        for sd in (1, 2):
            load_net(f"v3a_s{sd}", REAL / "out" / f"resnet18_v3a_s{sd}" / "image_refiner.pt")
            for n in VD:
                load_net(f"own-{n}_s{sd}", REAL / "out" / f"resnet18_v3a_{n}_s{sd}" / "image_refiner.pt")
    WFS = [NWS[ds] if x == "native" else int(x) for x in args.wf.split(",") if x]
    CONFIGS = [c for c in args.configs.split(",") if c]
    pp, cp = P.PlannerParams(), P.CostParams()
    if ds == "nusc":
        CH, cheap_px, NW = Mode("ns_c320", 192, 320), 192 * 320, 1600
    elif ds == "av2":
        CH, cheap_px, NW = Mode("av2_c320", 320, 256), 320 * 256, 2048
    else:
        CH, cheap_px, NW = Mode("ahd_c320", 224, 320), 224 * 320, 1920
    assert args.split == "test" or ds != "ahd"
    od = REAL / "results" / "k28" / "p3" / (f"{ds}_{args.split}_{part}{'' if part in ('ours', 'instr') or vname == 'yolov8s' else '-' + vname}"
                                            + (f"-{args.setting}" if args.setting != "base" else "") + (f"_{args.tag}" if args.tag else ""))
    assert part == "instr" or args.setting == "base"
    V6R = {"v6S": V6["v6S"]} if args.only_s else V6
    od.mkdir(parents=True, exist_ok=True)
    CAP, FLOG = set(), []
    if args.capture:
        import pandas as pd
        CAP = {(str(r.unit), int(r.frame)) for r in pd.read_csv(args.capture).itertuples() if r.dataset == ds}
        _fcg = KL.field_crop_g

        def _fc_log(*a, **k):                              # logs the field crops the loop computes (panels only)
            r_ = _fcg(*a, **k)
            if r_ is not None:
                FLOG.append(np.asarray(r_, float))
            return r_
        KL.field_crop_g = _fc_log
    for ui, (unit, cam, gen) in enumerate((K24.units if args.split == "test" else K27.units)(ds, args.max_units)):
        if ui % sn != si or (CAP and unit not in {u for u, _ in CAP}):
            continue
        fo = od / f"{unit}.pkl"
        if fo.exists():
            continue
        R, t = cam.transform("ego")
        P2m = cam.calib.P2
        fx, fy, cx, cy = P2m[0, 0], P2m[1, 1], P2m[0, 2], P2m[1, 2]
        dt = float(cam.dt)
        st = {}
        prev_act, prev_c, imgs, rows = None, None, [], []
        for fr, path, v, g, cached in gen():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            NATIVE = W * H
            a_gt, a_gt_obj = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, np.zeros(0))
            rec = {"unit": unit, "frame": fr, "v": v, "gt": not isinstance(cached, str), "act_gt": P.discrete_action(a_gt, pp)}
            if part == "instr":
                gt_t = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc")) if len(g) else tuple(np.zeros(0) for _ in range(4))
                act2_gt = IE.c2_action(gt_t, pp)
                gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
                gtc = np.asarray(g["coarse"]).astype(str) if len(g) else np.zeros(0, str)
                relevant = np.flatnonzero(np.asarray(a_gt_obj) >= pp.thr_decel) if len(g) else np.zeros(0, int)
                g_ln = np.asarray(g["long_near"], float) if len(g) else np.zeros(0)
                rec.update({"act2_gt": act2_gt, "n_relevant": len(relevant), "rel_range": [float(g_ln[k]) for k in relevant]})

            def geom(dd, pv=None):
                gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def keep(d):
                k = d["conf"] >= IE.THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k],
                        "entropy": np.asarray(d.get("entropy", np.zeros(len(d["conf"]))))[k]}

            def act(tt):
                return P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)

            def cost(a):
                return P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]

            def ckey(cr, scale):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                xe, ye = min(x2, W), min(y2, H)
                nw = max(32, int(math.ceil((xe - x1) * scale / 32)) * 32)
                nh = max(32, int(math.ceil((ye - y1) * scale / 32)) * 32)
                return (x1, y1, x2, y2, nh, nw)
            cache, rcache, last, capH, Fdet = {}, {}, {}, {}, {}

            def run_det(dt_, im, key):
                x1, y1, x2, y2, nh, nw = key
                d = keep(dt_.detect(im[y1:y2, x1:x2], Mode(f"c{nh}x{nw}", nh, nw))[0])
                return {"xyxy": d["xyxy"] + np.array([x1, y1, x1, y1], float), "conf": d["conf"], "coarse": d["coarse"]}

            def make_detect(tpx, scale, log=None, dt_=None, cnt=None):
                dt_ = dt_ or det

                def detect_crop(cr):
                    key = ckey(cr, scale)
                    tpx[0] += key[4] * key[5]
                    if cnt is not None:
                        cnt[0] += 1
                    ck = (id(dt_),) + key
                    if ck not in cache:
                        cache[ck] = run_det(dt_, img, key)
                    out = {k: x.copy() for k, x in cache[ck].items()}
                    if log is not None:
                        for b in out["xyxy"]:
                            log[tuple(b.tolist())] = key
                    return out
                return detect_crop

            def make_retro(log, used, dt_=None):
                dt_ = dt_ or det

                def retro(box, lag):
                    key = log.get(tuple(np.asarray(box, float).tolist()))
                    if key is None or len(imgs) < lag:
                        return None
                    rk = (lag, id(dt_)) + key
                    used.add(rk)
                    if rk not in rcache:
                        rcache[rk] = run_det(dt_, imgs[-lag], key)["xyxy"]
                    return rcache[rk]
                return retro

            def lift_pol(name, pol, sc, retro=None):
                h = st.get(name, {})

                def f(full):
                    if pol is None:
                        return geom(full, {"xyxy": h["merged"]} if "merged" in h else None)
                    z, lo, hi, _ = geom(full)
                    tt, _ = ttc_ss(full["xyxy"], h.get("s1"), h.get("s2"), sc, dt, pol["gate"], pol["t4"],
                                   retro if pol["t3"] else None)
                    return z, lo, hi, tt
                return f

            def lift_c(name, mode, sc):
                """C1 / C2 lift of the detections of pipeline `name` (its same-scale history s1 / s2)"""
                h = st.get(name, {})

                def f(full):
                    X = np.asarray(full["xyxy"], float).reshape(-1, 4)
                    z, lo, hi, _ = geom(full)
                    t4, _ = ttc_ss(X, h.get("s1"), h.get("s2"), sc, dt, 1.0, True)
                    ttc = np.full(len(X), np.inf) if mode == "C1" else t4.copy()
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
                    for n, (i, j1, j2) in enumerate(chn):
                        if mode == "C1":
                            r = rh[n]
                        else:
                            r4 = 1 / t4[i] if np.isfinite(t4[i]) and t4[i] > 0 else 0.0
                            r = (rh[n] + r4) / 2
                            hh0, hh2 = X[i][3] - X[i][1], h["s2"][j2][3] - h["s2"][j2][1]
                            if not (hh0 - hh2) * sc > 1.0:
                                ttc[i] = np.inf
                                continue
                        ttc[i] = min(1 / r, 1e3) if r > 1e-3 / dt else np.inf
                    return z, lo, hi, ttc
                return f

            def push(name, scene):
                h = st.setdefault(name, {})
                ff = np.asarray(scene["from_full"], bool)
                h["s2"] = h.get("s1")
                h["s1"] = np.asarray(scene["xyxy"], float).reshape(-1, 4)[ff]
                h["merged"] = np.asarray(scene["xyxy"], float).reshape(-1, 4)

            def refine(scene, name):
                tt = [np.asarray(a, float).copy() for a in scene["mono"]]
                if name == "none":
                    return tuple(tt)
                sup = [p_ for p_ in SP.support(scene["mono"], v, pp) if scene["from_full"][p_]]
                if not sup:
                    return tuple(tt)
                net, fmu, fsd = nets[name]
                zz = predicted_geometry({"xyxy": scene["xyxy"][sup], "coarse": scene["coarse"][sup]}, None, cam.calib, cam.cam_h,
                                        dt, cam_to_ego=(R, t))
                Fm = []
                for n, p_ in enumerate(sup):
                    x1, y1, x2, y2 = scene["xyxy"][p_]
                    zg, zh = float(zz["z_ground"][n]), float(zz["z_height"][n])
                    Fm.append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(scene["conf"][p_]),
                               float(CLS[str(scene["coarse"][p_])]), float(scene["mono"][0][p_]), zg, zh,
                               float(np.log(max(zg, .5) / max(zh, .5))), 1.0])
                Fm = ((np.array(Fm, np.float32) - fmu) / fsd).astype(np.float32)
                x = torch.from_numpy(np.stack([crop64(img, scene["xyxy"][p_]) for p_ in sup])).permute(0, 3, 1, 2).float().to(dev) / 255.0
                with torch.no_grad():
                    o = net(x, torch.from_numpy(Fm).to(dev)).float().cpu().numpy().reshape(len(sup), -1)
                for n, p_ in enumerate(sup):
                    c = float(np.exp(o[n, 0]))
                    for a in tt[:3]:
                        a[p_] *= c
                return tuple(tt)

            def score(prefix, scene, refs=("none", "v3a")):
                for r_ in refs:
                    tr_ = refine(scene, r_)
                    a1 = act(tr_)
                    rec[f"a_{prefix}|{r_}"], rec[f"J_{prefix}|{r_}"] = a1, cost(a1)
                    if part == "instr":
                        a2 = IE.c2_action(tr_, pp)
                        rec[f"a2_{prefix}|{r_}"], rec[f"J2_{prefix}|{r_}"] = a2, float(IE.C2_COST[act2_gt, a2])
                if part == "instr" and len(relevant):
                    rec[f"found_{prefix}"] = IE.matched(gtb[relevant], gtc[relevant], scene["xyxy"], scene["coarse"]) >= 0

            def scene_dict(dd, mono):
                return {"xyxy": dd["xyxy"], "conf": dd["conf"], "coarse": dd["coarse"], "mono": mono,
                        "from_full": np.ones(len(dd["conf"]), bool)}

            def loop(nm, arm, vmode, vval, ctx, side, max_calls, dt_=None, refs=("none", "v3a"), sup_fn=None, cmode=None, **kw):
                scale = side / NW
                tpx, log, used, cnt = [0.0], {}, set(), [0]
                vc = vval if vmode == "c" else vval * v
                if sup_fn is not None:
                    KL.SP = _Proxy(SP, sup_fn)
                try:
                    sc, crops, grown, rounds, conv = KL.run_g(base, v, cam, W, H, make_detect(tpx, scale, log, dt_, cnt),
                                                              (lift_c(nm, cmode, scale) if cmode else lift_pol(nm, parse(arm), scale, make_retro(log, used, dt_))), pp,
                                                              v_closes=(vc,), ctx=ctx, max_calls=max_calls, **kw)
                finally:
                    KL.SP = SP
                rpx = sum(k_[-2] * k_[-1] for k_ in used)
                score(nm, sc, refs)
                rec[f"px_{nm}"] = (cheap_px + tpx[0] + rpx) / NATIVE
                rec[f"cpx_{nm}"] = (tpx[0] + rpx) / NATIVE
                rec[f"ncrops_{nm}"], rec[f"nguard_{nm}"], rec[f"rounds_{nm}"] = len(crops), len(grown), rounds
                rec[f"ncalls_{nm}"] = cnt[0] + len(used)
                rec[f"conv_{nm}"] = conv
                last[nm] = (crops, grown, sc)
                push(nm, sc)
                return tpx[0] + rpx
            # ---- draft
            if isinstance(cached, dict) and not online:
                ch = cached
            else:
                ch = keep(ddet.detect(img, CH)[0]); ch["mono"] = geom(ch, prev_c)
            ch_raw = ch
            if args.setting == "del_all" and len(relevant):           # K4 B (instr_eval): delete the matched draft detections
                mm = IE.matched(gtb[relevant], gtc[relevant], ch["xyxy"], ch["coarse"])
                dl = [j for j in mm if j >= 0]
                rec.update({"deleted": len(dl), "deleted_gt_range": [float(g_ln[relevant[i]]) for i, j in enumerate(mm) if j in dl],
                            "deleted_mask": np.array([j in dl for j in mm], bool)})
                if dl:
                    k_ = np.ones(len(ch["conf"]), bool); k_[dl] = False
                    ch = {kk: (vv[k_] if not isinstance(vv, tuple) else tuple(np.asarray(a)[k_] for a in vv)) for kk, vv in ch.items()}
                rec["J_cheap_unmod"] = cost(act(ch_raw["mono"]))
            elif args.setting == "del_all":
                rec["deleted"] = 0
            if not (isinstance(cached, dict) and not online):
                prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            base = {k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}
            a_c = act(ch["mono"])
            rec.update({"a_cheap": a_c, "J_cheap": cost(a_c), "px_cheap": cheap_px / NATIVE})
            if part == "instr":
                a2c = IE.c2_action(ch["mono"], pp)
                rec.update({"a2_cheap": a2c, "J2_cheap": float(IE.C2_COST[act2_gt, a2c]),
                            "nosup_cheap": len(SP.support(ch["mono"], v, pp)) == 0})
                if len(relevant):
                    rec["found_cheap"] = IE.matched(gtb[relevant], gtc[relevant], ch["xyxy"], ch["coarse"]) >= 0
            tp = parse(TPOL)
            crop51 = loop("v51S", TPOL, "c", 0.0, 2.0, 960, None)
            if part == "ours":
                for nm, (cdet, side, cmode, rf) in V6R.items():
                    loop(nm, TPOL, "c", 0.0, 2.0, side, None, dt_=dets.get(cdet), cmode=cmode,
                         refs=("none", rf) + ((f"{rf}_s1", f"{rf}_s2") if args.seeds else ()))
            elif part == "basewf":
                for dt_, pre, Ls, refs in ((det, "wf", WFS, ("none", "v3a")),) + (((vdet, "vwf", VER_WF[ds], ("none", "own")),) if args.det else ()):
                    for L in Ls:
                        md, ms = wf_mode(ds, L)
                        nm = f"{pre}{L}"
                        d = keep(dt_.detect(img, md)[0])
                        mono = geom(d, {"xyxy": st[nm]["merged"]} if nm in st else None)
                        score(nm, scene_dict(d, mono), refs)
                        tt, _ = ttc_ss(d["xyxy"], st.get(nm, {}).get("s1"), st.get(nm, {}).get("s2"), ms, dt, tp["gate"], tp["t4"])
                        score(f"{nm}+T4", scene_dict(d, mono[:3] + (tt,)), refs)
                        score(f"{nm}+C1", scene_dict(d, lift_c(nm, "C1", ms)(d)), refs)
                        rec[f"px_{nm}"] = md.net_h * md.net_w / NATIVE
                        push(nm, scene_dict(d, None))
            c2px = [0.0]
            if part == "instr":
                none_ = lambda *a, **k: []                                        # noqa: E731
                F960 = FC1 = None
                for L in F_SIDES:                                  # F passes: k28_test basewf's rows (same names, same histories)
                    L = NWS[ds] if L == "native" else L
                    md, ms = wf_mode(ds, L)
                    nm = f"wf{L}"
                    d = keep(det.detect(img, md)[0])
                    mono = geom(d, {"xyxy": st[nm]["merged"]} if nm in st else None)
                    score(nm, scene_dict(d, mono))
                    tc1 = lift_c(nm, "C1", ms)(d)
                    score(f"{nm}+C1", scene_dict(d, tc1))
                    rec[f"px_{nm}"] = md.net_h * md.net_w / NATIVE
                    if L == 960:
                        F960, FC1 = d, tc1
                    Fdet[nm] = d
                    push(nm, scene_dict(d, None))
                FLOG.clear()
                loop("v6S", TPOL, "c", 0.0, 2.0, 960, None, cmode="C1")
                fl6 = list(FLOG)
                loop("so6S", TPOL, "c", 0.0, 2.0, 960, None, cmode="C1", field=False)
                loop("fo6S", TPOL, "c", 0.0, 2.0, 960, None, cmode="C1", sup_fn=none_)
                if args.setting == "base":                          # the three K3 rows of phase 0 (1), for v6-S
                    loop("pv6S", TPOL, "c", 0.0, 2.0, 960, None, cmode="C1",
                         sup_fn=lambda mono, v_, pp_: np.atleast_1d(dsf.brake_pivots(mono, v_, pp_)).astype(int))
                    loop("or6S", TPOL, "c", 0.0, 2.0, 960, None, cmode="C1", max_rounds=1)
                    loop("ng6S", TPOL, "c", 0.0, 2.0, 960, None, cmode="C1", guard=False)
                if args.setting == "base":
                    # K9: ours driven by the TTC-warning consumer (its own field / support, spec_general), + C1
                    sc2, cr2, _ = SG.run(base, v, cam, W, H, make_detect(c2px, 960 / NW), lift_c("c2", "C1", 960 / NW),
                                         IE.TTCWarn(pp, 0.0), max_rounds=3, guard=True)
                    score("c2", sc2)
                    rec["px_c2"], rec["ncrops_c2"] = (cheap_px + c2px[0]) / NATIVE, len(cr2)
                    push("c2", sc2)
                    # K16: counterfactual chain, the crop detector replaced by F = u960 + C1 (no refiner)
                    fxy = np.asarray(F960["xyxy"], float).reshape(-1, 4)

                    def detect_F(cr):
                        k = SP.centres_in(fxy, [cr]) if len(fxy) else np.zeros(0, bool)
                        return {"xyxy": fxy[k], "conf": F960["conf"][k], "coarse": F960["coarse"][k]}

                    def lift_F(full):
                        X = np.asarray(full["xyxy"], float).reshape(-1, 4)
                        ii = [int(np.argmax(box_iou(X[n:n + 1], fxy)[0])) for n in range(len(X))]
                        return tuple(np.asarray(a, float)[ii] for a in FC1)
                    for sn_, kw, fl in CHAIN:
                        cn = f"ch_{sn_}"
                        sd, _, _, rd, cv = KL.run_g(base, v, cam, W, H, detect_F, lift_F if fl else lift_c(cn, "C1", 960 / NW), pp, **kw)
                        a_s = act(sd["mono"])
                        rec.update({f"a_{cn}": a_s, f"J_{cn}": cost(a_s), f"rounds_{cn}": rd, f"conv_{cn}": cv})
                        push(cn, sd)
                    # K16 coverage: F's objects that would raise ours' action (ours without refiner)
                    a_o = rec["a_v6S|none"]
                    _, demF = P.required_decel(*FC1, v, pp) if len(fxy) else (0.0, np.zeros(0))
                    demF = np.asarray(demF, float)
                    drv = int(np.argmax(demF)) if rec["a_wf960+C1|none"] != P.KEEP and len(demF) else -1
                    cov = []
                    if a_o < P.HARD_BRAKE and len(demF):
                        thr = pp.thr_decel if a_o == P.KEEP else pp.thr_hard
                        cr6, gr6, _ = last["v6S"]
                        inR = SP.centres_in(fxy, list(cr6) + list(gr6))
                        for j in np.flatnonzero(demF >= thr):
                            cov.append({"z": float(FC1[0][j]), "y": float((FC1[1][j] + FC1[2][j]) / 2), "cls": str(F960["coarse"][j]),
                                        "inR": bool(inR[j]), "is_F_driver": bool(j == drv), "demand": float(demF[j])})
                    rec["cov"] = cov
            if part in ("baseheur", "instr"):                      # E construction (k28_dev.py) + C1
                scale = 960 / NW

                def canvas(cr):
                    k_ = ckey(cr, scale)
                    return k_[4] * k_[5]

                def fill_px(cands, bud):
                    sel, used_ = [], 0
                    for c in cands:
                        cv_ = canvas(c)
                        if used_ + cv_ <= bud:
                            sel.append(c); used_ += cv_
                    return sel, used_

                def tube(bud, near, far, wmode):
                    vb_n = min(cy + fy * cam.cam_h / near, H)
                    top = max(cy + fy * cam.cam_h / far - fy * 1.6 / far, 0)
                    best = []
                    for half_m in np.arange(0.5, 12.01, 0.25):
                        if wmode == "lane" and half_m > 2.5:
                            break
                        half = fx * half_m / near
                        c = np.array([max(cx - half, 0), top, min(cx + half, W), vb_n])
                        if canvas(c) > bud:
                            break
                        best = [c]
                    return best
                ent = np.asarray(ch.get("entropy", np.zeros(len(ch["conf"]))))
                sup_d = SP.support(ch["mono"], v, pp)
                for conf_nm in (CONFIGS if part == "baseheur" else CFG_S + (("unc2@c2",) if args.setting == "base" else ())):
                    vn, cap_nm = conf_nm.split("@")
                    bud = CAPS[cap_nm] * (cheap_px + crop51) - cheap_px if cap_nm != "c2" else c2px[0]
                    fam, i = vn[:3], (int(vn[3:]) if vn != "rnd" else 0)
                    hsuf, hrefs = ("", ("none", "v3a")) if vname == "yolov8s" or part == "instr" else (f"-{vname}", ("none", "own"))

                    def select(name):
                        if fam == "unc":
                            ctx, o = UNC[i]
                            order = np.lexsort((-ent, ch["conf"])) if o == "conf" else np.lexsort((ch["conf"], -ent))
                            return fill_px([KL.box_crop_ctx(ch["xyxy"][k], W, H, ctx) for k in order], bud)[0]
                        if fam == "tcr":                           # previous merged objects of this row
                            gr, n = TCR[i]
                            ps_ = st.get(name, {}).get("state")
                            cands = []
                            if ps_ is not None and len(ps_[0]):
                                pxy, pmono = ps_
                                _, dem = P.required_decel(*pmono, v, pp)
                                order = np.lexsort((pmono[0], -np.asarray(dem)))
                                if n:
                                    order = order[:n]
                                cands = [SP.box_crop(np.r_[pxy[k][:2] - gr * (pxy[k][2:] - pxy[k][:2]),
                                                           pxy[k][2:] + gr * (pxy[k][2:] - pxy[k][:2])], W, H) for k in order]
                            return fill_px(cands, bud)[0]
                        if fam == "cor":
                            near, far, wm = TUBE[i]
                            return tube(bud, near, far, wm) if bud > 0 else []
                        if fam == "tsp":
                            near, far, wm = TUBE[i]
                            sp_, used_ = fill_px([SP.box_crop(ch["xyxy"][k], W, H) for k in sup_d], bud) if bud > 0 else ([], 0)
                            return sp_ + (tube(bud - used_, near, far, wm) if bud - used_ > 0 else [])
                        rr = np.random.default_rng(zlib.crc32(f"{unit}/{fr}".encode()))        # rnd
                        cands = []
                        for _ in range(40):
                            if len(ch["xyxy"]):
                                b = ch["xyxy"][rr.integers(len(ch["xyxy"]))]
                                w_, h_ = min((b[2] - b[0]) * 2, W), min((b[3] - b[1]) * 2, H)
                            else:
                                w_, h_ = 200.0, 120.0
                            h_ = min(h_, H * 0.65)
                            x0, y0 = rr.uniform(0, max(W - w_, 1)), rr.uniform(H * 0.35, max(H - h_, H * 0.35 + 1))
                            cands.append(np.array([x0, y0, x0 + w_, y0 + h_]))
                        return fill_px(cands, bud)[0]
                    rows_ = [(f"{vn}@{cap_nm}{hsuf}", None), (f"{vn}@{cap_nm}{hsuf}+T4", "T4"), (f"{vn}@{cap_nm}{hsuf}+C1", "C1")]
                    if part == "instr":
                        rows_ = rows_[2:]
                    for name, lk in rows_:
                        tpx = [0.0]
                        lf = lift_c(name, "C1", scale) if lk == "C1" else lift_pol(name, tp if lk else None, scale)
                        sel_ = select(name)
                        sh, _ = IE.merge_once(base, sel_, SP.guarded(make_detect(tpx, scale, dt_=vdet), W, H), lf)
                        if (unit, fr) in CAP:
                            capH[name] = {"crops": [np.asarray(c, float) for c in sel_], "xyxy": np.asarray(sh["xyxy"], float),
                                          "from_full": np.asarray(sh["from_full"], bool)}
                        score(name, sh, hrefs)
                        rec[f"px_{name}"] = (cheap_px + tpx[0]) / NATIVE
                        push(name, sh)
                        if fam == "tcr":
                            st[name]["state"] = (np.asarray(sh["xyxy"], float).reshape(-1, 4), tuple(np.asarray(q, float) for q in sh["mono"]))
            if part == "instr" and (unit, fr) in CAP:
                cr6, gr6, sc6 = last["v6S"]
                rec["cap"] = {
                    "path": path, "W": W, "H": H, "v": v,
                    "draft": {k: np.asarray(ch[k]) for k in ("xyxy", "conf", "coarse")}, "draft_mono": tuple(np.asarray(a, float) for a in ch["mono"]),
                    "sens": dsf.insertion_grid(ch["mono"], v, SP.R_GRID, SP.Y_GRID, SP.OBJ_W, pp, 0.0),
                    "v6S_crops": [np.asarray(c, float) for c in cr6], "v6S_grown": [np.asarray(c, float) for c in gr6],
                    "v6S_field": [np.asarray(c, float) for c in cr6 if any(np.allclose(c, f_) for f_ in fl6)],
                    "scenes": {k_: {"xyxy": np.asarray(last[k_][2]["xyxy"], float), "coarse": np.asarray(last[k_][2]["coarse"]),
                                    "from_full": np.asarray(last[k_][2]["from_full"], bool),
                                    "mono": tuple(np.asarray(a, float) for a in last[k_][2]["mono"])} for k_ in ("v6S", "so6S", "fo6S")},
                    "F": {k_: {"xyxy": np.asarray(x["xyxy"], float), "coarse": np.asarray(x["coarse"])} for k_, x in Fdet.items()},
                    "heur": capH,
                    "gt": {"xyxy": gtb, "coarse": gtc, "relevant": relevant,
                           "range": g_ln}}
            imgs = (imgs + [img])[-2:]
            prev_act = act(ch_raw["mono"])
            rows.append(rec)
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{ui}] {unit}: {len(rows)} frames", flush=True)
    print(f"K28 P3 {part.upper()} DONE", flush=True)


if __name__ == "__main__":
    main()
