#!/usr/bin/env python
"""K28 PHASE 3 (post-hoc) — Jetson latency of the frozen v6 configurations (method_freeze_draft §2n), the bounded
variant, and the whole-frame controls plain / + T4 + refiner / + C1 + refiner. = 60_k28_latency.py with the C1 lift:
the learned temporal head (out/k28_temporal/temporal_head.pt, run in torch fp32) on the last three same-scale crops of
every object with a 3-observation chain on the pipeline's own history (k28_test.lift_c, mode C1, copied verbatim);
objects without a chain get no ttc. One configuration per process (one verifier engine in memory):
  --config v51S | v6S | v6M | v6L     crop detector / crop long side / lift / refiner of §2n (v51S: T4, v3a)
Pipelines: cheap; <wf> (plain), <wf>+comp (T4 + v3a), <wf>+c1comp (C1 + v3a); ours / ours_noref; ours_bN (59's bounded
loop, at most N calls) with the same lift + refiner. Per frame: ms, calls, tensor px, actions, cost; parity with the VUW
per-frame results of the same configuration (--ref_dir, key J_<ref_key>) on the timed frames.
--dataset kitti: DEEP's KITTI split (--kitti_split, default test) with the shared configuration (crop long side / 1242);
every pipeline is also scored on DEEP's four KITTI cells (brake / Planner B x mono / oracle range, DEEP's cheap histories,
as 50_kitti_baseline_cells.score); cells_bootstrap.csv: G per cell, mean of the four, sequence-bootstrap intervals.
--dataset ahd: Argoverse-HD val (24 sequences, every frame, ~/datasets/argoverse1) for the streaming-AP simulation on VUW
(per-frame ms of ours with image_id; whole-frame timings of K7 are reused, --wf none); a warm-up pass over the first
--warmup frames of the first sequence runs first and is discarded, so every recorded frame is warm (as 56_ahd_latency).
usage: 61_k28_v6_latency.py --dataset nusc|av2|kitti|ahd --config v6S [--wf u640,u960,...] [--bounds 2] [--mask_y0 1830]
                            [--ref_dir dir --ref_key v6S|v3a] [--max_frames n] [--tag name]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("k5", ROOT / "scripts" / "55_large_latency.py")
K = importlib.util.module_from_spec(_s); _s.loader.exec_module(K)
_f = importlib.util.spec_from_file_location("k11", ROOT / "scripts" / "58_large_latency_fast.py")
KF = importlib.util.module_from_spec(_f); _f.loader.exec_module(KF)
_b = importlib.util.spec_from_file_location("k18b", ROOT / "scripts" / "59_bounded_latency.py")
KB = importlib.util.module_from_spec(_b); _b.loader.exec_module(KB)
from rap.mono import box_iou, ttc_from_scale                                     # noqa: E402
from vtf import gpupost as GP                                                    # noqa: E402
from vtf.gpuprep import letterbox                                                # noqa: E402
from vtf.trtdyn import DynTRT                                                    # noqa: E402
from nusc_crop import crop64                                                     # noqa: E402

P, SP, THR, Mode, ENG = K.P, K.SP, K.THR, K.Mode, K.ENG
STATS = KF.STATS
KB.STATS = STATS


def sync():
    torch.cuda.synchronize()


# ---------------------------------------------------------------- k27_dev.assoc_idx / ttc_ss (verbatim, without T3)
def assoc_idx(curr, prev, thr=0.3):
    out = -np.ones(len(curr), int)
    if len(prev) == 0 or len(curr) == 0:
        return out
    iou = box_iou(curr, prev)
    taken = set()
    for ci in np.argsort(-iou.max(axis=1)):
        for pj in np.argsort(-iou[ci]):
            if iou[ci, pj] < thr:
                break
            if pj not in taken:
                taken.add(int(pj)); out[ci] = pj
                break
    return out


def ttc_ss(X, s1, s2, sc, dt, gate=None, t4=False):
    X = np.asarray(X, float).reshape(-1, 4)
    n = len(X)
    h = X[:, 3] - X[:, 1]
    P1, src = np.full((n, 4), np.nan), np.zeros(n, int)
    j1 = assoc_idx(X, s1) if s1 is not None and len(s1) else -np.ones(n, int)
    if (j1 >= 0).any():
        P1[j1 >= 0], src[j1 >= 0] = s1[j1[j1 >= 0]], 1
    h1 = P1[:, 3] - P1[:, 1]
    if not t4:
        ttc = ttc_from_scale(h, h1, dt)
        if gate is not None:
            with np.errstate(invalid="ignore"):
                ttc[~((h - h1) * sc > gate)] = np.inf
        return ttc, src
    P2 = np.full((n, 4), np.nan)
    if s1 is not None and len(s1) and s2 is not None and len(s2):
        a12 = assoc_idx(s1, s2)
        for i in np.flatnonzero(src == 1):
            if a12[j1[i]] >= 0:
                P2[i] = s2[a12[j1[i]]]
    h2 = P2[:, 3] - P2[:, 1]
    ttc = np.full(n, np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.isfinite(h1) & np.isfinite(h2) & (h > 1) & (h1 > 1) & (h2 > 1)
        r = np.median(np.stack([(h1 / h2 - 1) / dt, (h / h1 - 1) / dt, (h / h2 - 1) / (2 * dt)]), 0)
        ex = ok & (r > 1e-3 / dt)
        ttc[ex] = np.minimum(1 / r[ex], 1e3)
        if gate is not None:
            ttc[~((h - h2) * sc > gate)] = np.inf
    return ttc, src


CFG = {"v51S": ("yolov8s", 960, "T4", "v3a_vuw"), "v6S": ("yolov8s", 960, "C1", "v3a_vuw"),
       "v6M": ("yolov8m", 768, "C1", "own_yolov8m_vuw"), "v6L": ("yolo11m", 960, "C1", "own_yolo11m_vuw")}
NW = {"nusc": 1600, "av2": 2048}


class TempHead(torch.nn.Module):          # = remote/A_method/k28_ttrain.TempHead, without the weight download
    def __init__(self, n_feat):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet18(weights=None)
        m.fc = torch.nn.Identity()
        self.cnn = m
        self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.head = torch.nn.Sequential(torch.nn.Linear(3 * 512 + n_feat, 256), torch.nn.ReLU(inplace=True), torch.nn.Linear(256, 1))

    def forward(self, x, f):
        n = x.shape[0]
        x = x.reshape(n * 3, 3, 64, 64)
        x = torch.nn.functional.interpolate(x.flip(1), size=128, mode="bilinear", align_corners=False)
        e = self.cnn((x - self.mu) / self.sd).reshape(n, 3 * 512)
        return self.head(torch.cat([e, f], 1))[:, 0]


def frames_kitti(args):
    from rap.cache import Camera
    from rap.decision import KittiAdapter
    from rap.paths import KITTI_IMAGES
    dv = pd.read_csv(K.runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz", dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[(dv.track == "KITTI") & (dv.geometry == "mono") & (dv.system == "brake")].copy()
    dv["frame"] = dv.frame.astype(int)
    for n_sc, s in enumerate(sorted(set(dv[dv.split == args.kitti_split].seq))):
        if args.max_units and n_sc >= args.max_units:
            break
        geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s)
        cam = Camera("KITTI", s)
        R, t = cam.transform("ego")
        for fr in sorted(int(f) for f in dv[dv.seq == s].frame):
            g = geom[geom["frame"] == fr]
            yield s, fr, str(Path(KITTI_IMAGES) / s / f"{fr:06d}.png"), float(speeds[fr]) if fr < len(speeds) else float(speeds[-1]), \
                g[(g["y2"] - g["y1"]) >= 10], cam, (R, t)


def frames_ahd(args):
    from vtf import av1 as A1
    KA = importlib.util.spec_from_file_location("k7", ROOT / "scripts" / "56_ahd_latency.py")
    K7 = importlib.util.module_from_spec(KA); KA.loader.exec_module(K7)
    S = K7.seqs()[: args.max_units or None]
    plan = [(S[0], args.warmup, "warmup")] + [(s_, None, None) for s_ in S]
    for (sid, log, names, ids), nmax, lab in plan:
        names, ids = (names[:nmax], ids[:nmax]) if nmax else (names, ids)
        L = A1.AV1Log(K7.AV1, log, names)
        geom, speeds = L.geometry(), L.speeds()
        gtf = L.gt_frames()
        R, t = L.transform("ego")
        for fr, iid in enumerate(ids):
            g = geom[geom["frame"] == fr] if len(geom) else geom
            g = g[(g["y2"] - g["y1"]) >= 10] if len(g) else g
            AHD_META[(lab or f"seq{sid:02d}", fr)] = (iid, fr in gtf)
            yield lab or f"seq{sid:02d}", fr, str(L.image_path(fr)), float(speeds[fr]), g, L, (R, t)


AHD_META = {}


def chains(X, s1, s2):
    """k28_tdata.chains: same-scale 3-observation chains (i, j1, j2)"""
    if s1 is None or s2 is None or not len(s1) or not len(s2) or not len(X):
        return []
    j1 = assoc_idx(X, s1)
    a12 = assoc_idx(s1, s2)
    return [(i, int(j1[i]), int(a12[j1[i]])) for i in range(len(X)) if j1[i] >= 0 and a12[j1[i]] >= 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="v6S", choices=list(CFG))
    ap.add_argument("--head", default=str(ENG / "k28_temporal_head.pt"))
    ap.add_argument("--dataset", required=True, choices=["av2", "nusc", "kitti", "ahd"])
    ap.add_argument("--kitti_split", default="test")
    ap.add_argument("--wf_det", default="yolov8s")
    ap.add_argument("--wf", default="")
    ap.add_argument("--bounds", default="2,3")
    ap.add_argument("--v_close", type=float, default=0.0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--max_units", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--mask_y0", type=float, default=0.0)
    ap.add_argument("--ref_dir", default="")
    ap.add_argument("--ref_key", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--post_only", default="", help="run dir: recompute the tables from its k28_latency.pkl (no timing)")
    args = ap.parse_args()
    if args.post_only:
        run = Path(args.post_only)
        args = argparse.Namespace(**{**json.loads((run / "config.json").read_text()), "post_only": args.post_only})
        d = pd.read_pickle(run / "k28_latency.pkl")
        M = [c[3:-5] for c in d.columns if c.startswith("ms_") and c.endswith("+comp")]
        summarise(d, run, args, M, [int(b) for b in args.bounds.split(",") if b], args.dataset == "kitti")
        return
    args.verifier, side, LIFT, args.refiner = CFG[args.config]
    args.ref_key = args.ref_key or f"{args.config}|{'v3a' if args.refiner == 'v3a_vuw' else 'own-' + args.verifier}"
    run = K.runmeta.new_run(f"k28_v6_latency_{args.dataset}_{args.config}" + (f"_{args.tag}" if args.tag else ""), vars(args))
    ckh = torch.load(args.head, map_location="cuda", weights_only=False)
    head = TempHead(ckh["n_feat"]); head.load_state_dict(ckh["state"]); head = head.cuda().eval()
    hmu, hsd = ckh["fmu"], ckh["fsd"]
    if args.mask_y0:
        _pg = K.predicted_geometry

        def masked_pg(dets, *a, **k):
            g = _pg(dets, *a, **k)
            b = np.asarray(dets["xyxy"], float).reshape(-1, 4)
            if not len(b):
                return g
            m = (b[:, 3] - np.maximum(b[:, 1], args.mask_y0)).clip(0, None) / np.maximum(b[:, 3] - b[:, 1], 1e-6) >= 0.5
            if m.any():
                g = dict(g)
                for key in ("z", "z_ground", "z_height", "ttc"):
                    if key in g:
                        arr = np.asarray(g[key], float).copy(); arr[m] = np.inf; g[key] = arr
            return g
        K.predicted_geometry = masked_pg
    det_s = K.FastDetector(str(Path(K.MODELS) / "yolov8s.pt"), backend="trt", engine_dir=str(ENG), engine_stem="yolov8s_crop")
    det_w = det_s if args.wf_det == "yolov8s" else K.FastDetector(str(Path(K.MODELS) / f"{args.wf_det}.pt"), backend="trt",
                                                                  engine_dir=str(ENG), engine_stem=f"{args.wf_det}_crop")
    KIT = args.dataset == "kitti"
    if args.dataset == "ahd":
        WF = {}
        CH, scale, gen, NATIVE = Mode("224x320", 224, 320), side / 1920, frames_ahd(args), 1920 * 1200
        MS = {}
    elif KIT:
        from rap import planner_b as PB
        from rap.decision import _apply_range_source
        WF = {"u960": (288, 960), "native": (384, 1248)}
        CH, scale, gen, NATIVE = Mode("96x320", 96, 320), side / 1242, frames_kitti(args), 1242 * 375
        MS = {"u960": 960 / 1242, "native": 1.0}
        pbp, pbc = PB.PARAMS_B["static_obstacles"], PB.COSTS_B["default"]
        rng = np.random.default_rng(0)
    elif args.dataset == "av2":
        WF = {"u512": (512, 416), "u640": (640, 512), "u960": (960, 768), "u1280": (1280, 992), "u1536": (1536, 1184),
              "native": (2048, 1568)}
        CH, scale, gen, NATIVE = Mode("320x256", 320, 256), side / 2048, K.frames_av2(args), 1550 * 2048
        MS = {"u512": 512 / 2048, "u640": 640 / 2048, "u960": 960 / 2048, "u1280": 1280 / 2048, "u1536": 1536 / 2048, "native": 1.0}
    else:
        WF = {"u448": (256, 448), "u512": (288, 512), "u640": (384, 640), "u768": (448, 768), "u960": (576, 960),
              "u1280": (736, 1280), "u1536": (864, 1536), "native": (928, 1600)}
        CH, scale, gen, NATIVE = Mode("192x320", 192, 320), side / 1600, K.frames_nusc(args), 1600 * 900
        MS = {"u448": 0.28, "u512": 0.32, "u640": 0.4, "u768": 0.48, "u960": 0.6, "u1280": 0.8, "u1536": 0.96, "native": 1.0}
    keep_wf = [x for x in args.wf.split(",") if x != "none"] if args.wf else list(WF)
    WF = {k: v for k, v in WF.items() if k in keep_wf}
    M = {nm: Mode(f"{h}x{w}", h, w) for nm, (h, w) in WF.items()}
    missing = [m.name for m in M.values() if not (ENG / f"{args.wf_det}_crop_{m.name}.engine").exists()]
    assert not missing, ("missing whole-frame engines", args.wf_det, missing)
    refb = KF.RefinerBatch(args.refiner)
    refw = refb if args.refiner == "v3a_vuw" else KF.RefinerBatch("v3a_vuw")      # whole-frame controls: YOLOv8s + v3a
    G = K.GPUDetect()
    many = KF.Crops(G, scale)
    if args.verifier != "yolov8s":
        many.eng = DynTRT(ENG / f"{args.verifier}_dyn.engine")
    one = KB.OneCall(G, scale, many.eng)
    G.x, G.shape = torch.full((3, 2048, 2048), K.PAD, device="cuda"), (2048, 2048, 3)
    many([[0, 0, 64, 64]] * 2 + [[0, 0, 400, 400], [0, 0, 1000, 1000]])
    bounds = [int(b) for b in args.bounds.split(",") if b]
    pp, cp = P.PlannerParams(), P.CostParams()

    def gdetect(dt_, m):
        xb, r, pad = letterbox(G.x, m.net_h, m.net_w)
        raw = dt_.forward(xb[None], m)
        return GP.post_lean(GP.host_columns(raw, m, THR)[0], m, r, pad, G.shape, min_conf=THR)
    rows, hist, cur, prev_act, imgs = [], {}, None, None, []
    prev_act_o = prev_plan = prev_plan_o = None
    for unit, fr, path, v, g, cam, (R, t) in gen:
        if args.max_units and unit != cur and len({r["unit"] for r in rows}) >= args.max_units:
            break
        if unit != cur:
            hist, cur, prev_act, imgs = {}, unit, None, []
            prev_act_o = prev_plan = prev_plan_o = None
        img = cv2.imread(path)
        H, W = img.shape[:2]
        dt = float(cam.dt)
        a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, None)
        rec = {"unit": unit, "frame": fr}

        def lift(xyxy, coarse, pv):
            gk = K.predicted_geometry({"xyxy": xyxy, "coarse": coarse}, pv, cam.calib, cam.cam_h, dt, cam_to_ego=(R, t))
            return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

        def lift_t4(name, sc):
            h = hist.get(name, {})

            def f(full):
                z, lo, hi, _ = lift(full["xyxy"], full["coarse"], None)
                tt, _ = ttc_ss(full["xyxy"], h.get("s1"), h.get("s2"), sc, dt, 1.0, True)
                return z, lo, hi, tt
            return f

        gt_t = tuple(np.asarray(g[k_], float) for k_ in ("long_near", "lat_min", "lat_max", "ttc")) if len(g) else tuple(np.zeros(0) for _ in range(4))

        def cells(name, xyxy, tt):
            """DEEP's other three KITTI cells (brake oracle, Planner B mono / oracle) as 50_kitti_baseline_cells.score;
            brake mono = J_<name>"""
            if not KIT:
                return None
            geo = dict(zip(("z", "lat_min", "lat_max", "ttc"), (np.asarray(a, float) for a in tt)))
            to = tuple(np.asarray(a, float) for a in _apply_range_source(np.asarray(xyxy, float).reshape(-1, 4), geo, g, "gt", rng, 0.0))
            rec[f"Jbo_{name}"] = P.decision_cost(act(to), a_gt, prev_act_o, pp, cp)["J"]
            for tag, t_, pv in (("Jtm", tt, prev_plan), ("Jto", to, prev_plan_o)):
                rec[f"{tag}_{name}"] = PB.executed_cost(PB.plan(*t_, v, pv, pbp, pbc), *gt_t, v, pv, pbp, pbc)["J"]
            return to

        def lift_c1(name, sc):
            """k28_test.lift_c (mode C1) on the history of pipeline `name`"""
            h = hist.get(name, {})

            def f(full):
                X = np.asarray(full["xyxy"], float).reshape(-1, 4)
                z, lo, hi, _ = lift(X, full["coarse"], None)
                ttc = np.full(len(X), np.inf)
                chn = chains(X, h.get("s1"), h.get("s2")) if len(imgs) == 2 else []
                if not chn:
                    return z, lo, hi, ttc
                idx = np.array([c[0] for c in chn])
                zz = K.predicted_geometry({"xyxy": X[idx], "coarse": np.asarray(full["coarse"])[idx]}, None, cam.calib, cam.cam_h, dt,
                                          cam_to_ego=(R, t))
                cr, fe = [], []
                for n, (i, j1, j2) in enumerate(chn):
                    b1, b2 = h["s1"][j1], h["s2"][j2]
                    x1, y1, x2, y2 = X[i]
                    zg, zh = float(zz["z_ground"][n]), float(zz["z_height"][n])
                    hh = (y2 - y1, b1[3] - b1[1], b2[3] - b2[1])
                    cr.append(np.stack([crop64(img, X[i]), crop64(imgs[-1], b1), crop64(imgs[-2], b2)]))
                    fe.append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(full["conf"][i]),
                               float(K.CLS[str(full["coarse"][i])]), float(z[i]), zg, zh, float(np.log(max(zg, .5) / max(zh, .5))),
                               float(np.log(max(hh[0] / max(hh[1], 1e-3), 1e-3))), float(np.log(max(hh[1] / max(hh[2], 1e-3), 1e-3))), dt, 1.0])
                xin = torch.from_numpy(np.stack(cr)).permute(0, 1, 4, 2, 3).float().cuda() / 255.0
                fin = torch.from_numpy(((np.array(fe, np.float32) - hmu) / hsd).astype(np.float32)).cuda()
                with torch.no_grad():
                    rh = np.clip(head(xin, fin).float().cpu().numpy(), 0, 2)
                STATS["calls"] += 1
                for n, (i, _, _) in enumerate(chn):
                    r = rh[n]
                    ttc[i] = min(1 / r, 1e3) if r > 1e-3 / dt else np.inf
                return z, lo, hi, ttc
            return f

        def push(name, xyxy):
            h = hist.setdefault(name, {})
            h["s2"] = h.get("s1")
            h["s1"] = np.asarray(xyxy, float).reshape(-1, 4)

        def act(tt):
            return P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)

        def J(a):
            return P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]

        def keep(d):
            k = d["conf"] >= THR
            return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k]}

        def refine(scene, rb=None):
            rb = rb or refb
            refined = [np.asarray(a, float).copy() for a in scene["mono"]]
            sup = [p_ for p_ in SP.support(scene["mono"], v, pp) if scene["from_full"][p_]]
            corr = rb(img, [(scene["xyxy"][p_], scene["conf"][p_], scene["coarse"][p_], scene["mono"][0][p_]) for p_ in sup], cam, R, t, W, H)
            for p_, c in zip(sup, corr):
                for a in refined[:3]:
                    a[p_] *= c
            return tuple(refined), len(sup)
        # ---- cheap draft
        sync(); t0 = time.perf_counter()
        G.frame(img)
        d = keep(gdetect(det_s, CH))
        tc = lift(d["xyxy"], d["coarse"], hist.get("cheap_prev"))
        a_c = act(tc)
        sync(); rec["ms_cheap"] = 1e3 * (time.perf_counter() - t0)
        rec.update({"a_cheap": a_c, "J_cheap": J(a_c), "px_cheap": CH.net_h * CH.net_w / NATIVE})
        to_c = cells("cheap", d["xyxy"], tc)
        # ---- whole frame, plain and + components
        for nm, m in M.items():
            sync(); t0 = time.perf_counter()
            G.frame(img)
            dw = keep(gdetect(det_w, m))
            tt = lift(dw["xyxy"], dw["coarse"], hist.get(f"{nm}_prev"))
            a = act(tt)
            sync(); rec[f"ms_{nm}"] = 1e3 * (time.perf_counter() - t0)
            rec[f"a_{nm}"], rec[f"J_{nm}"], rec[f"px_{nm}"] = a, J(a), m.net_h * m.net_w / NATIVE
            cells(nm, dw["xyxy"], tt)
            sync(); t0 = time.perf_counter()
            G.frame(img)
            dw2 = keep(gdetect(det_w, m))
            sc_ = {"xyxy": dw2["xyxy"], "conf": dw2["conf"], "coarse": dw2["coarse"], "mono": lift_t4(f"{nm}+comp", MS[nm])(dw2),
                   "from_full": np.ones(len(dw2["conf"]), bool)}
            tt2, _ = refine(sc_, refw)
            a2 = act(tt2)
            sync(); rec[f"ms_{nm}+comp"] = 1e3 * (time.perf_counter() - t0)
            rec[f"a_{nm}+comp"], rec[f"J_{nm}+comp"], rec[f"px_{nm}+comp"] = a2, J(a2), m.net_h * m.net_w / NATIVE
            cells(f"{nm}+comp", dw2["xyxy"], tt2)
            sync(); t0 = time.perf_counter()
            G.frame(img)
            dw3 = keep(gdetect(det_w, m))
            sc3 = {"xyxy": dw3["xyxy"], "conf": dw3["conf"], "coarse": dw3["coarse"], "mono": lift_c1(f"{nm}+c1comp", MS[nm])(dw3),
                   "from_full": np.ones(len(dw3["conf"]), bool)}
            tt3, _ = refine(sc3, refw)
            a3 = act(tt3)
            sync(); rec[f"ms_{nm}+c1comp"] = 1e3 * (time.perf_counter() - t0)
            rec[f"a_{nm}+c1comp"], rec[f"J_{nm}+c1comp"], rec[f"px_{nm}+c1comp"] = a3, J(a3), m.net_h * m.net_w / NATIVE
            cells(f"{nm}+c1comp", dw3["xyxy"], tt3)
            hist[f"{nm}_prev"] = dw
            push(f"{nm}+comp", dw2["xyxy"])
            push(f"{nm}+c1comp", dw3["xyxy"])
        # ---- ours (v5.1-S) and the bounded variants
        for nm, N in [("ours", None)] + [(f"ours_b{b}", b) for b in bounds]:
            STATS["calls"], STATS["tpx"], STATS["exec"], STATS["tpx_b"] = 0, 0, 0, 0
            sync(); t0 = time.perf_counter()
            G.frame(img)
            dd = keep(gdetect(det_s, CH)); STATS["calls"] += 1
            dd["mono"] = lift(dd["xyxy"], dd["coarse"], hist.get("cheap_prev"))
            lf = lift_c1(nm, scale) if LIFT == "C1" else lift_t4(nm, scale)
            if N is None:
                scene, crops, rounds = SP.run(dd, v, cam, W, H, lambda cr: many([cr])[0], lf, pp,
                                              v_close=args.v_close, max_rounds=3, guard=True, detect_many=many, batch_round=True)
                info = {"n_crops": len(crops), "rounds": rounds}
            else:
                scene, calls, hit = KB.bounded_run(dd, v, cam, W, H, one, lf, pp, args.v_close, N)
                info = {"calls_loop": calls, "bound_hit": hit}
            a_noref = act(scene["mono"])
            t_loop = time.perf_counter()
            tt, n_ref = refine(scene)
            a_o = act(tt)
            sync(); t_end = time.perf_counter()
            px = (CH.net_h * CH.net_w + (STATS["tpx"] if N is None else STATS["tpx_b"])) / NATIVE
            rec.update({f"ms_{nm}_noref": 1e3 * (t_loop - t0), f"ms_{nm}": 1e3 * (t_end - t0), f"a_{nm}_noref": a_noref,
                        f"a_{nm}": a_o, f"J_{nm}_noref": J(a_noref), f"J_{nm}": J(a_o), f"px_{nm}": px, f"px_{nm}_noref": px,
                        f"calls_{nm}": STATS["calls"] + STATS["exec"], f"n_refined_{nm}": n_ref,
                        **{f"{k}_{nm}": v_ for k, v_ in info.items()}})
            cells(nm, scene["xyxy"], tt)
            cells(f"{nm}_noref", scene["xyxy"], scene["mono"])
            ff = np.asarray(scene["from_full"], bool)
            push(nm, np.asarray(scene["xyxy"], float).reshape(-1, 4)[ff])
        hist["cheap_prev"] = {"xyxy": d["xyxy"], "coarse": d["coarse"]}
        prev_act = a_c
        if KIT:
            prev_act_o = act(to_c)
            prev_plan = PB.plan(*tc, v, prev_plan, pbp, pbc)
            prev_plan_o = PB.plan(*to_c, v, prev_plan_o, pbp, pbc)
        imgs = (imgs + [img])[-2:]
        rec["warm"] = len(rows) < args.warmup if args.dataset != "ahd" else unit == "warmup"
        if args.dataset == "ahd":
            rec["image_id"], rec["gt"] = AHD_META[(unit, fr)]
        rows.append(rec)
        if len(rows) % 200 == 0:
            print(f"  {len(rows)} frames", flush=True)
        if args.max_frames and len(rows) >= args.max_frames:
            break
    d = pd.DataFrame(rows)
    d.to_pickle(run / "k28_latency.pkl")
    summarise(d, run, args, list(M), bounds, KIT)


def summarise(d, run, args, M, bounds, KIT):
    dd = d[~d.warm].reset_index(drop=True)
    if args.dataset == "ahd":                              # G on the frames with 3D labels (as K24); every frame is timed
        dd = dd[dd["gt"].astype(bool)].reset_index(drop=True)
    L = dd.J_cheap.to_numpy()
    uq = dd.unit.unique(); idx = [np.flatnonzero(dd.unit.to_numpy() == u) for u in uq]
    rng = np.random.default_rng(0); boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
    Vo = L - dd.J_ours.to_numpy()
    names = ["cheap"] + [x for nm in M for x in (nm, f"{nm}+comp", f"{nm}+c1comp")] + ["ours_noref", "ours"] + [x for b in bounds for x in (f"ours_b{b}_noref", f"ours_b{b}")]
    out = []
    for nm in names:
        ms = dd[f"ms_{nm}"]; V = L - dd[f"J_{nm}"].to_numpy()
        r = {"pipeline": nm, "verifier": args.verifier if nm.startswith("ours") else (args.wf_det if nm != "cheap" else "yolov8s"),
             "config": args.config if nm.startswith("ours") else "",
             "refiner": args.refiner if (nm.startswith("ours") and not nm.endswith("noref")) else ("v3a_vuw" if nm.endswith("comp") else "none"),
             "frames": len(dd), "G": V.sum() / L.sum() if nm != "cheap" else 0.0, "mean_ms": ms.mean(), "median_ms": ms.median(),
             "p95_ms": ms.quantile(.95), "p99_ms": ms.quantile(.99), "max_ms": ms.max(), "tensor_px_mean": dd[f"px_{nm}"].mean(),
             "tensor_px_p95": dd[f"px_{nm}"].quantile(.95)}
        if nm != "ours":
            dr = [(Vo[b].sum() - V[b].sum()) / L[b].sum() for b in boots]
            r["ci_lo_ours_minus"], r["ci_hi_ours_minus"] = np.percentile(dr, 2.5), np.percentile(dr, 97.5)
        out.append(r)
    res = pd.DataFrame(out); res.to_csv(run / "latency.csv", index=False)
    if KIT:
        cl = []
        tags = (("brake mono", "J"), ("brake oracle", "Jbo"), ("traj mono", "Jtm"), ("traj oracle", "Jto"))
        Vs = {nm: {tg: (dd[f"{tg}_cheap"] - dd[f"{tg}_{nm}"]).to_numpy() for _, tg in tags} for nm in names if nm != "cheap"}
        Ls = {tg: dd[f"{tg}_cheap"].to_numpy() for _, tg in tags}
        for nm, V in Vs.items():
            r = {"pipeline": nm, **{cell: V[tg].sum() / Ls[tg].sum() for cell, tg in tags}}
            r["mean4"] = np.mean([r[c] for c, _ in tags])
            if nm != "ours":
                dr = [np.mean([(Vs["ours"][tg][b].sum() - V[tg][b].sum()) / Ls[tg][b].sum() for _, tg in tags]) for b in boots]
                r["ours_minus_mean4_lo"], r["ours_minus_mean4_hi"] = np.percentile(dr, 2.5), np.percentile(dr, 97.5)
            cl.append(r)
        pd.DataFrame(cl).to_csv(run / "cells_bootstrap.csv", index=False)
        print(pd.DataFrame(cl).round(3).to_string(index=False))
    pd.set_option("display.width", 220)
    print(f"{args.dataset}: {len(dd)} timed frames; verifier {args.verifier}, refiner {args.refiner}, wf {args.wf_det}")
    print(res.round(3).to_string(index=False))
    if args.ref_dir:
        rf = []
        for f in sorted(glob.glob(str(Path(args.ref_dir) / "*.pkl"))):
            rf += pickle.load(open(f, "rb"))
        rf = pd.DataFrame(rf)
        m = dd.merge(rf[["unit", "frame", f"J_{args.ref_key}", "J_cheap"]], on=["unit", "frame"], suffixes=("", "_ref"))
        par = {"frames_matched": len(m), "identical_cost_ours": float(np.isclose(m.J_ours, m[f"J_{args.ref_key}"]).mean()),
               "identical_cost_cheap": float(np.isclose(m.J_cheap, m.J_cheap_ref).mean()),
               "G_ours_timed": (m.J_cheap.sum() - m.J_ours.sum()) / max(m.J_cheap.sum(), 1e-9),
               "G_ref_same_frames": (m.J_cheap_ref.sum() - m[f"J_{args.ref_key}"].sum()) / max(m.J_cheap_ref.sum(), 1e-9)}
        pd.DataFrame([par]).to_csv(run / "parity.csv", index=False)
        print("parity:", {k: round(v, 4) for k, v in par.items()})


if __name__ == "__main__":
    main()
