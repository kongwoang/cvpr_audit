#!/usr/bin/env python
"""K31 re-runs (post-hoc; v6-S frozen, method_freeze_draft §2n; nothing here selects anything; pre-registered in
handoff/K31/STATUS.md). = k30_eval.py (md5 of the copy logged in STATUS) with two new parts; the per-frame code (draft,
detectors, crop keys, C1 lift, refiner v3a, loop k_loop.run_g, heuristics with the guard, + C1, own history) is unchanged.
  --part ttcpx  R2a: DESP's TTC loop c2 (spec_general.run with the TTC-warning consumer, + C1; as k28_p3, scored as there)
                and the hybrid tsp8 with the TTC consumer's own support at the TTC loop's per-frame crop px (x1.0 cap:
                budget = c2's crop px), scored by both consumers, refiner none / v3a (on the TTC support, as K30 V1c)
  --part idmS   R2b: IDMFollow (K30 V2b): ours-IDM, the frozen v6-S, v5.1-S (the S budget) and the hybrid tsp8 with the IDM
                support at K28's S budget; field size on the draft (round-1 field cells and field-crop px) of both loops;
                the decisive IDM object's estimated range / ttc, whether it came from a crop and whether the refiner changed it
  --part curvdev  item 3 dev check (--split dev: nuScenes new val, AV2 CFG-DEV with the ego mask; draft only): curvature,
                  field cells of the straight and of the curved consumer on the draft scene
  --part curv     item 3 test: every row scored by the curved-path AEB consumer (pre-registered rows; see STATUS)
  --part b2       item 4: v6-S with at most two detector calls (k_loop.run_bounded, N = 2) and v6-S (parity with K28)
results/k31/r2/<ds>_<part>[_<tag>]/<unit>.pkl (ttcpx, idmS), results/k31/r3/<ds>_<part>[_<tag>]/ (curvdev, curv),
results/k31/r4/<ds>_b2[_<tag>]/
usage: k31_eval.py --dataset nusc|av2|ahd --part ttcpx|idmS|curvdev|curv|b2 [--split test|dev] [--shard i/n] [--max_units n]
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

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

CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
TPOL = "T4+T2_1"
UNC = [(c, o) for c in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0) for o in ("conf", "ent")]
TCR = [(g, n) for g in (0.0, 0.25, 0.5, 1.0) for n in (0, 3, 1)]                 # n = 0: all previous objects
TUBE = [(near, far, w) for near in (5.0, 10.0) for far in (40.0, 70.0, 100.0) for w in ("fill", "lane")]
NWS = {"nusc": 1600, "av2": 2048, "ahd": 1920}
HEUR8 = ("unc2", "unc4", "tcr0", "tcr3", "cor0", "cor8", "tsp2", "tsp8")
MULTS = (0.75, 1.0, 1.25, 1.5)
AXIS = [(s, c) for s in (960, 1152, 1280) for c in (2.0, 2.5, 3.0)]
CONS = {"tau0.8": dict(thr_decel=0.8, thr_hard=2.8), "tau1.2": dict(thr_decel=1.2, thr_hard=4.2),
        "tr0.3": dict(reaction_t=0.3), "tr0.6": dict(reaction_t=0.6)}
PROBE = {"vc5": dict(v_closes=(5.0,)), "vc10": dict(v_closes=(10.0,)), "ped": dict(probes=(KL.PED,)),
         "truck": dict(probes=(KL.TRUCK,)), "union": dict(probes=(KL.PED, KL.CAR, KL.TRUCK))}
CONS_HEUR = ("tsp8", "tcr0", "tcr3", "unc2", "unc4")


# ------------------------------------------------------------------------- V2b: leader following, pre-registered (STATUS)
IDM_A, IDM_B, IDM_S0, IDM_T = 1.5, 2.0, 2.0, 1.5


def idm_demand(t, v, pp):
    z, lmin, lmax, ttc = (np.asarray(a, float) for a in t)
    d = np.zeros(len(z))
    if not len(z) or v <= 0.1:
        return d
    sel = P.in_corridor(lmin, lmax, pp.corridor_half_w) & (z <= pp.max_range) & (z > 0)
    if not sel.any():
        return d
    dv = np.clip(z[sel] / np.clip(ttc[sel], 0.1, 1e3), 0.0, 60.0)
    s = np.maximum(z[sel], 0.5)
    sstar = IDM_S0 + v * IDM_T + v * dv / (2.0 * math.sqrt(IDM_A * IDM_B))
    d[sel] = np.clip(IDM_A * (sstar / s) ** 2, 0.0, pp.a_cap)
    return d


def idm_action(t, v, pp):
    d = idm_demand(t, v, pp)
    return P.discrete_action(float(d.max()) if len(d) else 0.0, pp)


class IDMFollow:
    """max-composable consumer for spec_general: field = probe cells (static probe closing at v + v_close, in the corridor)
    whose IDM demand reaches the next action's threshold; support = objects whose demand reaches the current one."""

    def __init__(self, pp, v_close=0.0):
        self.pp, self.vc = pp, v_close

    def decide(self, t, v):
        return idm_action(t, v, self.pp)

    def field(self, t, v, rg, yg):
        a = idm_action(t, v, self.pp)
        if a == P.HARD_BRAKE or v <= 0.1:
            return np.zeros((len(rg), len(yg)), bool)
        thr = (self.pp.thr_decel, self.pp.thr_hard)[a]
        rr, yy = np.meshgrid(rg, yg, indexing="ij")
        inc = (yy - SP.OBJ_W / 2 < self.pp.corridor_half_w) & (yy + SP.OBJ_W / 2 > -self.pp.corridor_half_w)
        dv = v + self.vc
        sstar = IDM_S0 + v * IDM_T + v * dv / (2.0 * math.sqrt(IDM_A * IDM_B))
        dem = np.clip(IDM_A * (sstar / np.maximum(rr, 0.5)) ** 2, 0.0, self.pp.a_cap)
        return inc & (dem >= thr) & (rr <= self.pp.max_range)

    def support(self, t, v):
        a = idm_action(t, v, self.pp)
        if a == P.KEEP:
            return np.zeros(0, int)
        thr = (0.0, self.pp.thr_decel, self.pp.thr_hard)[a]
        return np.flatnonzero(idm_demand(t, v, self.pp) >= thr)


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


# ------------------------------------------------------------------- item 3: curved-path AEB consumer (pre-registered)
KMAX = 0.1


def kappa(yaw, v):
    return float(np.clip(yaw / max(v, 1.0), -KMAX, KMAX)) if np.isfinite(yaw) else 0.0


def path_sd(x, y, k):
    """ego (x fwd, y left) -> (arc length s along the constant-curvature path, lateral offset d, left positive)"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if abs(k) < 1e-6:
        return x, y
    R = 1.0 / k
    rho = np.hypot(x, y - R)
    d = np.sign(R) * (abs(R) - rho)
    th = np.arctan2(x, np.sign(R) * (R - y))
    s = abs(R) * th
    return np.where(x > 0, s, -1.0), d


def curved(t, k):
    z, lo, hi, ttc = (np.asarray(a, float) for a in t)
    if abs(k) < 1e-6 or not len(z):
        return z, lo, hi, ttc
    s, _ = path_sd(z, (lo + hi) / 2, k)
    _, d1 = path_sd(z, lo, k)
    _, d2 = path_sd(z, hi, k)
    s = np.where(s < 0, 1e6, s)                                           # behind along the path: beyond the range limit
    return s, np.minimum(d1, d2), np.maximum(d1, d2), ttc


def curved_action(t, v, k, pp):
    return P.discrete_action(P.required_decel(*curved(t, k), v, pp)[0], pp)


def curved_support(t, v, pp, k):
    tt = curved(t, k)
    a_max, a_obj = P.required_decel(*tt, v, pp)
    act = P.discrete_action(a_max, pp)
    if act == P.KEEP or not len(a_obj):
        return np.zeros(0, int)
    thr = pp.thr_decel if act == P.DECELERATE else pp.thr_hard
    return np.flatnonzero(np.asarray(a_obj) >= thr)


def curved_grid(k):
    """dsf.insertion_grid of the curved consumer (identical to it for kappa = 0)"""
    def f(mono_t, v, r_grid, y_grid, obj_w, p, v_close=0.0):
        a_cur, _ = P.required_decel(*curved(mono_t, k), v, p)
        act = P.discrete_action(a_cur, p)
        if act == P.HARD_BRAKE:
            return np.zeros((len(r_grid), len(y_grid)), bool)
        thr_next = p.thr_decel if act == P.KEEP else p.thr_hard
        rr, yy = np.meshgrid(np.asarray(r_grid, float), np.asarray(y_grid, float), indexing="ij")
        s, d = path_sd(rr, yy, k)
        a_ins = dsf._insert_demand(np.where(s > 0, s, 1e6), v, p, v_close)
        in_corr = (d - obj_w / 2 < p.corridor_half_w) & (d + obj_w / 2 > -p.corridor_half_w) & (s > 0)
        return (a_ins >= thr_next) & in_corr
    return f


def run_bounded_keep(cheap, v, cam, W, H, detect_crop, lift, pp, N, mode, v_closes=(0.0,), r_grid=SP.R_GRID, y_grid=SP.Y_GRID,
                     probes=None, field=True, ctx=SP.CTX, max_grow=2):
    """K31 variant B2K (pre-registered): k_loop.run_bounded (verbatim logic) whose pending edge-truncated crop detections are
    not dropped at the bound: mode "A" adds them to the merged scene as detected, mode "B" restores the draft box with the
    largest IoU (>= 0.3) to each of them. No extra detector call."""
    from rap.mono import box_iou
    probes = probes or (KL.CAR,)
    crops, grown, pending = [], [], []                  # pending: (box, crop, growth step, conf, coarse)
    full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
    scene, rounds, converged = dict(cheap), 0, False

    def merged(full_, restore=None):
        out = SP.centres_in(cheap["xyxy"], crops)
        if grown and len(full_["conf"]) and len(cheap["xyxy"]):
            out |= box_iou(np.asarray(cheap["xyxy"], float), full_["xyxy"]).max(1) >= 0.5
        if restore is not None:
            out &= ~restore
        fz = lift(full_) if len(full_["conf"]) else tuple(np.zeros(0) for _ in range(4))
        return {"xyxy": np.r_[cheap["xyxy"][~out], full_["xyxy"]], "conf": np.r_[cheap["conf"][~out], full_["conf"]],
                "coarse": np.r_[cheap["coarse"][~out], full_["coarse"]],
                "mono": tuple(np.r_[a[~out], b] for a, b in zip(cheap["mono"], fz)),
                "from_full": np.r_[np.zeros(int((~out).sum()), bool), np.ones(len(full_["conf"]), bool)]}

    def nms(f_):
        if len(f_["conf"]) <= 1:
            return f_
        o = np.argsort(-f_["conf"], kind="stable"); iou = box_iou(f_["xyxy"], f_["xyxy"]); keep = []
        for i in o:
            if not keep or iou[i, keep].max() < 0.5:
                keep.append(i)
        return {k: f_[k][np.sort(np.array(keep))] for k in f_}
    while True:
        new = []
        if field:
            fc = KL.field_crop_g(scene["mono"], v, cam, W, H, pp, v_closes, r_grid, y_grid, probes)
            if fc is not None and not SP.covered(fc, crops):
                new.append(fc)
        for k in SP.support(scene["mono"], v, pp):
            bc = KL.box_crop_ctx(scene["xyxy"][k], W, H, ctx)
            if not SP.covered(bc, crops + grown + new):
                new.append(bc)
        gs = [SP.grow(b, c, W, H) for b, c, *_ in pending]
        if not new and not gs:
            converged = True
            break
        if rounds >= N:
            if pending and mode == "A":
                add = {"xyxy": np.array([p_[0] for p_ in pending], float).reshape(-1, 4), "conf": np.array([p_[3] for p_ in pending], float),
                       "coarse": np.array([p_[4] for p_ in pending]).astype("<U8")}
                scene = merged(nms({k: np.concatenate([full[k], add[k]]) for k in full}))
            elif pending and mode == "B" and len(cheap["xyxy"]):
                M = box_iou(np.asarray(cheap["xyxy"], float), np.array([p_[0] for p_ in pending], float).reshape(-1, 4))
                restore = np.zeros(len(cheap["xyxy"]), bool)
                for j in range(M.shape[1]):
                    i = int(np.argmax(M[:, j]))
                    if M[i, j] >= 0.3:
                        restore[i] = True
                scene = merged(full, restore)
            break
        rounds += 1
        nxt = []
        for c in new:
            d = {k: np.asarray(x) for k, x in detect_crop(c).items()}
            if len(d["conf"]):
                top, other = SP.truncated(d["xyxy"], c, W, H, top_only=True)
                tr = other | top
                nxt += [(d["xyxy"][i], c, 0, d["conf"][i], d["coarse"][i]) for i in np.flatnonzero(tr)]
                d = {k: x[~tr] for k, x in d.items()}
            full = {k: np.concatenate([full[k], d[k]]) for k in full}
        for (b, _, step, _, _), g in zip(pending, gs):
            dg = {k: np.asarray(x) for k, x in detect_crop(g).items()}
            if not len(dg["conf"]):
                continue
            ix = np.clip(np.minimum(dg["xyxy"][:, 2], b[2]) - np.maximum(dg["xyxy"][:, 0], b[0]), 0, None)
            iy = np.clip(np.minimum(dg["xyxy"][:, 3], b[3]) - np.maximum(dg["xyxy"][:, 1], b[1]), 0, None)
            io = ix * iy / max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
            j = int(np.argmax(io))
            if io[j] < 0.5:
                continue
            if not SP.truncated(dg["xyxy"][j:j + 1], g, W, H)[0] or step == max_grow - 1:
                full = {k: np.concatenate([full[k], dg[k][j:j + 1]]) for k in full}
            else:
                nxt.append((dg["xyxy"][j], g, step + 1, dg["conf"][j], dg["coarse"][j]))
        crops += new
        grown += gs
        pending = nxt
        full = nms(full)
        scene = merged(full)
    scene.setdefault("from_full", np.zeros(len(scene["conf"]), bool))
    return scene, crops, grown, rounds, converged


class _DsfProxy:
    def __init__(self, fn):
        self.insertion_grid = fn

    def __getattr__(self, n):
        return getattr(dsf, n)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2", "ahd"])
    ap.add_argument("--part", required=True, choices=["ttcpx", "idmS", "curvdev", "curv", "b2", "b2dev", "b2v"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max_units", type=int, default=0)
    ap.add_argument("--units", default="", help="comma list of unit names (smoke)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--b2k_modes", default="A,B", help="part b2dev / b2v: B2K candidates to run")
    args = ap.parse_args()
    ds, part = args.dataset, args.part
    si, sn = (int(x) for x in args.shard.split("/"))
    sys.path.insert(0, str(B / "scripts" / "common"))
    from nusc_refiner_crop import crop64
    dev = torch.device("cuda:0")
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device="cuda:0", backend="torch")
    nets = {}
    ck = torch.load(REAL / "out" / "resnet18_v3a" / "image_refiner.pt", map_location=dev, weights_only=False)
    n3 = KT.Res18x3(len(ck["fmu"])); n3.load_state_dict(ck["state"]); nets["v3a"] = (n3.to(dev).eval(), ck["fmu"], ck["fsd"])
    ckh = torch.load(REAL / "out" / "k28_temporal" / "temporal_head.pt", map_location=dev, weights_only=False)
    head = KTT.TempHead(ckh["n_feat"]); head.load_state_dict(ckh["state"]); head = head.to(dev).eval()
    hmu, hsd = ckh["fmu"], ckh["fsd"]
    PP0, cp = P.PlannerParams(), P.CostParams()
    if ds == "nusc":
        CH, cheap_px, NW = Mode("ns_c320", 192, 320), 192 * 320, 1600
    elif ds == "av2":
        CH, cheap_px, NW = Mode("av2_c320", 320, 256), 320 * 256, 2048
    else:
        CH, cheap_px, NW = Mode("ahd_c320", 224, 320), 224 * 320, 1920
    sub = {"ttcpx": "r2", "idmS": "r2", "curvdev": "r3", "curv": "r3", "b2": "r4", "b2dev": "r4", "b2v": "r4"}[part]
    od = REAL / "results" / "k31" / sub / (f"{ds}_{part}" + (f"_{args.tag}" if args.tag else ""))
    GTY = {}
    if part == "curv":
        g_ = pd.read_pickle(REAL / "results" / "k30" / "gt" / f"{ds}.pkl")
        GTY = dict(zip(zip(g_["unit"], g_["frame"].astype(int)), g_["yaw_rate"].astype(float)))
    KG = None
    if part == "curvdev":
        _kg = importlib.util.spec_from_file_location("k30g", B / "scripts" / "A_method" / "k30_gt.py")
        KG = importlib.util.module_from_spec(_kg); _kg.loader.exec_module(KG)
        ndb = None
        if ds == "nusc":
            from rap.nusc import NuScenesDB
            from rap.paths import NUSCENES_TRAINVAL
            ndb = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
            nby = {ndb.scene_name(t_): t_ for t_ in ndb.scenes}
    od.mkdir(parents=True, exist_ok=True)
    FLOG = []
    _fcg = KL.field_crop_g

    def _fc_log(*a, **k):                                  # logs the field crops the loop computes
        r_ = _fcg(*a, **k)
        if r_ is not None:
            FLOG.append(np.asarray(r_, float))
        return r_
    KL.field_crop_g = _fc_log
    UNITS = {u for u in args.units.split(",") if u}
    SETTINGS = {"base": PP0} if part != "cons" else {"base": PP0, **{k: replace(PP0, **v) for k, v in CONS.items()}}
    unit_iter = K27.units(ds, args.max_units) if args.split == "dev" else K24.units(ds, args.max_units)
    for ui, (unit, cam, gen) in enumerate(unit_iter):
        if ui % sn != si or (UNITS and unit not in UNITS):
            continue
        fo = od / f"{unit}.pkl"
        if fo.exists():
            continue
        R, t = cam.transform("ego")
        P2m = cam.calib.P2
        fx, fy, cx, cy = P2m[0, 0], P2m[1, 1], P2m[0, 2], P2m[1, 2]
        dt = float(cam.dt)
        st = {}
        prev_c, imgs, rows = None, [], []
        prev_a = {k: None for k in SETTINGS}                   # cheap's previous action per setting (the cost's jerk term)
        prev_a3 = None
        prev_ac = None
        YAW = {}
        if part == "curvdev":
            if ds == "nusc":
                toks = ndb.samples(nby[unit])
                ts_ = [ndb._t["sample"][tk]["timestamp"] / 1e6 for tk in toks]
                yr_ = KG.yaw_rates([ndb._global_to_ego(tk)[0] for tk in toks], ts_)
            else:
                ts_ = [x_ * 1e-9 for x_ in cam.frame_ts]
                yr_ = KG.yaw_rates([cam._pose(x_)[0] for x_ in cam.frame_ts], ts_)
            YAW = {i_: float(y_) for i_, y_ in enumerate(yr_)}
        for fr, path, v, g, cached in gen():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            NATIVE = W * H
            rec = {"unit": unit, "frame": fr, "v": v, "gt": not isinstance(cached, str)}
            gt_t = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc")) if len(g) else tuple(np.zeros(0) for _ in range(4))
            gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
            gtc = np.asarray(g["coarse"]).astype(str) if len(g) else np.zeros(0, str)
            GT = {}
            for sn_, pp_ in SETTINGS.items():
                a_gt, a_obj = P.required_decel(*gt_t, v, pp_) if len(g) else (0.0, np.zeros(0))
                rel = np.flatnonzero(np.asarray(a_obj) >= pp_.thr_decel) if len(g) else np.zeros(0, int)
                dec = int(np.argmax(a_obj)) if a_gt > 0 and len(g) else -1
                GT[sn_] = (a_gt, rel, dec)
                sfx = "" if sn_ == "base" else f"@{sn_}"
                rec.update({f"a_gt{sfx}": a_gt, f"act_gt{sfx}": P.discrete_action(a_gt, pp_), f"n_relevant{sfx}": len(rel), f"dec{sfx}": dec})
            rec["act2_gt"] = IE.c2_action(gt_t, PP0)
            if part == "idmS":
                d3 = idm_demand(gt_t, v, PP0) if len(g) else np.zeros(0)
                a3_gt = float(d3.max()) if len(d3) else 0.0
                rec.update({"a3_gt": a3_gt, "act3_gt": P.discrete_action(a3_gt, PP0), "dec3": int(np.argmax(d3)) if a3_gt > 0 else -1})

            def geom(dd, pv=None):
                gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def keep(d):
                k = d["conf"] >= IE.THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k],
                        "entropy": np.asarray(d.get("entropy", np.zeros(len(d["conf"]))))[k]}

            def act(tt, pp_=PP0):
                return P.discrete_action(P.required_decel(*tt, v, pp_)[0], pp_)

            def cost(a, sn_="base"):
                return P.decision_cost(a, GT[sn_][0], prev_a[sn_], SETTINGS[sn_], cp)["J"]

            def ckey(cr, scale):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                xe, ye = min(x2, W), min(y2, H)
                nw = max(32, int(math.ceil((xe - x1) * scale / 32)) * 32)
                nh = max(32, int(math.ceil((ye - y1) * scale / 32)) * 32)
                return (x1, y1, x2, y2, nh, nw)
            cache, rcache, last = {}, {}, {}

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
                    ck_ = (id(dt_),) + key
                    if ck_ not in cache:
                        cache[ck_] = run_det(dt_, img, key)
                    out = {k: x.copy() for k, x in cache[ck_].items()}
                    if log is not None:
                        for b in out["xyxy"]:
                            log[tuple(b.tolist())] = key
                    return out
                return detect_crop

            def lift_pol(name, pol, sc):
                """k28_p3.lift_pol without the T3 retro-detection (TPOL = T4+T2_1 has t3 False)"""
                h = st.get(name, {})
                assert not pol["t3"]

                def f(full):
                    z, lo, hi, _ = geom(full)
                    tt, _ = ttc_ss(full["xyxy"], h.get("s1"), h.get("s2"), sc, dt, pol["gate"], pol["t4"], None)
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
                        r = rh[n]
                        ttc[i] = min(1 / r, 1e3) if r > 1e-3 / dt else np.inf
                    return z, lo, hi, ttc
                return f

            def push(name, scene):
                h = st.setdefault(name, {})
                ff = np.asarray(scene["from_full"], bool)
                h["s2"] = h.get("s1")
                h["s1"] = np.asarray(scene["xyxy"], float).reshape(-1, 4)[ff]
                h["merged"] = np.asarray(scene["xyxy"], float).reshape(-1, 4)

            def refine(scene, name, sup_fn):
                tt = [np.asarray(a, float).copy() for a in scene["mono"]]
                if name == "none":
                    return tuple(tt)
                sup = [p_ for p_ in sup_fn(scene["mono"]) if scene["from_full"][p_]]
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

            def brake_sup(pp_):
                return lambda mono: SP.support(mono, v, pp_)

            def dec_info(prefix, scene, tr_, crops, sn_="base", pp_=PP0, field=None):
                """the decisive annotated object of setting sn_ in the pipeline's final scene"""
                dec = GT[sn_][2]
                if dec < 0:
                    return
                b = gtb[dec:dec + 1]
                rec[f"dec_in_crop_{prefix}"] = bool(SP.centres_in(b, list(crops))[0]) if len(crops) else False
                if field is not None:
                    rec[f"dec_in_field_{prefix}"] = bool(SP.centres_in(b, list(field))[0]) if len(field) else False
                m = int(IE.matched(b, gtc[dec:dec + 1], scene["xyxy"], scene["coarse"])[0])
                rec[f"dec_found_{prefix}"] = m >= 0
                if m >= 0:
                    _, dem = P.required_decel(*tr_, v, pp_)
                    rec.update({f"dec_z_{prefix}": float(tr_[0][m]), f"dec_ttc_{prefix}": float(tr_[3][m]),
                                f"dec_lmin_{prefix}": float(tr_[1][m]), f"dec_lmax_{prefix}": float(tr_[2][m]), f"dec_dem_{prefix}": float(np.asarray(dem)[m]),
                                f"dec_from_crop_{prefix}": bool(scene["from_full"][m])})

            def score(prefix, scene, refs=("none", "v3a"), sn_="base", sup_fn=None, c2=True, crops=None, field=None, dec=False):
                pp_ = SETTINGS[sn_]
                sfx = "" if sn_ == "base" else f"@{sn_}"
                sup_fn = sup_fn or brake_sup(pp_)
                for r_ in refs:
                    tr_ = refine(scene, r_, sup_fn)
                    a1 = act(tr_, pp_)
                    rec[f"a_{prefix}|{r_}{sfx}"], rec[f"J_{prefix}|{r_}{sfx}"] = a1, cost(a1, sn_)
                    if c2 and sn_ == "base":
                        a2 = IE.c2_action(tr_, PP0)
                        rec[f"a2_{prefix}|{r_}"], rec[f"J2_{prefix}|{r_}"] = a2, float(IE.C2_COST[rec["act2_gt"], a2])
                    if dec and r_ == refs[-1]:
                        dec_info(prefix + sfx, scene, tr_, crops or [], sn_, pp_, field)
                rel = GT[sn_][1]
                if len(rel):
                    rec[f"found_{prefix}{sfx}"] = IE.matched(gtb[rel], gtc[rel], scene["xyxy"], scene["coarse"]) >= 0

            def score3(prefix, scene, sup_fn, refs=("none", "v3a")):
                """part idm: scored by IDMFollow; refiner on sup_fn's objects"""
                for r_ in refs:
                    tr_ = refine(scene, r_, sup_fn)
                    a3 = idm_action(tr_, v, PP0)
                    rec[f"a3_{prefix}|{r_}"] = a3
                    rec[f"J3_{prefix}|{r_}"] = P.decision_cost(a3, rec["a3_gt"], prev_a3, PP0, cp)["J"]
                if rec.get("dec3", -1) >= 0:
                    d_ = rec["dec3"]
                    m_ = int(IE.matched(gtb[d_:d_ + 1], gtc[d_:d_ + 1], scene["xyxy"], scene["coarse"])[0])
                    rec[f"dec3_found_{prefix}"] = m_ >= 0
                    if m_ >= 0:
                        t0, t1 = refine(scene, "none", sup_fn), refine(scene, refs[-1], sup_fn)
                        rec.update({f"dec3_z_{prefix}": float(t1[0][m_]), f"dec3_ttc_{prefix}": float(t1[3][m_]),
                                    f"dec3_from_crop_{prefix}": bool(scene["from_full"][m_]),
                                    f"dec3_refined_{prefix}": bool(abs(float(t1[0][m_]) - float(t0[0][m_])) > 1e-9)})

            def scene_dict(dd, mono):
                return {"xyxy": dd["xyxy"], "conf": dd["conf"], "coarse": dd["coarse"], "mono": mono,
                        "from_full": np.ones(len(dd["conf"]), bool)}

            def loop(nm, ctx, side, sn_="base", refs=("none", "v3a"), dec=False, score_it=True, t4=False, mc=None, **kw):
                """k28_p3.loop: C1 lift (v6-S), or the T4 policy TPOL (v5.1-S, t4=True); keys and history '<nm>[@<setting>]'"""
                scale = side / NW
                key = nm + ("" if sn_ == "base" else f"@{sn_}")
                tpx, log, cnt = [0.0], {}, [0]
                FLOG.clear()
                lf = lift_pol(key, parse(TPOL), scale) if t4 else lift_c(key, "C1", scale)
                sc, crops, grown, rounds, conv = KL.run_g(base, v, cam, W, H, make_detect(tpx, scale, log, None, cnt), lf,
                                                          SETTINGS[sn_], v_closes=kw.pop("v_closes", (0.0,)), ctx=ctx, max_calls=mc, **kw)
                fl = list(FLOG)
                if score_it:
                    score(nm, sc, refs, sn_, crops=list(crops) + list(grown), field=fl, dec=dec)
                rec[f"px_{key}"] = (cheap_px + tpx[0]) / NATIVE
                rec[f"ncrops_{key}"], rec[f"nguard_{key}"], rec[f"rounds_{key}"] = len(crops), len(grown), rounds
                rec[f"conv_{key}"] = bool(conv)
                last[key] = (crops, grown, sc, fl)
                push(key, sc)
                return tpx[0]
            # ---- draft (as k28_p3: the cached draft on nuScenes, online elsewhere)
            if isinstance(cached, dict):
                ch = cached
            else:
                ch = keep(det.detect(img, CH)[0]); ch["mono"] = geom(ch, prev_c)
                prev_c = {"xyxy": ch["xyxy"], "coarse": ch["coarse"]}
            base = {k: ch[k] for k in ("xyxy", "conf", "coarse", "mono")}
            for sn_, pp_ in SETTINGS.items():
                a_c = act(ch["mono"], pp_)
                sfx = "" if sn_ == "base" else f"@{sn_}"
                rec[f"a_cheap{sfx}"], rec[f"J_cheap{sfx}"] = a_c, cost(a_c, sn_)
                rel = GT[sn_][1]
                if len(rel):
                    rec[f"found_cheap{sfx}"] = IE.matched(gtb[rel], gtc[rel], ch["xyxy"], ch["coarse"]) >= 0
            rec["a2_cheap"] = IE.c2_action(ch["mono"], PP0)
            rec["J2_cheap"] = float(IE.C2_COST[rec["act2_gt"], rec["a2_cheap"]])
            rec["px_cheap"] = cheap_px / NATIVE
            scale9 = 960 / NW

            def canvas(cr):
                k_ = ckey(cr, scale9)
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

            def select(fam, i, name, bud, sup_d, pp_):
                """E construction (k28_dev.py / k28_p3.py), for one budget; returns (crops, tube rect or None, support crops)"""
                if fam == "unc":
                    ctx, o = UNC[i]
                    order = np.lexsort((-ent, ch["conf"])) if o == "conf" else np.lexsort((ch["conf"], -ent))
                    return fill_px([KL.box_crop_ctx(ch["xyxy"][k], W, H, ctx) for k in order], bud)[0], None, []
                if fam == "tcr":                                   # previous merged objects of this row
                    gr, n = TCR[i]
                    ps_ = st.get(name, {}).get("state")
                    cands = []
                    if ps_ is not None and len(ps_[0]):
                        pxy, pmono = ps_
                        _, dem = P.required_decel(*pmono, v, pp_)
                        order = np.lexsort((pmono[0], -np.asarray(dem)))
                        if n:
                            order = order[:n]
                        cands = [SP.box_crop(np.r_[pxy[k][:2] - gr * (pxy[k][2:] - pxy[k][:2]),
                                                   pxy[k][2:] + gr * (pxy[k][2:] - pxy[k][:2])], W, H) for k in order]
                    return fill_px(cands, bud)[0], None, []
                if fam == "cor":
                    near, far, wm = TUBE[i]
                    tb = tube(bud, near, far, wm) if bud > 0 else []
                    return tb, (tb[0] if tb else None), []
                if fam == "tsp":
                    near, far, wm = TUBE[i]
                    sp_, used_ = fill_px([SP.box_crop(ch["xyxy"][k], W, H) for k in sup_d], bud) if bud > 0 else ([], 0)
                    tb = tube(bud - used_, near, far, wm) if bud - used_ > 0 else []
                    return sp_ + tb, (tb[0] if tb else None), sp_
                raise ValueError(fam)

            def heur(cfg, tag, bud, sn_="base", sup_d=None, sup_fn=None, dec=False, c2=True, scorer=None):
                """one heuristic + C1, refiner none / v3a; name '<cfg>@<tag>+C1'"""
                fam, i = cfg[:3], int(cfg[3:])
                pp_ = SETTINGS[sn_]
                name = f"{cfg}@{tag}+C1" + ("" if sn_ == "base" else f"@{sn_}")
                sup_d = SP.support(ch["mono"], v, pp_) if sup_d is None else sup_d
                sel_, tb, spc = select(fam, i, name, bud, sup_d, pp_)
                tpx = [0.0]
                sh, allc = IE.merge_once(base, sel_, SP.guarded(make_detect(tpx, scale9), W, H), lift_c(name, "C1", scale9))
                if scorer is not None:
                    scorer(name, sh)
                else:
                    score(f"{cfg}@{tag}+C1", sh, ("none", "v3a"), sn_, sup_fn=sup_fn, c2=c2, crops=allc, dec=dec)
                rec[f"px_{name}"] = (cheap_px + tpx[0]) / NATIVE
                if dec:
                    rec[f"tube_{name}"] = None if tb is None else np.asarray(tb, float)
                    rec[f"supcrops_{name}"] = [np.asarray(c, float) for c in spc]
                push(name, sh)
                if fam == "tcr":
                    st[name]["state"] = (np.asarray(sh["xyxy"], float).reshape(-1, 4), tuple(np.asarray(q, float) for q in sh["mono"]))
            # ================================================================================================= parts
            if part == "ttcpx":
                # R2a. DESP's TTC loop as k28_p3 (refiner on the braking support there; score() default), then the hybrid with
                # the TTC consumer's own support at the TTC loop's crop px (x1.0: 1.0 x (cheap + c2 crop px) - cheap px)
                c2px = [0.0]
                sc2, cr2, _ = SG.run(base, v, cam, W, H, make_detect(c2px, scale9), lift_c("c2", "C1", scale9),
                                     IE.TTCWarn(PP0, 0.0), max_rounds=3, guard=True)
                score("c2", sc2)
                rec["px_c2"], rec["ncrops_c2"] = (cheap_px + c2px[0]) / NATIVE, len(cr2)
                push("c2", sc2)
                sup2 = IE.TTCWarn(PP0, 0.0).support(ch["mono"], v)
                fam_name = "tsp8c2@c2+C1"
                sel_, tb, _ = select("tsp", 8, fam_name, c2px[0], sup2, PP0)
                tpx = [0.0]
                sh, allc = IE.merge_once(base, sel_, SP.guarded(make_detect(tpx, scale9), W, H), lift_c(fam_name, "C1", scale9))
                score(fam_name, sh, ("none", "v3a"), "base", sup_fn=lambda mono: IE.TTCWarn(PP0, 0.0).support(mono, v))
                rec[f"px_{fam_name}"] = (cheap_px + tpx[0]) / NATIVE
                push(fam_name, sh)
            elif part == "idmS":
                cons_ = IDMFollow(PP0, 0.0)
                rec["a3_cheap"] = idm_action(ch["mono"], v, PP0)
                rec["J3_cheap"] = P.decision_cost(rec["a3_cheap"], rec["a3_gt"], prev_a3, PP0, cp)["J"]
                isup = lambda mono: cons_.support(mono, v)                       # noqa: E731
                # field size on the draft (round 1): cells of the consumer's field and the px of its field crop
                f_i = cons_.field(ch["mono"], v, SP.R_GRID, SP.Y_GRID)
                fp = SG.footprint(f_i, cam, W, H, SP.R_GRID, SP.Y_GRID)
                rec["field_cells_r1_oursIDM"] = int(np.asarray(f_i).sum())
                rec["fieldpx_r1_oursIDM"] = 0.0 if fp is None else float((fp[2] - fp[0]) * (fp[3] - fp[1])) / NATIVE
                f_b = dsf.insertion_grid(ch["mono"], v, SP.R_GRID, SP.Y_GRID, KL.CAR[0], PP0, 0.0)
                fb = _fcg(ch["mono"], v, cam, W, H, PP0, (0.0,), SP.R_GRID, SP.Y_GRID, (KL.CAR,))
                rec["field_cells_r1_v6S"] = int(np.asarray(f_b).sum())
                rec["fieldpx_r1_v6S"] = 0.0 if fb is None else float((fb[2] - fb[0]) * (fb[3] - fb[1])) / NATIVE
                # ours driven by IDMFollow (as K30 V2b), refiner on IDMFollow's support
                ipx = [0.0]
                sc3, cr3, rd3 = SG.run(base, v, cam, W, H, make_detect(ipx, scale9), lift_c("idm", "C1", scale9), cons_,
                                       max_rounds=3, guard=True)
                score3("oursIDM", sc3, isup)
                rec["px_oursIDM"], rec["ncrops_oursIDM"], rec["rounds_oursIDM"] = (cheap_px + ipx[0]) / NATIVE, len(cr3), rd3
                push("idm", sc3)
                # the frozen v6-S (braking-driven), deployed refiner (braking support), scored by IDMFollow
                loop("v6S", 2.0, 960, score_it=False)
                score3("v6S", last["v6S"][2], brake_sup(PP0))
                rec["nfield_v6S"] = len(last["v6S"][3])
                # K28's S budget (v5.1-S crop px) and the hybrid (IDMFollow support + TUBE[8]) at it
                crop51 = loop("v51S", 2.0, 960, t4=True, score_it=False)
                sup_i = cons_.support(ch["mono"], v)
                heur("tsp8", "S", crop51, sup_d=sup_i, scorer=lambda nm_, sh_: score3(nm_, sh_, isup))
                prev_a3 = rec["a3_cheap"]
            elif part == "curvdev":
                k_ = kappa(YAW.get(int(fr), 0.0), v)
                f_s = dsf.insertion_grid(ch["mono"], v, SP.R_GRID, SP.Y_GRID, KL.CAR[0], PP0, 0.0)
                f_c = curved_grid(k_)(ch["mono"], v, SP.R_GRID, SP.Y_GRID, KL.CAR[0], PP0, 0.0)
                rec.update({"yaw_rate": YAW.get(int(fr), np.nan), "kappa": k_, "field_cells_straight": int(f_s.sum()),
                            "field_cells_curved": int(f_c.sum()), "a_cheap_curved": curved_action(ch["mono"], v, k_, PP0),
                            "act_gt_curved": curved_action(gt_t, v, k_, PP0) if len(g) else 0})
            elif part == "curv":
                k_ = kappa(GTY.get((unit, int(fr)), np.nan), v)
                rec["kappa"] = k_
                a_gc, aobj_c = P.required_decel(*curved(gt_t, k_), v, PP0) if len(g) else (0.0, np.zeros(0))
                rec["a_gt_c"], rec["act_gt_c"] = float(a_gc), P.discrete_action(a_gc, PP0)
                decc = int(np.argmax(aobj_c)) if a_gc > 0 and len(g) else -1
                rec["decc"] = decc
                csup = lambda mono: curved_support(mono, v, PP0, k_)                      # noqa: E731

                def score_c(prefix, scene, sup_fn=None):
                    for r_ in ("none", "v3a"):
                        tr_ = refine(scene, r_, sup_fn or csup)
                        a_ = curved_action(tr_, v, k_, PP0)
                        rec[f"ac_{prefix}|{r_}"] = a_
                        rec[f"Jc_{prefix}|{r_}"] = P.decision_cost(a_, rec["a_gt_c"], prev_ac, PP0, cp)["J"]
                    if decc >= 0:
                        rec[f"decc_found_{prefix}"] = int(IE.matched(gtb[decc:decc + 1], gtc[decc:decc + 1], scene["xyxy"], scene["coarse"])[0]) >= 0
                rec["ac_cheap"] = curved_action(ch["mono"], v, k_, PP0)
                rec["Jc_cheap"] = P.decision_cost(rec["ac_cheap"], rec["a_gt_c"], prev_ac, PP0, cp)["J"]
                for L in (960, NWS[ds]):
                    md, ms = wf_mode(ds, L)
                    nm = f"wf{L}"
                    d = keep(det.detect(img, md)[0])
                    score_c(f"{nm}+C1", scene_dict(d, lift_c(nm, "C1", ms)(d)))
                    rec[f"px_{nm}"] = md.net_h * md.net_w / NATIVE
                    push(nm, scene_dict(d, None))
                crop51 = loop("v51S", 2.0, 960, t4=True, score_it=False)
                # frozen straight v6-S, deployed refiner (straight support), scored by the curved consumer
                loop("v6S", 2.0, 960, score_it=False)
                score_c("v6S", last["v6S"][2], brake_sup(PP0))
                # v6-S with the curved consumer: the frozen loop with its field and support replaced
                o_dsf, o_sp = KL.dsf, KL.SP
                KL.dsf = _DsfProxy(curved_grid(k_))
                KL.SP = _Proxy(SP, lambda mono, v_, pp_: curved_support(mono, v_, pp_, k_))
                try:
                    loop("v6Sc", 2.0, 960, score_it=False)
                finally:
                    KL.dsf, KL.SP = o_dsf, o_sp
                score_c("v6Sc", last["v6Sc"][2])
                rec["nfield_v6Sc"] = len(last["v6Sc"][3])
                # straight hybrid as K28 (straight support + straight tube), uncertainty, at the S budget
                heur("tsp8", "S", crop51, scorer=lambda nm_, sh_: score_c(nm_, sh_))
                heur("unc2", "S", crop51, scorer=lambda nm_, sh_: score_c(nm_, sh_))
                tb_s = tube(crop51 - fill_px([SP.box_crop(ch["xyxy"][q], W, H) for q in SP.support(ch["mono"], v, PP0)], crop51)[1], *TUBE[8])

                # curved hybrid: the curved consumer's support + a tube along the curved path, at the S budget
                def tube_c(bud):
                    near, far = TUBE[8][0], TUBE[8][1]
                    if abs(k_) < 1e-6:
                        return tube(bud, near, far, "fill")
                    ss = np.arange(near, far + 1e-6, 2.0)
                    R_ = 1.0 / k_
                    px_, py_ = R_ * np.sin(ss / R_), R_ * (1 - np.cos(ss / R_))
                    nx_, ny_ = -np.sin(ss / R_), np.cos(ss / R_)
                    best = []
                    for half_m in np.arange(0.5, 12.01, 0.25):
                        X_ = np.r_[px_ + half_m * nx_, px_ - half_m * nx_]
                        Y_ = np.r_[py_ + half_m * ny_, py_ - half_m * ny_]
                        ok = X_ > 1.0
                        if not ok.any():
                            break
                        X_, Y_ = X_[ok], Y_[ok]
                        u_ = cx - fx * Y_ / X_
                        vb_ = cy + fy * cam.cam_h / X_
                        vt_ = vb_ - fy * 1.6 / X_
                        c = np.array([max(u_.min(), 0), max(vt_.min(), 0), min(u_.max(), W), min(vb_.max(), H)])
                        if c[2] - c[0] < 8 or c[3] - c[1] < 8 or canvas(c) > bud:
                            break
                        best = [c]
                    return best
                sup_c = curved_support(ch["mono"], v, PP0, k_)
                spc_, usedc = fill_px([SP.box_crop(ch["xyxy"][q], W, H) for q in sup_c], crop51) if crop51 > 0 else ([], 0)
                tb_c = tube_c(crop51 - usedc) if crop51 - usedc > 0 else []
                nmc = "tsp8c@S+C1"
                tpx = [0.0]
                sh, allc = IE.merge_once(base, spc_ + tb_c, SP.guarded(make_detect(tpx, scale9), W, H), lift_c(nmc, "C1", scale9))
                score_c(nmc, sh)
                rec[f"px_{nmc}"] = (cheap_px + tpx[0]) / NATIVE
                push(nmc, sh)
                if decc >= 0:
                    bdec = gtb[decc:decc + 1]
                    rec["decc_in_straight_tube"] = bool(SP.centres_in(bdec, tb_s)[0]) if len(tb_s) else False
                    rec["decc_in_curved_tube"] = bool(SP.centres_in(bdec, tb_c)[0]) if len(tb_c) else False
                    rec["decc_range"], rec["decc_cls"] = float(gt_t[0][decc]), str(gtc[decc])
                prev_ac = rec["ac_cheap"]
            elif part == "b2":
                loop("v6S", 2.0, 960)
                loop("v6Sb2", 2.0, 960, mc=2)
            elif part in ("b2dev", "b2v"):
                loop("v6S", 2.0, 960)
                loop("v6Sb2", 2.0, 960, mc=2)
                for md_ in [m_ for m_ in args.b2k_modes.split(",") if m_]:
                    nm = f"v6Sb2{md_}"
                    tpx, cnt = [0.0], [0]
                    sc_k, crk, grk, rdk, cvk = run_bounded_keep(base, v, cam, W, H, make_detect(tpx, 960 / NW, None, None, cnt),
                                                                lift_c(nm, "C1", 960 / NW), PP0, 2, md_, ctx=2.0)
                    score(nm, sc_k)
                    rec[f"px_{nm}"], rec[f"rounds_{nm}"], rec[f"conv_{nm}"] = (cheap_px + tpx[0]) / NATIVE, rdk, bool(cvk)
                    push(nm, sc_k)
            imgs = (imgs + [img])[-2:]
            for sn_, pp_ in SETTINGS.items():
                prev_a[sn_] = act(ch["mono"], pp_)
            rows.append(rec)
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{ui}] {unit}: {len(rows)} frames", flush=True)
    print(f"K30 {part.upper()} {ds} DONE", flush=True)


if __name__ == "__main__":
    main()
