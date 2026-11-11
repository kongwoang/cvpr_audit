#!/usr/bin/env python
"""K42 harness (k42_vuw_20261111/build.py from k39_eval.py; part k42 added). K30 (post-hoc; v6-S frozen, method_freeze_draft §2n; nothing here selects anything). GPU re-runs on the K28 test units
(k24_test.units: nuScenes new test 228 scenes, AV2 val 150 logs with the ego mask through k10_masked_run.py, Argoverse-HD val
24 sequences). The per-frame code is k28_p3.py's (md5 8b791fa2): same draft, detectors, crop keys, C1 lift, refiner v3a,
loop (k_loop.run_g), heuristics (E construction, merge_once with the guard, + C1, own history). Parts:
  --part match  V1a / V1c / V2 / V3: v5.1-S, v6-S (field crops logged), and the 8 S heuristics unc2 / unc4 / tcr0 / tcr3 / cor0
                / cor8 / tsp2 / tsp8 + C1 at a per-frame crop budget m x (cheap px + v6-S crop px) - cheap px,
                m in {0.75, 1.0, 1.25, 1.5}; the hybrid tsp8 with the TTC consumer's own support (instr_eval.TTCWarn) and with
                the braking support, at K28's S budget (v5.1-S px) and at m = 1.0. Every pipeline scored by the braking
                controller (a / J) and the TTC-warning consumer (a2 / J2), refiner none / v3a; decision-relevant recall and the
                decisive annotated object: matched?, inside a crop?, its estimated range / ttc / demand (v6-S, hybrid, corridor).
  --part axis   V1b: v6-S with crop long side {960, 1152, 1280} x support context {2, 2.5, 3} (a post-hoc axis, not a
                configuration; (960, 2) is v6-S itself)
  --part idm    V2b: the leader-following consumer IDMFollow (pre-registered in handoff/K30/STATUS.md): cheap, whole frame
                u960 / native + C1 + v3a, ours driven by IDMFollow (spec_general.run, + C1), the frozen v6-S, and the hybrid
                (IDMFollow support + TUBE[8]) / uncertainty unc2 at ours-IDM's crop px; refiner on the scoring consumer's support
  --part cons   V5b / V5c: consumer parameters (thr_decel, thr_hard) x {0.8, 1.2}, reaction_t {0.3, 0.6} (GT action, field,
                support, refiner support and heuristic rankings all recomputed), and probe assumptions for v6-S only (v_close 5 /
                10 m/s; probe pedestrian / truck / union of pedestrian, car, truck); pipelines v6-S, v5.1-S (the S budget),
                whole frame u960 / native + C1, tsp8 / tcr0 / tcr3 / unc2 / unc4 @S + C1; refiner none / v3a
results/k30/<part>/<ds>/<unit>.pkl
usage: k30_eval.py --dataset nusc|av2|ahd --part match|axis|idm|cons [--shard i/n] [--max_units n] [--units u1,u2]
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
from rap.mono import box_iou, predicted_geometry                                          # noqa: E402
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
R_EXT = np.array([8.0, 12.5, 17.5, 22.5, 27.5, 35.0, 45.0, 55.0, 70.0, 80.0])   # = k16_attr / k28_p3 (K16 d)
Y_EXT = np.arange(-9.0, 9.01, 1.5)
CHAIN42 = (("a", {}, False), ("b", {}, True), ("c1", dict(v_closes=(10.0,)), True), ("c2", dict(v_closes=(20.0,)), True),
           ("d", dict(v_closes=(20.0,), r_grid=R_EXT, y_grid=Y_EXT), True),
           ("e", dict(v_closes=(20.0,), r_grid=R_EXT, y_grid=Y_EXT, probes=(KL.PED, KL.CAR, KL.TRUCK)), True),
           ("f", dict(v_closes=(20.0,), r_grid=R_EXT, y_grid=Y_EXT, probes=(KL.PED, KL.CAR, KL.TRUCK), max_rounds=10), True))

SMOKE_CHECK = os.environ.get("K39_SMOKE_CHECK") == "1"
AXIS = [(s, c) for s in (960, 1152, 1280) for c in (2.0, 2.5, 3.0)]
CONS = {"tau0.8": dict(thr_decel=0.8, thr_hard=2.8), "tau1.2": dict(thr_decel=1.2, thr_hard=4.2),
        "tr0.3": dict(reaction_t=0.3), "tr0.6": dict(reaction_t=0.6)}
PROBE = {"vc5": dict(v_closes=(5.0,)), "vc10": dict(v_closes=(10.0,)), "ped": dict(probes=(KL.PED,)),
         "truck": dict(probes=(KL.TRUCK,)), "union": dict(probes=(KL.PED, KL.CAR, KL.TRUCK))}
CONS_HEUR = ("tsp8", "tcr0", "tcr3", "unc2", "unc4")


# ------------------------------------------------------------------------- V2b: leader following, pre-registered (STATUS)
IDM_A, IDM_B, IDM_S0, IDM_T = 1.5, 2.0, 2.0, 1.5



# ---------------------------------------------------------------------------------------------------- K42 helpers
_COV_ORIG = SP.covered
COVLOG = {"on": False, "calls": []}


def union_frac(crop, crops):
    """exact share of crop's area inside the union of crops (coordinate compression)"""
    x1, y1, x2, y2 = (float(a) for a in crop)
    area = max((x2 - x1) * (y2 - y1), 1e-6)
    R = []
    for c in crops:
        a, b, cc, d = max(x1, float(c[0])), max(y1, float(c[1])), min(x2, float(c[2])), min(y2, float(c[3]))
        if cc > a and d > b:
            R.append((a, b, cc, d))
    if not R:
        return 0.0
    xs = sorted({r[0] for r in R} | {r[2] for r in R})
    ys = sorted({r[1] for r in R} | {r[3] for r in R})
    tot = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            mx, my = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(r[0] <= mx <= r[2] and r[1] <= my <= r[3] for r in R):
                tot += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    return tot / area


def covered_logged(crop, crops, tol=0.95):
    r = _COV_ORIG(crop, crops, tol)
    if COVLOG["on"]:
        uf = union_frac(crop, crops) if crops else 0.0
        COVLOG["calls"].append((bool(r), uf, len(crops)))
    return r


def covered_union(crop, crops, tol=0.95):
    """K42-4(b) strict rule: skip only if the candidate lies entirely inside the union of earlier crops"""
    return bool(crops) and union_frac(crop, crops) >= 1.0 - 1e-9


SP.covered = covered_logged

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


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2", "ahd"])
    ap.add_argument("--part", required=True, choices=["k39", "k42"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max_units", type=int, default=0)
    ap.add_argument("--units", default="", help="comma list of unit names (smoke)")
    ap.add_argument("--tag", default="")
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
    od = REAL / "results" / ("k42" if part == "k42" else "k39") / ("vuw" + (f"_{args.tag}" if args.tag else "")) / ds
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
    for ui, (unit, cam, gen) in enumerate(K24.units(ds, args.max_units)):
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
            if part == "idm":
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
                    rec[f"dec3_found_{prefix}"] = int(IE.matched(gtb[d_:d_ + 1], gtc[d_:d_ + 1], scene["xyxy"], scene["coarse"])[0]) >= 0

            def scene_dict(dd, mono):
                return {"xyxy": dd["xyxy"], "conf": dd["conf"], "coarse": dd["coarse"], "mono": mono,
                        "from_full": np.ones(len(dd["conf"]), bool)}

            def loop(nm, ctx, side, sn_="base", refs=("none", "v3a"), dec=False, score_it=True, t4=False, **kw):
                """k28_p3.loop: C1 lift (v6-S), or the T4 policy TPOL (v5.1-S, t4=True); keys and history '<nm>[@<setting>]'"""
                scale = side / NW
                key = nm + ("" if sn_ == "base" else f"@{sn_}")
                tpx, log, cnt = [0.0], {}, [0]
                FLOG.clear()
                lf = lift_pol(key, parse(TPOL), scale) if t4 else lift_c(key, "C1", scale)
                sc, crops, grown, rounds, conv = KL.run_g(base, v, cam, W, H, make_detect(tpx, scale, log, None, cnt), lf,
                                                          SETTINGS[sn_], v_closes=kw.pop("v_closes", (0.0,)), ctx=ctx, max_calls=None, **kw)
                fl = list(FLOG)
                if score_it:
                    score(nm, sc, refs, sn_, crops=list(crops) + list(grown), field=fl, dec=dec)
                rec[f"px_{key}"] = (cheap_px + tpx[0]) / NATIVE
                rec[f"ncrops_{key}"], rec[f"nguard_{key}"], rec[f"rounds_{key}"] = len(crops), len(grown), rounds
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
            def strict_guarded(tpx, bud_px, skips):
                """SP.guarded (max_grow 2, relevant None) with the strict cap: a call whose canvas would push tpx above
                bud_px is skipped; a skipped growth keeps its pending truncated detection as detected"""
                dcrop = make_detect(tpx, scale9)

                def over(cr):
                    k_ = ckey(cr, scale9)
                    return tpx[0] + k_[4] * k_[5] > bud_px + 1e-6

                def det_(crop):
                    if over(crop):
                        skips[0] += 1
                        return {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}, []
                    d = {k: np.asarray(x) for k, x in dcrop(crop).items()}
                    if not len(d["conf"]):
                        return d, []
                    top, other = SP.truncated(d["xyxy"], crop, W, H, top_only=True)
                    tr = other | top
                    out = {k: x[~tr] for k, x in d.items()}
                    extra = []
                    live = [({k: d[k][i:i + 1] for k in d}, crop) for i in np.flatnonzero(tr)]
                    for step in range(2):
                        if not live:
                            break
                        nxt = []
                        for row, c in live:
                            b = row["xyxy"][0]
                            g_ = SP.grow(b, c, W, H)
                            if over(g_):
                                skips[1] += 1
                                out = {k: np.concatenate([out[k], row[k]]) for k in out}
                                continue
                            extra.append(g_)
                            dg = {k: np.asarray(x) for k, x in dcrop(g_).items()}
                            if not len(dg["conf"]):
                                continue
                            ix = np.clip(np.minimum(dg["xyxy"][:, 2], b[2]) - np.maximum(dg["xyxy"][:, 0], b[0]), 0, None)
                            iy = np.clip(np.minimum(dg["xyxy"][:, 3], b[3]) - np.maximum(dg["xyxy"][:, 1], b[1]), 0, None)
                            io = ix * iy / max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
                            j = int(np.argmax(io))
                            if io[j] < 0.5:
                                continue
                            rj = {k: dg[k][j:j + 1] for k in dg}
                            if not SP.truncated(dg["xyxy"][j:j + 1], g_, W, H)[0] or step == 1:
                                out = {k: np.concatenate([out[k], rj[k]]) for k in out}
                            else:
                                nxt.append((rj, g_))
                        live = nxt
                    return out, extra
                return det_

            def heur_strict(cfg, tag, bud, bud_px):
                """heur() with the strict post-guard px cap (selection as heur at budget bud); name '<cfg>@<tag>+C1'"""
                fam, i = cfg[:3], int(cfg[3:])
                name = f"{cfg}@{tag}+C1"
                sup_d = SP.support(ch["mono"], v, PP0)
                sel_, tb, spc = select(fam, i, name, bud, sup_d, PP0)
                tpx, skips = [0.0], [0, 0]
                sh, allc = IE.merge_once(base, sel_, strict_guarded(tpx, bud_px, skips), lift_c(name, "C1", scale9))
                score(name, sh, ("none", "v3a"), c2=False)
                rec[f"px_{name}"] = (cheap_px + tpx[0]) / NATIVE
                rec[f"cpx_{name}"], rec[f"skip_crop_{name}"], rec[f"skip_guard_{name}"] = tpx[0], skips[0], skips[1]
                push(name, sh)
                if fam == "tcr":
                    st[name]["state"] = (np.asarray(sh["xyxy"], float).reshape(-1, 4), tuple(np.asarray(q, float) for q in sh["mono"]))
            # ================================================================================================= parts
            if part == "match":
                crop51 = loop("v51S", 2.0, 960, t4=True)
                crop6 = loop("v6S", 2.0, 960, dec=True)
                cr6, gr6, _, fl6 = last["v6S"]
                rec["crops_v6S"] = [np.asarray(c, float) for c in list(cr6) + list(gr6)]
                rec["field_v6S"] = [np.asarray(c, float) for c in fl6]
                for m in MULTS:
                    bud = m * (cheap_px + crop6) - cheap_px
                    for cfg in HEUR8:
                        heur(cfg, f"x{m:.2f}", bud, dec=(m == 1.0 and cfg in ("tsp8", "cor8", "cor0", "tsp2")))
                # V1c: the hybrid with the TTC consumer's own support, at K28's S budget and at m = 1.0 (and the braking-support
                # form at the S budget = K28 phase 3's tsp8@S + C1, for parity)
                sup2 = IE.TTCWarn(PP0, 0.0).support(ch["mono"], v)
                budS = 1.0 * (cheap_px + crop51) - cheap_px
                heur("tsp8", "S", budS)
                for tag, bud in (("S", budS), ("x1.00", 1.0 * (cheap_px + crop6) - cheap_px)):
                    fam_name = f"tsp8c2@{tag}+C1"
                    sel_, tb, _ = select("tsp", 8, fam_name, bud, sup2, PP0)
                    tpx = [0.0]
                    sh, allc = IE.merge_once(base, sel_, SP.guarded(make_detect(tpx, scale9), W, H), lift_c(fam_name, "C1", scale9))
                    score(f"tsp8c2@{tag}+C1", sh, ("none", "v3a"), "base",
                          sup_fn=lambda mono: IE.TTCWarn(PP0, 0.0).support(mono, v))
                    rec[f"px_{fam_name}"] = (cheap_px + tpx[0]) / NATIVE
                    push(fam_name, sh)
            elif part == "axis":
                for side, ctx in AXIS:
                    loop(f"v6S_s{side}_c{ctx:g}", ctx, side)
            elif part == "idm":
                cons_ = IDMFollow(PP0, 0.0)
                rec["a3_cheap"] = idm_action(ch["mono"], v, PP0)
                rec["J3_cheap"] = P.decision_cost(rec["a3_cheap"], rec["a3_gt"], prev_a3, PP0, cp)["J"]
                isup = lambda mono: cons_.support(mono, v)                       # noqa: E731
                for L in (960, NWS[ds]):
                    md, ms = wf_mode(ds, L)
                    nm = f"wf{L}"
                    d = keep(det.detect(img, md)[0])
                    score3(f"{nm}+C1", scene_dict(d, lift_c(nm, "C1", ms)(d)), isup)
                    rec[f"px_{nm}"] = md.net_h * md.net_w / NATIVE
                    push(nm, scene_dict(d, None))
                # ours driven by IDMFollow (spec_general.run, + C1), refiner on IDMFollow's support
                ipx = [0.0]
                sc3, cr3, rd3 = SG.run(base, v, cam, W, H, make_detect(ipx, scale9), lift_c("idm", "C1", scale9), cons_,
                                       max_rounds=3, guard=True)
                score3("oursIDM", sc3, isup)
                rec["px_oursIDM"], rec["ncrops_oursIDM"], rec["rounds_oursIDM"] = (cheap_px + ipx[0]) / NATIVE, len(cr3), rd3
                push("idm", sc3)
                # the frozen v6-S (braking-driven), deployed refiner (braking support), scored by IDMFollow
                loop("v6S", 2.0, 960, score_it=False)
                sc6 = last["v6S"][2]
                score3("v6S", sc6, brake_sup(PP0))
                # hybrid (IDMFollow support + TUBE[8]) and uncertainty unc2 at ours-IDM's crop px
                sup_i = cons_.support(ch["mono"], v)
                heur("tsp8", "IDM", ipx[0], sup_d=sup_i, scorer=lambda nm_, sh_: score3(nm_, sh_, isup))
                heur("unc2", "IDM", ipx[0], scorer=lambda nm_, sh_: score3(nm_, sh_, isup))
                prev_a3 = rec["a3_cheap"]
            elif part == "k39":
                crop6 = loop("v6S", 2.0, 960)
                md, ms = wf_mode(ds, 960)
                dF = keep(det.detect(img, md)[0])
                FC1 = lift_c("wf960", "C1", ms)(dF)
                score("wf960+C1", scene_dict(dF, FC1), ("none", "v3a"), c2=False)
                rec["px_wf960"] = md.net_h * md.net_w / NATIVE
                push("wf960", scene_dict(dF, None))
                rec["budget_k39"] = crop6
                # ---- K39-2: nominal m = 1 (K30 parity) and strict post-guard cap, budget = v6-S's crop px of the frame
                for cfg in HEUR8:
                    heur(cfg, "x1.00", 1.0 * (cheap_px + crop6) - cheap_px)
                    heur_strict(cfg, "strict", 1.0 * (cheap_px + crop6) - cheap_px, crop6)
                    if SMOKE_CHECK:                               # strict machinery with an unlimited px cap = nominal
                        heur_strict(cfg, "chk", 1.0 * (cheap_px + crop6) - cheap_px, float("inf"))
                # ---- K39-4: single interventions (no refiner)
                fxy = np.asarray(dF["xyxy"], float).reshape(-1, 4)

                def detect_F(cr):
                    k_ = SP.centres_in(fxy, [cr]) if len(fxy) else np.zeros(0, bool)
                    return {"xyxy": fxy[k_], "conf": np.asarray(dF["conf"])[k_], "coarse": np.asarray(dF["coarse"])[k_]}
                sd_, _, _, rd_, cv_ = KL.run_g(base, v, cam, W, H, detect_F, lift_c("sa", "C1", 960 / NW), PP0)
                score("sa", sd_, ("none",), c2=False)
                rec["rounds_sa"], rec["conv_sa"] = rd_, cv_
                push("sa", sd_)

                def lift_Fm(name, sc):
                    own = lift_c(name, "C1", sc)

                    def f(full):
                        o = own(full)
                        X = np.asarray(full["xyxy"], float).reshape(-1, 4)
                        if not len(fxy) or not len(X):
                            return o
                        iou = box_iou(X, fxy)
                        j, ok = iou.argmax(1), iou.max(1) >= 0.5
                        return tuple(np.where(ok, np.asarray(fq, float)[j], np.asarray(oq, float)) for fq, oq in zip(FC1, o))
                    return f
                tpx_b = [0.0]
                sb_, crb, grb, rdb, cvb = KL.run_g(base, v, cam, W, H, make_detect(tpx_b, 960 / NW), lift_Fm("sb", 960 / NW), PP0,
                                                   v_closes=(0.0,), ctx=2.0, max_calls=None)
                score("sb", sb_, ("none",), c2=False)
                rec["px_sb"], rec["rounds_sb"] = (cheap_px + tpx_b[0]) / NATIVE, rdb
                push("sb", sb_)
                loop("sc1", 2.0, 960, refs=("none",), v_closes=(10.0,))
                loop("sc2", 2.0, 960, refs=("none",), v_closes=(20.0,))
                loop("sd", 2.0, 960, refs=("none",), r_grid=R_EXT, y_grid=Y_EXT)
                loop("se", 2.0, 960, refs=("none",), probes=(KL.PED, KL.CAR, KL.TRUCK))
                loop("sf", 2.0, 960, refs=("none",), max_rounds=10)
            elif part == "k42":
                # ---- v6-S frozen, stopping test logged
                COVLOG["on"], COVLOG["calls"] = True, []
                try:
                    loop("v6S", 2.0, 960)
                finally:
                    COVLOG["on"] = False
                cl = COVLOG["calls"]
                rec["cov_calls_v6S"] = len(cl)
                rec["cov_skip_v6S"] = sum(1 for r_, u_, n_ in cl if r_)
                rec["cov_skip_union_req_v6S"] = sum(1 for r_, u_, n_ in cl if r_ and u_ < 1.0 - 1e-9)   # skipped now, union would require
                rec["cov_req_union_skip_v6S"] = sum(1 for r_, u_, n_ in cl if (not r_) and n_ and u_ >= 1.0 - 1e-9)  # run now, union would skip
                rec["cov_skip_union_fracs_v6S"] = [u_ for r_, u_, n_ in cl if r_ and u_ < 1.0 - 1e-9]
                # ---- v6-S with the strict union-containment stopping test (own history)
                SP.covered = covered_union
                try:
                    loop("v6U", 2.0, 960)
                finally:
                    SP.covered = covered_logged
                # ---- pi(F)
                md, ms = wf_mode(ds, 960)
                dF = keep(det.detect(img, md)[0])
                FC1 = lift_c("wf960", "C1", ms)(dF)
                score("wf960+C1", scene_dict(dF, FC1), ("none", "v3a"), c2=False)
                rec["px_wf960"] = md.net_h * md.net_w / NATIVE
                push("wf960", scene_dict(dF, None))
                fxy = np.asarray(dF["xyxy"], float).reshape(-1, 4)

                def detect_F(cr):
                    k_ = SP.centres_in(fxy, [cr]) if len(fxy) else np.zeros(0, bool)
                    return {"xyxy": fxy[k_], "conf": np.asarray(dF["conf"])[k_], "coarse": np.asarray(dF["coarse"])[k_]}
                # ---- K39-4 single interventions, none AND v3a
                sd_, _, _, rd_, cv_ = KL.run_g(base, v, cam, W, H, detect_F, lift_c("sa", "C1", 960 / NW), PP0)
                score("sa", sd_, ("none", "v3a"), c2=False)
                rec["rounds_sa"], rec["conv_sa"] = rd_, cv_
                push("sa", sd_)

                def lift_Fm(name, sc):
                    own = lift_c(name, "C1", sc)

                    def f(full):
                        o = own(full)
                        X = np.asarray(full["xyxy"], float).reshape(-1, 4)
                        if not len(fxy) or not len(X):
                            return o
                        iou = box_iou(X, fxy)
                        j, ok = iou.argmax(1), iou.max(1) >= 0.5
                        return tuple(np.where(ok, np.asarray(fq, float)[j], np.asarray(oq, float)) for fq, oq in zip(FC1, o))
                    return f
                tpx_b = [0.0]
                sb_, crb, grb, rdb, cvb = KL.run_g(base, v, cam, W, H, make_detect(tpx_b, 960 / NW), lift_Fm("sb", 960 / NW), PP0,
                                                   v_closes=(0.0,), ctx=2.0, max_calls=None)
                score("sb", sb_, ("none", "v3a"), c2=False)
                rec["px_sb"], rec["rounds_sb"] = (cheap_px + tpx_b[0]) / NATIVE, rdb
                push("sb", sb_)
                loop("sc1", 2.0, 960, v_closes=(10.0,))
                loop("sc2", 2.0, 960, v_closes=(20.0,))
                loop("sd", 2.0, 960, r_grid=R_EXT, y_grid=Y_EXT)
                loop("se", 2.0, 960, probes=(KL.PED, KL.CAR, KL.TRUCK))
                loop("sf", 2.0, 960, max_rounds=10)
                # ---- K28 p3 / K16 counterfactual chain (k28_p3.py CHAIN), none AND v3a

                def lift_F(full):
                    X = np.asarray(full["xyxy"], float).reshape(-1, 4)
                    ii = [int(np.argmax(box_iou(X[n:n + 1], fxy)[0])) for n in range(len(X))]
                    return tuple(np.asarray(a_, float)[ii] for a_ in FC1)
                for cn_, kw_, fl_ in CHAIN42:
                    cn = f"ch_{cn_}"
                    sdc, _, _, rdc, cvc = KL.run_g(base, v, cam, W, H, detect_F, lift_F if fl_ else lift_c(cn, "C1", 960 / NW), PP0, **kw_)
                    score(cn, sdc, ("none", "v3a"), c2=False)
                    rec[f"rounds_{cn}"], rec[f"conv_{cn}"] = rdc, cvc
                    push(cn, sdc)
            elif part == "cons":
                for sn_ in SETTINGS:
                    c51 = loop("v51S", 2.0, 960, sn_=sn_, t4=True)
                    loop("v6S", 2.0, 960, sn_=sn_, dec=True)
                    budS = 1.0 * (cheap_px + c51) - cheap_px
                    for cfg in CONS_HEUR:
                        heur(cfg, "S", budS, sn_=sn_, c2=False)
                for pn, kw in PROBE.items():
                    loop(f"v6S@{pn}", 2.0, 960, **dict(kw))
                for L in (960, NWS[ds]):
                    md, ms = wf_mode(ds, L)
                    nm = f"wf{L}"
                    d = keep(det.detect(img, md)[0])
                    tc1 = lift_c(nm, "C1", ms)(d)
                    for sn_ in SETTINGS:
                        score(f"{nm}+C1", scene_dict(d, tc1), ("none", "v3a"), sn_, c2=False)
                    rec[f"px_{nm}"] = md.net_h * md.net_w / NATIVE
                    push(nm, scene_dict(d, None))
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
