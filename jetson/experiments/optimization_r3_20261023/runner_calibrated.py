#!/usr/bin/env python
"""Generated isolated W1 development harness. See protocol.md and fork_manifest.json."""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pickle
import json
import time
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import Retro, RefineClosure, obligations, draft_recheck
from proactive import Pending
import bridge
from inference import Runtime
from estimators import motion_floors, add_floor, scale_refinement
_kl = importlib.util.spec_from_file_location('sandbox_loop', Path(__file__).with_name('sandbox_loop.py'))
KL = importlib.util.module_from_spec(_kl); _kl.loader.exec_module(KL)
import bounded_variant
bounded_variant.KL = KL
from bounded_variant import run_bounded_keep
ttc_ss, parse = K27.ttc_ss, K27.parse
from rap import planner as P                                                     # noqa: E402
from rap.detect import Mode, TwoFidelityDetector                                 # noqa: E402
from rap.mono import predicted_geometry                                          # noqa: E402
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


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2"])
    ap.add_argument("--part", default="devref", choices=["devref"])
    ap.add_argument("--split", default="dev", choices=["dev"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max_units", type=int, default=0)
    ap.add_argument("--units", default="", help="comma list of unit names (smoke)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--suite",default="bridge",choices=["bridge","learned"])
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
    runtime=Runtime(ds,head,crop64,K27.assoc_idx,calibrated=True)
    PP0, cp = P.PlannerParams(), P.CostParams()
    if ds == "nusc":
        CH, cheap_px, NW = Mode("ns_c320", 192, 320), 192 * 320, 1600
    elif ds == "av2":
        CH, cheap_px, NW = Mode("av2_c320", 320, 256), 320 * 256, 2048
    else:
        CH, cheap_px, NW = Mode("ahd_c320", 224, 320), 224 * 320, 1920
    od = REAL / "experiments/optimization_r3_20261023/results" / (args.tag or "dev") / ds
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
    reference_dir = REAL / 'results/k30/dev_devref' / ds
    allowed = {p.stem for p in reference_dir.glob('*.pkl')}
    assert len(allowed) == {'nusc':152,'av2':300}[ds], (ds,len(allowed))
    if ds == 'av2':
        assert os.environ.get('K10_MASK_Y0') == '1830', 'fixed AV2 mask required'
        import pandas as pd
        split = pd.read_csv(REAL/'results/k23/av2_split.csv')
        assert allowed == set(split[split.role=='CFG-DEV'].log)
        assert not allowed.intersection(split[split.role.isin(['REF-TRAIN','REF-VAL'])].log)
    else:
        import pandas as pd
        split = pd.read_csv(REAL/'results/nusc_full/split.csv')
        train = set(split[(split['set']=='new') & (split.split=='train')].seq)
        assert allowed == set(split[(split['set']=='new') & (split.split=='val')].seq)
        assert not allowed.intersection(train), 'nuScenes dev/train leakage'
    (od.parent / (ds + '_split_audit.json')).write_text(json.dumps({
        'dataset':ds,'allowed_units':sorted(allowed),'training_disjoint':True},indent=2))
    SETTINGS = {"base": PP0} if part != "cons" else {"base": PP0, **{k: replace(PP0, **v) for k, v in CONS.items()}}
    for ui, (unit, cam, gen) in enumerate(K27.units(ds, args.max_units)):
        if ui % sn != si or (UNITS and unit not in UNITS):
            continue
        assert unit in allowed, ('non-dev unit',unit)
        fo = od / f"{unit}.pkl"
        if fo.exists():
            prior=json.loads(fo.with_suffix('.parity.json').read_text())
            assert not any(prior['differences'].values()),prior
            continue
        R, t = cam.transform("ego")
        P2m = cam.calib.P2
        fx, fy, cx, cy = P2m[0, 0], P2m[1, 1], P2m[0, 2], P2m[1, 2]
        dt = float(cam.dt)
        st = {}
        runtime.bank={}
        prev_c, imgs, rows = None, [], []
        prev_a = {k: None for k in SETTINGS}                   # cheap's previous action per setting (the cost's jerk term)
        prev_a3 = None
        reference_map = {int(r['frame']):r for r in pickle.load((REAL/'experiments/w1_bounded_20261023/results/dev'/ds/f'{unit}.pkl').open('rb'))}
        case_units={c['unit'] for c in json.loads(Path(__file__).with_name('case_manifest_with_severity.json').read_text())['cases']}
        trace_case=True  # all-row mechanism attribution, no outcome-based recording
        for fr, path, v, g, cached in gen():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            NATIVE = W * H
            runtime.frame(int(fr));learned_maps={}
            rec = {}  # recompute controls; do not import outcomes into policies
            rec.update({"unit": unit, "frame": fr, "v": v, "dt":dt, "gt": not isinstance(cached, str)})
            gt_t = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc")) if len(g) else tuple(np.zeros(0) for _ in range(4))
            gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
            gtc = np.asarray(g["coarse"]).astype(str) if len(g) else np.zeros(0, str)
            rec['gt_boxes']=gtb;rec['gt_classes']=gtc;rec['gt_ids']=np.asarray(g['track_id']) if len(g) else np.zeros(0,int)
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
            cache, rcache, last, RETRO = {}, {}, {}, {}

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
                original_h = st.get(name, {})

                def finish(full,mono):
                    if '_floor' not in name:return mono
                    h=original_h
                    h1=(h['s1'],h['c1']) if h.get('s1') is not None else None
                    h2=(h['s2'],h['c2']) if h.get('s2') is not None else None
                    _,rates,_=motion_floors(full['xyxy'],full['coarse'],h1,h2,sc,dt)
                    return add_floor(mono,rates,np.ones(len(full['xyxy']),bool))[0]

                def apply_adapter(full,h,chn,mono):
                    if not any(tag in name for tag in ('_missing','_allgeo','_alldec','_upper','_query')):return finish(full,mono)
                    objective='decision' if '_alldec' in name else 'geometry'
                    zz=predicted_geometry({'xyxy':full['xyxy'],'coarse':full['coarse']},None,cam.calib,cam.cam_h,dt,cam_to_ego=(R,t))
                    pred,mask=runtime.infer(objective,full,mono,zz,h,chn,st.get('cheap_history',{}),imgs,img,W,H,v,dt)
                    out=[np.asarray(q).copy() for q in mono];known={i for i,_,_ in chn}
                    for i in range(len(full['xyxy'])):
                        if not np.isfinite(pred[i,0]):continue
                        learned_maps.setdefault(name,{})[tuple(full['xyxy'][i])]=pred[i].copy()
                        if ('_missing' in name or '_upper' in name) and i in known:continue
                        rate=float(pred[i,4] if '_upper' in name else pred[i,0])
                        out[3][i]=min(1/rate,1e3) if rate>1e-3/dt else np.inf
                    rec['adapter_items_'+name]=rec.get('adapter_items_'+name,0)+int(np.isfinite(pred[:,0]).sum())
                    rec['adapter_missing_'+name]=rec.get('adapter_missing_'+name,0)+int((mask.sum(1)<3).sum())
                    return tuple(out)

                def f(full):
                    h = dict(original_h)
                    X = np.asarray(full["xyxy"], float).reshape(-1, 4)
                    z, lo, hi, _ = geom(full)
                    t4, _ = ttc_ss(X, h.get("s1"), h.get("s2"), sc, dt, 1.0, True)
                    ttc = np.full(len(X), np.inf) if mode == "C1" else t4.copy()
                    chn = KTD.chains(X, h.get("s1"), h.get("s2")) if len(imgs) == 2 else []
                    if name in RETRO:
                        h, chn = RETRO[name].augment(full,(z,lo,hi,ttc),h)
                    if '_bridge' in name:
                        h,chn,nbridge=bridge.augment(full,h,chn,st.get('cheap_history',{}))
                        rec['bridge_chains_'+name]=rec.get('bridge_chains_'+name,0)+nbridge
                    if '_transfer' in name:
                        (z,lo,hi,ttc),ntransfer=bridge.transfer(full,(z,lo,hi,ttc),chn,base)
                        rec['transferred_'+name]=rec.get('transferred_'+name,0)+ntransfer
                    if not chn:
                        return apply_adapter(full,h,chn,(z,lo,hi,ttc))
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
                    return apply_adapter(full,h,chn,(z,lo,hi,ttc))
                return f

            def push(name, scene):
                h = st.setdefault(name, {})
                ff = np.asarray(scene["from_full"], bool)
                h["c2"] = h.get("c1")
                h["c1"] = np.asarray(scene["coarse"])[ff].copy()
                h["s2"] = h.get("s1")
                h["s1"] = np.asarray(scene["xyxy"], float).reshape(-1, 4)[ff]
                h["merged"] = np.asarray(scene["xyxy"], float).reshape(-1, 4)

            def refine(scene, name, sup_fn):
                tt = [np.asarray(a, float).copy() for a in scene["mono"]]
                if name=='adapt':
                    factor=np.ones(len(tt[0]));sup=sup_fn(scene['mono'])
                    for i in sup:
                        if scene['from_full'][i]:factor[i]=scene.get('_adapter_factors',np.ones(len(tt[0])))[i]
                    return scale_refinement(tt,factor)
                if name in ("none", "closed"):
                    return tuple(tt)
                if name in ('speed','speedhalf'):
                    rf=[np.asarray(q).copy() for q in refine(scene,'v3a',sup_fn)]
                    good=np.isfinite(tt[0])&(tt[0]>0)&np.isfinite(rf[0])
                    power=1. if name=='speed' else .5
                    rf[3][good]*=np.power(rf[0][good]/tt[0][good],power)
                    return tuple(rf)
                if name == "half":
                    rf=refine(scene,"v3a",sup_fn)
                    factor=np.ones(len(tt[0]));valid=np.isfinite(tt[0])&(tt[0]>0)&np.isfinite(rf[0])
                    factor[valid]=np.sqrt(np.asarray(rf[0])[valid]/tt[0][valid])
                    return scale_refinement(tt,factor)
                if name.startswith("v3a_"):                  # K30-DEV conservative variants of v3a
                    rf = [np.asarray(a, float).copy() for a in refine(scene, "v3a", sup_fn)]
                    if not len(tt[0]):
                        return tuple(rf)
                    if name in ("v3a_n15", "v3a_n30"):
                        D = 15.0 if name == "v3a_n15" else 30.0
                        back = (rf[0] > tt[0]) & (tt[0] < D)
                    else:
                        du = np.asarray(P.required_decel(*tt, v, PP0)[1], float)
                        dr = np.asarray(P.required_decel(*rf, v, PP0)[1], float)
                        back = (du >= PP0.thr_hard) & (dr < du)
                    for a, b in zip(rf, tt):
                        a[back] = b[back]
                    return tuple(rf)
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

            def audit(nm, sc, crops, grown, tr):
                h = st.get(nm,{})
                full_ids = np.flatnonzero(sc['from_full'])
                chains = KTD.chains(sc['xyxy'][full_ids],h.get('s1'),h.get('s2')) if len(imgs)==2 else []
                observed_ids = {int(full_ids[i]) for i,_,_ in chains}
                fc,ns = obligations(dict(sc,mono=tr),crops,grown,
                    lambda tt: KL.field_crop_g(tt,v,cam,W,H,PP0,(0.0,),SP.R_GRID,SP.Y_GRID,(KL.CAR,)),W,H,v,PP0)
                raw = sc.get('_raw_mono',sc['mono'])
                rec.update({f'diag_post_field_{nm}':fc, f'diag_post_support_{nm}':ns,
                    f'diag_action_drop_{nm}':int(act(tr)<act(raw)),
                    f'diag_full_objects_{nm}':len(full_ids), f'diag_chain_objects_{nm}':len(observed_ids)})
                if GT['base'][2] >= 0:
                    di=GT['base'][2]
                    m=int(IE.matched(gtb[di:di+1],gtc[di:di+1],sc['xyxy'],sc['coarse'])[0])
                    rec[f'diag_dec_chain_{nm}'] = m in observed_ids
                if nm == 'v6S' or trace_case:
                    rec['trace_'+nm] = dict(xyxy=sc['xyxy'],coarse=sc['coarse'],conf=sc['conf'],from_full=sc['from_full'],
                        raw=raw,refined=tr,crops=list(crops)+list(grown),chain_ids=sorted(observed_ids),
                        retro_events=RETRO[nm].events if nm in RETRO else [])
                return sorted(observed_ids)

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
                scale = side / NW
                key = nm
                tpx, log, cnt = [0.0], {}, [0]
                FLOG.clear()
                if nm.endswith('_retro'):
                    RETRO[nm] = Retro(base,imgs,log,lambda im,k: run_det(det,im,k),KTD.chains,v,PP0)
                pending=st.setdefault(nm,{}).setdefault('pending',Pending()) if '_pending' in nm else None
                seed=pending.seed(base,W,H,KL.box_crop_ctx) if pending else []
                if '_extent' in nm:seed+=bridge.extent_seed(st.setdefault(nm,{}),base,W,H,KL.box_crop_ctx)
                vc=bridge.motion_probe(base,PP0) if '_motion' in nm else 0.
                rec['probe_vclose_'+nm]=vc
                if '_query' in nm:
                    chi=st.get('cheap_history',{})
                    chains=KTD.chains(base['xyxy'],chi.get('s1'),chi.get('s2')) if len(imgs)==2 else []
                    zz=predicted_geometry({'xyxy':base['xyxy'],'coarse':base['coarse']},None,cam.calib,cam.cam_h,dt,cam_to_ego=(R,t))
                    pred,mask=runtime.infer('geometry',base,base['mono'],zz,chi,chains,chi,imgs,img,W,H,v,dt,kind='cheap')
                    factor=np.where(np.isfinite(pred[:,1]),pred[:,1],1.)
                    lower=list(scale_refinement(base['mono'],factor));upper=[q.copy() for q in lower]
                    for i in range(len(pred)):
                        if np.isfinite(pred[i,0]):
                            lower[3][i]=1/max(float(pred[i,2]),.001);upper[3][i]=1/max(float(pred[i,4]),.001)
                    _,dl=P.required_decel(*lower,v,PP0);_,du=P.required_decel(*upper,v,PP0)
                    candidates=[i for i in range(len(pred)) if np.isfinite(pred[i,0]) and P.discrete_action(dl[i],PP0)!=P.discrete_action(du[i],PP0)]
                    if candidates:
                        i=max(candidates,key=lambda j:(du[j]-dl[j],-j));seed.append(KL.box_crop_ctx(base['xyxy'][i],W,H,2.))
                    rec['uncertain_query_'+nm]=int(bool(candidates))
                lf = lift_pol(key, parse(TPOL), scale) if t4 else lift_c(key, "C1", scale)
                close = RefineClosure(refine,v,PP0) if nm.endswith('_close') else None
                recheck = (lambda cr,gr,sc: draft_recheck(base,sc,cr,gr,W,H,v,PP0,KL.box_crop_ctx)) if nm.endswith('_recheck') else None
                if 'b2A' in nm:
                    sc,crops,grown,rounds,conv = run_bounded_keep(base,v,cam,W,H,
                        make_detect(tpx,scale,log,None,cnt),lf,PP0,2,'A',ctx=ctx,seed=seed,v_closes=(0.,vc))
                    assert rounds <= 2
                else:
                    sc,crops,grown,rounds,conv = KL.run_g(base,v,cam,W,H,make_detect(tpx,scale,log,None,cnt),lf,
                        PP0,v_closes=(0.0,),ctx=ctx,max_calls=None,post=close,post_rounds=1,recheck=recheck)
                fl=list(FLOG)
                adapted=any(tag in nm for tag in ('_allgeo','_alldec','_query'))
                if adapted:
                    estimates=learned_maps.get(nm,{})
                    sc['_adapter_factors']=np.array([estimates[tuple(b)][1] if tuple(b) in estimates else 1. for b in sc['xyxy']])
                    refs=('none','v3a','adapt')
                if close is not None: refs=('closed',)
                score(nm,sc,refs,'base',crops=list(crops)+list(grown),field=fl,dec=True)
                tr = sc['mono'] if close else refine(sc,'adapt' if adapted else 'v3a',brake_sup(PP0))
                observed=audit(nm,sc,crops,grown,tr)
                if nm=='v6S_b2A_retro':score(nm,sc,('half','speed','speedhalf'),'base',c2=False)
                if '_extent' in nm:bridge.extent_update(st[nm],sc,tr,v,PP0)
                if pending:
                    pending.update(sc,base,observed,v,PP0)
                    rec.update({f'pending_requested_{nm}':pending.requested,f'pending_armed_{nm}':pending.armed,
                                f'pending_completed_{nm}':pending.completed,f'pending_expired_{nm}':pending.expired})
                r=RETRO.get(nm)
                rpx=r.pixels if r else 0
                assert r is None or r.calls <= 2
                rec.update({f'px_{nm}':(cheap_px+tpx[0]+rpx)/NATIVE,
                    f'ncalls_{nm}':cnt[0]+(r.calls if r else 0),f'ncrops_{nm}':len(crops),
                    f'nguard_{nm}':len(grown),f'rounds_{nm}':rounds,f'conv_{nm}':conv,
                    f'retro_calls_{nm}':r.calls if r else 0,f'retro_attempted_{nm}':r.attempted if r else 0,
                    f'retro_success_{nm}':r.success if r else 0,
                    f'closure_corrections_{nm}':close.new_corrections if close else 0})
                last[nm]=(crops,grown,sc,fl)
                push(nm,sc)
                return tpx[0]+rpx
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
            # Every fixed acquisition row, independent history; all three refiner controls.
            if trace_case:rec['trace_cheap']={k:np.asarray(v).copy() if k!='mono' else tuple(np.asarray(q).copy() for q in v) for k,v in base.items()}
            for nm in ('v6S','v6S_b2A_retro','v6S_b2A_upper_retro','v6S_b2A_query_retro'):
                loop(nm,2.0,960,dec=True)
            chist=st.setdefault('cheap_history',{})
            chist['s2']=chist.get('s1');chist['c2']=chist.get('c1')
            chist['s1']=np.asarray(base['xyxy']).copy();chist['c1']=np.asarray(base['coarse']).copy()
            imgs = (imgs + [img])[-2:]
            for sn_, pp_ in SETTINGS.items():
                prev_a[sn_] = act(ch["mono"], pp_)
            rows.append(rec)
        oldmap=reference_map
        parity={'unit':unit,'dataset':ds,'frames':len(rows),'differences':{},'recomputed':['cheap','v6S','v6S_b2A_retro']}
        assert {int(r['frame']) for r in rows} == set(oldmap), ('frames',unit)
        for col in ('act_gt','a_gt','a_cheap','J_cheap','a_v6S|none','a_v6S|v3a','J_v6S|v3a','px_v6S',
                    'a_v6S_b2A_retro|none','a_v6S_b2A_retro|v3a','J_v6S_b2A_retro|v3a','px_v6S_b2A_retro'):
            delta=[abs(float(r[col])-float(oldmap[int(r['frame'])][col])) for r in rows]
            parity['differences'][col]=sum(x>1e-8 for x in delta)
        (od/f'{unit}.parity.json').write_text(json.dumps(parity,indent=2))
        # Preserve the failed-unit data for review, but fail closed before main launch.
        tmp=fo.with_suffix('.tmp')
        with tmp.open('wb') as fh: pickle.dump(rows,fh)
        tmp.replace(fo)
        assert not any(parity['differences'].values()), parity

        print(f"  [{ui}] {unit}: {len(rows)} frames", flush=True)
    print(f"W1 SANDBOX {ds} DONE", flush=True)


if __name__ == "__main__":
    main()
