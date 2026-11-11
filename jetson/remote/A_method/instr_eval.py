#!/usr/bin/env python
"""Instrumented re-run (K2 agreement, K3 necessity ablations, K4 field contribution, K9 second consumer).

Re-runs the FINAL configuration of each dataset with the frozen code (vtf.speculative.run for the full method) and
logs, per frame and pipeline, the ACTION (not only the cost), crops, rounds, guard re-detections and detector tensor
pixels. Variants of the loop (K3) are a copy of speculative.run with switches (run_variant below; the 'full' switches
reproduce speculative.run exactly — asserted on every frame). Nothing in src/vtf is modified.

Datasets / configurations (frozen): nuScenes (cached cheap draft as the confirmatory run, v3: v_close 10, crops at
0.6, refiner v1 out/resnet18) and AV2 (online cheap draft as av2_eval_v4, v4-lite: v_close 0, crops at 960/2048,
refiner v2 out/resnet18_v2). Settings: base | del_all | del_half (K4 B: matched draft detections of decision-relevant
GT objects deleted before the policy runs, from the draft and from its history; del_half with probability 0.5, seeded).
"""
from __future__ import annotations

import argparse
import math
import zlib
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from rap import planner as P
from rap.detect import Mode, TwoFidelityDetector
from rap.mono import box_iou, predicted_geometry
from vtf import dsf
from vtf import spec_general as SG
from vtf import speculative as SP

B = Path(os.environ["B"])
OUT = B / "results" / "instr"
THR = 0.25
CLS = {"vehicle": 0, "person": 1, "cyclist": 2}
COMPAT = {"vehicle": {"vehicle"}, "person": {"person", "cyclist"}, "cyclist": {"cyclist", "person"}}
VARIANTS = ("full", "field_only", "support_only", "pivots", "one_round", "no_guard", "draft_hist", "no_guard_draft_hist")
HEUR = ("uncertainty", "temporal_crit", "random")
# ---------------------------------------------------------------------------- consumer 2: TTC warning (K9)
TTC_WARN, TTC_BRAKE = 4.0, 2.0                    # actions: 0 none, 1 warn (ttc <= 4 s), 2 brake (ttc <= 2 s)
C2_COST = np.array([[0, 1, 5],                    # rows: required (gt) action 0/1/2; cols: taken action 0/1/2
                    [3, 0, 2],
                    [10, 4, 0]], float)


def c2_demand(t, pp):
    z, lmin, lmax, ttc = (np.asarray(a, float) for a in t)
    sel = P.in_corridor(lmin, lmax, pp.corridor_half_w) & (z <= pp.max_range) & (z > 0)
    d = np.zeros(len(z))
    d[sel] = 1.0 / np.clip(ttc[sel], 0.05, np.inf)
    return d


def c2_action(t, pp):
    d = c2_demand(t, pp)
    m = d.max() if len(d) else 0.0
    return 2 if m >= 1 / TTC_BRAKE else 1 if m >= 1 / TTC_WARN else 0


class TTCWarn:
    """max-composable consumer for spec_general: field = probe cells (static probe closing at v + v_close, inside the
    corridor) whose 1/ttc reaches the next action's threshold; support = objects whose 1/ttc reaches the current one."""

    def __init__(self, pp, v_close):
        self.pp, self.vc = pp, v_close

    def decide(self, t, v):
        return c2_action(t, self.pp)

    def field(self, t, v, rg, yg):
        a = c2_action(t, self.pp)
        if a == 2:
            return np.zeros((len(rg), len(yg)), bool)
        thr = (1 / TTC_WARN, 1 / TTC_BRAKE)[a]
        rr, yy = np.meshgrid(rg, yg, indexing="ij")
        inc = (yy - SP.OBJ_W / 2 < self.pp.corridor_half_w) & (yy + SP.OBJ_W / 2 > -self.pp.corridor_half_w)
        return inc & ((v + self.vc) / rr >= thr) & (rr <= self.pp.max_range)

    def support(self, t, v):
        a = c2_action(t, self.pp)
        if a == 0:
            return np.zeros(0, int)
        thr = (0, 1 / TTC_WARN, 1 / TTC_BRAKE)[a]
        return np.flatnonzero(c2_demand(t, self.pp) >= thr)


# ---------------------------------------------------------------------------- the loop with switches (K3)
def run_variant(cheap, v, cam, W, H, detect_crop, lift, pp, v_close, max_rounds=3, guard=True, use_field=True,
                support_mode="support", obj_h=SP.OBJ_H):
    """speculative.run with switches: use_field (field crops), support_mode in {support, pivots, none}."""
    gdet = SP.guarded(detect_crop, W, H) if guard else None
    crops, grown, full = [], [], {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
    scene, rounds = dict(cheap), 0
    for _ in range(max_rounds):
        new = []
        if use_field:
            fc = SP.field_crop(scene["mono"], v, cam, W, H, pp, v_close, None, obj_h)
            if fc is not None and not SP.covered(fc, crops):
                new.append(fc)
        if support_mode == "support":
            idx = SP.support(scene["mono"], v, pp)
        elif support_mode == "pivots":
            idx = np.atleast_1d(dsf.brake_pivots(scene["mono"], v, pp)).astype(int)
        else:
            idx = []
        for k in idx:
            bc = SP.box_crop(scene["xyxy"][k], W, H)
            if not SP.covered(bc, crops + grown + new):
                new.append(bc)
        if not new:
            break
        rounds += 1
        for c in new:
            if gdet is None:
                d = detect_crop(c)
            else:
                d, ex = gdet(c)
                grown += ex
            full = {k: np.concatenate([full[k], np.asarray(d[k])]) for k in full}
        crops += new
        if len(full["conf"]) > 1:
            o = np.argsort(-full["conf"], kind="stable")
            iou = box_iou(full["xyxy"], full["xyxy"])
            keep = []
            for i in o:
                if not keep or iou[i, keep].max() < 0.5:
                    keep.append(i)
            keep = np.sort(np.array(keep))
            full = {k: full[k][keep] for k in full}
        out = SP.centres_in(cheap["xyxy"], crops)
        if grown and len(full["conf"]) and len(cheap["xyxy"]):
            out |= box_iou(np.asarray(cheap["xyxy"], float), full["xyxy"]).max(1) >= 0.5
        fz = lift(full) if len(full["conf"]) else tuple(np.zeros(0) for _ in range(4))
        scene = {"xyxy": np.r_[cheap["xyxy"][~out], full["xyxy"]], "conf": np.r_[cheap["conf"][~out], full["conf"]],
                 "coarse": np.r_[cheap["coarse"][~out], full["coarse"]],
                 "mono": tuple(np.r_[a[~out], b] for a, b in zip(cheap["mono"], fz)),
                 "from_full": np.r_[np.zeros(int((~out).sum()), bool), np.ones(len(full["conf"]), bool)]}
    scene.setdefault("from_full", np.zeros(len(scene["conf"]), bool))
    return scene, crops + grown, rounds


def merge_once(cheap, crops, gdet, lift):
    """baselines_eq.merge_once (heuristic crops, guarded, one pass)."""
    full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
    grown = []
    for c in crops:
        d, ex = gdet(c)
        grown += ex
        full = {k: np.concatenate([full[k], np.asarray(d[k])]) for k in full}
    if len(full["conf"]) > 1:
        o = np.argsort(-full["conf"], kind="stable"); iou = box_iou(full["xyxy"], full["xyxy"]); keep = []
        for i in o:
            if not keep or iou[i, keep].max() < 0.5:
                keep.append(i)
        full = {k: full[k][np.sort(np.array(keep))] for k in full}
    out = SP.centres_in(cheap["xyxy"], crops) if crops else np.zeros(len(cheap["xyxy"]), bool)
    if grown and len(full["conf"]) and len(cheap["xyxy"]):
        out |= box_iou(np.asarray(cheap["xyxy"], float), full["xyxy"]).max(1) >= 0.5
    fz = lift(full) if len(full["conf"]) else tuple(np.zeros(0) for _ in range(4))
    return {"xyxy": np.r_[cheap["xyxy"][~out], full["xyxy"]], "conf": np.r_[cheap["conf"][~out], full["conf"]],
            "coarse": np.r_[cheap["coarse"][~out], full["coarse"]],
            "mono": tuple(np.r_[a[~out], b] for a, b in zip(cheap["mono"], fz)),
            "from_full": np.r_[np.zeros(int((~out).sum()), bool), np.ones(len(full["conf"]), bool)]}, crops + grown


def area(c):
    return max(c[2] - c[0], 0) * max(c[3] - c[1], 0)


def fill_strict(cands, budget):
    out, used = [], 0.0
    for c in cands:
        if used + area(c) <= budget:
            out.append(c); used += area(c)
    return out


def matched(gt_boxes, gt_cls, det_xyxy, det_cls, thr=0.3):
    """per GT box: index of the matched detection (class-compatible, IoU >= thr, Hungarian) or -1."""
    from scipy.optimize import linear_sum_assignment
    res = -np.ones(len(gt_boxes), int)
    if not len(gt_boxes) or not len(det_xyxy):
        return res
    M = box_iou(np.asarray(gt_boxes, float), np.asarray(det_xyxy, float))
    for i, c in enumerate(gt_cls):
        for j, dc in enumerate(det_cls):
            if str(dc) not in COMPAT.get(str(c), {str(c)}):
                M[i, j] = 0
    r, cc = linear_sum_assignment(-M)
    for i, j in zip(r, cc):
        if M[i, j] >= thr:
            res[i] = j
    return res


# ---------------------------------------------------------------------------- dataset adapters
def units_nusc(split):
    from rap.cache import Camera, DetCache
    from rap.decision import _apply_range_source
    from rap.nusc import NuScenesDB, make_adapter
    from rap.paths import CACHE, NUSCENES_TRAINVAL
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    by = {db.scene_name(t): t for t in db.scenes}
    sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
    for s in sorted(sp[(sp["set"] == "new") & (sp.split == split)].seq):
        cam = Camera("nuScenes", s)
        c = DetCache(Path(CACHE) / "nusc_det_tv" / "ns_cheap_320" / f"{s}.npz")
        toks = db.samples(by[s])
        geom, speeds = ad.geometry(s), ad.speeds(s)

        def frames(c=c, toks=toks, geom=geom, speeds=speeds):
            rng = np.random.default_rng(0)
            for i, fr in enumerate(c.frames):
                fr = int(fr)
                dc = c.det(i)
                kc = dc["conf"] >= THR
                geo_c = {kk: vv[kc] for kk, vv in c.geo(i).items()}
                g = geom[geom["frame"] == fr]
                g = g[(g["y2"] - g["y1"]) >= 10]
                bxc = dc["xyxy"][kc].astype(float)
                cons = tuple(np.asarray(a, float) for a in _apply_range_source(bxc, geo_c, g, "mono", rng, 0.0))
                yield fr, str(db.image_path(toks[fr])), float(speeds[fr]) if fr < len(speeds) else float(speeds[-1]), g, \
                    {"xyxy": bxc, "conf": dc["conf"][kc].astype(float), "coarse": dc["coarse"][kc],
                     "entropy": dc["entropy"][kc] if "entropy" in dc else np.zeros(int(kc.sum())), "mono": cons}
        yield s, cam, frames


def units_av2(split):
    from vtf import av2 as AV
    root = B / "datasets" / "av2" / "sensor" / ("val" if split == "test" else "train")
    if split == "test":
        logs = sorted(p.name for p in root.iterdir() if (p / "annotations.feather").exists())
    else:                                           # AV2 dev subset: train logs of av2_sota_labels' "val" half
        import hashlib
        logs = [l for l in sorted(p.stem for p in (B / "datasets" / "av2" / "lists" / "train").glob("*.done"))
                if hashlib.md5(l.encode()).hexdigest()[0] in "012"]
    for log in logs:
        L = AV.AV2Log(root, log)

        def frames(L=L):
            geom, speeds = L.geometry(), L.speeds()
            for fr in range(len(L.frame_ts)):
                g = geom[geom["frame"] == fr] if len(geom) else geom
                g = g[(g["y2"] - g["y1"]) >= 10] if len(g) else g
                yield fr, str(L.image_path(fr)), float(speeds[fr]), g, None
        yield log, L, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nusc", "av2"])
    ap.add_argument("--split", required=True, help="nusc: val | test; av2: dev | test (= AV2 val)")
    ap.add_argument("--setting", default="base", choices=["base", "del_all", "del_half"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max_units", type=int, default=0)
    ap.add_argument("--diag_units", type=int, default=0, help="K2.5 locality diagnostics on the first N units")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))
    dev = torch.device(args.device)
    det = TwoFidelityDetector(str(B / "models" / "yolov8s.pt"), device=args.device, backend="torch")
    sys.path.insert(0, str(B / "scripts" / "common"))
    import train_refiner_remote as TR
    from nusc_refiner_crop import crop64
    rname = "resnet18" if args.dataset == "nusc" else "resnet18_v2"
    ck = torch.load(B / "out" / rname / "image_refiner.pt", map_location=dev, weights_only=False)
    rnet = TR.Res18(len(ck["fmu"])); rnet.load_state_dict(ck["state"]); rnet = rnet.to(dev).eval()
    fmu, fsd = ck["fmu"], ck["fsd"]
    pp, cp = P.PlannerParams(), P.CostParams()
    if args.dataset == "nusc":
        W0, H0 = 1600, 900
        V_CLOSE, SCALE = 10.0, 0.6
        M = {"cheap": None, "u640": Mode("f384x640", 384, 640), "u960": Mode("f576x960", 576, 960), "native": Mode("ns_u1600", 928, 1600)}
        units = units_nusc(args.split)
    else:
        W0, H0 = 1550, 2048
        V_CLOSE, SCALE = 0.0, 960 / 2048
        M = {"cheap": Mode("av2_c320", 320, 256), "u640": Mode("av2_f640", 640, 512), "u960": Mode("av2_f960", 960, 768),
             "native": Mode("av2_u2048", 2048, 1568)}
        units = units_av2(args.split)
    NATIVE = W0 * H0
    CHEAP_PX = (192 * 320 if args.dataset == "nusc" else 320 * 256) / NATIVE
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_{args.split}_{args.setting}"
    grow_count = [0]
    _grow = SP.grow

    def counting_grow(*a, **k):
        grow_count[0] += 1
        return _grow(*a, **k)
    SP.grow = counting_grow                               # counts guard re-detections (one call per grown crop)
    for ui, (unit, cam, frames) in enumerate(units):
        if ui % sn != si or (args.max_units and ui >= args.max_units):
            continue
        fo = OUT / f"{tag}__{unit}.pkl"
        if fo.exists():
            continue
        R, t = cam.transform("ego")
        hist = {}
        prev_act, prev_act2, prev_cheap, prev_cheap_boxes = None, None, None, None
        rows = []
        rng_del = np.random.default_rng(zlib.crc32(unit.encode()))
        for fr, path, v, g, cheap_cached in frames():
            img = cv2.imread(path)
            if img is None:
                continue
            H, W = img.shape[:2]
            a_gt, a_gt_obj = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp) if len(g) else (0.0, np.zeros(0))
            act_gt = P.discrete_action(a_gt, pp)
            gt_t = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc")) if len(g) else tuple(np.zeros(0) for _ in range(4))
            act2_gt = c2_action(gt_t, pp)
            gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
            gtc = np.asarray(g["coarse"]).astype(str) if len(g) else np.zeros(0, str)
            relevant = np.flatnonzero(np.asarray(a_gt_obj) >= pp.thr_decel) if len(g) else np.zeros(0, int)
            rec = {"unit": unit, "frame": fr, "v": v, "act_gt": act_gt, "act2_gt": act2_gt, "n_relevant": len(relevant)}

            def lift(dd, pv):
                gk = predicted_geometry({"xyxy": dd["xyxy"], "coarse": dd["coarse"]}, pv, cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
                return tuple(np.asarray(gk[x], float) for x in ("z", "lat_min", "lat_max", "ttc"))

            def keep(d):
                k = d["conf"] >= THR
                return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k],
                        "entropy": np.asarray(d.get("entropy", np.zeros(len(d["conf"]))))[k]}

            def act(tt):
                return P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)

            def cost(a):
                return P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]

            def cost2(a2):
                return float(C2_COST[act2_gt, a2]) + (0.0 if prev_act2 is None or a2 == prev_act2 else 0.0)

            def record(name, scene_xyxy, scene_coarse, tt, px=None, extra=None):
                a = act(tt)
                rec[f"a_{name}"], rec[f"J_{name}"] = a, cost(a)
                rec[f"a2_{name}"] = c2_action(tt, pp)
                rec[f"J2_{name}"] = cost2(rec[f"a2_{name}"])
                if px is not None:
                    rec[f"px_{name}"] = px
                if len(relevant):
                    mm = matched(gtb[relevant], gtc[relevant], scene_xyxy, scene_coarse)
                    rec[f"found_{name}"] = mm >= 0
                if extra:
                    rec.update({f"{k}_{name}": x for k, x in extra.items()})
            # ---- cheap draft
            if cheap_cached is not None:
                cheap = cheap_cached
            else:
                cheap = keep(det.detect(img, M["cheap"])[0])
                cheap["mono"] = lift(cheap, prev_cheap)
            cheap_raw = cheap
            # ---- K4 B: delete matched draft detections of decision-relevant GT objects
            if args.setting != "base" and len(relevant):
                mm = matched(gtb[relevant], gtc[relevant], cheap["xyxy"], cheap["coarse"])
                dl = [j for j in mm if j >= 0 and (args.setting == "del_all" or rng_del.random() < 0.5)]
                rec["deleted"] = len(dl)
                rec["deleted_range"] = [float(cheap["mono"][0][j]) for j in dl]
                rec["deleted_gt_range"] = [float(g["long_near"][relevant[i]]) for i, j in enumerate(mm) if j in dl]
                rec["deleted_mask"] = np.array([j in dl for j in mm], bool)
                if dl:
                    k_ = np.ones(len(cheap["conf"]), bool); k_[dl] = False
                    cheap = {kk: (vv[k_] if not isinstance(vv, tuple) else tuple(a[k_] for a in vv)) for kk, vv in cheap.items()}
            else:
                rec["deleted"] = 0
            record("cheap", cheap["xyxy"], cheap["coarse"], cheap["mono"], CHEAP_PX)
            if args.setting != "base":
                rec["J_cheap_unmod"] = cost(act(cheap_raw["mono"]))
            # ---- whole-frame passes (pi(F)); u640 / native only in the base setting
            for nm, m in M.items():
                if nm == "cheap" or (args.setting != "base" and nm != "u960"):
                    continue
                d = keep(det.detect(img, m)[0])
                tt = lift(d, hist.get(nm))
                record(nm, d["xyxy"], d["coarse"], tt, m.net_h * m.net_w / NATIVE)
                hist[nm] = d
                if nm == "u960":
                    F960 = d

            tpx = [0.0]

            def detect_crop(cr):
                x1, y1 = int(cr[0]), int(cr[1])
                x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
                sub = img[y1:y2, x1:x2]
                nw = max(32, int(math.ceil(sub.shape[1] * SCALE / 32)) * 32)
                nh = max(32, int(math.ceil(sub.shape[0] * SCALE / 32)) * 32)
                tpx[0] += nh * nw
                d = keep(det.detect(sub, Mode(f"c{nh}x{nw}", nh, nw))[0])
                d["xyxy"] = d["xyxy"] + np.array([x1, y1, x1, y1], float)
                return {k: d[k] for k in ("xyxy", "conf", "coarse")}

            def refine(scene):
                refined = [np.asarray(a, float).copy() for a in scene["mono"]]
                for p_ in SP.support(scene["mono"], v, pp):
                    if not scene["from_full"][p_]:
                        continue
                    x1, y1, x2, y2 = scene["xyxy"][p_]
                    zz = predicted_geometry({"xyxy": scene["xyxy"][p_:p_ + 1], "coarse": scene["coarse"][p_:p_ + 1]}, None,
                                            cam.calib, cam.cam_h, cam.dt, cam_to_ego=(R, t))
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
                return tuple(refined)

            # ---- ours and its variants
            variants = VARIANTS if args.setting == "base" else ("full", "field_only", "support_only")
            budget = 0.0
            for vn in variants:
                cfg = {"full": {}, "field_only": {"support_mode": "none"}, "support_only": {"use_field": False},
                       "pivots": {"support_mode": "pivots"}, "one_round": {"max_rounds": 1}, "no_guard": {"guard": False},
                       "draft_hist": {}, "no_guard_draft_hist": {"guard": False}}[vn]
                hkey = "cheap_hist" if vn.endswith("draft_hist") else vn
                pv = prev_cheap_boxes if vn.endswith("draft_hist") else hist.get(f"scene_{vn}")
                lf = (lambda full, pv=pv: lift(full, pv))
                tpx[0] = 0.0; grow_count[0] = 0
                if vn == "full":
                    scene, crops, rounds = SP.run({k: cheap[k] for k in ("xyxy", "conf", "coarse", "mono")}, v, cam, W, H,
                                                  detect_crop, lf, pp, v_close=V_CLOSE, guard=True)
                    budget = sum(area(c) for c in crops)
                else:
                    scene, crops, rounds = run_variant({k: cheap[k] for k in ("xyxy", "conf", "coarse", "mono")}, v, cam, W, H,
                                                       detect_crop, lf, pp, V_CLOSE, **cfg)
                px = CHEAP_PX + tpx[0] / NATIVE
                record(vn, scene["xyxy"], scene["coarse"], scene["mono"], px,
                       {"crops": len(crops), "rounds": rounds, "guard_redet": grow_count[0], "crop_area": sum(area(c) for c in crops) / NATIVE})
                tr = refine(scene)
                a_r = act(tr)
                rec[f"a_{vn}_ref"], rec[f"J_{vn}_ref"] = a_r, cost(a_r)
                hist[f"scene_{vn}"] = {"xyxy": scene["xyxy"]}
                if vn == "full":
                    full_scene, full_crops = scene, crops
            # ---- heuristics at ours' per-frame crop area (strict)
            gdet = SP.guarded(detect_crop, W, H)
            P2 = cam.calib.P2
            for hn in HEUR:
                if hn == "uncertainty":
                    order = np.lexsort((-np.asarray(cheap.get("entropy", np.zeros(len(cheap["conf"])))), cheap["conf"]))
                    cands = [SP.box_crop(cheap["xyxy"][k], W, H) for k in order]
                elif hn == "temporal_crit":
                    cands = []
                    ps = hist.get("state_temporal_crit")
                    if ps is not None and len(ps[0]):
                        pxy, pmono = ps
                        _, dem = P.required_decel(*pmono, v, pp)
                        order = np.lexsort((pmono[0], -np.asarray(dem)))
                        cands = [SP.box_crop(np.r_[pxy[k][:2] - 0.25 * (pxy[k][2:] - pxy[k][:2]), pxy[k][2:] + 0.25 * (pxy[k][2:] - pxy[k][:2])], W, H) for k in order]
                else:
                    rr = np.random.default_rng(zlib.crc32(f"{unit}/{fr}".encode()))
                    cands = []
                    for _ in range(40):
                        if len(cheap["xyxy"]):
                            b = cheap["xyxy"][rr.integers(len(cheap["xyxy"]))]
                            w_, h_ = min((b[2] - b[0]) * 2, W), min((b[3] - b[1]) * 2, H)
                        else:
                            w_, h_ = 200.0, 120.0
                        h_ = min(h_, H * 0.65)
                        x0, y0 = rr.uniform(0, max(W - w_, 1)), rr.uniform(H * 0.35, max(H - h_, H * 0.35 + 1))
                        cands.append(np.array([x0, y0, x0 + w_, y0 + h_]))
                sel = fill_strict(cands, budget)
                tpx[0] = 0.0; grow_count[0] = 0
                scene_h, cr_h = merge_once({k: cheap[k] for k in ("xyxy", "conf", "coarse", "mono")}, sel, gdet,
                                           lambda full, hn=hn: lift(full, hist.get(f"scene_{hn}")))
                record(hn, scene_h["xyxy"], scene_h["coarse"], scene_h["mono"], CHEAP_PX + tpx[0] / NATIVE,
                       {"crops": len(cr_h), "guard_redet": grow_count[0]})
                hist[f"scene_{hn}"] = {"xyxy": scene_h["xyxy"]}
                if hn == "temporal_crit":
                    hist["state_temporal_crit"] = (np.asarray(scene_h["xyxy"], float).reshape(-1, 4), tuple(np.asarray(a, float) for a in scene_h["mono"]))
            # ---- K9: second consumer (TTC warning), ours with its own field / support (closed form) + uncertainty
            if args.setting == "base":
                cons2 = TTCWarn(pp, V_CLOSE)
                tpx[0] = 0.0
                scene2, crops2, rounds2 = SG.run({k: cheap[k] for k in ("xyxy", "conf", "coarse", "mono")}, v, cam, W, H,
                                                 detect_crop, lambda full: lift(full, hist.get("scene_c2")), cons2, max_rounds=3, guard=True)
                a2 = c2_action(scene2["mono"], pp)
                rec.update({"a2_ours_c2": a2, "J2_ours_c2": cost2(a2), "px_ours_c2": CHEAP_PX + tpx[0] / NATIVE, "crops_ours_c2": len(crops2)})
                hist["scene_c2"] = {"xyxy": scene2["xyxy"]}
                order = np.lexsort((-np.asarray(cheap.get("entropy", np.zeros(len(cheap["conf"])))), cheap["conf"]))
                sel = fill_strict([SP.box_crop(cheap["xyxy"][k], W, H) for k in order], sum(area(c) for c in crops2))
                tpx[0] = 0.0
                sc_u2, _ = merge_once({k: cheap[k] for k in ("xyxy", "conf", "coarse", "mono")}, sel, gdet,
                                      lambda full: lift(full, hist.get("scene_unc_c2")))
                a2 = c2_action(sc_u2["mono"], pp)
                rec.update({"a2_unc_c2": a2, "J2_unc_c2": cost2(a2), "px_unc_c2": CHEAP_PX + tpx[0] / NATIVE})
                hist["scene_unc_c2"] = {"xyxy": sc_u2["xyxy"]}
            # ---- K2.5 locality diagnostics (first diag_units units): F's objects inside the crops (+ F's ttc history)
            if args.setting == "base" and ui < args.diag_units:
                for dn, fhist in (("loc", False), ("loc_Fhist", True)):
                    def detect_crop_F(cr):
                        k = SP.centres_in(F960["xyxy"], [cr]) if len(F960["xyxy"]) else np.zeros(0, bool)
                        return {"xyxy": F960["xyxy"][k], "conf": F960["conf"][k], "coarse": F960["coarse"][k]}
                    pv = hist.get("u960_prev") if fhist else hist.get(f"scene_{dn}")
                    sc_d, _, _ = SP.run({k: cheap[k] for k in ("xyxy", "conf", "coarse", "mono")}, v, cam, W, H, detect_crop_F,
                                        lambda full, pv=pv: lift(full, pv), pp, v_close=V_CLOSE, guard=True)
                    record(dn, sc_d["xyxy"], sc_d["coarse"], sc_d["mono"])
                    hist[f"scene_{dn}"] = {"xyxy": sc_d["xyxy"]}
                hist["u960_prev"] = F960
            rows.append(rec)
            prev_act = act(cheap_raw["mono"])
            prev_act2 = c2_action(cheap_raw["mono"], pp)
            prev_cheap = {"xyxy": cheap["xyxy"], "coarse": cheap["coarse"]} if cheap_cached is None else None
            prev_cheap_boxes = {"xyxy": cheap["xyxy"]}
        with open(fo, "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{ui}] {unit}: {len(rows)} frames", flush=True)
    print("INSTR DONE")


if __name__ == "__main__":
    main()
