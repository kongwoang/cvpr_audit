#!/usr/bin/env python
"""K31 item 6 (P2, registered in docs/paper/handoff/K31/STATUS.md): empirical check of decision exactness for a
BOTTLENECK selector on NAVSIM navtest. pi(S) = argmax over 15 fixed candidates (route centreline offsets {0, -1.5, +1.5} m
x constant accelerations {+2, +1, 0, -1.5, -3} m/s^2) of s(tau, S) = min(b(tau), min_o h(tau, o)), b = 5 s x normalised
progress, h = time of first overlap of the ego rectangle with object o along tau (constant velocity, cap 5 s).
Stages (env: scripts/common/env_navsim.sh):
  run    per token: pi(cheap), pi(F = full960), pi(native); DESP loop (field by single probes on K20's 7 x 7 grid, support
         = objects with h(tau, o) <= s(tau*) for some tau != tau*, crops as spec_pdm_dets, <= 3 rounds) with
         (i) oracle locality (objects of F's detections whose box centre lies in the crops, the draft's elsewhere) and
         (ii) real crops (scale 0.5, truncation guard; GPU). -> results/k31/p2/<log>.pkl
  pairs  K20's 1,000-token subset, draft scene: pairs of probe cells whose joint insertion changes pi while neither
         single insertion does (full re-evaluation of pi). -> results/k31/p2_pairs/<log>.pkl
  table  -> results/k31/tables/p2_agreement.csv, p2_disagreements.csv, p2_pairs.csv
usage: k31_nav_p2.py --stage run|pairs|table [--shard i/n] [--max_tokens N] [--only_log L] [--no_real]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
from pathlib import Path

import numpy as np

B = Path(os.environ["B"])
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
DATA = Path(os.environ["OPENSCENE_DATA_ROOT"])
OUT = B / "results" / "k31" / ("p2" + os.environ.get("K31_P2_TAG", ""))
X_GRID = np.array([8.0, 12.0, 17.0, 23.0, 30.0, 40.0, 55.0])
Y_GRID = np.array([-10.5, -7.0, -3.5, 0.0, 3.5, 7.0, 10.5])
T = np.round(np.arange(51) * 0.1, 2)
OFFS, ACCS = (0.0, -1.5, 1.5), (2.0, 1.0, 0.0, -1.5, -3.0)
EGO_L, EGO_W, EGO_RA = 5.176, 2.297, 1.461
CAP, VMAX, RAMP_S, MAX_ROUNDS = 5.0, 15.0, 2.0, 3


def load_mod(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


NP = load_mod("np_", B / "scripts" / "A_navsim" / "navsim_pdms.py")
NH = load_mod("nh_", B / "scripts" / "A_navsim" / "k30_nav_hybrid.py")


# ---------------------------------------------------------------- scenes
def branch_objects(ann, det, cam, gz, priors):
    """navsim_pdms.branch_annotations split into the objects kept in every branch (outside the camera or not eligible:
    annotation indices) and the object of each detection: ("gt", i) matched to annotation i, ("fp", row) lifted, None"""
    from scipy.optimize import linear_sum_assignment
    boxes, names = np.asarray(ann["gt_boxes"], float).reshape(-1, 7), list(ann["gt_names"])
    elig = [i for i, n in enumerate(names) if n in NP.ELIGIBLE]
    proj = {i: cam.project(boxes[i]) for i in elig}
    inc = [i for i in elig if proj[i] is not None]
    dx, dc = np.asarray(det["xyxy"], float).reshape(-1, 4), np.asarray(det["coarse"]).astype(str)
    per = [None] * len(dx)
    if inc and len(dx):
        M = NP.iou([proj[i] for i in inc], dx)
        for a, i in enumerate(inc):
            M[a, [j for j in range(len(dx)) if dc[j] not in NP.COMPAT[names[i]]]] = 0
        r, c = linear_sum_assignment(-M)
        for a, j in zip(r, c):
            if M[a, j] >= NP.IOU_THR:
                per[j] = ("gt", inc[a])
    for j in range(len(dx)):
        if per[j] is not None:
            continue
        b = dx[j]
        gp = cam.ground_point(0.5 * (b[0] + b[2]), b[3], gz)
        if gp is None:
            continue
        (gx, gy), ray = gp
        if gx < NP.FP_MIN_AHEAD or np.hypot(gx, gy) > NP.FP_MAX_RANGE:
            continue
        L, Wd, _ = priors[NP.DET_TO_NAME[dc[j]]]
        per[j] = ("fp", np.array([gx + ray[0] * L / 2, gy + ray[1] * L / 2, float(np.arctan2(ray[1], ray[0])), L, Wd, 0.0, 0.0]))
    base = [i for i in range(len(names)) if i not in inc]
    return base, per


def gt_row(ann, i):
    b, v = np.asarray(ann["gt_boxes"], float)[i], np.asarray(ann["gt_velocity_3d"], float)[i]
    return np.array([b[0], b[1], b[6], b[3], b[4], v[0], v[1]])


class Scene:
    """objects [x, y, yaw, l, w, vx, vy]; dbox[k] = the detection box an object came from (None for kept annotations)"""

    def __init__(self, ann, base, items):
        rows, dbox, seen = [gt_row(ann, i) for i in base], [None] * len(base), set(base)
        for obj, box in items:
            if obj is None:
                continue
            if obj[0] == "gt":
                if obj[1] in seen:
                    continue
                seen.add(obj[1]); rows.append(gt_row(ann, obj[1]))
            else:
                rows.append(obj[1])
            dbox.append(np.asarray(box, float))
        self.o = np.asarray(rows, float).reshape(-1, 7)
        self.dbox = dbox


# ---------------------------------------------------------------- selector
def path_of(cl):
    """forward centreline resampled every 0.2 m, extended straight by 250 m: (s, xy, heading, normal)"""
    p = NH.forward_path(cl) if cl is not None and len(cl) >= 2 else np.zeros((0, 2))
    if len(p) < 2:
        p = np.array([[0.0, 0.0], [1.0, 0.0]])
    seg = np.hypot(*np.diff(p, axis=0).T)
    p = p[np.r_[True, seg > 1e-6]]
    t_end = (p[-1] - p[-2]) / np.hypot(*(p[-1] - p[-2]))
    p = np.vstack([p, p[-1] + 250.0 * t_end])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    sg = np.arange(0.0, s[-1], 0.2)
    xy = np.stack([np.interp(sg, s, p[:, 0]), np.interp(sg, s, p[:, 1])], 1)
    d = np.gradient(xy, axis=0)
    hd = np.arctan2(d[:, 1], d[:, 0])
    return sg, xy, hd, np.stack([-np.sin(hd), np.cos(hd)], 1)


class Selector:
    def __init__(self, cl, v0):
        sg, xy, hd, nr = path_of(cl)
        e0 = float(np.dot(-xy[0], nr[0]))                                   # ego's own lateral offset from the path
        poses, prog = [], []
        for d in OFFS:
            lat = e0 + (d - e0) * np.minimum(1.0, T / RAMP_S)
            for a in ACCS:
                v = np.clip(v0 + a * T, 0.0, max(v0, VMAX))
                S = np.r_[0.0, np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(T))]
                k = np.minimum(np.searchsorted(sg, S), len(sg) - 1)
                ra = xy[k] + lat[:, None] * nr[k]
                poses.append(np.stack([ra[:, 0] + EGO_RA * np.cos(hd[k]), ra[:, 1] + EGO_RA * np.sin(hd[k]), hd[k]], 1))
                prog.append(S[-1])
        self.P = np.asarray(poses)                                           # (15, 51, 3) ego rectangle centres
        prog = np.asarray(prog)
        self.b = CAP * prog / prog.max() if prog.max() >= 0.1 else np.full(len(prog), CAP)
        self.reach = float(np.hypot(self.P[..., 0], self.P[..., 1]).max()) + 15.0

    def h(self, o):
        """(15, K) time of first overlap (cap 5 s)"""
        o = np.asarray(o, float).reshape(-1, 7)
        H = np.full((len(self.P), len(o)), CAP)
        near = np.hypot(o[:, 0], o[:, 1]) <= self.reach + np.hypot(o[:, 5], o[:, 6]) * CAP
        if not near.any():
            return H
        q = o[near]
        ax, ay, ah = (self.P[:, None, :, i] for i in range(3))                # (15, 1, 51)
        bx = q[None, :, 0, None] + q[None, :, 5, None] * T
        by = q[None, :, 1, None] + q[None, :, 6, None] * T
        bh, bl, bw = q[None, :, 2, None], q[None, :, 3, None], q[None, :, 4, None]
        dx, dy = bx - ax, by - ay
        ca, sa, cb, sb = np.cos(ah), np.sin(ah), np.cos(bh), np.sin(bh)
        sep = np.zeros(np.broadcast_shapes(dx.shape, bh.shape), bool)
        for ux, uy in ((ca, sa), (-sa, ca), (cb, sb), (-sb, cb)):
            ra = EGO_L / 2 * np.abs(ca * ux + sa * uy) + EGO_W / 2 * np.abs(-sa * ux + ca * uy)
            rb = bl / 2 * np.abs(cb * ux + sb * uy) + bw / 2 * np.abs(-sb * ux + cb * uy)
            sep |= np.abs(dx * ux + dy * uy) > ra + rb
        hit = ~sep                                                            # (15, Kn, 51)
        first = np.where(hit.any(-1), T[np.argmax(hit, -1)], CAP)
        H[:, near] = np.minimum(first, CAP)
        return H

    def score(self, o):
        Hm = self.h(o)
        return np.minimum(self.b, Hm.min(1) if Hm.shape[1] else CAP), Hm

    def pi(self, o):
        return int(np.argmax(self.score(o)[0]))


# ---------------------------------------------------------------- DESP
def probes(cam, gz, priors):
    L, Wd, Ht = priors["vehicle"]
    out = []
    for x in X_GRID:
        for y in Y_GRID:
            ib = cam.project(np.array([x, y, gz + Ht / 2, L, Wd, Ht, 0.0]))
            if ib is not None:
                out.append((np.array([x, y, 0.0, L, Wd, 0.0, 0.0]), np.asarray(ib, float)))
    return out


def field_support(sel, sc, prb):
    s, Hm = sel.score(sc.o)
    a = int(np.argmax(s))
    fb = [ib for row, ib in prb if sel.pi(np.vstack([sc.o, row])) != a]     # full re-evaluation per probe
    others = np.arange(len(s)) != a
    sup = [sc.dbox[k] for k in range(len(sc.o)) if sc.dbox[k] is not None and (Hm[others, k] <= s[a]).any()]
    return a, fb, sup


def crops_of(fb, sup, R, VS, W, H):
    cols = {}
    for b in fb:
        cols.setdefault(round(0.5 * (b[0] + b[2]) / 160), []).append(b)
    new = []
    for bs in cols.values():
        bs = np.asarray(bs)
        c = np.array([bs[:, 0].min(), bs[:, 1].min(), bs[:, 2].max(), bs[:, 3].max()])
        if not VS.covered(c, R + new):
            new.append(c)
    for b in sup:
        c = VS.box_crop(np.asarray(b, float), W, H)
        if not VS.covered(c, R + new):
            new.append(c)
    return new


def loop(sel, prb, merge, draft, VS, W, H):
    """Algorithm 1; merge(R) -> Scene. Stops when the field and support of the current merged scene add no crop
    (then R covers them, as the proposition requires), at most MAX_ROUNDS detection rounds.
    Returns (pi, rounds, R, scene, capped, per-round log); capped = crops were still being added after the last round"""
    R, M, log = [], draft, []
    for k in range(MAX_ROUNDS + 1):
        a, fb, sup = field_support(sel, M, prb)
        new = crops_of(fb, sup, R, VS, W, H)
        log.append({"a": a, "field_cells": len(fb), "support": len(sup), "new_crops": len(new)})
        if not new or k == MAX_ROUNDS:
            return a, k, R, M, bool(new), log
        R = R + new
        M = merge(R)


def area(cs, W, H):
    return float(sum((c[2] - c[0]) * (c[3] - c[1]) for c in cs) / (W * H))


def classify(sel, ann, M, mapF, detF, R, VS):
    """why the oracle loop disagrees with pi(F): objects of F outside R whose single insertion into M changes pi"""
    a = sel.pi(M.o)
    out = ~VS.centres_in(np.asarray(detF["xyxy"], float).reshape(-1, 4), R) if R else np.ones(len(mapF), bool)
    have = {tuple(np.round(r, 4)) for r in M.o}
    n_field = 0
    for j in np.flatnonzero(out):
        obj = mapF[j]
        if obj is None:
            continue
        row = gt_row(ann, obj[1]) if obj[0] == "gt" else obj[1]
        if tuple(np.round(row, 4)) in have:
            continue
        n_field += int(sel.pi(np.vstack([M.o, row])) != a)
    return n_field


def run(args):
    NS = load_mod("nsr", B / "scripts" / "A_nuplan" / "nuplan_spec_real.py")
    VS, W, H = NS.VS, NS.W, NS.H
    import cv2
    from rap.mono import box_iou
    gz = float((NP.DETS / "ground_z.txt").read_text()) if (NP.DETS / "ground_z.txt").exists() else \
        float((EXP / "dets" / "ground_z.txt").read_text())
    priors = pickle.load(open(EXP / "dets" / "size_priors.pkl", "rb"))
    si, sn = (int(x) for x in args.shard.split("/"))
    OUT.mkdir(parents=True, exist_ok=True)
    logs = sorted(p.stem for p in (EXP / "dets_k30b").glob("*.pkl"))
    if args.only_log:
        logs = [args.only_log]
    n = 0
    for li, log in enumerate(logs):
        if li % sn != si or (OUT / f"{log}.pkl").exists():
            continue
        dets = pickle.load(open(EXP / "dets_k30b" / f"{log}.pkl", "rb"))
        cen = pickle.load(open(EXP / "k30_center" / f"{log}.pkl", "rb")) if (EXP / "k30_center" / f"{log}.pkl").exists() else {}
        frames = {fr["token"]: fr for fr in pickle.load(open(DATA / "navsim_logs" / "test" / f"{log}.pkl", "rb"))}
        rows = []
        for tok in sorted(dets):
            rec = dets[tok]
            if "cam" not in rec or "full960" not in rec or tok not in frames:
                continue
            fr = frames[tok]
            ann, cam = fr["anns"], NP.Cam(*rec["cam"])
            cl = cen.get(tok)
            cl_ok = isinstance(cl, np.ndarray) and len(cl) >= 2
            v0 = float(np.hypot(*np.asarray(fr["ego_dynamic_state"], float)[:2]))
            sel = Selector(cl if cl_ok else None, v0)
            prb = probes(cam, gz, priors)
            maps = {b: branch_objects(ann, rec[b], cam, gz, priors) for b in ("cheap", "full960", "native")}
            sc = {b: Scene(ann, maps[b][0], zip(maps[b][1], np.asarray(rec[b]["xyxy"], float).reshape(-1, 4))) for b in maps}
            r = {"log": log, "token": tok, "v0": v0, "centerline": bool(cl_ok), "n_obj": len(sc["full960"].o),
                 "n_probe_cells": len(prb)}
            for b in maps:
                r[f"pi_{b}"] = sel.pi(sc[b].o)
            # (i) oracle locality
            bC, pC = maps["cheap"]; bF, pF = maps["full960"]
            xC, xF = np.asarray(rec["cheap"]["xyxy"], float).reshape(-1, 4), np.asarray(rec["full960"]["xyxy"], float).reshape(-1, 4)

            def merge_oracle(R):
                inC, inF = VS.centres_in(xC, R), VS.centres_in(xF, R)
                items = [(pC[j], xC[j]) for j in range(len(xC)) if not inC[j]] + [(pF[j], xF[j]) for j in range(len(xF)) if inF[j]]
                return Scene(ann, bC, items)
            a, k, R, M, capped, lg = loop(sel, prb, merge_oracle, sc["cheap"], VS, W, H)
            r.update({"pi_oracle": a, "rounds_oracle": k, "capped_oracle": capped, "px_oracle": area(R, W, H),
                      "field_cells_r1": lg[0]["field_cells"], "support_r1": lg[0]["support"]})
            if a != r["pi_full960"]:
                r["oracle_field_miss_objects"] = classify(sel, ann, M, pF, rec["full960"], R, VS)
            # (ii) real crops
            if not args.no_real:
                img = cv2.imread(str(DATA / "sensor_blobs" / "test" / fr["cams"]["CAM_F0"]["data_path"]))
                if img is None:
                    r["pi_real"] = -1
                    rows.append(r)
                    continue
                gdet = VS.guarded(NS.real_crop_detector(img), W, H)
                cache = {}

                def merge_real(R):
                    full = {"xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "coarse": np.zeros(0, "<U8")}
                    grown = []
                    for c_ in R:
                        key = tuple(np.round(c_, 2))
                        if key not in cache:
                            cache[key] = gdet(c_)
                        d_, ex_ = cache[key]
                        grown += ex_
                        full = {kk: np.concatenate([full[kk], np.asarray(d_[kk])]) for kk in full}
                    if len(full["conf"]) > 1:
                        o_, iou_ = np.argsort(-full["conf"], kind="stable"), box_iou(full["xyxy"], full["xyxy"])
                        kp = []
                        for i_ in o_:
                            if not kp or iou_[i_, kp].max() < 0.5:
                                kp.append(i_)
                        full = {kk: full[kk][np.sort(np.array(kp))] for kk in full}
                    out = VS.centres_in(xC, R) if R else np.zeros(len(xC), bool)
                    if grown and len(full["conf"]) and len(xC):
                        out |= box_iou(xC, full["xyxy"]).max(1) >= 0.5
                    det = {kk: np.concatenate([np.asarray(rec["cheap"][kk])[~out], full[kk]]) for kk in ("xyxy", "conf", "coarse")}
                    bm, pm = branch_objects(ann, det, cam, gz, priors)
                    merge_real.grown = grown
                    return Scene(ann, bm, zip(pm, np.asarray(det["xyxy"], float).reshape(-1, 4)))
                merge_real.grown = []
                a2, k2, R2, _, cap2, _ = loop(sel, prb, merge_real, sc["cheap"], VS, W, H)
                r.update({"pi_real": a2, "rounds_real": k2, "capped_real": cap2,
                          "px_real": area(R2 + list(merge_real.grown), W, H)})
            rows.append(r)
            n += 1
            if args.max_tokens and n >= args.max_tokens:
                break
        with open(OUT / f"{log}.pkl", "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{li + 1}/{len(logs)}] {log}: {len(rows)} tokens; oracle agreement "
              f"{np.mean([x['pi_oracle'] == x['pi_full960'] for x in rows]) if rows else float('nan'):.3f}", flush=True)
        if args.max_tokens and n >= args.max_tokens:
            break
    print("K31 P2 RUN DONE", flush=True)


def pairs(args):
    KP_FIELD = EXP / "field_test"
    toks = []
    for f in sorted(KP_FIELD.glob("*.pkl")):
        d = pickle.load(open(f, "rb"))
        toks += [(f.stem, t) for t, v in sorted(d.items()) if "err" not in v]
    rng = np.random.default_rng(0)
    pick = [toks[i] for i in sorted(rng.choice(len(toks), size=min(1000, len(toks)), replace=False))]
    si, sn = (int(x) for x in args.shard.split("/"))
    gz = float((EXP / "dets" / "ground_z.txt").read_text())
    priors = pickle.load(open(EXP / "dets" / "size_priors.pkl", "rb"))
    od = B / "results" / "k31" / ("p2_pairs" + os.environ.get("K31_P2_TAG", ""))
    od.mkdir(parents=True, exist_ok=True)
    logs = sorted({lg for lg, _ in pick})
    for li, log in enumerate(logs):
        if li % sn != si or (od / f"{log}.pkl").exists():
            continue
        dets = pickle.load(open(EXP / "dets_k30b" / f"{log}.pkl", "rb"))
        cen = pickle.load(open(EXP / "k30_center" / f"{log}.pkl", "rb"))
        frames = {fr["token"]: fr for fr in pickle.load(open(DATA / "navsim_logs" / "test" / f"{log}.pkl", "rb"))}
        rows = []
        for tok in [t for lg, t in pick if lg == log]:
            rec, fr = dets.get(tok), frames.get(tok)
            if rec is None or fr is None or "cam" not in rec:
                rows.append({"log": log, "token": tok, "err": "missing"})
                continue
            cam, ann = NP.Cam(*rec["cam"]), fr["anns"]
            cl = cen.get(tok)
            sel = Selector(cl if isinstance(cl, np.ndarray) and len(cl) >= 2 else None,
                           float(np.hypot(*np.asarray(fr["ego_dynamic_state"], float)[:2])))
            base, per = branch_objects(ann, rec["cheap"], cam, gz, priors)
            o = Scene(ann, base, zip(per, np.asarray(rec["cheap"]["xyxy"], float).reshape(-1, 4))).o
            a = sel.pi(o)
            prb = [p for p, _ in probes(cam, gz, priors)]
            single = [sel.pi(np.vstack([o, p])) != a for p in prb]
            n_pairs = n_quiet = n_joint = 0
            for i in range(len(prb)):
                for j in range(i + 1, len(prb)):
                    n_pairs += 1
                    if single[i] or single[j]:
                        continue
                    n_quiet += 1
                    n_joint += int(sel.pi(np.vstack([o, prb[i], prb[j]])) != a)
            rows.append({"log": log, "token": tok, "cells": len(prb), "single_sensitive": int(sum(single)), "pairs": n_pairs,
                         "pairs_neither_single": n_quiet, "pairs_joint_change": n_joint})
        with open(od / f"{log}.pkl", "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{li + 1}/{len(logs)}] {log}: {len(rows)} tokens, joint-only changes "
              f"{sum(r.get('pairs_joint_change', 0) for r in rows)}", flush=True)
    print("K31 P2 PAIRS DONE", flush=True)


def table(args):
    import pandas as pd
    rows = [r for f in sorted(OUT.glob("*.pkl")) for r in pickle.load(open(f, "rb"))]
    d = pd.DataFrame(rows)
    logs = d.log.to_numpy()
    ul = np.unique(logs)
    idx = {u: np.flatnonzero(logs == u) for u in ul}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([idx[u] for u in rng.choice(ul, len(ul))]) for _ in range(2000)]

    def agree(col, m=None):
        x = (d[col] == d["pi_full960"]).to_numpy(float)
        mm = np.ones(len(d), bool) if m is None else m
        bs = [x[k][mm[k]].mean() for k in draws if mm[k].any()]
        return x[mm].mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), int(mm.sum())
    out = []
    dis = (d.pi_cheap != d.pi_full960).to_numpy()
    for lab, col, px in (("pi(cheap draft)", "pi_cheap", None), ("pi(native)", "pi_native", None),
                         ("DESP, oracle locality", "pi_oracle", "px_oracle"), ("DESP, real crops", "pi_real", "px_real")):
        if col not in d:
            continue
        for sub, m in (("all tokens", None), ("tokens where pi(cheap) != pi(F)", dis)):
            a, lo, hi, nt = agree(col, m)
            mm = np.ones(len(d), bool) if m is None else m
            out.append({"row": lab, "tokens": sub, "n_tokens": nt, "logs": d[mm].log.nunique(), "agreement_with_pi_F": a,
                        "ci_lo": lo, "ci_hi": hi, "px_mean": d.loc[mm, px].mean() if px else np.nan,
                        "rounds_mean": d.loc[mm, col.replace("pi_", "rounds_")].mean() if px else np.nan,
                        "share_capped": d.loc[mm, col.replace("pi_", "capped_")].mean() if px else np.nan})
    tb = B / "results" / "k31" / "tables"
    pd.DataFrame(out).to_csv(tb / "p2_agreement.csv", index=False)
    cols = [c for c in ("log", "token", "v0", "centerline", "n_obj", "pi_cheap", "pi_full960", "pi_native", "pi_oracle",
                        "rounds_oracle", "capped_oracle", "oracle_field_miss_objects", "pi_real") if c in d]
    d.loc[d.pi_oracle != d.pi_full960, cols].to_csv(tb / "p2_disagreements.csv", index=False)
    pr = [r for f in sorted((B / "results" / "k31" / "p2_pairs").glob("*.pkl")) for r in pickle.load(open(f, "rb"))]
    if pr:
        p = pd.DataFrame(pr)
        p = p[p.get("err").isna()] if "err" in p else p
        pd.DataFrame([{"tokens": len(p), "probe_cells_mean": p.cells.mean(), "single_sensitive_mean": p.single_sensitive.mean(),
                       "pairs": int(p.pairs.sum()), "pairs_neither_single": int(p.pairs_neither_single.sum()),
                       "pairs_joint_change": int(p.pairs_joint_change.sum()),
                       "tokens_with_joint_change": int((p.pairs_joint_change > 0).sum())}]).to_csv(tb / "p2_pairs.csv", index=False)
    print(pd.DataFrame(out).round(4).to_string())
    print("K31 P2 TABLE DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "pairs", "table"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max_tokens", type=int, default=0)
    ap.add_argument("--only_log", default="")
    ap.add_argument("--no_real", action="store_true")
    a = ap.parse_args()
    {"run": run, "pairs": pairs, "table": table}[a.stage](a)


if __name__ == "__main__":
    main()
