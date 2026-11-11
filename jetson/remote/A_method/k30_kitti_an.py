#!/usr/bin/env python
"""K30 V8c + the K28 KITTI flag re-run: tables from results/k30/kitti/<seq>.pkl (k30_kitti.py; post-hoc, CPU).
(1) DEEP's four KITTI cells (brake mono J, brake oracle Jbo, Planner B mono Jtm / oracle Jto) as 61_k28_v6_latency.summarise:
    G_cell = (sum cheap - sum pipeline) / sum cheap, mean of the four; v6-S minus each pipeline's mean of 4, sequence bootstrap
    2,000, seed 0 (the 61 construction). Every refiner row with KITTI's camera flag 0.0 (as v3a was trained) and 1.0 (K28's
    Jetson run). Frame sets: all 3,536 DEEP test frames, and K28's 3,336 (61 dropped the first 200 rows, sequence 0007 frames
    0-199, as warm-up).
(2) KITTI 2D AP (official object protocol, port of the devkit's python evaluation, bbox only): Car IoU 0.7 (Van ignored),
    Pedestrian 0.5 (Person = Person_sitting ignored), Cyclist 0.5; easy / moderate / hard = min height 40 / 25 / 25 px,
    occlusion <= 0 / 1 / 2, truncation <= 0.15 / 0.30 / 0.50 (tracking labels code truncation 0 / 1 / 2, so only truncation-0
    objects count; the others are ignored, never false negatives); DontCare regions remove detections (intersection over the
    detection's area > the IoU threshold); detections lower than the minimum height ignored; 40 recall points (R40).
    COCO detector classes: vehicle (car / bus / truck) -> Car, person -> Pedestrian, cyclist (bicycle / motorcycle) -> Cyclist.
    Pipelines: cheap (96x320 draft), u960, native, v6-S (the K7 definition stored by k30_kitti.py); boxes with conf >= 0.1.
    Intervals: sequence bootstrap (same 2,000 resamples), each resample's 41 score thresholds snapped down to a 0.0025 grid
    (the per-sequence counts are precomputed on that grid; the full-sample AP on the grid is reported next to the exact one).
results/k30/tables/v8c_kitti_cells.csv, v8c_kitti_ap.csv, v8c_kitti_ap_diff.csv
"""
from __future__ import annotations

import glob
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ.get("B_REAL", os.environ["B"]))
R30 = B / "results" / "k30"
LAB = B / "datasets" / "kitti_tracking" / "training" / "label_02"
TAGS = (("brake mono", "J"), ("brake oracle", "Jbo"), ("traj mono", "Jtm"), ("traj oracle", "Jto"))
CELL_PIPES = ["u960", "u960+C1", "u960+T4+ref_f0", "u960+T4+ref_f1", "u960+C1+ref_f0", "u960+C1+ref_f1",
              "native", "native+C1", "native+T4+ref_f0", "native+T4+ref_f1", "native+C1+ref_f0", "native+C1+ref_f1",
              "v6S_noref", "v6S_f0", "v6S_f1"]
K28_NAME = {"u960+T4+ref_f1": "u960+comp", "u960+C1+ref_f1": "u960+c1comp", "native+T4+ref_f1": "native+comp",
            "native+C1+ref_f1": "native+c1comp", "v6S_noref": "ours_noref", "v6S_f1": "ours", "u960": "u960", "native": "native"}
CLASSES = {"Car": ("vehicle", 0.7, ("van",)), "Pedestrian": ("person", 0.5, ("person_sitting", "person")), "Cyclist": ("cyclist", 0.5, ())}
DIFF = ("easy", "moderate", "hard")
MIN_H, MAX_OCC, MAX_TR = (40, 25, 25), (0, 1, 2), (0.15, 0.3, 0.5)
AP_PIPES = ("cheap", "u960", "native", "v6S")
GRID = np.round(np.arange(0.1, 1.0 + 1e-9, 0.0025), 4)
NB, SEED = 2000, 0


def load():
    rows, dets = [], {}
    for f in sorted(glob.glob(str(R30 / "kitti" / "*.pkl"))):
        x = pickle.load(open(f, "rb"))
        rows += x["rows"]
        s = Path(f).stem
        for (fr, nm), o in x["dets"].items():
            dets[(s, int(fr), nm)] = o
    return pd.DataFrame(rows), dets


def boots(units):
    uq = sorted(set(units))
    rng = np.random.default_rng(SEED)
    return uq, [rng.integers(0, len(uq), len(uq)) for _ in range(NB)]


# ------------------------------------------------------------------ (1) decision cells
def cells(d, tag):
    d = d.reset_index(drop=True)
    uq = list(d.unit.unique())
    idx = [np.flatnonzero(d.unit.to_numpy() == u) for u in uq]
    rng = np.random.default_rng(SEED)
    bs = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(NB)]
    Ls = {tg: d[f"{tg}_cheap"].to_numpy(float) for _, tg in TAGS}
    Vs = {nm: {tg: Ls[tg] - d[f"{tg}_{nm}"].to_numpy(float) for _, tg in TAGS} for nm in CELL_PIPES}
    out = []
    for nm in CELL_PIPES:
        r = {"frames_set": tag, "frames": len(d), "pipeline": nm, "k28_name": K28_NAME.get(nm, ""),
             **{c: Vs[nm][tg].sum() / Ls[tg].sum() for c, tg in TAGS}}
        r["mean4"] = float(np.mean([r[c] for c, _ in TAGS]))
        if nm.startswith("v6S"):
            r["px_mean"] = d.px_v6S.mean()
        for ref in ("v6S_f0", "v6S_f1"):
            if nm == ref:
                continue
            dr = [np.mean([(Vs[ref][tg][b].sum() - Vs[nm][tg][b].sum()) / Ls[tg][b].sum() for _, tg in TAGS]) for b in bs]
            r[f"{ref}_minus_mean4"] = r_ = float(np.mean([(Vs[ref][tg].sum() - Vs[nm][tg].sum()) / Ls[tg].sum() for _, tg in TAGS]))
            r[f"{ref}_minus_lo"], r[f"{ref}_minus_hi"] = float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))
        out.append(r)
    return out


# ------------------------------------------------------------------ (2) KITTI 2D AP
def labels(seq):
    g = pd.read_csv(LAB / f"{seq}.txt", sep=" ", header=None, usecols=range(10),
                    names=["frame", "tid", "type", "trunc", "occ", "alpha", "x1", "y1", "x2", "y2"])
    return {f: x for f, x in g.groupby("frame")}


def iou(a, b, crit=-1):
    """devkit image_box_overlap: crit -1 IoU, 0 intersection over the area of a"""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    den = aa[:, None] + bb[None, :] - inter if crit == -1 else aa[:, None] * np.ones((1, len(b)))
    return inter / np.maximum(den, 1e-12)


def prep(g, det, cls, di):
    """devkit clean_data for one image -> (n_valid_gt, ignored_gt, ignored_dt (only dets != -1 kept), scores, ov, ov_dc)"""
    name, _, neigh = CLASSES[cls]
    ig_gt, gtb, dcb = [], [], []
    nv = 0
    if g is not None:
        for t, tr, oc, x1, y1, x2, y2 in zip(g["type"].str.lower(), g.trunc, g.occ, g.x1, g.y1, g.x2, g.y2):
            if t == "dontcare":
                dcb.append((x1, y1, x2, y2))
                continue
            vc = 1 if t == cls.lower() else (0 if t in neigh else -1)
            if vc == -1:
                continue
            ign = oc > MAX_OCC[di] or tr > MAX_TR[di] or (y2 - y1) <= MIN_H[di]
            if vc == 1 and not ign:
                ig_gt.append(0); nv += 1
            else:
                ig_gt.append(1)
            gtb.append((x1, y1, x2, y2))
    X = np.asarray(det["xyxy"], float).reshape(-1, 4)
    k = np.asarray(det["coarse"]).astype(str) == name
    X, sc = X[k], np.asarray(det["conf"], float)[k]
    ig_dt = np.where((X[:, 3] - X[:, 1]) < MIN_H[di], 1, 0) if len(X) else np.zeros(0, int)
    gtb, dcb = np.asarray(gtb, float).reshape(-1, 4), np.asarray(dcb, float).reshape(-1, 4)
    return nv, np.asarray(ig_gt, int), ig_dt, sc, iou(X, gtb), iou(X, dcb, 0)


def stats(im, mo, thresh=0.0, fp_mode=False):
    """devkit compute_statistics_jit (metric bbox); returns tp, fp, fn, scores of the true positives"""
    nv, ig_gt, ig_dt, sc, ov, ovdc = im
    nd = len(sc)
    assigned = np.zeros(nd, bool)
    below = sc < thresh if fp_mode else np.zeros(nd, bool)
    tp = fp = fn = 0
    tps = []
    for i in range(len(ig_gt)):
        det_idx, valid, maxo, ass_ign = -1, None, 0.0, False
        for j in range(nd):
            if assigned[j] or below[j]:
                continue
            o = ov[j, i]
            if o <= mo:
                continue
            if not fp_mode:
                if valid is None or sc[j] > valid:
                    det_idx, valid = j, sc[j]
            elif (o > maxo or ass_ign) and ig_dt[j] == 0:
                maxo, det_idx, valid, ass_ign = o, j, 1, False
            elif valid is None and ig_dt[j] == 1:
                det_idx, valid, ass_ign = j, 1, True
        if valid is None:
            if ig_gt[i] == 0:
                fn += 1
        elif ig_gt[i] == 1 or ig_dt[det_idx] == 1:
            assigned[det_idx] = True
        else:
            tp += 1
            tps.append(sc[det_idx])
            assigned[det_idx] = True
    if fp_mode:
        free = ~(assigned | (ig_dt == 1) | below)
        fp = int(free.sum())
        if ovdc.shape[1]:
            for i in range(ovdc.shape[1]):
                for j in range(nd):
                    if free[j] and ovdc[j, i] > mo:
                        free[j] = False
                        fp -= 1
    return tp, fp, fn, tps


def get_thresholds(scores, num_gt, n_pts=41):
    scores = np.sort(np.asarray(scores, float))[::-1]
    cur, th = 0.0, []
    for i, s in enumerate(scores):
        lr = (i + 1) / num_gt
        rr = (i + 2) / num_gt if i < len(scores) - 1 else lr
        if (rr - cur) < (cur - lr) and i < len(scores) - 1:
            continue
        th.append(s)
        cur += 1.0 / (n_pts - 1.0)
    return th


def ap_r40(tp, fp):
    prec = np.zeros(41)
    n = len(tp)
    for i in range(n):
        prec[i] = tp[i] / max(tp[i] + fp[i], 1e-12)
    for i in range(n):
        prec[i] = prec[i:].max()
    return float(prec[1:].sum() / 40 * 100)


def combo(args):
    pipe, cls, di, seqs, imgs = args
    mo = CLASSES[cls][1]
    per = {}
    for s in seqs:
        ims = [prep(g, det, cls, di) for g, det in imgs[s]]
        nv = sum(im[0] for im in ims)
        tps = [x for im in ims for x in stats(im, mo)[3]]
        cnt = np.zeros((len(GRID), 2))
        for im in ims:
            if not len(im[3]):
                continue
            hi = np.searchsorted(GRID, im[3].max(), side="right")      # thresholds above every score: no detection survives
            for k in range(hi):
                tp, fp, _, _ = stats(im, mo, GRID[k], True)
                cnt[k] += (tp, fp)
        per[s] = (nv, np.asarray(tps), cnt, ims)
    # exact AP on the full sample (devkit)
    nv = sum(p[0] for p in per.values())
    th = get_thresholds(np.concatenate([p[1] for p in per.values()]), nv) if nv else []
    tp_, fp_ = [], []
    for t in th:
        a = np.zeros(2)
        for p in per.values():
            for im in p[3]:
                a += stats(im, mo, t, True)[:2]
        tp_.append(a[0]); fp_.append(a[1])
    exact = ap_r40(tp_, fp_) if nv else np.nan

    def grid_ap(mult):
        nv_ = sum(m * per[s][0] for s, m in mult.items())
        if nv_ == 0:
            return np.nan
        th_ = get_thresholds(np.concatenate([np.repeat(per[s][1], m) for s, m in mult.items()]), nv_)
        cnt = sum(m * per[s][2] for s, m in mult.items())
        ks = np.clip(np.searchsorted(GRID, np.asarray(th_) + 1e-9, side="right") - 1, 0, len(GRID) - 1)
        return ap_r40(cnt[ks, 0], cnt[ks, 1])
    full_grid = grid_ap({s: 1 for s in seqs})
    uq, bs = boots(seqs)
    bo = np.array([grid_ap({uq[i]: int(c) for i, c in zip(*np.unique(b, return_counts=True))}) for b in bs])
    return {"pipeline": pipe, "class": cls, "difficulty": DIFF[di], "gt_count": nv, "AP_R40": exact, "AP_R40_grid": full_grid}, bo


def main():
    d, dets = load()
    OUT = R30 / "tables"
    out = cells(d, "all DEEP test frames") + cells(d[~((d.unit == sorted(d.unit.unique())[0]) & (d.frame < 200))], "K28 set (0007 frames 0-199 dropped)")
    co = pd.DataFrame(out)
    k28 = B / "results" / "k30" / "kitti" / "k28_kitti_v6S_cells.csv"
    if k28.exists():                                           # K28 Jetson table (flag 1.0, TensorRT) for the check
        j = pd.read_csv(k28).set_index("pipeline")
        co["k28_jetson_mean4"] = co.k28_name.map(lambda n: j["mean4"].get(n, np.nan) if n else np.nan)
    co.to_csv(OUT / "v8c_kitti_cells.csv", index=False)
    pd.set_option("display.width", 250)
    print(co.round(3).to_string(index=False), flush=True)
    seqs = sorted(d.unit.unique())
    lab = {s: labels(s) for s in seqs}
    imgs = {p: {s: [(lab[s].get(int(fr)), dets[(s, int(fr), p)]) for fr in d[d.unit == s].frame] for s in seqs} for p in AP_PIPES}
    jobs = [(p, c, di, seqs, imgs[p]) for p in AP_PIPES for c in CLASSES for di in range(3)]
    with Pool(int(os.environ.get("NPROC", 8))) as pool:
        res = pool.map(combo, jobs)
    ap = pd.DataFrame([r for r, _ in res])
    G = {p: (d.J_cheap.sum() - d[f"J_{k}"].sum()) / d.J_cheap.sum() for p, k in (("cheap", "cheap"), ("u960", "u960"), ("native", "native"), ("v6S", "v6S_f0"))}
    ap["G_brake_mono"] = ap.pipeline.map(G)
    ap["px_mean"] = ap.pipeline.map({"cheap": 96 * 320 / (375 * 1242), "u960": 288 * 960 / (375 * 1242), "native": 384 * 1248 / (375 * 1242), "v6S": d.px_v6S.mean()})
    ap.to_csv(OUT / "v8c_kitti_ap.csv", index=False)
    print(ap.round(3).to_string(index=False), flush=True)
    bo = {(r["pipeline"], r["class"], r["difficulty"]): b for r, b in res}
    df = []
    for c in CLASSES:
        for dn in DIFF:
            for p in ("cheap", "u960", "native"):
                a, b = bo[("v6S", c, dn)], bo[(p, c, dn)]
                ok = np.isfinite(a) & np.isfinite(b)
                ra = ap.set_index(["pipeline", "class", "difficulty"])
                df.append({"class": c, "difficulty": dn, "other": p, "v6S_minus": ra.loc[("v6S", c, dn), "AP_R40"] - ra.loc[(p, c, dn), "AP_R40"],
                           "ci_lo": float(np.percentile((a - b)[ok], 2.5)), "ci_hi": float(np.percentile((a - b)[ok], 97.5)), "resamples": int(ok.sum())})
    df = pd.DataFrame(df)
    df.to_csv(OUT / "v8c_kitti_ap_diff.csv", index=False)
    print(df.round(2).to_string(index=False))
    print("K30 KITTI ANALYZE DONE", flush=True)


if __name__ == "__main__":
    main()
