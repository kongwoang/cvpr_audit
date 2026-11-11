#!/usr/bin/env python
"""K30 V7 / V8e — detection metrics from the k30_eval2.py apdump outputs (pre-registered in handoff/K30/STATUS.md). CPU.
Ground truth: the benchmark's projected 2D boxes per frame (k24_test.units, height >= 10 px; classes vehicle / person /
cyclist; instr_eval.COMPAT class compatibility), with the annotated geometry (long_near, lat_min, lat_max, ttc) and ego speed.
Outputs: "ap" (conf >= 0.1; AV2 / Argoverse-HD) or "sc" (conf >= 0.25; nuScenes, whose cached draft has no low-confidence
boxes; also every recall / relevance metric). AV2: predictions at least half inside the ego-mask band (y >= 1830) are removed,
as the K10 protocol removes them from the decision. Per dataset and pipeline:
  AP, AP50              COCO-style (IoU 0.50:0.05:0.95, 101-point), mean over the three classes; nuScenes frames: every 4th
  corridor_AP / _recall GT in the ego corridor (in_corridor(lat_min, lat_max, 1.2 m), long_near <= 80 m); predictions matched
                        to other GT ignored; unmatched predictions are FP only if their own estimate is in that corridor
  rel_recall / rel_precision   GT with braking demand >= thr_decel matched (IoU >= 0.3, Hungarian, instr_eval.matched);
                        predictions with estimated demand >= thr_decel that match a relevant GT / all such predictions
  recall_r<bin>, recall_h<bin> recall at IoU >= 0.5 by GT range bin and by native box height (sc outputs)
  spurious_brake_per_1k predictions with estimated demand >= thr_decel matching no GT box of any class (IoU >= 0.3)
V8e (--corner): frames with a non-COCO object class in the ego corridor (raw annotations; nuScenes barrier / traffic cone /
construction vehicle, AV2 construction cone / barrel / bollard): G per pipeline on those frames (from results/k30/match and
k28 p3 instr) and class-agnostic recall (IoU >= 0.3) of those objects by any output box.
results/k30/tables/v7_detection.csv (, v8e_corner.csv)
usage: k30_v7.py [--datasets nusc,av2,ahd] [--src apdump] [--partial]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", B))
_k24 = importlib.util.spec_from_file_location("k24t", B / "scripts" / "A_method" / "k24_test.py")
K24 = importlib.util.module_from_spec(_k24); _k24.loader.exec_module(K24)
IE = K24.IE
from rap import planner as P                                                     # noqa: E402
from rap.mono import box_iou                                                     # noqa: E402

NU = {"nusc": 228, "av2": 150, "ahd": 24}
NWS = {"nusc": 1600, "av2": 2048, "ahd": 1920}
CLASSES = ("vehicle", "person", "cyclist")
RBINS = [(0, 30), (30, 50), (50, 80), (80, np.inf)]
HBINS = [(0, 16), (16, 32), (32, np.inf)]
MASK_Y0 = 1830.0


def compat(gc, pc):
    return str(pc) in IE.COMPAT.get(str(gc), {str(gc)})


def greedy(gt_b, gt_c, pr_b, pr_c, pr_s, thr):
    """COCO-style greedy matching by score: per prediction, the best unmatched class-compatible GT with IoU >= thr"""
    m_pr = -np.ones(len(pr_b), int)
    if not len(gt_b) or not len(pr_b):
        return m_pr
    iou = box_iou(pr_b, gt_b)
    used = np.zeros(len(gt_b), bool)
    for i in np.argsort(-pr_s, kind="stable"):
        best, bj = thr, -1
        for j in range(len(gt_b)):
            if not used[j] and compat(gt_c[j], pr_c[i]) and iou[i, j] >= best:
                best, bj = iou[i, j], j
        if bj >= 0:
            used[bj] = True; m_pr[i] = bj
    return m_pr


def ap101(tp, scores, n_gt):
    if n_gt == 0:
        return np.nan
    o = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.asarray(tp, float)[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    out = []
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(rec, r, side="left")
        out.append(prec[k] if k < len(prec) else 0.0)
    return float(np.mean(out))


def in_corr(geo, pp):
    z, lo, hi = (np.asarray(a, float) for a in geo[:3])
    return P.in_corridor(lo, hi, pp.corridor_half_w) & (z <= pp.max_range) & (z > 0)


def load_dumps(ds, src, partial):
    fs = sorted(glob.glob(str(REAL / "results" / "k30" / src / ds / "*.pkl")))
    if not fs or (len(fs) < NU[ds] and not partial):
        return None, len(fs)
    dets, rows = {}, []
    for f in fs:
        x = pickle.load(open(f, "rb"))
        u = Path(f).stem
        rows += x["rows"]
        for (fr, nm), o in x["dets"].items():
            dets[(u, fr, nm)] = o
    return (pd.DataFrame(rows), dets), len(fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="nusc,av2,ahd")
    ap.add_argument("--src", default="apdump")
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()
    pp = P.PlannerParams()
    out = []
    for ds in args.datasets.split(","):
        x, n = load_dumps(ds, args.src, args.partial)
        print(ds, "units", n, flush=True)
        if x is None:
            continue
        rows, dets = x
        units = set(rows.unit)
        pipes = sorted({k[2] for k in dets})
        nat = NWS[ds]
        acc = {p: {"apl": {t: {c: [] for c in CLASSES} for t in np.round(np.arange(0.5, 0.96, 0.05), 2)}, "ngt": {c: 0 for c in CLASSES},
                   "corr": {"tp": [], "s": [], "ngt": 0, "hit": 0}, "rel": [0, 0, 0, 0], "rb": {b: [0, 0] for b in RBINS},
                   "hb": {b: [0, 0] for b in HBINS}, "spur": 0, "frames": 0} for p in pipes}
        for unit, cam, gen in K24.units(ds):
            if unit not in units:
                continue
            for fr, path, v, g, cached in gen():
                if isinstance(cached, str):                      # Argoverse-HD frames without labels
                    continue
                if ds == "nusc" and fr % 4:
                    continue
                gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
                gtc = np.asarray(g["coarse"]).astype(str) if len(g) else np.zeros(0, str)
                gln = np.asarray(g["long_near"], float) if len(g) else np.zeros(0)
                gt_t = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc")) if len(g) else tuple(np.zeros(0) for _ in range(4))
                _, dem_gt = P.required_decel(*gt_t, v, pp) if len(g) else (0.0, np.zeros(0))
                rel = np.asarray(dem_gt, float) >= pp.thr_decel if len(g) else np.zeros(0, bool)
                gcorr = in_corr(gt_t, pp) if len(g) else np.zeros(0, bool)
                gh = gtb[:, 3] - gtb[:, 1] if len(g) else np.zeros(0)
                for p in pipes:
                    o = dets.get((unit, fr, p))
                    if o is None:
                        continue
                    A = acc[p]; A["frames"] += 1
                    sc = o["sc"]
                    out_ = o["ap"] if (o["ap"] is not None and ds != "nusc") else {k_: sc[k_] for k_ in ("xyxy", "conf", "coarse")}
                    pb, pc, ps = np.asarray(out_["xyxy"], float).reshape(-1, 4), np.asarray(out_["coarse"]).astype(str), np.asarray(out_["conf"], float)
                    if ds == "av2" and len(pb):
                        h = np.maximum(pb[:, 3] - pb[:, 1], 1e-6)
                        km = (pb[:, 3] - np.maximum(pb[:, 1], MASK_Y0)).clip(0, None) / h < 0.5
                        pb, pc, ps = pb[km], pc[km], ps[km]
                    # whole-image AP per class and IoU threshold
                    for c in CLASSES:
                        gi = np.flatnonzero(np.array([compat(c, q) for q in gtc], bool)) if len(gtc) else np.zeros(0, int)
                        gi = np.flatnonzero(gtc == c) if len(gtc) else np.zeros(0, int)
                        pi = np.flatnonzero(np.array([compat(c, q) for q in pc], bool)) if len(pc) else np.zeros(0, int)
                        A["ngt"][c] += len(gi)
                        for t in A["apl"]:
                            m = greedy(gtb[gi], gtc[gi], pb[pi], pc[pi], ps[pi], t)
                            A["apl"][t][c] += list(zip((m >= 0).astype(float), ps[pi]))
                    # corridor AP (IoU 0.5) / recall
                    m = greedy(gtb, gtc, pb, pc, ps, 0.5)
                    # estimated geometry of the AP outputs: sc carries the deployed geometry; for ap outputs use sc's geometry by box identity
                    sgeo = sc["geo"]
                    sb = np.asarray(sc["xyxy"], float).reshape(-1, 4)
                    est_in = np.zeros(len(pb), bool)
                    if len(pb) and len(sb):
                        ii = box_iou(pb, sb)
                        j = ii.argmax(1)
                        ok = ii[np.arange(len(pb)), j] > 0.99
                        ic = in_corr(sgeo, pp)
                        est_in = ok & ic[j]
                    A["corr"]["ngt"] += int(gcorr.sum())
                    for i in range(len(pb)):
                        if m[i] >= 0:
                            if gcorr[m[i]]:
                                A["corr"]["tp"].append(1.0); A["corr"]["s"].append(ps[i])
                        elif est_in[i]:
                            A["corr"]["tp"].append(0.0); A["corr"]["s"].append(ps[i])
                    A["corr"]["hit"] += int(sum(1 for i in range(len(pb)) if m[i] >= 0 and gcorr[m[i]]))
                    # decision relevance / spurious braking, sc outputs with the deployed geometry
                    scc = np.asarray(sc["coarse"]).astype(str)
                    _, dem = P.required_decel(*sgeo, v, pp) if len(sb) else (0.0, np.zeros(0))
                    dem = np.asarray(dem, float)
                    if rel.any():
                        mm = IE.matched(gtb[rel], gtc[rel], sb, scc)
                        A["rel"][0] += int((mm >= 0).sum()); A["rel"][1] += int(rel.sum())
                    hi_d = np.flatnonzero(dem >= pp.thr_decel) if len(dem) else np.zeros(0, int)
                    if len(hi_d):
                        mm_all = IE.matched(sb[hi_d], scc[hi_d], gtb, gtc) if len(gtb) else -np.ones(len(hi_d), int)
                        # a predicted relevant object is correct if it matches a relevant GT
                        mm_rel = IE.matched(sb[hi_d], scc[hi_d], gtb[rel], gtc[rel]) if rel.any() else -np.ones(len(hi_d), int)
                        A["rel"][2] += int((mm_rel >= 0).sum()); A["rel"][3] += len(hi_d)
                        if len(gtb):
                            ia = box_iou(sb[hi_d], gtb).max(1)
                            A["spur"] += int((ia < 0.3).sum())
                        else:
                            A["spur"] += len(hi_d)
                    # recall by range bin and box height (sc outputs, IoU 0.5)
                    ms = greedy(gtb, gtc, sb, scc, np.asarray(sc["conf"], float), 0.5)
                    hitg = np.zeros(len(gtb), bool)
                    hitg[ms[ms >= 0]] = True
                    for b in RBINS:
                        k_ = (gln >= b[0]) & (gln < b[1])
                        A["rb"][b][0] += int(hitg[k_].sum()); A["rb"][b][1] += int(k_.sum())
                    for b in HBINS:
                        k_ = (gh >= b[0]) & (gh < b[1])
                        A["hb"][b][0] += int(hitg[k_].sum()); A["hb"][b][1] += int(k_.sum())
        for p, A in acc.items():
            aps = {t: np.nanmean([ap101([q[0] for q in A["apl"][t][c]], [q[1] for q in A["apl"][t][c]], A["ngt"][c]) for c in CLASSES]) for t in A["apl"]}
            r = {"dataset": ds, "pipeline": p, "frames": A["frames"], "output": "sc (conf >= 0.25)" if ds == "nusc" else "ap (conf >= 0.1)",
                 "AP": float(np.nanmean(list(aps.values()))), "AP50": float(aps[0.5]),
                 "corridor_AP50": ap101(A["corr"]["tp"], A["corr"]["s"], A["corr"]["ngt"]),
                 "corridor_recall50": A["corr"]["hit"] / A["corr"]["ngt"] if A["corr"]["ngt"] else np.nan,
                 "rel_recall": A["rel"][0] / A["rel"][1] if A["rel"][1] else np.nan, "rel_precision": A["rel"][2] / max(A["rel"][3], 1) if A["rel"][3] else np.nan,
                 "spurious_brake_per_1k": A["spur"] / max(A["frames"], 1) * 1000}
            for b in RBINS:
                r[f"recall_r{b[0]}-{b[1]}"] = A["rb"][b][0] / A["rb"][b][1] if A["rb"][b][1] else np.nan
            for b in HBINS:
                r[f"recall_h{b[0]}-{b[1]}"] = A["hb"][b][0] / A["hb"][b][1] if A["hb"][b][1] else np.nan
            out.append(r)
        print(pd.DataFrame([q for q in out if q["dataset"] == ds]).round(3).to_string(index=False), flush=True)
    od = REAL / "results" / "k30" / "tables"
    od.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(od / f"v7_detection{'' if args.src == 'apdump' else '_' + args.src}.csv", index=False)
    print("K30 V7 DONE", flush=True)


if __name__ == "__main__":
    main()
