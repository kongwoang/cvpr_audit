#!/usr/bin/env python
"""Detection metrics of the pipelines whose merged scenes were dumped by nusc_ablate.py --dump (v3, v3 + refiner,
v4-lite, ...), on the same new-TEST frames and with the same definitions as det_metrics.py (every --step-th keyframe,
conf >= 0.25, class-aware IoU 0.5; AP50, recall all / small / decision-relevant, FP per frame). CPU only.
usage: det_metrics_dump.py <scenes__*.pkl> [<more dumps>] [--step 4]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
sys.path.insert(0, str(B / "scripts" / "A_method"))
from det_metrics import CLASSES, THR, ap_allpoint, keep, match                   # noqa: E402
from rap import planner as P                                                     # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                    # noqa: E402
from rap.paths import NUSCENES_TRAINVAL                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--step", type=int, default=4)
    args = ap.parse_args()
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    ad = make_adapter(db)
    pp = P.PlannerParams()
    entries = {}
    for p in args.dumps:
        tag = Path(p).stem.replace("scenes__new_test", "") or "dump"
        for e in pd.read_pickle(p):
            for k, d in e.items():
                if not isinstance(d, dict):
                    continue
                name = k if k in ("cheap", "u960") else f"{k}{tag}"          # cheap / u960 are the same in every dump
                entries.setdefault(name, {})[(e["seq"], int(e["frame"]))] = d
    names = sorted(entries)
    acc = {m: {"conf": {c: [] for c in CLASSES}, "tp": {c: [] for c in CLASSES}, "hit_all": [], "hit_small": [],
               "hit_rel": [], "fp": 0, "rel_det": 0, "rel_tp": 0} for m in names}
    ngt = {c: 0 for c in CLASSES}
    nfr = 0
    keys = sorted(next(iter(entries.values())))
    by_seq = {}
    for s, f in keys:
        by_seq.setdefault(s, []).append(f)
    for s, frames in by_seq.items():
        geom, speeds = ad.geometry(s), ad.speeds(s)
        for i, fr in enumerate(sorted(frames)):
            if i % args.step:
                continue
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= 10]
            gx = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
            gc = np.asarray(g["coarse"]).astype(str) if len(g) else np.zeros(0, "<U8")
            small = (gx[:, 3] - gx[:, 1]) < 32
            rel = np.array([P.discrete_action(P.required_decel(g["long_near"][j:j + 1], g["lat_min"][j:j + 1],
                                                               g["lat_max"][j:j + 1], g["ttc"][j:j + 1], v, pp)[0], pp) > 0
                            for j in range(len(g))], bool)
            for c in CLASSES:
                ngt[c] += int((gc == c).sum())
            nfr += 1
            for m in names:
                d = keep(entries[m][(s, fr)])
                tp, hit = match(d, gx, gc)
                A = acc[m]
                for c in CLASSES:
                    kk = d["coarse"] == c
                    A["conf"][c].append(d["conf"][kk]); A["tp"][c].append(tp[kk])
                A["hit_all"].append(hit); A["hit_small"].append(hit[small]); A["hit_rel"].append(hit[rel])
                A["fp"] += int((~tp).sum())
                # relevant precision: detections whose own lifted state demands braking, and how many are real objects
                raw = entries[m][(s, fr)]
                kk = np.asarray(raw["conf"], float) >= THR
                mono = [np.asarray(a, float)[kk] for a in raw["mono"]]
                own = np.array([P.discrete_action(P.required_decel(*(a[j:j + 1] for a in mono), v, pp)[0], pp) > 0
                                for j in range(len(d["conf"]))], bool)
                A["rel_det"] += int(own.sum()); A["rel_tp"] += int((own & tp).sum())
    rows = []
    for m in names:
        A = acc[m]
        aps = [ap_allpoint(np.concatenate(A["conf"][c]), np.concatenate(A["tp"][c]), ngt[c]) for c in CLASSES]
        cat = lambda k: np.concatenate(A[k]) if A[k] else np.zeros(0)                    # noqa: E731
        rows.append({"method": m, "AP50": np.nanmean(aps), **{f"AP50_{c}": a for c, a in zip(CLASSES, aps)},
                     "R_all": cat("hit_all").mean(), "R_small": cat("hit_small").mean(), "R_rel": cat("hit_rel").mean(),
                     "n_rel": len(cat("hit_rel")), "FP_per_frame": A["fp"] / nfr,
                     "P_rel": A["rel_tp"] / max(A["rel_det"], 1), "rel_FP_per_1k_frames": 1000 * (A["rel_det"] - A["rel_tp"]) / nfr})
    res = pd.DataFrame(rows)
    out = B / "results" / "det_metrics"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"newtest_step{args.step}_dumps.csv", index=False)
    print(f"{nfr} frames (threshold {THR}), GT per class {ngt}")
    print(res.round(3).to_string(index=False))
    print("DET METRICS DUMP DONE")


if __name__ == "__main__":
    main()
