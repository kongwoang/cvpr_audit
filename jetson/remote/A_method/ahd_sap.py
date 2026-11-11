#!/usr/bin/env python
"""K7 streaming AP (sAP, Li et al., "Towards Streaming Perception", ECCV 2020) on Argoverse-HD val.

Inputs: the offline detections of every pipeline (results/argoverse_hd/seq*.pkl, from argoverse_hd_eval.py --stage run,
unchanged) and the per-frame Jetson AGX Xavier latencies of the same pipelines on the same frames
(results/argoverse_hd/ahd_latency_per_frame.csv, from scripts/56_ahd_latency.py on the Jetson, TensorRT FP16).
Simulation per sequence and pipeline (single sequential processor, greedy policy): frames arrive at their camera
timestamps; whenever the processor is idle it takes the most recent frame that has arrived (frames that arrived while it
was busy are skipped); the output of frame i is available at start_i + latency_i. At every annotated frame j the
prediction is the most recent output with finish time <= t_j (empty before the first output). sAP = COCO AP of these
streamed predictions against frame j's annotations (same three road-user groups as the offline AP).
usage: ahd_sap.py <latency_per_frame.csv>  ->  results/argoverse_hd/summary_sap.csv (+ streaming lag stats)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
HD = B / "datasets" / "argoverse_hd" / "Argoverse-HD" / "annotations"
OUT = B / "results" / "argoverse_hd"
GROUP = {0: 1, 1: 2, 2: 0, 3: 2, 4: 0, 5: 0}
CGROUP = {"vehicle": 0, "person": 1, "cyclist": 2}
UNIF = (640, 960, 1280, 1600, 1920)


def simulate(t, lat):
    """t: arrival times (s) of the frames of one sequence, lat: latency (s) of each frame.
    returns for each frame j the index of the frame whose output is shown at t_j (-1 = none), and finished pairs."""
    n = len(t)
    done = []                                        # (finish time, frame index)
    tau, last = t[0], -1
    while True:
        cand = np.flatnonzero(t <= tau + 1e-9)
        i = cand[-1] if len(cand) else -1
        if i <= last:                                # nothing new: wait for the next arrival
            if last + 1 >= n:
                break
            tau = t[last + 1]
            continue
        f = tau + lat[i]
        done.append((f, i)); last = i; tau = f
        if tau > t[-1]:
            break
    fin = np.array([d[0] for d in done]); idx = np.array([d[1] for d in done])
    show = np.full(n, -1)
    for j in range(n):
        k = np.searchsorted(fin, t[j], side="right") - 1
        show[j] = idx[k] if k >= 0 else -1
    return show, len(done)


def main():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    lat = pd.read_csv(sys.argv[1])
    d = json.loads((HD / "val.json").read_text())
    ims = pd.DataFrame(d["images"])
    ims["t"] = ims.name.str.rsplit("_", n=1).str[1].str.split(".").str[0].astype(np.int64) * 1e-9
    cg = COCO(str(OUT / "gt_groups.json"))
    dets = {}
    for p in sorted(OUT.glob("seq*.pkl")):
        dets.update(pickle.load(open(p, "rb"))["dets"])
    pipes = {"cheap": "cheap", **{f"u{s}": f"u{s}" for s in UNIF}, "ours": "ours", "ours_noref_timing": "ours"}
    res, lagrows = [], []
    for nm, dname in pipes.items():
        col = "ms_ours_noref" if nm == "ours_noref_timing" else f"ms_{nm}"
        out, lags, n_proc = [], [], 0
        for sid, g in ims.groupby("sid"):
            g = g.sort_values("fid")
            l_ = lat[lat.sid == sid].set_index("frame")[col]
            # frame index in the latency table = position in the sequence (fid order); images missing there are skipped
            t = g.t.to_numpy() - g.t.to_numpy()[0]
            pos = np.arange(len(g))
            ok = np.isin(pos, l_.index.to_numpy())
            assert ok.mean() > 0.99, (sid, ok.mean())
            L = np.array([l_.get(p_, np.nan) for p_ in pos]) / 1e3
            L[np.isnan(L)] = np.nanmedian(L)
            show, n_ = simulate(t, L)
            n_proc += n_
            for j, (iid, s) in enumerate(zip(g.id.to_numpy(), show)):
                if s < 0:
                    continue
                lags.append(t[j] - t[s])
                dd = dets.get((int(g.id.to_numpy()[s]), dname))
                if dd is None:
                    continue
                for b, c, k in zip(dd["xyxy"], dd["conf"], dd["coarse"]):
                    out.append({"image_id": int(iid), "category_id": CGROUP[str(k)], "score": float(c),
                                "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
        E = COCOeval(cg, cg.loadRes(out), "bbox")
        E.params.imgIds = sorted(int(i) for i in ims.id)
        E.evaluate(); E.accumulate(); E.summarize()
        lags = np.array(lags)
        res.append({"pipeline": nm, "sAP": E.stats[0], "sAP50": E.stats[1], "sAP_small": E.stats[3],
                    "mean_latency_ms": lat[col].mean(), "frames_processed_share": n_proc / len(ims),
                    "mean_staleness_ms": 1e3 * lags.mean(), "p95_staleness_ms": 1e3 * np.percentile(lags, 95)})
    r = pd.DataFrame(res)
    r.to_csv(OUT / "summary_sap.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.round(4).to_string(index=False))
    print("AHD SAP DONE")


if __name__ == "__main__":
    main()
