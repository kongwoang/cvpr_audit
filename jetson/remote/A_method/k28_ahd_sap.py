#!/usr/bin/env python
"""K28 PHASE 3 (post-hoc) — streaming AP of v6-S on Argoverse-HD val: ahd_sap.py's simulation (Li et al., ECCV 2020; single
sequential processor, greedy newest-frame policy, the newest finished output shown at every annotated frame) with
  detections: the K28 AP dump of the phase-2 configuration (results/k28/ahd_test_v6apdump/<seq>.pkl, method v6S, the K7
              AP definition for crop pipelines: conf >= 0.1, draft boxes outside the crops + crop boxes);
  latency:    per-frame Jetson ms of v6-S (61_k28_v6_latency.py --dataset ahd --config v6S; ms_ours = with refiner,
              ms_ours_noref) from <latency.csv> (columns unit, frame, image_id, ms_ours, ms_ours_noref, ms_cheap).
The whole-frame / v4-lite rows of K7 (results/argoverse_hd/summary_sap.csv, K7 detections + K7 Jetson timings) are copied
next to them for the comparison. usage: k28_ahd_sap.py <latency.csv>  ->  results/k28/p3/tables/sap_ahd_v6.csv
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
REAL = Path(os.environ.get("B_REAL", str(B)))
SAP = runpy.run_path(str(B / "scripts" / "A_method" / "ahd_sap.py"), run_name="sap_lib")
HDm = runpy.run_path(str(B / "scripts" / "A_method" / "argoverse_hd_eval.py"), run_name="ahd_lib")


def main():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    lat = pd.read_csv(sys.argv[1])
    d = json.loads((HDm["HD"] / "val.json").read_text())
    ims = pd.DataFrame(d["images"])
    ims["t"] = ims.name.str.rsplit("_", n=1).str[1].str.split(".").str[0].astype(np.int64) * 1e-9
    gt = {"images": d["images"], "categories": [{"id": i, "name": n} for i, n in enumerate(("vehicle", "person", "two-wheeler"))],
          "annotations": [{**a, "category_id": HDm["GROUP"][a["category_id"]]} for a in d["annotations"] if a["category_id"] in HDm["GROUP"]]}
    tmp = REAL / "results" / "k28" / "p3" / "tables" / "ahd_gt_groups_sap.json"
    tmp.write_text(json.dumps(gt))
    cg = COCO(str(tmp))
    dets, fr_of = {}, {}
    for f in sorted(glob.glob(str(REAL / "results" / "k28" / "ahd_test_v6apdump" / "*.pkl"))):
        z = pickle.load(open(f, "rb"))
        ids = z["image_ids"]
        for (fr, m), o in z["dets"].items():
            if m == "v6S" and o["ap"] is not None:
                dets[int(ids[fr])] = o["ap"]
        for fr, iid in enumerate(ids):
            fr_of[int(iid)] = fr
    lat = lat.set_index("image_id")
    res = []
    for nm, col in (("v6-S + refiner v3a (Jetson timing with refiner)", "ms_ours"), ("v6-S, no-refiner timing", "ms_ours_noref")):
        out, lags, n_proc, cover = [], [], 0, []
        for sid, g in ims.groupby("sid"):
            g = g.sort_values("fid")
            t = g.t.to_numpy() - g.t.to_numpy()[0]
            L = lat[col].reindex(g.id.to_numpy()).to_numpy(float) / 1e3
            cover.append(np.isfinite(L).mean())
            L[~np.isfinite(L)] = np.nanmedian(L)
            show, n_ = SAP["simulate"](t, L)
            n_proc += n_
            for j, (iid, s) in enumerate(zip(g.id.to_numpy(), show)):
                if s < 0:
                    continue
                lags.append(t[j] - t[s])
                dd = dets.get(int(g.id.to_numpy()[s]))
                if dd is None:
                    continue
                for b, c, k in zip(dd["xyxy"], dd["conf"], dd["coarse"]):
                    out.append({"image_id": int(iid), "category_id": HDm["CGROUP"][str(k)], "score": float(c),
                                "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
        E = COCOeval(cg, cg.loadRes(out), "bbox")
        E.params.imgIds = sorted(int(i) for i in ims.id)
        E.evaluate(); E.accumulate(); E.summarize()
        lags = np.array(lags)
        res.append({"pipeline": nm, "source": "K28 (v6-S AP dump + Jetson 61)", "sAP": E.stats[0], "sAP50": E.stats[1],
                    "sAP_small": E.stats[3], "mean_latency_ms": float(lat[col].mean()), "p95_latency_ms": float(lat[col].quantile(.95)),
                    "frames_processed_share": n_proc / len(ims), "mean_staleness_ms": 1e3 * lags.mean(),
                    "p95_staleness_ms": 1e3 * np.percentile(lags, 95), "latency_coverage_min": float(min(cover))})
    k7 = pd.read_csv(REAL / "results" / "argoverse_hd" / "summary_sap.csv").assign(source="K7 (AV2 configuration v4-lite + v2; whole frame)")
    r = pd.concat([pd.DataFrame(res), k7], ignore_index=True, sort=False)
    r.to_csv(REAL / "results" / "k28" / "p3" / "tables" / "sap_ahd_v6.csv", index=False)
    pd.set_option("display.width", 220)
    print(r.round(4).to_string(index=False))
    print("K28 AHD SAP DONE")


if __name__ == "__main__":
    main()
