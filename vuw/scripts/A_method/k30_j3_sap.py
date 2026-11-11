#!/usr/bin/env python
"""K30 J3 (post-hoc): streaming AP on Argoverse-HD val with the K29 fast path. k28_ahd_sap.py's simulation (ahd_sap.py:
Li et al., ECCV 2020; single sequential processor, greedy newest-frame policy, the newest finished output shown at every
annotated frame) per pipeline with
  latency     per-frame Jetson ms of scripts/67_k30_jetson.py --dataset ahd (fast path trt32; results/k30/j3_latency.csv)
  detections  K30 apdump (results/k30/apdump/ahd: cheap, wf960 / wf1280 / wf1920 + C1, v6S; the K7 AP definition, conf >= 0.1)
              -- the <= 2-call bound has no dump; its row uses v6-S's detections with the bounded loop's latency
  metrics     sAP / sAP50 / sAP_small (COCO, AHD's 3 road-user groups, as K28) and the ego-corridor streaming AP50
              (k30_v7 rule: GT of the benchmark's 3D labels in the corridor <= 80 m; a shown prediction is TP when it
              matches such a GT at IoU >= 0.5, FP when unmatched and its own deployed estimate lies in the corridor)
K28's slow-C1 v6-S rows (results/k28/p3/tables/sap_ahd_v6.csv) are appended for comparison.
usage: k30_j3_sap.py <j3_latency.csv>  ->  results/k30/tables/j3_sap.csv
"""
from __future__ import annotations

import glob
import importlib.util
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
_v = importlib.util.spec_from_file_location("k30v7", B / "scripts" / "A_method" / "k30_v7.py")
V7 = importlib.util.module_from_spec(_v); _v.loader.exec_module(V7)
PIPES = {"cheap (u320)": ("ms_cheap", "cheap"),
         "u960 + C1 + v3a": ("ms_u960+c1comp@trt32", "wf960+C1"),
         "u1280 + C1 + v3a": ("ms_u1280+c1comp@trt32", "wf1280+C1"),
         "native + C1 + v3a": ("ms_native+c1comp@trt32", "wf1920+C1"),
         "v6-S (fast path)": ("ms_ours@trt32", "v6S"),
         "v6-S <= 2 calls (latency; v6-S detections)": ("ms_ours_b2@trt32", "v6S")}


def main():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    lat = pd.read_csv(sys.argv[1]).drop_duplicates("image_id").set_index("image_id")
    d = json.loads((HDm["HD"] / "val.json").read_text())
    ims = pd.DataFrame(d["images"])
    ims["t"] = ims.name.str.rsplit("_", n=1).str[1].str.split(".").str[0].astype(np.int64) * 1e-9
    tmp = REAL / "results" / "k28" / "p3" / "tables" / "ahd_gt_groups_sap.json"
    cg = COCO(str(tmp))
    id2uf = {}
    for f in sorted(glob.glob(str(REAL / "results" / "k28" / "ahd_test_v6apdump" / "*.pkl"))):
        for fr, iid in enumerate(pickle.load(open(f, "rb"))["image_ids"]):
            id2uf[int(iid)] = (Path(f).stem, fr)
    dets = {}
    for f in sorted(glob.glob(str(REAL / "results" / "k30" / "apdump" / "ahd" / "*.pkl"))):
        for (fr, nm), o in pickle.load(open(f, "rb"))["dets"].items():
            dets[(Path(f).stem, int(fr), nm)] = o
    # corridor GT (benchmark 3D labels) per labelled frame
    pp = V7.P.PlannerParams()
    corr_gt = {}
    for unit, cam, gen in V7.K24.units("ahd"):
        for fr, path, v, g, cached in gen():
            if isinstance(cached, str) or not len(g):
                continue
            gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
            gt_t = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc"))
            corr_gt[(unit, fr)] = (gtb, np.asarray(g["coarse"]).astype(str), V7.in_corr(gt_t, pp))
    res = []
    for lab, (col, key) in PIPES.items():
        if col not in lat:
            print("missing latency column", col); continue
        out, lags, n_proc, cover = [], [], 0, []
        ctp, cs, cngt = [], [], 0
        for sid, g in ims.groupby("sid"):
            g = g.sort_values("fid")
            t = g.t.to_numpy() - g.t.to_numpy()[0]
            L = lat[col].reindex(g.id.to_numpy()).to_numpy(float) / 1e3
            cover.append(np.isfinite(L).mean())
            L[~np.isfinite(L)] = np.nanmedian(L)
            show, n_ = SAP["simulate"](t, L)
            n_proc += n_
            ids = g.id.to_numpy()
            for j, (iid, s) in enumerate(zip(ids, show)):
                uf_j = id2uf.get(int(iid))
                cgt = corr_gt.get(uf_j) if uf_j else None
                if cgt is not None:
                    cngt += int(cgt[2].sum())
                if s < 0:
                    continue
                lags.append(t[j] - t[s])
                uf_s = id2uf.get(int(ids[s]))
                o = dets.get((uf_s[0], uf_s[1], key)) if uf_s else None
                if o is None:
                    continue
                dd = o["ap"] if o["ap"] is not None else o["sc"]
                pb = np.asarray(dd["xyxy"], float).reshape(-1, 4)
                pc, ps = np.asarray(dd["coarse"]).astype(str), np.asarray(dd["conf"], float)
                for b, c, k in zip(pb, ps, pc):
                    out.append({"image_id": int(iid), "category_id": HDm["CGROUP"][str(k)], "score": float(c),
                                "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
                if cgt is not None and len(pb):                  # corridor streaming AP50 (k30_v7 rule)
                    gtb, gtc, gcorr = cgt
                    m = V7.greedy(gtb, gtc, pb, pc, ps, 0.5)
                    sb = np.asarray(o["sc"]["xyxy"], float).reshape(-1, 4)
                    est_in = np.zeros(len(pb), bool)
                    if len(sb):
                        ii = V7.box_iou(pb, sb); jj = ii.argmax(1)
                        est_in = (ii[np.arange(len(pb)), jj] > 0.99) & V7.in_corr(o["sc"]["geo"], pp)[jj]
                    for i in range(len(pb)):
                        if m[i] >= 0:
                            if gcorr[m[i]]:
                                ctp.append(1.0); cs.append(ps[i])
                        elif est_in[i]:
                            ctp.append(0.0); cs.append(ps[i])
        E = COCOeval(cg, cg.loadRes(out), "bbox")
        E.params.imgIds = sorted(int(i) for i in ims.id)
        E.evaluate(); E.accumulate(); E.summarize()
        lags = np.array(lags)
        res.append({"pipeline": lab, "source": "K30 J3 (fast path trt32, 67_k30_jetson.py)", "sAP": E.stats[0], "sAP50": E.stats[1],
                    "sAP_small": E.stats[3], "corridor_sAP50": V7.ap101(ctp, cs, cngt), "corridor_gt": cngt,
                    "mean_latency_ms": float(lat[col].mean()), "p95_latency_ms": float(lat[col].quantile(.95)),
                    "frames_processed_share": n_proc / len(ims), "mean_staleness_ms": 1e3 * lags.mean(),
                    "p95_staleness_ms": 1e3 * np.percentile(lags, 95), "latency_coverage_min": float(min(cover))})
    k28 = REAL / "results" / "k28" / "p3" / "tables" / "sap_ahd_v6.csv"
    r = pd.concat([pd.DataFrame(res), pd.read_csv(k28)] if k28.exists() else [pd.DataFrame(res)], ignore_index=True, sort=False)
    r.to_csv(REAL / "results" / "k30" / "tables" / "j3_sap.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.round(4).to_string(index=False))
    print("K30 J3 SAP DONE")


if __name__ == "__main__":
    main()
