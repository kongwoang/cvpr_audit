#!/usr/bin/env python
"""K40 (private helper): detection records of tab:det from the k30_eval2.py apdump outputs, with the matching of
k30_v7.py (same functions, same frames, same outputs) -> <out>/det_predictions_<dataset>.parquet (pipeline, unit, frame,
cls, score, tp50: whole-image predictions per evaluated class and their greedy IoU >= 0.5 match) and
<out>/det_frames_<dataset>.parquet (pipeline, unit, frame, ngt_<cls>, rel_hit / rel_gt: relevant GT matched / all,
relpred_hit / relpred_n: predictions with estimated demand >= thr_decel matching a relevant GT / all, spur: such
predictions matching no GT box, IoU < 0.3). Pipelines of the table only.
usage: k40_export_det.py <out_dir> [datasets]"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
_s = importlib.util.spec_from_file_location("k30v7", B / "scripts/A_method/k30_v7.py")
V = importlib.util.module_from_spec(_s); _s.loader.exec_module(V)
P, K24, IE, box_iou = V.P, V.K24, V.IE, V.box_iou
NAME = {"nusc": "nuscenes", "av2": "av2", "ahd": "argoverse_hd"}
PIPES = {"nusc": ["cheap", "wf960+C1", "wf1600+C1", "tsp8@S+C1", "v6S"], "av2": ["cheap", "wf960+C1", "wf2048+C1", "tsp8@S+C1", "v6S"],
         "ahd": ["cheap", "wf960+C1", "wf1920+C1", "tsp8@S+C1", "v6S"]}


def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    pp = P.PlannerParams()
    for ds in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["nusc", "av2", "ahd"]):
        (rows, dets), n = V.load_dumps(ds, "apdump", False)
        units = set(rows.unit)
        preds, frames = [], []
        for unit, cam, gen in K24.units(ds):
            if unit not in units:
                continue
            for fr, path, v, g, cached in gen():
                if isinstance(cached, str) or (ds == "nusc" and fr % 4):
                    continue
                gtb = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
                gtc = np.asarray(g["coarse"]).astype(str) if len(g) else np.zeros(0, str)
                gt_t = tuple(np.asarray(g[k], float) for k in ("long_near", "lat_min", "lat_max", "ttc")) if len(g) else tuple(np.zeros(0) for _ in range(4))
                _, dem_gt = P.required_decel(*gt_t, v, pp) if len(g) else (0.0, np.zeros(0))
                rel = np.asarray(dem_gt, float) >= pp.thr_decel if len(g) else np.zeros(0, bool)
                for p in PIPES[ds]:
                    o = dets.get((unit, fr, p))
                    if o is None:
                        continue
                    sc = o["sc"]
                    out_ = o["ap"] if (o["ap"] is not None and ds != "nusc") else {k_: sc[k_] for k_ in ("xyxy", "conf", "coarse")}
                    pb, pc, ps = np.asarray(out_["xyxy"], float).reshape(-1, 4), np.asarray(out_["coarse"]).astype(str), np.asarray(out_["conf"], float)
                    if ds == "av2" and len(pb):
                        h = np.maximum(pb[:, 3] - pb[:, 1], 1e-6)
                        km = (pb[:, 3] - np.maximum(pb[:, 1], V.MASK_Y0)).clip(0, None) / h < 0.5
                        pb, pc, ps = pb[km], pc[km], ps[km]
                    f = {"pipeline": p, "unit": str(unit), "frame": int(fr)}
                    for c in V.CLASSES:
                        gi = np.flatnonzero(gtc == c) if len(gtc) else np.zeros(0, int)
                        pi = np.flatnonzero(np.array([V.compat(c, q) for q in pc], bool)) if len(pc) else np.zeros(0, int)
                        f[f"ngt_{c}"] = len(gi)
                        m = V.greedy(gtb[gi], gtc[gi], pb[pi], pc[pi], ps[pi], 0.5)
                        preds += [{"pipeline": p, "unit": str(unit), "frame": int(fr), "cls": c, "score": float(s_), "tp50": bool(t_ >= 0)}
                                  for s_, t_ in zip(ps[pi], m)]
                    sgeo, sb = sc["geo"], np.asarray(sc["xyxy"], float).reshape(-1, 4)
                    scc = np.asarray(sc["coarse"]).astype(str)
                    _, dem = P.required_decel(*sgeo, v, pp) if len(sb) else (0.0, np.zeros(0))
                    dem = np.asarray(dem, float)
                    f.update(rel_hit=0, rel_gt=0, relpred_hit=0, relpred_n=0, spur=0)
                    if rel.any():
                        mm = IE.matched(gtb[rel], gtc[rel], sb, scc)
                        f["rel_hit"], f["rel_gt"] = int((mm >= 0).sum()), int(rel.sum())
                    hi_d = np.flatnonzero(dem >= pp.thr_decel) if len(dem) else np.zeros(0, int)
                    if len(hi_d):
                        mm_rel = IE.matched(sb[hi_d], scc[hi_d], gtb[rel], gtc[rel]) if rel.any() else -np.ones(len(hi_d), int)
                        f["relpred_hit"], f["relpred_n"] = int((mm_rel >= 0).sum()), len(hi_d)
                        f["spur"] = int((box_iou(sb[hi_d], gtb).max(1) < 0.3).sum()) if len(gtb) else len(hi_d)
                    frames.append(f)
        pd.DataFrame(preds).to_parquet(od / f"det_predictions_{NAME[ds]}.parquet", index=False, compression="zstd")
        F = pd.DataFrame(frames)
        F.to_parquet(od / f"det_frames_{NAME[ds]}.parquet", index=False, compression="zstd")
        print(ds, len(preds), "predictions", F.groupby("pipeline").size().to_dict(), flush=True)


if __name__ == "__main__":
    main()
