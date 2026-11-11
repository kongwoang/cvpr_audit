#!/usr/bin/env python
"""Check the proved speculative loop (support + field crops, iterated) under exact locality: KITTI, cached full-640
detections stand in for running the full model on a crop.  Decision agreement with the full mode, value, pixels."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import KittiAdapter
from rap.paths import CACHE
from rap.scoring import topk_expect
from vtf import runmeta, speculative as SP

v_close = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
near = len(sys.argv) > 2 and sys.argv[2] == "near"
r_grid = np.r_[[7.5, 10.0] if near else [], SP.R_GRID]
pp, cp = P.PlannerParams(), P.CostParams()
dv = pd.read_csv(runmeta.RAP_REPO / "results/final/benchmark_decision_values.csv.gz", dtype={"seq": str, "frame": str})
dv = dv[(dv.track == "KITTI") & (dv.geometry == "mono") & (dv.system == "brake")].copy(); dv["frame"] = dv.frame.astype(int)
rows = []
for s in sorted(set(dv.seq)):
    geom, speeds = KittiAdapter.geometry(s), KittiAdapter.speeds(s); cam = Camera("KITTI", s)
    c = DetCache(Path(CACHE) / "det" / "cheap_320" / f"{s}.npz"); f = DetCache(Path(CACHE) / "det" / "full_640" / f"{s}.npz")
    prev_act = None
    for i, fr in enumerate(c.frames):
        fr = int(fr); v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
        g = geom[geom["frame"] == fr]; g = g[(g["y2"] - g["y1"]) >= 10]
        a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
        sc = c.scalars(i); W, H = float(sc["img_w"]), float(sc["img_h"])
        def pack(cache, idx):
            d, gg = cache.det(idx), cache.geo(idx); k = d["conf"] >= 0.25
            return {"xyxy": d["xyxy"][k].astype(float), "conf": d["conf"][k].astype(float), "coarse": d["coarse"][k],
                    "mono": tuple(np.asarray(gg[x][k], float) for x in ("z", "lat_min", "lat_max", "ttc"))}
        ch, fu = pack(c, i), pack(f, f.index[fr])
        def detect_crop(crop):
            m = SP.centres_in(fu["xyxy"], [crop]); return {"xyxy": fu["xyxy"][m], "conf": fu["conf"][m], "coarse": fu["coarse"][m]}
        # lift: the proxy reuses the full detections' own lifted geometry (matched by box identity)
        def lift(full):
            idx = [int(np.flatnonzero((fu["xyxy"] == b).all(1))[0]) for b in full["xyxy"]]
            return tuple(a[idx] for a in fu["mono"])
        scene, crops, rounds = SP.run(ch, v, cam, W, H, detect_crop, lift, pp, v_close=v_close, r_grid=r_grid)
        act = lambda t: P.discrete_action(P.required_decel(*t, v, pp)[0], pp)
        cost = lambda a: P.decision_cost(a, a_gt, prev_act, pp, cp)["J"]
        a_c, a_f, a_s = act(ch["mono"]), act(fu["mono"]), act(scene["mono"])
        rows.append({"seq": s, "frame": fr, "J_c": cost(a_c), "J_f": cost(a_f), "J_s": cost(a_s), "agree": int(a_s == a_f),
                     "flip": int(a_c != a_f), "rounds": rounds, "px": SP.union_area(crops, W, H) if hasattr(SP, "union_area") else sum((cc[2]-cc[0])*(cc[3]-cc[1]) for cc in crops) / (W * H)})
        prev_act = a_c
d = pd.DataFrame(rows).merge(dv[["seq", "frame", "split", "V", "J_cheap"]], on=["seq", "frame"])
assert np.allclose(d.J_c - d.J_f, d.V, atol=1e-9)
for sp in ("all", "test"):
    q = d if sp == "all" else d[d.split == "test"]; L = q.J_cheap.sum()
    print(f"v_close {v_close} near {near} {sp}: agree {q.agree.mean():.4f} (on flips {q.agree[q.flip == 1].mean():.4f}), G_full {q.V.sum()/L:.4f}, "
          f"G_spec {(q.J_c - q.J_s).sum()/L:.4f}, mean pixels (sum of crop areas) {q.px.mean():.4f}, rounds mean {q.rounds.mean():.2f} max {q.rounds.max()}")
dis = d[d.agree == 0]
print("disagreements:", len(dis), "of which cheap==full action:", int((dis.flip == 0).sum()))
d.to_pickle(runmeta.FINAL / f"spec_loop_proxy_kitti_vc{v_close}{'_near' if near else ''}.pkl")
