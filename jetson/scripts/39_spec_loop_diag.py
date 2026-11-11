#!/usr/bin/env python
"""Why the proved loop disagrees with the full decision on some KITTI frames: which assumption fails."""
from pathlib import Path
import numpy as np, pandas as pd
from rap import planner as P
from rap.cache import Camera, DetCache
from rap.decision import KittiAdapter
from rap.paths import CACHE
from vtf import runmeta, speculative as SP, dsf

pp = P.PlannerParams()
d = pd.read_pickle(runmeta.FINAL / "spec_loop_proxy_kitti.pkl")
dis = d[d.agree == 0]
cnt = {}
for s, g in dis.groupby("seq"):
    speeds = KittiAdapter.speeds(s); cam = Camera("KITTI", s)
    c = DetCache(Path(CACHE) / "det" / "cheap_320" / f"{s}.npz"); f = DetCache(Path(CACHE) / "det" / "full_640" / f"{s}.npz")
    for fr in g.frame:
        i = c.index[fr]; v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
        df, gf = f.det(f.index[fr]), f.geo(f.index[fr]); k = df["conf"] >= 0.25
        t = tuple(np.asarray(gf[x][k], float) for x in ("z", "lat_min", "lat_max", "ttc"))
        a_max, a_obj = P.required_decel(*t, v, pp)
        if not len(a_obj):
            cnt["full has no object (cheap phantom kept?)"] = cnt.get("full has no object (cheap phantom kept?)", 0) + 1; continue
        j = int(np.argmax(a_obj)); z, lo, hi, tt = (x[j] for x in t)
        # the same object as a static probe would have demand:
        static = dsf._insert_demand(np.array([z]), v, pp)[0]
        in_corr = lo < pp.corridor_half_w and hi > -pp.corridor_half_w
        if not in_corr:
            key = "decisive full object not in corridor (?)"
        elif z < 12.5:
            key = "decisive full object nearer than 12.5 m (outside probe grid)"
        elif a_obj[j] > static + 1e-6 and P.discrete_action(static, pp) < P.discrete_action(a_obj[j], pp):
            key = "closing speed: static probe underestimates demand"
        elif (hi - lo) > 1.8 + 1e-6 or abs((hi + lo) / 2) > 0.0:
            key = "lateral: object wider/offset relative to probe grid"
        else:
            key = "other"
        cnt[key] = cnt.get(key, 0) + 1
for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
    print(f"{v:4d}  {k}")
