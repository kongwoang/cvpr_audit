#!/usr/bin/env python
"""Exactness and cost of the decomposable Planner B field and pivots vs probing, on every KITTI frame."""
import time
from pathlib import Path
import numpy as np
from rap import planner as P, planner_b as B
from rap.cache import DetCache
from rap.decision import KittiAdapter
from rap.paths import CACHE
from vtf import dsf

pb, cb = B.PARAMS_B["static_obstacles"], B.COSTS_B["default"]
R = np.array([12.5, 17.5, 22.5, 27.5, 35.0, 45.0, 55.0, 70.0]); Y = np.array([-6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6.])
n = mg = mp = 0; tp = tc = 0.0
for s in [q.stem for q in sorted((Path(CACHE) / "det" / "cheap_320").glob("*.npz"))]:
    c = DetCache(Path(CACHE) / "det" / "cheap_320" / f"{s}.npz"); sp = KittiAdapter.speeds(s); prev = None
    for i, fr in enumerate(c.frames):
        fr = int(fr); v = float(sp[fr]) if fr < len(sp) else float(sp[-1])
        d, g = c.det(i), c.geo(i); k = d["conf"] >= 0.25
        t = tuple(np.asarray(g[x][k], float) for x in ("z", "lat_min", "lat_max", "ttc"))
        t0 = time.perf_counter(); base = B.plan(*t, v, prev, pb, cb)
        probe = np.array([[B.plan(*tuple(np.r_[a, b] for a, b in zip(t, (r, y - .9, y + .9, np.inf))), v, prev, pb, cb) != base for y in Y] for r in R])
        piv_p = [j for j in range(len(t[0])) if B.plan(*tuple(np.delete(a, j) for a in t), v, prev, pb, cb) != base]
        t1 = time.perf_counter()
        grid = dsf.planner_b_insertion_grid(t, v, prev, R, Y, 1.8, pb, cb); piv_c = list(dsf.planner_b_pivots(t, v, prev, pb, cb))
        t2 = time.perf_counter()
        tp += t1 - t0; tc += t2 - t1; n += 1
        mg += int(not np.array_equal(probe, grid)); mp += int(sorted(piv_p) != sorted(piv_c))
        prev = base
print(f"frames {n}: grid mismatches {mg}, pivot mismatches {mp}; per frame probing {1e3*tp/n:.2f} ms, decomposed {1e3*tc/n:.2f} ms")
