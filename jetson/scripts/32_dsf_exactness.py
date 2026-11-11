#!/usr/bin/env python
"""B2 step 1: the closed-form braking field and pivots agree with probing on every KITTI frame; and cost."""
import time
from pathlib import Path
import numpy as np
from rap import planner as P
from rap.cache import DetCache
from rap.decision import KittiAdapter
from rap.paths import CACHE
from vtf import dsf

pp = P.PlannerParams()
R = np.array([12.5, 17.5, 22.5, 27.5, 35.0, 45.0, 55.0, 70.0]); Y = np.array([-6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6.])
dec = lambda t, v: P.discrete_action(P.required_decel(*t, v, pp)[0], pp)
n = mism_grid = mism_piv = 0; t_probe = t_closed = 0.0
for s in [p.stem for p in sorted((Path(CACHE) / "det" / "cheap_320").glob("*.npz"))]:
    c = DetCache(Path(CACHE) / "det" / "cheap_320" / f"{s}.npz"); sp = KittiAdapter.speeds(s)
    for i, fr in enumerate(c.frames):
        fr = int(fr); v = float(sp[fr]) if fr < len(sp) else float(sp[-1])
        d, g = c.det(i), c.geo(i); k = d["conf"] >= 0.25
        t = tuple(np.asarray(g[x][k], float) for x in ("z", "lat_min", "lat_max", "ttc"))
        t0 = time.perf_counter(); base = dec(t, v)
        probe = np.array([[dec(tuple(np.r_[a, b] for a, b in zip(t, (r, y - .9, y + .9, np.inf))), v) != base for y in Y] for r in R])
        piv_p = [j for j in range(len(t[0])) if dec(tuple(np.delete(a, j) for a in t), v) != base]
        t1 = time.perf_counter()
        closed = dsf.insertion_grid(t, v, R, Y, 1.8, pp); piv_c = list(dsf.brake_pivots(t, v, pp))
        t2 = time.perf_counter()
        t_probe += t1 - t0; t_closed += t2 - t1; n += 1
        mism_grid += int(not np.array_equal(probe, closed)); mism_piv += int(sorted(piv_p) != sorted(piv_c))
print(f"frames {n}: grid mismatches {mism_grid}, pivot mismatches {mism_piv}")
print(f"per frame: probing {1e3 * t_probe / n:.2f} ms, closed form {1e3 * t_closed / n:.3f} ms")
