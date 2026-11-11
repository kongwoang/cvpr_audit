"""K33 (docs/paper/handoff/K33/table1_design.md): crop selection of the AnA-style (D) and counterfactual-importance (E)
baselines, shared by the VUW evaluation (torch) and the Jetson timing (TensorRT), so that both run the same rows.

Scenes are lifted object tuples mono = (z, lat_min, lat_max, ttc) as everywhere in vtf; pp = rap.planner.PlannerParams.
"""
from __future__ import annotations

import numpy as np

from rap import planner as P

from . import speculative as SP

ANA_T = (0.05, 0.10, 0.15, 0.25)
ANA_TH = (3.0, 5.0, np.inf)
CF_P = (0.10, 0.20, 0.30, 0.40)
CF_Q = (0.5, 1.0, 2.0)


def ana_cands():
    """the 12 AnA candidates (t, T_h); the official form is (0.25, inf) only as a reported row"""
    return [(t, th) for t in ANA_T for th in ANA_TH]


def cf_cands():
    return [(p, q) for p in CF_P for q in CF_Q]


def act(mono, v, pp):
    return P.discrete_action(P.required_decel(*mono, v, pp)[0], pp)


def ana_targets(conf, mono, v, pp, t, th):
    """pass-1 detections (conf >= t) in the ego corridor within the range limit with TTC < T_h, most urgent first
    (TTC ascending, then range)"""
    conf = np.asarray(conf, float)
    z, lo, hi, ttc = (np.asarray(a, float) for a in mono)
    ok = (conf >= t) & P.in_corridor(lo, hi, pp.corridor_half_w) & (z <= pp.max_range) & (ttc < th)
    idx = np.flatnonzero(ok)
    return idx[np.lexsort((z[idx], ttc[idx]))]


def _drop(mono, i):
    k = np.ones(len(mono[0]), bool); k[i] = False
    return tuple(np.asarray(a, float)[k] for a in mono)


def _set(mono, i, j, val):
    m = [np.asarray(a, float).copy() for a in mono]
    m[j][i] = val
    return tuple(m)


def cf_targets(mono, v, pp, p, q):
    """objects whose removal, range x (1 +- p) or TTC +- q (s) changes the action; order: removal-critical first, then
    range, then TTC; within a class by braking demand descending"""
    mono = tuple(np.asarray(a, float) for a in mono)
    n = len(mono[0])
    if not n:
        return np.zeros(0, int)
    a0 = act(mono, v, pp)
    _, dem = P.required_decel(*mono, v, pp)
    dem = np.asarray(dem, float)
    cls = np.full(n, 9)
    for i in range(n):
        z, ttc = mono[0][i], mono[3][i]
        if act(_drop(mono, i), v, pp) != a0:
            cls[i] = 0
        elif any(act(_set(mono, i, 0, z * f), v, pp) != a0 for f in (1 - p, 1 + p)):
            cls[i] = 1
        elif np.isfinite(ttc) and any(act(_set(mono, i, 3, max(ttc + d, 0.1)), v, pp) != a0 for d in (-q, q)):
            cls[i] = 2
    idx = np.flatnonzero(cls < 9)
    return idx[np.lexsort((-dem[idx], cls[idx]))]


def crops_for(xyxy, idx, sup, W, H, have=()):
    """box_crop (context 2) of the targets, then of the support objects not already targeted; crops already covered by
    `have` (earlier rounds) are skipped"""
    out = []
    for k in list(idx) + [s for s in sup if s not in set(idx)]:
        c = SP.box_crop(np.asarray(xyxy[k], float), W, H)
        if not SP.covered(c, list(have) + out):
            out.append(c)
    return out


def fill_cap(crops, cost, cap):
    """crops in their priority order while the summed canvas px (cost(crop)) stays within the cap"""
    sel, used = [], 0.0
    for c in crops:
        cv = cost(c)
        if used + cv <= cap:
            sel.append(c); used += cv
    return sel
