#!/usr/bin/env python
"""K10 — post-hoc robustness check: run an existing evaluation script unchanged with a fixed ego-region (hood) image
mask applied to EVERY pipeline. Detections whose box lies at least half inside the mask (containment >= 0.5; the mask
is the full-width band of image rows y >= K10_MASK_Y0) are removed from the decision: rap.mono.predicted_geometry is
wrapped so that their range / ground range / height range and ttc are +inf, which the braking controller excludes
(rap.planner.required_decel keeps only dist <= max_range), i.e. they carry no demand, are never support and seed no
crop. Everything else (detector calls, crops, merge, refiner on support objects, lift history) is unchanged.
Run inside the shadow tree $B/k10_masked (B points there; datasets / models / scripts are symlinks) so the masked
results never overwrite the original ones.
usage: K10_MASK_Y0=1830 python k10_masked_run.py <script.py> [script args...]
"""
from __future__ import annotations

import os
import runpy
import sys

import numpy as np

import rap.mono as M

Y0 = float(os.environ["K10_MASK_Y0"])
_ORIG = M.predicted_geometry
STATS = {"calls": 0, "boxes": 0, "masked": 0}


def in_mask(xyxy):
    b = np.asarray(xyxy, float).reshape(-1, 4)
    h = np.maximum(b[:, 3] - b[:, 1], 1e-6)
    return (b[:, 3] - np.maximum(b[:, 1], Y0)).clip(0, None) / h >= 0.5


def predicted_geometry(dets, *a, **k):
    g = _ORIG(dets, *a, **k)
    m = in_mask(dets["xyxy"]) if len(dets["xyxy"]) else np.zeros(0, bool)
    STATS["calls"] += 1; STATS["boxes"] += len(m); STATS["masked"] += int(m.sum())
    if m.any():
        g = dict(g)
        for key in ("z", "z_ground", "z_height", "ttc"):
            if key in g:
                arr = np.asarray(g[key], float).copy(); arr[m] = np.inf; g[key] = arr
    return g


M.predicted_geometry = predicted_geometry
script = sys.argv[1]
sys.argv = sys.argv[1:]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    print(f"K10 MASK y0={Y0:g}: lift calls {STATS['calls']}, boxes {STATS['boxes']}, masked {STATS['masked']}", flush=True)
