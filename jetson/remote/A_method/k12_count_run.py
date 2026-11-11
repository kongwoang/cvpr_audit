#!/usr/bin/env python
"""K12(b) — run an evaluation script unchanged while counting the detector input tensor pixels of every detection call
(rap.detect.TwoFidelityDetector.detect: mode.net_h x mode.net_w per call, i.e. 32-px rounded canvases, guard
re-detections included). Prints the total, the number of calls and the frames of the output table.
usage: python k12_count_run.py <script.py> [args...]"""
import runpy
import sys

from rap.detect import TwoFidelityDetector

C = {"px": 0, "calls": 0}
_orig = TwoFidelityDetector.detect


def detect(self, img, mode, *a, **k):
    C["px"] += mode.net_h * mode.net_w; C["calls"] += 1
    return _orig(self, img, mode, *a, **k)


TwoFidelityDetector.detect = detect
sys.argv = sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    print(f"K12 COUNT: detector tensor px total {C['px']}, calls {C['calls']}", flush=True)
