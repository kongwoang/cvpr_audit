#!/usr/bin/env python
"""Fidelity (A) for ESOD: their `scripts/data_prepare.py --dataset VisDrone` on VisDrone2019-DET-val, unchanged except
the SAM result moved to the CPU before `.numpy()` (the same wrapper as scripts/A_method/esod_prepare.py).
Run inside envs/esod with cwd = third_party/esod, VisDrone -> $B/repro/visdrone."""
import argparse
import os
import sys
from pathlib import Path

B = Path(os.environ["B"])
sys.path.insert(0, str(B / "third_party" / "esod"))
sys.path.insert(0, str(B / "third_party" / "esod" / "scripts"))
import data_prepare as DP                                                     # noqa: E402

_seg = DP.segment_image


def _segment_cpu(*a, **k):
    res, inv = _seg(*a, **k)
    return res.cpu(), (inv.cpu() if hasattr(inv, "cpu") else inv)


DP.segment_image = _segment_cpu
DP.opt = argparse.Namespace(dataset="VisDrone")
DP.prepare_visdrone()
print("ESOD VISDRONE PREPARE DONE")
