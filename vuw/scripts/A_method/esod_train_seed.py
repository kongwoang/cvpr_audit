#!/usr/bin/env python
"""ESOD's train.py with another RNG seed, official code untouched: their `init_seeds(2 + rank)` is offset by $ESOD_SEED
(patched on utils.general before train.py imports it). Run with cwd = third_party/esod, same arguments as train.py."""
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
import utils.general as G                                                         # noqa: E402

_orig, _off = G.init_seeds, int(os.environ["ESOD_SEED"])
G.init_seeds = lambda seed=0: _orig(seed + _off)
sys.argv = ["train.py"] + sys.argv[1:]
runpy.run_path("train.py", run_name="__main__")
