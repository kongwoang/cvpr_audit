#!/usr/bin/env python
"""K40 (private helper): per-frame field / tube records of fig:tube (handoff/K36b/f2_frames.csv.gz: all three test
splits; draft action a_cheap, ego speed v, whether the field is non-empty, whether the hybrid's tube is formed, and the
footprint / tube shares and pixels) -> <out>/tube_frames.parquet
usage: scripts/py scripts/k40_export_tube.py <out_dir>"""
import sys
from pathlib import Path

import pandas as pd

t = pd.read_csv(Path(__file__).resolve().parents[1] / "docs/paper/handoff/K36b/f2_frames.csv.gz")
t["unit"] = t.unit.astype(str)
t["dataset"] = t.dataset.map({"nusc": "nuscenes", "av2": "av2", "ahd": "argoverse_hd"})
od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
t.to_parquet(od / "tube_frames.parquet", index=False, compression="zstd")
print(t.shape, t.groupby(["dataset", "a_cheap"])[["field_nonempty", "tube_formed"]].mean().round(2).to_string())
