#!/usr/bin/env python
"""K40 (private helper): Bench2Drive per-route records (tab:b2d) -> <out>/b2d_routes.parquet (three runs x branch x route:
official DS / RC / success / Multi-Ability flags and infraction counts; handoff/K30/b2d_per_route.csv) and
<out>/b2d_inloop.parquet (fourth run with the logging agent: per route the TTC < 3 s actor pairs in the camera view and
those seen by the perception branch; handoff/K30/inloop_per_route.csv). Branches of the table only.
usage: scripts/py scripts/k40_export_b2d.py <out_dir>"""
import sys
from pathlib import Path

import pandas as pd

HO = Path(__file__).resolve().parents[1] / "docs/paper/handoff/K30"
BR = ["privileged", "cheap", "u960", "u1600", "ours_v6"]

od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
r = pd.read_csv(HO / "b2d_per_route.csv")
r[r.branch.isin(BR)].to_parquet(od / "b2d_routes.parquet", index=False, compression="zstd")
i = pd.read_csv(HO / "inloop_per_route.csv").drop(columns=["chunk"])
i[i.branch.isin(BR)].to_parquet(od / "b2d_inloop.parquet", index=False, compression="zstd")
print(r[r.branch.isin(BR)].groupby(["branch", "run"]).size().unstack().to_string(), "\n", i.branch.value_counts().to_dict())
