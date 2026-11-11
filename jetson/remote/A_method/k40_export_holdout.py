#!/usr/bin/env python
"""K40 (private helper): the held-out Argoverse-HD training sequences (65 units, GT frames) -> records/holdout.parquet,
the rows of tab:holdout (k35_k4_analyze.load: K28-style runner, AnA iterated, bounded + temporal recovery)."""
import importlib.util
import os
import sys
from pathlib import Path

B = Path(os.environ["B"])
sys.path.insert(0, str(B / "scripts/A_method"))
s = importlib.util.spec_from_file_location("k4", B / "scripts/A_method/k35_k4_analyze.py")
K4 = importlib.util.module_from_spec(s); s.loader.exec_module(K4)
d = K4.load()
cols = ["unit", "frame", "act_gt", "a_gt_k0"] + [c for r in K4.ROWS.values() for c in r] + [c for r in K4.CAP.values() for c in r]
d = d[list(dict.fromkeys(cols))].rename(columns={"a_gt_k0": "a_gt"})
d.to_parquet(Path(sys.argv[1]) / "holdout.parquet", index=False, compression="zstd")
print(d.shape)
