#!/usr/bin/env python
"""K40 R1 parity (private helper): release run_rows.py records vs the stored paper records, per row, every differing
frame listed. Stored sources: K28 p3 test instr (cheap, loop rows, heuristics @S, wf640 / 960 / native + C1),
K28 test v6basewf-yolo11m (whole frame + plain / T4 / C1 at every side), K30 match (heuristics x1.00),
K39 vuw (heuristics strict).
usage: k40_parity.py <release run dir> <dataset nusc|av2|ahd> <out.json>"""
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
NAT = {"nusc": 1600, "av2": 2048, "ahd": 1920}


def recs(d):
    rr = []
    for f in sorted(glob.glob(str(Path(d) / "*.pkl"))):
        rr += pickle.load(open(f, "rb"))
    x = pd.DataFrame(rr)
    x["unit"] = x.unit.astype(str)
    return x


def main():
    run, ds, outp = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    r = recs(run)
    units = set(r.unit)
    src = {"k28instr": B / "results/k28/p3" / f"{ds}_test_instr", "k28basewf": B / "results/k28" / f"{ds}_test_v6basewf-yolo11m",
           "k30match": B / "results/k30/match" / ds, "k39vuw": B / "results/k39/vuw" / ds}
    res = {}
    for nm, d in src.items():
        s = recs(d)
        s = s[s.unit.isin(units)]
        m = r.merge(s, on=["unit", "frame"], suffixes=("", "@ref"), how="inner")
        cols = [c for c in r.columns if (c.startswith(("a_", "px_")) and c + "@ref" in m)]
        out = {"frames_release": len(r), "frames_matched": len(m)}
        for c in cols:
            x, y = m[c].astype(float).to_numpy(), m[c + "@ref"].astype(float).to_numpy()
            bad = ~np.isclose(x, y, rtol=0, atol=1e-12, equal_nan=True)
            out[c] = {"differing": int(bad.sum()), "frames": [f"{u}/{f}" for u, f in zip(m.unit[bad], m.frame[bad])][:50]}
        res[nm] = out
    outp.write_text(json.dumps(res, indent=1) + "\n")
    for nm, out in res.items():
        tot = {c: v["differing"] for c, v in out.items() if isinstance(v, dict)}
        print(nm, "matched", out["frames_matched"], "of", out["frames_release"], "| columns", len(tot), "| differing:",
              {c: n for c, n in tot.items() if n} or "none")


if __name__ == "__main__":
    main()
