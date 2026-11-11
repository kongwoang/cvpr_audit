#!/usr/bin/env python
"""K39 VUW parity checks (handoff/K39/PREREG.md). Per dataset, on the frames of results/k39/vuw[_tag]/<ds>:
  v6-S a / J / px and u960 + C1 a / J vs K28 p3 test instr; the 8 heuristics at x1.00 (a / J / px) vs results/k30/match;
  sa (exact locality alone) vs the stored chain ch_a (a, J); smoke only: <cfg>@chk (strict machinery, unlimited cap)
  == <cfg>@x1.00; strict px <= v6-S px on every frame.
usage: k39_vuw_check.py <tag ('' or smoke)> [datasets]"""
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
HEUR8 = ("unc2", "unc4", "tcr0", "tcr3", "cor0", "cor8", "tsp2", "tsp8")


def recs(d):
    rr = []
    for f in sorted(glob.glob(str(d / "*.pkl"))):
        rr += pickle.load(open(f, "rb"))
    x = pd.DataFrame(rr)
    x["unit"] = x.unit.astype(str)
    return x


def ndiff(m, a, b):
    x, y = m[a], m[b]
    if x.dtype == object or y.dtype == object:
        return int((x.astype(str) != y.astype(str)).sum())
    return int((~np.isclose(x.astype(float), y.astype(float), rtol=0, atol=1e-9, equal_nan=True)).sum())


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    dss = sys.argv[2].split(",") if len(sys.argv) > 2 else ["nusc", "av2", "ahd"]
    out = {}
    for ds in dss:
        d = recs(B / "results/k39" / ("vuw" + (f"_{tag}" if tag else "")) / ds)
        units = set(d.unit)
        r = {"frames": len(d), "units": len(units)}
        k28 = recs(B / "results/k28/p3" / f"{ds}_test_instr")
        k28 = k28[k28.unit.isin(units)]
        m = d.merge(k28, on=["unit", "frame"], suffixes=("", "_ref"), validate="1:1")
        r["k28_frames_matched"] = len(m)
        for c in ("a_cheap", "J_cheap", "a_v6S|none", "J_v6S|none", "a_v6S|v3a", "J_v6S|v3a", "px_v6S",
                  "a_wf960+C1|none", "a_wf960+C1|v3a", "J_wf960+C1|v3a"):
            r[f"k28 {c}"] = ndiff(m, c, c + "_ref")
        r["k28 sa vs ch_a (a)"] = ndiff(m.assign(_x=m["a_sa|none"]), "_x", "a_ch_a")
        r["k28 sa vs ch_a (J)"] = ndiff(m.assign(_x=m["J_sa|none"]), "_x", "J_ch_a")
        k30 = recs(B / "results/k30/match" / ds)
        k30 = k30[k30.unit.isin(units)]
        m3 = d.merge(k30, on=["unit", "frame"], suffixes=("", "_ref"), validate="1:1")
        r["k30_frames_matched"] = len(m3)
        for cfg in HEUR8:
            nm = f"{cfg}@x1.00+C1"
            for c in (f"a_{nm}|none", f"J_{nm}|v3a", f"a_{nm}|v3a", f"px_{nm}"):
                r[f"k30 {c}"] = ndiff(m3, c, c + "_ref")
            s = f"{cfg}@strict+C1"
            ratio = d[f"px_{s}"] / d["px_v6S"]
            r[f"strict {cfg} px ratio max"] = float(ratio.max())
            r[f"strict {cfg} px ratio mean"] = float(ratio.mean())
            r[f"strict {cfg} violations"] = int((ratio > 1 + 1e-9).sum())
            if f"a_{cfg}@chk+C1|v3a" in d:
                for c in ("a_{}|none", "J_{}|v3a", "px_{}"):
                    r[f"chk {cfg} {c.format('')}"] = ndiff(d, c.format(f"{cfg}@chk+C1"), c.format(nm))
        out[ds] = r
    print(json.dumps(out, indent=1))
    od = B / "results/k39" / ("vuw" + (f"_{tag}" if tag else ""))
    (od / "checks.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
