#!/usr/bin/env python
"""Summarise the exploratory pivot-verification headroom (03) per cell, as shares of the all-cheap loss (all units)."""
from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

from rap.scoring import topk_expect
from vtf import runmeta

DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"


def main():
    dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
    dv = dv[dv.track.isin(["KITTI", "nuScenes"])].copy()
    dv["frame"] = dv.frame.astype(int)
    out = []
    for f in sorted(glob.glob(str(runmeta.RAW / "*_headroom_*" / "headroom__*.pkl"))):
        _, ds, geo = f.rsplit("/", 1)[1][:-4].split("__")
        h = pd.read_pickle(f)
        for name in ("brake", "traj"):
            if f"{name}_J_cheap" not in h:
                continue
            ref = dv[(dv.track == ds) & (dv.geometry == geo) & (dv.system == name)]
            d = h.merge(ref[["seq", "frame", "V", "J_cheap"]], on=["seq", "frame"])
            assert np.allclose(d[f"{name}_J_cheap"], d.J_cheap, atol=1e-9)
            Lc = float(d.J_cheap.sum())
            V = d.V.to_numpy(float)
            k = int(round(0.2 * len(d)))
            G = lambda col: float((d[f"{name}_J_cheap"] - d[col]).sum()) / Lc            # noqa: E731
            r = {"cell": f"{ds} {geo} {name}", "n": len(d),
                 "full (frame escalation, all)": V.sum() / Lc,
                 "oracle frame escalation @20%": topk_expect(V, [V], [k])[0][0][0] / Lc,
                 "escalate+perfect arbiter (sum V+)": V[V > 0].sum() / Lc,
                 "cheap + verify 1 pivot": G(f"{name}_J_cheap_pK1"),
                 "cheap + verify 2 pivots": G(f"{name}_J_cheap_pK2"),
                 "full + verify 1 pivot": G(f"{name}_J_full_pK1"),
                 "full + verify 2 pivots": G(f"{name}_J_full_pK2"),
                 "cheap + verify all dets": G(f"{name}_J_cheap_all"),
                 "full + verify all dets": G(f"{name}_J_full_all"),
                 "verifications/frame (cheap,K=1)": float(d[f"{name}_n_cheap_pK1"].mean())}
            for e, s in ((0.05, 0.05), (0.1, 0.1), (0.2, 0.2)):
                r[f"cheap + 1 pivot, noisy e{e} s{s}"] = G(f"{name}_J_cheap_pK1_e{e}_s{s}")
                r[f"full + 1 pivot, noisy e{e} s{s}"] = G(f"{name}_J_full_pK1_e{e}_s{s}")
            out.append(r)
    res = pd.DataFrame(out).set_index("cell").T
    pd.set_option("display.width", 250)
    print(res.round(3).to_string())
    res.to_csv(runmeta.FINAL / "headroom_summary.csv")


if __name__ == "__main__":
    main()
