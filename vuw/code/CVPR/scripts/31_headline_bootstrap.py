#!/usr/bin/env python
"""Paired unit-bootstrap intervals for the headline speculative results vs uniform escalation (test split only)."""
import glob
import numpy as np
import pandas as pd
from vtf import runmeta

DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"
dv = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
dv = dv[dv.track.isin(["KITTI", "nuScenes"])].copy(); dv["frame"] = dv.frame.astype(int)

def boot(q, jc, js, n=2000, seed=0):
    units = q.unit.to_numpy(); uniq = np.unique(units)
    idx = [np.flatnonzero(units == u) for u in uniq]
    Lc, V, G = q[jc].to_numpy(), q.V.to_numpy(), (q[jc] - q[js]).to_numpy()
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        t = np.concatenate([idx[i] for i in rng.integers(0, len(uniq), len(uniq))])
        L = Lc[t].sum()
        if L > 0: d.append((G[t].sum() - V[t].sum()) / L)
    return (G.sum() / Lc.sum(), V.sum() / Lc.sum(), *np.percentile(d, [2.5, 97.5]), np.mean(np.array(d) > 0))

rows = []
def add(name, pkl, ds, geo, system, jc, js):
    f = sorted(glob.glob(str(runmeta.RAW / pkl)))
    if not f: return
    d = pd.read_pickle(f[-1]).drop(columns=["V", "unit", "split", "J_cheap"], errors="ignore")
    ref = dv[(dv.track == ds) & (dv.geometry == geo) & (dv.system == system) & (dv.split == "test")]
    q = ref[["seq", "frame", "unit", "V"]].merge(d, on=["seq", "frame"])
    g, u, lo, hi, p = boot(q, jc, js)
    rows.append(dict(result=name, n_units=q.unit.nunique(), G_method=g, G_uniform=u, diff_lo=lo, diff_hi=hi, share_draws_better=p))

add("spec+refiner, KITTI mono brake (proxy)", "*_spec_refine_KITTI_mono/spec_refine.pkl", "KITTI", "mono", "brake", "brake_J_c", "brake_J_spec_ref")
add("spec+refiner, KITTI mono traj (proxy)", "*_spec_refine_KITTI_mono/spec_refine.pkl", "KITTI", "mono", "traj", "traj_J_c", "traj_J_spec_ref")
add("spec, KITTI mono brake (proxy)", "*_spec_refine_KITTI_mono/spec_refine.pkl", "KITTI", "mono", "brake", "brake_J_c", "brake_J_spec")
add("real spec 640, KITTI mono brake", "*_realspec_KITTI_mono/realspec.pkl", "KITTI", "mono", "brake", "J_c", "J_spec")
add("real spec 960, KITTI mono brake", "*_realspec_s0.773_KITTI_mono/realspec.pkl", "KITTI", "mono", "brake", "J_c", "J_spec")
add("real spec, KITTI oracle brake", "*_realspec_KITTI_oracle/realspec.pkl", "KITTI", "oracle", "brake", "J_c", "J_spec")
add("real spec, nuScenes mono brake", "*_realspec_nuScenes_mono/realspec.pkl", "nuScenes", "mono", "brake", "J_c", "J_spec")
add("real spec, nuScenes oracle brake", "*_realspec_nuScenes_oracle/realspec.pkl", "nuScenes", "oracle", "brake", "J_c", "J_spec")
add("learned refiner only, KITTI mono traj", "*_learned_refiner_KITTI/learned_refiner.pkl", "KITTI", "mono", "traj", "traj_J_c", "traj_J_learn")
r = pd.DataFrame(rows); r.to_csv(runmeta.FINAL / "headline_bootstrap.csv", index=False)
pd.set_option("display.width", 250); print(r.round(3).to_string())
