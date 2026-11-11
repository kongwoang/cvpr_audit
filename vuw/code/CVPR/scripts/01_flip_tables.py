#!/usr/bin/env python
"""Phase 0H stage 1: per-frame escalation records for KITTI and nuScenes, and pre-registration gate V1.

Writes `flips__<dataset>__<geometry>.pkl` into the run directory.  Gate V1: the braking and Planner B losses recomputed
here must equal DEEP's canonical per-frame losses (`benchmark_decision_values.csv.gz`) within 1e-9 on every frame.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vtf import flips, runmeta

from rap.decision import KittiAdapter            # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL   # noqa: E402

DV = runmeta.RAP_REPO / "results" / "final" / "benchmark_decision_values.csv.gz"
TOL = 1e-9


def canonical() -> pd.DataFrame:
    d = pd.read_csv(DV, dtype={"seq": str, "frame": str, "unit": str})
    d = d[d.track.isin(["KITTI", "nuScenes"])].copy()
    d["frame"] = d.frame.astype(int)
    return d


def gate_v1(df: pd.DataFrame, ds: str, geo: str, dv: pd.DataFrame) -> dict:
    out = {}
    for sysname, col in (("brake", "brake"), ("traj", "traj")):
        if f"{col}_J_c" not in df:
            continue
        ref = dv[(dv.track == ds) & (dv.geometry == geo) & (dv.system == sysname)]
        m = df.merge(ref[["seq", "frame", "J_cheap", "J_full", "V", "split", "unit"]], on=["seq", "frame"],
                     how="outer", validate="one_to_one", indicator=True)
        missing = int((m._merge != "both").sum())
        dc = float(np.abs(m[f"{col}_J_c"] - m.J_cheap).max())
        dfl = float(np.abs(m[f"{col}_J_f"] - m.J_full).max())
        ok = missing == 0 and dc <= TOL and dfl <= TOL
        out[sysname] = {"frames": int(len(ref)), "unmatched": missing, "max_abs_dJ_cheap": dc,
                        "max_abs_dJ_full": dfl, "pass": bool(ok)}
        print(f"  V1 {ds}/{geo}/{sysname}: frames {len(ref)} unmatched {missing} "
              f"max|dJc| {dc:.2e} max|dJf| {dfl:.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["KITTI", "nuScenes"], required=True)
    ap.add_argument("--geometry", choices=["mono", "oracle"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N sequences only (smoke test; skips V1 verdict)")
    args = ap.parse_args()
    run = runmeta.new_run(f"flips_{args.dataset}_{args.geometry}", vars(args))

    if args.dataset == "KITTI":
        det_dir = Path(CACHE) / "det"
        cm, fm = "cheap_320", "full_640"
        adapter, with_traj, vc = KittiAdapter, True, None
    else:
        from rap import egospeed
        from rap.nusc import NuScenesDB, make_adapter
        det_dir = Path(CACHE) / "nusc_det_tv"
        cm, fm = "ns_cheap_320", "ns_full_640"
        adapter = make_adapter(NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval"))
        with_traj = False
        t = egospeed.table()
        tab = {s: dict(zip(g.frame.astype(int), g[egospeed.COL].astype(float))) for s, g in t.groupby("seq")}
        vc = lambda s: tab[s]                                                          # noqa: E731
    seqs = [p.stem for p in sorted((det_dir / cm).glob("*.npz"))]
    if args.limit:
        seqs = seqs[:args.limit]
    rows = flips.build(det_dir, cm, fm, seqs, adapter, args.geometry, v_causal=vc, with_traj=with_traj)
    df = pd.DataFrame(rows)
    df.to_pickle(run / f"flips__{args.dataset}__{args.geometry}.pkl")
    print(f"  {len(df)} frames, {df.brake_flip.sum()} braking flips"
          + (f", {df.traj_flip.sum()} Planner B flips" if with_traj else ""), flush=True)

    dv = canonical()
    if args.limit:
        dv = dv[dv.seq.isin(seqs)]
    v1 = gate_v1(df, args.dataset, args.geometry, dv)
    (run / "gate_v1.json").write_text(json.dumps(v1, indent=2))
    if not all(x["pass"] for x in v1.values()):
        print("GATE V1 FAILED", flush=True)
        sys.exit(2)
    print("GATE V1 PASSED", flush=True)


if __name__ == "__main__":
    main()
