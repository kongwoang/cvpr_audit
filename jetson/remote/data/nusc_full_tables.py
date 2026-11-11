#!/usr/bin/env python
"""nuScenes-full: causal ego speed, split, and per-frame decision tables (braking) for every detected scene.

Split: scenes of DEEP's frozen split keep their labels (`old`); the other scenes (`new`) get DEEP's rule applied among
themselves — sort by (location, scene name), position i -> i mod 10: 0-4 train, 5-6 val, 7-9 test.  The confirmatory
evaluation uses the `new` test scenes only.
Writes $B/results/nusc_full/{split.csv, flips__<geometry>.pkl, values__<geometry>.csv}.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from rap.nusc import NuScenesDB, make_adapter
from rap.paths import CACHE, NUSCENES_TRAINVAL
from vtf import flips

B = Path(os.environ["B"])
OUT = B / "results" / "nusc_full"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["mono", "oracle"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    names = sorted(p.stem for p in (Path(CACHE) / "nusc_det_tv" / "ns_cheap_320").glob("*.npz")
                   if (Path(CACHE) / "nusc_det_tv" / "ns_full_640" / p.name).exists())
    # ---- split
    deep = json.loads((Path(os.environ["RAP_REPO"]) / "configs" / "benchmark_splits.json").read_text())
    dv = pd.read_csv(Path(os.environ["RAP_REPO"]) / "results" / "final" / "benchmark_decision_values.csv.gz",
                     dtype={"seq": str}, usecols=["track", "seq", "split"])
    old = dict(dv[dv.track == "nuScenes"].drop_duplicates("seq")[["seq", "split"]].values)
    by = {db.scene_name(s): s for s in db.scenes}
    logs = {r["token"]: r for r in json.loads((Path(NUSCENES_TRAINVAL) / "v1.0-trainval" / "log.json").read_text())}
    loc = {n: logs[db._t["scene"][by[n]]["log_token"]]["location"] for n in names}
    new = sorted((n for n in names if n not in old), key=lambda n: (loc[n], n))
    lab = {n: ("train" if i % 10 < 5 else "val" if i % 10 < 7 else "test") for i, n in enumerate(new)}
    sp = pd.DataFrame([{"seq": n, "location": loc[n], "set": "old" if n in old else "new",
                        "split": old.get(n, lab.get(n))} for n in names])
    sp.to_csv(OUT / "split.csv", index=False)
    print(sp.groupby(["set", "split"]).size().to_string(), flush=True)
    # ---- causal ego speed for every scene (DEEP's definition)
    vc = {n: dict(enumerate(db.ego_speeds_causal(by[n], 0.5))) for n in names}
    rows = []
    for n in names:
        rows += [{"seq": n, "frame": f, "v_ego_causal": v} for f, v in vc[n].items()]
    pd.DataFrame(rows).to_csv(OUT / "v_ego_causal.csv", index=False)
    # ---- decision tables (braking), as DEEP's decision builder, plus the flip/pivot features
    adapter = make_adapter(db)
    recs = flips.build(Path(CACHE) / "nusc_det_tv", "ns_cheap_320", "ns_full_640", names, adapter, args.geometry,
                       v_causal=lambda s: vc[s], with_traj=False)
    d = pd.DataFrame(recs).merge(sp[["seq", "set", "split"]], on="seq")
    d.to_pickle(OUT / f"flips__{args.geometry}.pkl")
    v = d[["seq", "frame", "set", "split", "brake_J_c", "brake_J_f", "brake_V", "v_causal"]]
    v.to_csv(OUT / f"values__{args.geometry}.csv", index=False)
    for (st, s_), g in v.groupby(["set", "split"]):
        aff = (g.brake_V.abs() > 1e-9)
        print(f"  {args.geometry} {st:3s} {s_:5s}: frames {len(g):6d} affected {int(aff.sum()):5d} "
              f"harmed {float((g.brake_V < -1e-9).sum() / max(aff.sum(), 1)):.3f} "
              f"uniform {g.brake_V.sum() / g.brake_J_c.sum():+.4f}", flush=True)


if __name__ == "__main__":
    main()
