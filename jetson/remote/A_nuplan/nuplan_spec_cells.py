#!/usr/bin/env python
"""nuPlan (B3): decision value of speculative perception vs uniform FULL, PDM-Closed and IDM, DEEP's 1,440 states.

Reads data/cache/nuplan_real_spec/nuplan_spec_{planner}_raw.csv (branches spec, cheap, full planned through DEEP's 115
hook), checks that cheap/full reproduce DEEP's nuplan_real_perception_{planner}_raw.csv, and reports per planner x loss
x split: G_full = (C_cheap - C_full)/C_cheap, G_spec likewise, harm rates, and log-level paired bootstrap of
G_spec - G_full (1,000 draws).  Costs are DEEP's 83 `costs`, unchanged.
"""
from __future__ import annotations

import importlib.util, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ["RAP_REPO"])
sys.path.insert(0, str(ROOT / "src"))
from rap.paths import CACHE, RESULTS                      # noqa: E402

_s = importlib.util.spec_from_file_location("x83", ROOT / "scripts" / "83_external_transfer.py")
x83 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(x83)
OUTC = Path(CACHE) / os.environ.get("SPEC_OUT", "nuplan_real_spec")
LOSSES = ("collision", "clearance_shortfall", "log_deviation", "safety", "scalar_J")


def g(cc, cb):
    return (cc.sum() - cb.sum()) / cc.sum() if cc.sum() > 1e-12 else np.nan


def main():
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())["nuplan"]
    split_of = {l: k for k in ("train", "val", "test") for l in splits[k]}
    rows, repro = [], {}
    for p in ("pdm_closed", "idm"):
        f = OUTC / f"nuplan_spec_{p}_raw.csv"
        if not f.exists():
            print(f"  {p}: {f.name} missing, skipped")
            continue
        d = pd.read_csv(f)
        deep = pd.read_csv(Path(RESULTS) / "final" / f"nuplan_real_perception_{p}_raw.csv")
        dd = d.sort_values(["scenario", "iteration"]).reset_index(drop=True)
        de = deep.sort_values(["scenario", "iteration"]).reset_index(drop=True)
        assert (dd[["scenario", "iteration"]].values == de[["scenario", "iteration"]].values).all()
        repro[p] = {b: int((~np.isclose(x83.costs(dd, b)["scalar_J"], x83.costs(de, b)["scalar_J"], equal_nan=True)).sum())
                    for b in ("cheap", "full")}
        ok = (d.ok_cheap == 1) & (d.ok_full == 1) & (d.ok_spec == 1)
        d = d[ok].reset_index(drop=True)
        C = {b: x83.costs(d, b) for b in ("cheap", "full", "spec")}
        logs = d.log.to_numpy()
        for split in ("all", "test"):
            msk = np.ones(len(d), bool) if split == "all" else (d.log.map(split_of) == "test").to_numpy()
            lg = logs[msk]
            uq = np.unique(lg)
            idx = [np.flatnonzero(lg == u) for u in uq]
            for loss in LOSSES:
                cc, cf, cs = (C[b][loss][msk] for b in ("cheap", "full", "spec"))
                Vf, Vs = cc - cf, cc - cs
                r = {"planner": p, "split": split, "loss": loss, "states": int(msk.sum()), "logs": len(uq),
                     "G_full": g(cc, cf), "G_spec": g(cc, cs),
                     "aff_full": int((np.abs(Vf) > 1e-9).sum()), "aff_spec": int((np.abs(Vs) > 1e-9).sum()),
                     "harm_full": float((Vf < -1e-9).sum() / max((np.abs(Vf) > 1e-9).sum(), 1)),
                     "harm_spec": float((Vs < -1e-9).sum() / max((np.abs(Vs) > 1e-9).sum(), 1)),
                     "spec_eq_full_states": float(np.mean(np.abs(cs - cf) <= 1e-9))}
                rng, dr = np.random.default_rng(0), []
                for _ in range(1000):
                    t = np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))])
                    if cc[t].sum() > 1e-12:
                        dr.append(g(cc[t], cs[t]) - g(cc[t], cf[t]))
                r["spec_minus_full_lo"], r["spec_minus_full_hi"] = (np.percentile(dr, [2.5, 97.5]) if dr else (np.nan, np.nan))
                rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUTC / "nuplan_spec_cells.csv", index=False)
    crops = pd.read_csv(OUTC / "spec_crops.csv.gz")
    pd.set_option("display.width", 250)
    print("reproduction of DEEP's cheap/full scalar_J (mismatching states):", repro)
    print(f"crop footprint: {crops.describe().round(3).to_string()}")
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
