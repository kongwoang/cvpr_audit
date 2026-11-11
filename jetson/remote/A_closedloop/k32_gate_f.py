#!/usr/bin/env python
"""K32 amendment F (handoff/K32/STATUS.md): gate (c) on every registered K32 outcome metric. Paired route bootstrap
(2,000, seed 0) of cheap − privileged on the dev3 runs: collision rate, preventable collision rate (amendment D), mean per-
route min TTC (capped 10 s), missed required episodes per route, unnecessary hard-brake onsets per route. (c) passes if
any interval excludes 0. (b) and the components decision are read from dev3_gate.csv. Writes tables/dev3F_c.csv and
tables/dev3F_gate.csv (GATE_PASS = b_pass and c_pass).
usage: k32_gate_f.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

S = importlib.util.spec_from_file_location("g", Path(__file__).with_name("k32_dev_gate.py"))
g = importlib.util.module_from_spec(S); S.loader.exec_module(g)


def per_route(br, pj, starts):
    R, Lg = {}, {}
    for f in sorted((g.K / "dev3").glob(f"{br}_c*.json")):
        for r in json.load(open(f))["_checkpoint"]["records"]:
            R[r["route_id"].split("_")[1]] = r
        Lg.update(g.jsonl(g.K / "dev3" / f"{f.stem}_save" / pj, starts))
    rows = {}
    for rid, r in R.items():
        ln = Lg.get(rid)
        T = [t for t in (ln or {}).get("ticks", []) if "a_gt" in t]
        cl = g.preventable(r, ln)
        missed, i = 0, 0
        while i < len(T):
            if T[i]["a_gt"] >= 1:
                j = i
                while j < len(T) and T[j]["a_gt"] >= 1:
                    j += 1
                missed += not any(T[k]["a_applied"] >= 1 for k in range(i, j))
                i = j
            else:
                i += 1
        unn = sum(1 for k in range(1, len(T)) if T[k]["a_applied"] == 2 and T[k - 1]["a_applied"] < 2 and T[k]["a_gt"] == 0)
        mt = min([t["min_ttc"] for t in T if t.get("min_ttc") is not None] or [10.0])
        rows[rid] = {"collision": float(len(cl) > 0), "preventable_collision": float(any(x[4] for x in cl)),
                     "min_ttc": min(mt, 10.0), "missed_required_episodes": missed, "unnecessary_hard_onsets": unn}
    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    starts = g.route_starts(g.K / "dev_routes.xml")
    P = per_route("privileged", "aeb_privileged_L0.jsonl", starts)
    C = per_route("cheap", "aeb_cheap_L0.jsonl", starts)
    idx = P.index.intersection(C.index)
    out = []
    for m in P.columns:
        d, lo, hi = g.boot_diff(C.loc[idx, m], P.loc[idx, m])
        out.append({"metric": m, "routes": len(idx), "privileged_mean": P.loc[idx, m].mean(), "cheap_mean": C.loc[idx, m].mean(),
                    "cheap_minus_privileged": d, "ci_lo": lo, "ci_hi": hi, "excludes_0": bool(lo > 0 or hi < 0)})
    O = pd.DataFrame(out)
    O.to_csv(g.OUT / "dev3F_c.csv", index=False)
    gate = pd.read_csv(g.OUT / "dev3_gate.csv").iloc[0].to_dict()
    gate["c_pass_registered"] = gate.get("c_pass")
    gate["c_pass"] = bool(O.excludes_0.any())
    gate["c_metrics_passing"] = ";".join(O[O.excludes_0].metric)
    gate["GATE_PASS"] = bool(gate["b_pass"] and gate["c_pass"])
    pd.DataFrame([gate]).to_csv(g.OUT / "dev3F_gate.csv", index=False)
    pd.set_option("display.width", 200)
    print(O.round(3).to_string(index=False))
    print({k: gate[k] for k in ("b_pass", "c_pass_registered", "c_pass", "c_metrics_passing", "GATE_PASS")})


if __name__ == "__main__":
    main()
