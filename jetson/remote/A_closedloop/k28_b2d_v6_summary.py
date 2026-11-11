#!/usr/bin/env python
"""K28 PHASE 3 (post-hoc) — K6 Bench2Drive with v6-S in the loop (branch ours_v6, perception_pilot.py on branch ours_v6),
summarised as b2d_repeats_summary.py does (219 routes, route 11755 excluded; DS = mean score_composed over the routes,
success = Completed / Perfect and no infraction other than min_speed). ours_v6 is merged with the official
tools/merge_route_json.py (its own folder only); the K6 branches are read from their existing merged.json (not re-merged).
Only runs where ours_v6 has all 219 routes enter the comparison; every branch is averaged per route over those same runs, then
paired route bootstrap (2,000, seed 0) of ours_v6 - branch on the run-averaged per-route DS and success.
Writes results/k28/p3/tables/b2d_v6_per_route.csv, b2d_v6_per_run.csv, b2d_v6_summary.csv."""
from __future__ import annotations

import glob
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
TOOL = B / "third_party" / "carla_garage" / "Bench2Drive" / "tools" / "merge_route_json.py"
OUT = B / "results" / "k28" / "p3" / "tables"
RUNS = {1: B / "results" / "b2d" / "full", 2: B / "results" / "b2d" / "rep2", 3: B / "results" / "b2d" / "rep3"}
BRANCHES = ("privileged", "cheap", "u960", "u1600", "ours", "ours_v6")
NROUTES = 219


def records(d, merge):
    if merge:
        subprocess.run(["python", str(TOOL), "-f", str(d)], capture_output=True, text=True)
    m = json.loads((d / "merged.json").read_text()) if (d / "merged.json").exists() else None
    out = []
    for rd in (m["_checkpoint"]["records"] if m else []):
        inf = rd["infractions"]
        out.append({"route_id": rd["route_id"].split("_rep")[0], "status": rd["status"],
                    "driving_score": rd["scores"]["score_composed"], "route_completion": rd["scores"]["score_route"],
                    "infraction_penalty": rd["scores"]["score_penalty"],
                    "success": rd["status"] in ("Completed", "Perfect") and all(len(v) == 0 for k, v in inf.items() if k != "min_speed_infractions"),
                    **{f"n_{k}": len(v) for k, v in inf.items()}})
    return out


def perception(d, br):
    recs = []
    for f in glob.glob(str(d / br / "save_chunk*" / f"perception_{br}.jsonl")):
        recs += [json.loads(l) for l in open(f) if l.strip()]
    if not recs:
        return {}
    w = np.array([r["frames"] for r in recs], float)
    return {**{f"perception_{k}_per_frame": float(np.average([r["mean"][k] for r in recs], weights=w)) for k in ("in_cam", "hidden", "phantoms", "crops")},
            "perception_routes_logged": len(recs)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, perc = [], []
    for run, root in RUNS.items():
        for br in BRANCHES:
            if (root / br).exists():
                rows += [{"run": run, "branch": br, **r} for r in records(root / br, merge=br == "ours_v6")]
                p = perception(root, br)
                if p:
                    perc.append({"run": run, "branch": br, **p})
    pr = pd.DataFrame(rows)
    pr = pr[~pr.route_id.str.contains("11755")]
    pr.to_csv(OUT / "b2d_v6_per_route.csv", index=False)
    per_run = pr.groupby(["branch", "run"]).agg(routes=("route_id", "nunique"), DS=("driving_score", "mean"), SR=("success", "mean"),
                                                RC=("route_completion", "mean"), failed=("status", lambda s: int(s.str.startswith("Failed").sum()))).reset_index()
    if perc:
        per_run = per_run.merge(pd.DataFrame(perc), on=["branch", "run"], how="left")
    per_run.to_csv(OUT / "b2d_v6_per_run.csv", index=False)
    v6 = per_run[(per_run.branch == "ours_v6") & (per_run.routes == NROUTES)]
    runs = sorted(v6.run)
    q = pr[pr.run.isin(runs)]
    avg = q.groupby(["branch", "route_id"]).agg(DS=("driving_score", "mean"), SR=("success", "mean"), runs=("run", "nunique")).reset_index()
    rng = np.random.default_rng(0)
    a = avg[avg.branch == "ours_v6"].set_index("route_id")
    out = []
    for br in BRANCHES:
        p = per_run[(per_run.branch == br) & per_run.run.isin(runs)]
        if not len(p):
            continue
        r = {"branch": br, "runs_compared": "/".join(str(x) for x in runs), "routes_per_run": "/".join(str(x) for x in p.routes),
             "DS_mean": p.DS.mean(), "DS_std": p.DS.std(ddof=1) if len(p) > 1 else np.nan, "SR_mean": p.SR.mean(),
             "SR_std": p.SR.std(ddof=1) if len(p) > 1 else np.nan, "RC_mean": p.RC.mean(), **{f"DS_run{int(x.run)}": x.DS for _, x in p.iterrows()}}
        if br != "ours_v6":
            b = avg[avg.branch == br].set_index("route_id")
            common = sorted(set(a.index) & set(b.index))
            dd = (a.loc[common, "DS"] - b.loc[common, "DS"]).to_numpy(); ds = (a.loc[common, "SR"] - b.loc[common, "SR"]).to_numpy()
            bs = [rng.integers(0, len(common), len(common)) for _ in range(2000)]
            r.update({"paired_routes": len(common), "v6_minus_DS": dd.mean(),
                      "v6_minus_DS_ci_lo": np.percentile([dd[i].mean() for i in bs], 2.5), "v6_minus_DS_ci_hi": np.percentile([dd[i].mean() for i in bs], 97.5),
                      "v6_minus_SR": ds.mean(), "v6_minus_SR_ci_lo": np.percentile([ds[i].mean() for i in bs], 2.5),
                      "v6_minus_SR_ci_hi": np.percentile([ds[i].mean() for i in bs], 97.5)})
        out.append(r)
    s = pd.DataFrame(out)
    s.to_csv(OUT / "b2d_v6_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(per_run.round(3).to_string(index=False)); print(s.round(3).T.to_string())
    print(f"K28 B2D V6 SUMMARY DONE (runs {runs})")


if __name__ == "__main__":
    main()
