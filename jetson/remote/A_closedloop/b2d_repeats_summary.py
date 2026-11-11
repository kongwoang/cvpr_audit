#!/usr/bin/env python
"""K6 update — Bench2Drive over three runs (run 1 = results/b2d/full, runs 2 / 3 = results/b2d/rep2, rep3; 219 routes,
route 11755 excluded in every run): per branch and run the official merge (tools/merge_route_json.py) and the per-route
records; mean +- std over the runs of driving score (mean over the 219 routes), success rate and route completion;
paired route bootstrap (2,000) of ours - branch on the RUN-AVERAGED per-route driving score and success.
Writes results/b2d/summary/b2d_repeats_summary.csv, b2d_repeats_per_route.csv."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
TOOL = B / "third_party" / "carla_garage" / "Bench2Drive" / "tools" / "merge_route_json.py"
OUT = B / "results" / "b2d" / "summary"
RUNS = {1: B / "results" / "b2d" / "full", 2: B / "results" / "b2d" / "rep2", 3: B / "results" / "b2d" / "rep3"}
BRANCHES = ("privileged", "cheap", "u960", "u1600", "ours")


def records(d):
    subprocess.run(["python", str(TOOL), "-f", str(d)], capture_output=True, text=True)
    m = json.loads((d / "merged.json").read_text()) if (d / "merged.json").exists() else None
    out = []
    for rd in (m["_checkpoint"]["records"] if m else []):
        inf = rd["infractions"]
        out.append({"route_id": rd["route_id"].split("_rep")[0], "status": rd["status"],
                    "driving_score": rd["scores"]["score_composed"], "route_completion": rd["scores"]["score_route"],
                    "success": rd["status"] in ("Completed", "Perfect") and all(len(v) == 0 for k, v in inf.items() if k != "min_speed_infractions"),
                    "n_collisions_vehicle": len(inf.get("collisions_vehicle", [])), "n_collisions_pedestrian": len(inf.get("collisions_pedestrian", []))})
    return out


def main():
    rows = []
    for run, root in RUNS.items():
        for br in BRANCHES:
            if (root / br).exists():
                rows += [{"run": run, "branch": br, **r} for r in records(root / br)]
    pr = pd.DataFrame(rows)
    pr = pr[~pr.route_id.str.contains("11755")]
    pr.to_csv(OUT / "b2d_repeats_per_route.csv", index=False)
    per_run = pr.groupby(["branch", "run"]).agg(routes=("route_id", "nunique"), DS=("driving_score", "mean"), SR=("success", "mean"),
                                                RC=("route_completion", "mean")).reset_index()
    avg = pr.groupby(["branch", "route_id"]).agg(DS=("driving_score", "mean"), SR=("success", "mean"), runs=("run", "nunique")).reset_index()
    rng = np.random.default_rng(0)
    out = []
    for br in BRANCHES:
        q = per_run[per_run.branch == br]
        if not len(q):
            continue
        r = {"branch": br, "runs": len(q), "routes_per_run": "/".join(str(x) for x in q.routes),
             "DS_mean": q.DS.mean(), "DS_std": q.DS.std(ddof=1) if len(q) > 1 else np.nan,
             "SR_mean": q.SR.mean(), "SR_std": q.SR.std(ddof=1) if len(q) > 1 else np.nan,
             "RC_mean": q.RC.mean(), "RC_std": q.RC.std(ddof=1) if len(q) > 1 else np.nan,
             **{f"DS_run{int(x.run)}": x.DS for _, x in q.iterrows()}}
        if br != "ours":
            a = avg[avg.branch == "ours"].set_index("route_id"); b = avg[avg.branch == br].set_index("route_id")
            common = sorted(set(a.index) & set(b.index))
            dd = (a.loc[common, "DS"] - b.loc[common, "DS"]).to_numpy(); ds = (a.loc[common, "SR"] - b.loc[common, "SR"]).to_numpy()
            bs = [rng.integers(0, len(common), len(common)) for _ in range(2000)]
            r.update({"paired_routes": len(common), "ours_minus_DS": dd.mean(),
                      "ours_minus_DS_ci_lo": np.percentile([dd[i].mean() for i in bs], 2.5), "ours_minus_DS_ci_hi": np.percentile([dd[i].mean() for i in bs], 97.5),
                      "ours_minus_SR": ds.mean(), "ours_minus_SR_ci_lo": np.percentile([ds[i].mean() for i in bs], 2.5),
                      "ours_minus_SR_ci_hi": np.percentile([ds[i].mean() for i in bs], 97.5)})
        out.append(r)
    s = pd.DataFrame(out)
    s.to_csv(OUT / "b2d_repeats_summary.csv", index=False)
    per_run.to_csv(OUT / "b2d_repeats_per_run.csv", index=False)
    pd.set_option("display.width", 250)
    print(per_run.round(3).to_string(index=False)); print(s.round(3).T.to_string())


if __name__ == "__main__":
    main()
