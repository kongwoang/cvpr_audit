#!/usr/bin/env python
"""K6: Bench2Drive 220 routes, PDM-Lite with perception in the loop — merge (official tools/merge_route_json.py per branch)
and summarise: Driving Score, Success Rate (Bench2Drive's definitions: DS = Σ score_composed / 220; success = status
Completed / Perfect and no infraction other than min_speed), infraction counts, paired route bootstrap of ours − branch,
routes evaluated per branch, unmatched detections per frame (perception_<branch>.jsonl, routes run after 2026-10-03 01:15).
Outputs: results/b2d/summary/b2d_summary.csv, b2d_per_route.csv."""
from __future__ import annotations

import glob
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
O = B / "results" / "b2d" / "full"
TOOL = B / "third_party" / "carla_garage" / "Bench2Drive" / "tools" / "merge_route_json.py"
OUT = B / "results" / "b2d" / "summary"
BRANCHES = ("privileged", "cheap", "u960", "u1600", "ours")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, merged = [], {}
    for br in BRANCHES:
        d = O / br
        if not d.exists():
            continue
        subprocess.run(["python", str(TOOL), "-f", str(d)], capture_output=True, text=True)
        m = json.loads((d / "merged.json").read_text()) if (d / "merged.json").exists() else None
        merged[br] = m
        for rd in (m["_checkpoint"]["records"] if m else []):
            inf = rd["infractions"]
            succ = rd["status"] in ("Completed", "Perfect") and all(len(v) == 0 for k, v in inf.items() if k != "min_speed_infractions")
            r = {"branch": br, "route_id": rd["route_id"], "scenario": rd.get("scenario_name"), "town": rd.get("town_name"),
                 "status": rd["status"], "driving_score": rd["scores"]["score_composed"], "route_completion": rd["scores"]["score_route"],
                 "infraction_penalty": rd["scores"]["score_penalty"], "success": succ}
            r.update({f"n_{k}": len(v) for k, v in inf.items()})
            rows.append(r)
    pr = pd.DataFrame(rows)
    pr.to_csv(OUT / "b2d_per_route.csv", index=False)
    # unmatched detections per frame (routes logged after the per-route fix)
    um = {}
    for br in BRANCHES:
        recs = []
        for f in glob.glob(str(O / br / "save_chunk*" / f"perception_{br}.jsonl")):
            recs += [json.loads(l) for l in open(f) if l.strip()]
        if recs:
            w = np.array([r["frames"] for r in recs], float)
            um[br] = {k: float(np.average([r["mean"][k] for r in recs], weights=w)) for k in ("in_cam", "hidden", "phantoms", "crops")}
            um[br]["routes_logged"] = len(recs)
    srows = []
    rng = np.random.default_rng(0)
    for br in BRANCHES:
        if br not in merged or merged[br] is None:
            continue
        p = pr[pr.branch == br]
        r = {"branch": br, "routes_evaluated": len(p), "driving_score_official": merged[br]["driving score"],
             "success_rate_official": merged[br]["success rate"], "driving_score_mean_evaluated": p.driving_score.mean(),
             "success_rate_evaluated": p.success.mean(), "route_completion_mean": p.route_completion.mean()}
        for c in [c for c in p.columns if c.startswith("n_")]:
            r[f"{c}_total"] = int(p[c].sum())
        if br in um:
            r.update({f"perception_{k}_per_frame": v for k, v in um[br].items()})
        if br != "ours" and (pr.branch == "ours").any():
            a = pr[pr.branch == "ours"].set_index("route_id"); b = p.set_index("route_id")
            common = sorted(set(a.index) & set(b.index))
            if common:
                da = (a.loc[common, "driving_score"] - b.loc[common, "driving_score"]).to_numpy()
                ds = (a.loc[common, "success"].astype(float) - b.loc[common, "success"].astype(float)).to_numpy()
                bs = [rng.integers(0, len(common), len(common)) for _ in range(2000)]
                r.update({"paired_routes": len(common), "ours_minus_DS": da.mean(), "ours_minus_DS_ci_lo": np.percentile([da[i].mean() for i in bs], 2.5),
                          "ours_minus_DS_ci_hi": np.percentile([da[i].mean() for i in bs], 97.5), "ours_minus_SR": ds.mean(),
                          "ours_minus_SR_ci_lo": np.percentile([ds[i].mean() for i in bs], 2.5), "ours_minus_SR_ci_hi": np.percentile([ds[i].mean() for i in bs], 97.5)})
        srows.append(r)
    s = pd.DataFrame(srows)
    s.to_csv(OUT / "b2d_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.T.to_string())


if __name__ == "__main__":
    main()
