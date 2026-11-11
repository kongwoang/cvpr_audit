#!/usr/bin/env python
"""K30 V4 (post-hoc) — Bench2Drive, from the three finished K6 / K28 runs (results/b2d/{full,rep2,rep3}; branches privileged,
cheap, u960, u1600, ours (v3), ours_v6 = v6-S), route 11755 excluded as in K6 (219 routes). Nothing is re-run.
Per run x branch x route, from the stored records (merged.json) and the per-route metric_info.json:
  * DS, RC, infraction penalty, success (Completed / Perfect, no infraction except min_speed), status
  * infractions by type: count per route, and per km (route_length from the record's meta)
  * Bench2Drive's official metrics, computed with the official code (third_party/Bench2Drive/tools, offline-map version):
      - Multi-Ability success (Merging, Overtaking, Emergency_Brake, Give_Way, Traffic_Signs; ability_benchmark.py rules,
        Traffic_Signs' junction-completion rule with the route planner on the offline map), per route and aggregated
      - Driving Efficiency (mean of the min-speed percentages of a route, as efficiency_smoothness_benchmark.read_from_json)
      - Driving Smoothness / Comfortness (seg_compute_comfort_metric of the official script, per route)
  * ego hard-brake events per km: entries into longitudinal acceleration < -4.05 m/s^2 (the official min_lon_accel bound),
    acceleration from the logged positions (speed smoothed as the official comfort code); ego only: whether a brake was
    necessary cannot be told from the stored data (no actor state is logged), so this is a proxy, not "unnecessary" brakes
Per branch: mean +- sd over runs; per scenario type DS / SR; routes failed by cheap and completed by v6-S in >= 2 of 3 runs
and vice versa; paired route bootstrap (2,000, seed 0) of ours_v6 - branch on the run-averaged per-route value of every
metric. Also writes the official tools' merged json + ability json per run / branch under results/k30/b2d/.
Run inside the CARLA environment (scripts/A_closedloop/carla_env.sh).
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ["B"])
TOOLS = B / "third_party" / "Bench2Drive" / "tools"
XML = B / "third_party" / "carla_garage" / "Bench2Drive" / "leaderboard" / "data" / "bench2drive220.xml"
RUNS = {1: B / "results" / "b2d" / "full", 2: B / "results" / "b2d" / "rep2", 3: B / "results" / "b2d" / "rep3"}
BRANCHES = ("privileged", "cheap", "u960", "u1600", "ours", "ours_v6")
OUT = B / "results" / "k30" / "b2d"
INF = ("collisions_layout", "collisions_pedestrian", "collisions_vehicle", "red_light", "stop_infraction", "outside_route_lanes",
       "min_speed_infractions", "yield_emergency_vehicle_infractions", "scenario_timeouts", "route_dev", "vehicle_blocked", "route_timeout")


def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def main():
    ab = load(TOOLS / "ability_benchmark.py", "b2d_ability")
    es = load(TOOLS / "efficiency_smoothness_benchmark.py", "b2d_eff")
    routes = {r.get("id"): r for r in ET.parse(XML).getroot().findall("route")}
    stype = {k: r.find("scenarios").find("scenario").get("type") for k, r in routes.items()}
    town = {k: r.get("town") for k, r in routes.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    grp_cache, tsjc = {}, {}

    def ts_junction_completion(rid):
        """ability_benchmark's Traffic_Signs junction-completion threshold of route rid (official code, offline map)"""
        if rid not in tsjc:
            tn = town[rid]
            if tn not in grp_cache:
                cmap = ab.load_offline_map(tn, os.path.join(os.environ.get("CARLA_ROOT", "."), "CarlaUE4/Content/Carla/Maps"))
                grp_cache[tn] = ab.GlobalRoutePlanner(cmap, 1.0)
            wr = ab.get_waypoint_route(ab.get_position(routes[rid]), grp_cache[tn])
            count = 0
            for wp in wr:
                count += 1
                if wp.is_junction:
                    break
            tsjc[rid] = float(count + 8) / len(wr) if wp.is_junction else np.nan
        return tsjc[rid]

    rows = []
    for run, root in RUNS.items():
        for br in BRANCHES:
            d = root / br
            if not (d / "merged.json").exists():
                print("missing", d); continue
            m = json.loads((d / "merged.json").read_text())
            recs = [r for r in m["_checkpoint"]["records"] if "11755" not in r["route_id"]]
            # official merged json (route 11755 removed) + metric_info symlinks for the official tools
            od = OUT / f"run{run}" / br
            (od / "metric").mkdir(parents=True, exist_ok=True)
            mm = dict(m); mm["_checkpoint"] = dict(m["_checkpoint"]); mm["_checkpoint"]["records"] = recs
            (od / "merged.json").write_text(json.dumps(mm))
            for f in glob.glob(str(d / "save_chunk*" / "*" / "metric_info.json")):
                sd = Path(f).parent
                ln = od / "metric" / sd.name[sd.name.index("RouteScenario"):]        # = the record's save_name (dir = <chunk>_<save_name>)
                if not ln.exists():
                    ln.symlink_to(sd)
            for r in recs:
                rid = r["route_id"].split("_")[1]
                inf = r["infractions"]
                km = float(r["meta"].get("route_length", np.nan)) / 1000.0 if r.get("meta") else np.nan
                success = r["status"] in ("Completed", "Perfect") and not ab.get_infraction_status(r)
                rec = {"run": run, "branch": br, "route_id": rid, "scenario": stype.get(rid, ""), "town": town.get(rid, ""),
                       "status": r["status"], "DS": r["scores"]["score_composed"], "RC": r["scores"]["score_route"],
                       "penalty": r["scores"]["score_penalty"], "success": bool(success), "km": km,
                       **{f"n_{k}": len(inf.get(k, [])) for k in INF}}
                for k in INF:
                    rec[f"perkm_{k}"] = rec[f"n_{k}"] / km if km and km > 0 else np.nan
                # official efficiency (per route, as read_from_json; NaN when the route has no min-speed entry)
                ef = []
                for s in inf.get("min_speed_infractions", []):
                    n = re.search(r"\b\d+\.?\d*%", s)
                    if n and float(n.group().rstrip("%")) <= 1000:
                        ef.append(float(n.group().rstrip("%")))
                rec["efficiency"] = float(np.mean(ef)) if ef else np.nan
                # official smoothness + hard brakes from metric_info
                mf = od / "metric" / r["save_name"] / "metric_info.json"
                if mf.exists():
                    jd = json.loads(mf.read_text())
                    arr = {k: np.array([v[k] for v in jd.values()]) for k in ("acceleration", "angular_velocity", "forward_vector", "right_vector", "location", "rotation")}
                    rec["smoothness"] = float(es.seg_compute_comfort_metric(**arr))
                    # ego longitudinal deceleration from the logged positions (the logged acceleration has non-physical spikes
                    # down to -26 m/s^2): forward speed, Savitzky-Golay (7, 2) as the official comfort code, then its derivative
                    from scipy.signal import savgol_filter
                    n_ = len(arr["location"])
                    dt_ = float(r["meta"]["duration_game"]) / max(n_, 1) if r.get("meta") else 0.05
                    sp = np.r_[0.0, (np.diff(arr["location"][:, :2], axis=0) * arr["forward_vector"][1:, :2]).sum(1) / dt_]
                    sp = savgol_filter(sp, 7, 2) if n_ > 7 else sp
                    lon = np.gradient(sp, dt_) if n_ > 1 else np.zeros(n_)
                    lon = savgol_filter(lon, 7, 2) if n_ > 7 else lon
                    hb = lon < -4.05
                    rec["hard_brake_events"] = int(np.sum(hb[1:] & ~hb[:-1]) + (hb[0] if len(hb) else 0))
                    rec["dt_log"] = dt_
                    rec["hard_brake_per_km"] = rec["hard_brake_events"] / km if km and km > 0 else np.nan
                    rec["frames"] = len(lon)
                else:
                    rec.update({"smoothness": np.nan, "hard_brake_events": np.nan, "hard_brake_per_km": np.nan, "frames": np.nan})
                # official Multi-Ability flags of this route
                for a, scs in ab.Ability.items():
                    if rec["scenario"] not in scs:
                        rec[f"ab_{a}"] = np.nan
                    elif a != "Traffic_Signs" or success:
                        rec[f"ab_{a}"] = float(success)
                    else:
                        jc = ts_junction_completion(rid)
                        rec[f"ab_{a}"] = float(r["scores"]["score_route"] / 100.0 > jc and not inf.get("stop_infraction") and not inf.get("red_light"))
                rows.append(rec)
            print(f"run {run} {br}: {len(recs)} routes", flush=True)
    pr = pd.DataFrame(rows)
    pr.to_csv(OUT / "b2d_per_route.csv", index=False)
    METRICS = (["DS", "RC", "penalty", "success", "efficiency", "smoothness", "hard_brake_per_km"] + [f"perkm_{k}" for k in INF]
               + [f"n_{k}" for k in INF] + [f"ab_{a}" for a in ab.Ability])
    per_run = pr.groupby(["branch", "run"])[METRICS].mean().reset_index()
    per_run["ability_mean"] = per_run[[f"ab_{a}" for a in ab.Ability]].mean(1)
    per_run["routes"] = pr.groupby(["branch", "run"]).route_id.nunique().to_numpy()
    per_run.to_csv(OUT / "b2d_per_run.csv", index=False)
    summ = per_run.groupby("branch").agg(**{f"{m}_mean": (m, "mean") for m in METRICS + ["ability_mean"]},
                                         **{f"{m}_sd": (m, "std") for m in METRICS + ["ability_mean"]}).reset_index()
    # paired route bootstrap of ours_v6 - branch on the run-averaged per-route value
    avg = pr.groupby(["branch", "route_id"])[METRICS].mean()
    rng = np.random.default_rng(0)
    iv = []
    a = avg.loc["ours_v6"]
    for br in BRANCHES:
        if br == "ours_v6" or br not in avg.index.get_level_values(0):
            continue
        b = avg.loc[br]
        common = sorted(set(a.index) & set(b.index))
        bs = [rng.integers(0, len(common), len(common)) for _ in range(2000)]
        for mt in METRICS:
            dd = (a.loc[common, mt] - b.loc[common, mt]).to_numpy(float)
            ok = np.isfinite(dd)
            if not ok.any():
                continue
            vals = [np.nanmean(dd[i]) for i in bs]
            iv.append({"branch": br, "metric": mt, "routes": int(ok.sum()), "v6_minus": float(np.nanmean(dd)),
                       "ci_lo": float(np.nanpercentile(vals, 2.5)), "ci_hi": float(np.nanpercentile(vals, 97.5))})
    pd.DataFrame(iv).to_csv(OUT / "b2d_intervals.csv", index=False)
    summ.to_csv(OUT / "b2d_summary.csv", index=False)
    # per scenario type
    sc = pr.groupby(["branch", "scenario"]).agg(routes=("route_id", "nunique"), DS=("DS", "mean"), SR=("success", "mean")).reset_index()
    sc.to_csv(OUT / "b2d_per_scenario.csv", index=False)
    # routes failed by cheap and completed by v6-S in >= 2 of 3 runs, and vice versa ("completed" = success)
    s = pr.pivot_table(index="route_id", columns="branch", values="success", aggfunc="sum")
    flips = []
    for rid, r in s.iterrows():
        if r.get("ours_v6", 0) >= 2 and r.get("cheap", 0) <= 1:
            flips.append({"route_id": rid, "scenario": stype.get(rid, ""), "direction": "v6-S completes, cheap fails",
                          "v6_successes": int(r["ours_v6"]), "cheap_successes": int(r["cheap"])})
        if r.get("cheap", 0) >= 2 and r.get("ours_v6", 0) <= 1:
            flips.append({"route_id": rid, "scenario": stype.get(rid, ""), "direction": "cheap completes, v6-S fails",
                          "v6_successes": int(r["ours_v6"]), "cheap_successes": int(r["cheap"])})
    pd.DataFrame(flips).to_csv(OUT / "b2d_route_flips.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print(per_run[["branch", "run", "routes", "DS", "success", "efficiency", "smoothness", "hard_brake_per_km", "ability_mean"]].round(3).to_string(index=False))
    print(pd.DataFrame(flips).to_string(index=False))
    print("K30 B2D DONE", flush=True)


if __name__ == "__main__":
    main()
