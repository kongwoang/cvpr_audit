#!/usr/bin/env python
"""K30 V4 in-loop metrics (post-hoc; pre-registered in handoff/K30/STATUS.md) from the logging run (perception_pilot_log.py:
PerceptionPilot unchanged + per-frame logging; results/b2d/k30log/<branch>/save_chunk<k>/inloop_<branch>.jsonl, checkpoints
<branch>/chunk<k>.json; the rep2 routes, 219, route 11755 excluded as in K6).
Route id: each in-loop record holds the route's town and global-plan start point; it is matched to the route of the chunk's
routes file with the same town and the nearest first waypoint (<= 5 m). A route run twice (evaluator crash + resume) keeps
its last record (the one the checkpoint scored).
Per branch x route:
  km              path length from the logged ego positions
  hard brakes     rising edges of brake >= 0.5; per km. "unnecessary" = PDM-Lite's hazard actor on that frame is a phantom
                  (an unmatched detection made a static obstacle, id >= 10,000,000); "real actor" = a real vehicle / walker
                  hazard; "rule" = red light / stop sign hazard only; "other" = no hazard flag (e.g. leader following)
  recall          over (frame, actor) pairs of real actors ahead in the ego corridor with TTC <= 3 s that project into the
                  front camera within the detection radius: share the branch did not hide (privileged: 1 by construction)
  episodes        sensitivity, not pre-registered: brake >= 0.5 episodes separated by > 1 s (the controller toggles the
                  brake within one stop); unnecessary episodes = phantom hazard on the episode's first frame
  min TTC         minimum over frames of the closest corridor actor's TTC (inf if none); routes with min TTC < 1 s / 0.5 s
  DS / success    from the logging run's own checkpoint (a 4th run; DS of runs 1-3 are in v4 / K28)
Per branch: sums / means over routes; paired route bootstrap (2,000, seed 0) of ours_v6 - branch on routes present in both.
results/k30/b2d/inloop_per_route.csv, inloop_summary.csv
"""
from __future__ import annotations

import glob
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ.get("B_REAL", os.environ["B"]))
RUN = Path(os.environ.get("K30_B2D_LOG", B / "results" / "b2d" / "k30log"))
OUT = Path(os.environ.get("K30_B2D_INLOOP_OUT", B / "results" / "k30" / "b2d"))
BRANCHES = ("privileged", "cheap", "u960", "u1600", "ours", "ours_v6")
PHANTOM = 10_000_000


def routes_of(xml):
    out = []
    for r in ET.parse(xml).getroot().iter("route"):
        p = r.find("waypoints").find("position")
        out.append((r.get("id"), r.get("town"), float(p.get("x")), float(p.get("y"))))
    return out


def route_metrics(fr, branch):
    xy = np.array([f["xy"] for f in fr if "xy" in f], float).reshape(-1, 2)
    km = float(np.hypot(*np.diff(xy, axis=0).T).sum() / 1000) if len(xy) > 1 else 0.0
    ok = [f for f in fr if "brake" in f]
    br = np.array([f["brake"] for f in ok], float)
    ev = np.flatnonzero((br >= 0.5) & np.concatenate([[True], br[:-1] < 0.5])) if len(br) else np.zeros(0, int)
    kinds = {"unnecessary": 0, "real actor": 0, "rule": 0, "other": 0}
    for i in ev:
        f = ok[i]
        hz = f["haz"]
        if f.get("haz_phantom"):
            kinds["unnecessary"] += 1
        elif hz[0] or hz[1]:
            kinds["real actor"] += 1
        elif hz[2] or hz[3]:
            kinds["rule"] += 1
        else:
            kinds["other"] += 1
    # sensitivity (not pre-registered): brake episodes = rising edges at least 1 s (20 frames at 20 Hz) after the last frame
    # with brake >= 0.5, since PDM-Lite's longitudinal controller toggles the brake within one stop
    on = np.flatnonzero(br >= 0.5) if len(br) else np.zeros(0, int)
    epi = [i for k_, i in enumerate(on) if k_ == 0 or i - on[k_ - 1] > 20]
    epi_unn = sum(bool(ok[i].get("haz_phantom")) for i in epi)
    n3 = seen3 = 0
    mins = []
    for f in ok:
        tt = [a[3] for a in f["actors"] if a[3] is not None]
        mins.append(min(tt) if tt else math.inf)
        for a in f["actors"]:
            if a[3] is not None and a[3] <= 3.0 and a[4]:
                n3 += 1
                seen3 += 0 if (a[5] and branch != "privileged") else 1
    return {"frames": len(ok), "log_errors": sum("err" in f for f in fr), "km": km, "hard_brakes": len(ev),
            **{f"hb_{k.replace(' ', '_')}": v for k, v in kinds.items()}, "brake_episodes": len(epi), "brake_episodes_unnecessary": epi_unn,
            "pairs_ttc3_incam": n3, "pairs_ttc3_seen": seen3,
            "min_ttc": float(min(mins)) if mins else math.inf}


def main():
    rows = []
    for br in BRANCHES:
        for ck in sorted(glob.glob(str(RUN / br / "chunk*.json"))):
            k = Path(ck).stem
            xml = RUN / "routes" / f"{k}.xml"
            if not xml.exists():
                continue
            rts = routes_of(xml)
            rec = {r["route_id"].split("_")[1]: r for r in json.load(open(ck)).get("_checkpoint", {}).get("records", [])}
            last = {}
            jl = RUN / br / f"save_{k}" / f"inloop_{br}.jsonl"
            for line in open(jl) if jl.exists() else []:
                try:
                    x = json.loads(line)
                except json.JSONDecodeError:
                    continue
                st = x.get("start")
                if not st:
                    continue
                cand = [(math.hypot(st["x"] - rx, st["y"] - ry), rid) for rid, town, rx, ry in rts if town == st["town"]]
                if not cand or min(cand)[0] > 5.0:
                    continue
                last[min(cand)[1]] = x["frames"]
            for rid, fr in last.items():
                r = {"branch": br, "chunk": k, "route": rid, **route_metrics(fr, br)}
                cr = rec.get(rid)
                if cr is not None:
                    r.update({"DS": cr["scores"]["score_composed"], "RC": cr["scores"]["score_route"], "status": cr["status"],
                              "collisions": sum(cr["infractions"][c] if isinstance(cr["infractions"][c], int) else len(cr["infractions"][c])
                                                for c in ("collisions_layout", "collisions_pedestrian", "collisions_vehicle"))})
                rows.append(r)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "inloop_per_route.csv", index=False)
    # a route whose final checkpoint status is a simulator / agent crash (e.g. the last route of a chunk, which the
    # dispatcher does not re-run because the chunk's progress is complete) has truncated records: left out, counted
    if "status" in d:
        crashed = d.status.astype(str).str.contains("crashed", case=False)
        print("routes left out (final status crashed):", d[crashed].groupby("branch").size().to_dict())
        d = d[~crashed].reset_index(drop=True)
    if d.empty:
        print("no in-loop records"); return

    def agg(x):
        km = x.km.sum()
        return {"routes": len(x), "km": km, "hard_brakes_per_km": x.hard_brakes.sum() / km, "unnecessary_per_km": x.hb_unnecessary.sum() / km,
                "real_actor_per_km": x.hb_real_actor.sum() / km, "rule_per_km": x.hb_rule.sum() / km, "other_per_km": x.hb_other.sum() / km,
                "unnecessary_share": x.hb_unnecessary.sum() / max(x.hard_brakes.sum(), 1),
                "brake_episodes_per_km": x.brake_episodes.sum() / km, "unnecessary_episodes_per_km": x.brake_episodes_unnecessary.sum() / km,
                "recall_ttc3": x.pairs_ttc3_seen.sum() / max(x.pairs_ttc3_incam.sum(), 1), "pairs_ttc3": int(x.pairs_ttc3_incam.sum()),
                "routes_min_ttc_lt1": float((x.min_ttc < 1.0).mean()), "routes_min_ttc_lt0.5": float((x.min_ttc < 0.5).mean()),
                "DS_mean": x.DS.mean() if "DS" in x else np.nan, "log_errors": int(x.log_errors.sum())}
    out = []
    ref = d[d.branch == "ours_v6"].set_index("route")
    for br in BRANCHES:
        x = d[d.branch == br]
        if x.empty:
            continue
        r = {"branch": br, **agg(x)}
        if br != "ours_v6" and len(ref):
            com = sorted(set(ref.index) & set(x.route))
            a, b = ref.loc[com], x.set_index("route").loc[com]
            rng = np.random.default_rng(0)
            bs = [rng.integers(0, len(com), len(com)) for _ in range(2000)]
            r["paired_routes"] = len(com)
            for nm, f in (("hard_brakes_per_km", lambda q, s: q.hard_brakes.to_numpy()[s].sum() / q.km.to_numpy()[s].sum()),
                          ("unnecessary_per_km", lambda q, s: q.hb_unnecessary.to_numpy()[s].sum() / q.km.to_numpy()[s].sum()),
                          ("unnecessary_episodes_per_km", lambda q, s: q.brake_episodes_unnecessary.to_numpy()[s].sum() / q.km.to_numpy()[s].sum()),
                          ("recall_ttc3", lambda q, s: q.pairs_ttc3_seen.to_numpy()[s].sum() / max(q.pairs_ttc3_incam.to_numpy()[s].sum(), 1)),
                          ("routes_min_ttc_lt1", lambda q, s: float((q.min_ttc.to_numpy()[s] < 1.0).mean()))):
                al = np.arange(len(com))
                r[f"v6_minus_{nm}"] = f(a, al) - f(b, al)
                v = [f(a, s) - f(b, s) for s in bs]
                r[f"v6_minus_{nm}_lo"], r[f"v6_minus_{nm}_hi"] = np.percentile(v, 2.5), np.percentile(v, 97.5)
        out.append(r)
    o = pd.DataFrame(out)
    o.to_csv(OUT / "inloop_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(o.round(3).T.to_string())
    print("K30 B2D INLOOP DONE")


if __name__ == "__main__":
    main()
