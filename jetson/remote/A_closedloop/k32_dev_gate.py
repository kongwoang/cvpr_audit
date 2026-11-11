#!/usr/bin/env python
"""K32 dev gate (pre-registered in handoff/K32/STATUS.md): from the dev runs
  results/k32/dev/logall_c<k>.json (+ _save/aeb_privileged_logall_L0.jsonl)  privileged-AEB driving, every branch logged
  results/k32/dev/cheap_c<k>.json  (+ _save/aeb_cheap_L0.jsonl)               cheap branch driving (L0)
writes results/k32/tables/dev_lift_error.csv, dev_routes_outcome.csv, dev_gate.csv:
  (a) per branch: median |log(range_est / range_true)| on all matched actors and on the refined ones (with vs without the
      refiner on the same detections); median |TTC_est - TTC_true| (estimates capped at 20 s) on matched actors from a
      high-resolution pass with true TTC < 10 s, C1 vs box growth; pooled over the branches that use the component ->
      rule: disable the refiner if it increases the median range error, C1 if it increases the median TTC error
  (b) privileged-AEB collision share of the dev routes (<= 5%)
  (c) cheap - privileged: collision rate and mean per-route minimum TTC (capped at 10 s), paired route bootstrap
      (2,000, seed 0); passes if either interval excludes 0
usage (any env with pandas): k32_dev_gate.py [--dev dev2 --priv privileged --no_a]
  dev2 (amendment A, handoff/K32/STATUS.md): privileged-AEB without the log run (rule (a) is decided: components off), the
  same thresholds (b), (c); tables dev2_gate.csv, dev2_routes_outcome.csv and dev2_collisions.csv (every leaderboard
  collision matched to the agent's contact log by actor id: type, ego speed, hold state)
"""
from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ.get("B", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
K = B / "results" / "k32"
OUT = K / "tables"
COMP_BRANCHES = ("u960", "native", "v6S", "v6Sb2", "hybrid")


def route_starts(xml):
    out = {}
    for r in ET.parse(xml).getroot().findall("route"):
        p = r.find("waypoints").find("position")
        out[r.get("id")] = (float(p.get("x")), float(p.get("y")))
    return out


def route_of(start, starts):
    if not start:
        return None
    best = min(starts.items(), key=lambda kv: math.hypot(kv[1][0] - start["x"], kv[1][1] - start["y"]))
    return best[0] if math.hypot(best[1][0] - start["x"], best[1][1] - start["y"]) < 30.0 else None


def records(ckpt):
    js = json.load(open(ckpt))
    out = {}
    for r in js["_checkpoint"]["records"]:
        rid = r["route_id"].split("_")[1]
        inf = r.get("infractions", {})
        out[rid] = {"collision": any(len(inf.get(k, [])) for k in ("collisions_vehicle", "collisions_pedestrian", "collisions_layout")),
                    "coll_vehicle": len(inf.get("collisions_vehicle", [])), "coll_ped": len(inf.get("collisions_pedestrian", [])),
                    "coll_layout": len(inf.get("collisions_layout", [])), "RC": r["scores"]["score_route"], "status": r["status"],
                    "km": float(r.get("meta", {}).get("route_length", np.nan)) / 1000.0}
    return out


def jsonl(path, starts):
    out = {}
    if not path.exists():
        return out
    for line in open(path):
        d = json.loads(line)
        rid = route_of(d.get("start"), starts)
        if rid is not None:
            out[rid] = d
    return out


def preventable(rec, line):
    """amendment D: per leaderboard collision of a route, (actor, type, lead, v, preventable)"""
    import re as _re
    cons = {c["id"]: c for c in (line or {}).get("contacts", [])}
    ticks = (line or {}).get("ticks", [])
    out = []
    for k in ("collisions_vehicle", "collisions_pedestrian", "collisions_layout"):
        for txt in rec.get("infractions", {}).get(k, []):
            m = _re.search(r"id=(\d+)", txt)
            c = cons.get(int(m.group(1))) if m else None
            if c is None:
                out.append((None, "unmatched", None, None, True)); continue
            if c["type"] in ("side", "rear"):
                out.append((c["id"], c["type"], None, c["v"], False)); continue
            before = [t for t in ticks if t.get("t", 0) <= c["t"] and "a_gt" in t]
            onset = None
            for t in reversed(before):
                if t["a_gt"] < 1:
                    break
                onset = t
            if onset is None:                                   # no demand at all before contact
                out.append((c["id"], c["type"], 0.0, c["v"], False)); continue
            lead = (c["t"] - onset["t"]) * 0.05
            out.append((c["id"], c["type"], lead, onset["v"], bool(lead >= onset["v"] / 6.0 + 0.4)))
    return out


def boot_diff(a, b, n=2000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(a), len(a)) for _ in range(n)]
    v = [np.mean(a[i] - b[i]) for i in idx]
    return float(np.mean(a - b)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    import argparse
    import re
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="dev")
    ap.add_argument("--priv", default="logall")
    ap.add_argument("--no_a", action="store_true")
    ap.add_argument("--preventable", action="store_true", help="amendment D: (b) counts preventable collisions only")
    args = ap.parse_args()
    pre = "" if args.dev == "dev" else f"{args.dev}_"
    OUT.mkdir(parents=True, exist_ok=True)
    starts = route_starts(K / "dev_routes.xml")
    rp, rc, lp, lc = {}, {}, {}, {}
    pj = "aeb_privileged_logall_L0.jsonl" if args.priv == "logall" else "aeb_privileged_L0.jsonl"
    for f in sorted((K / args.dev).glob(f"{args.priv}_c*.json")):
        rp.update(records(f))
        lp.update(jsonl(K / args.dev / f"{f.stem}_save" / pj, starts))
    for f in sorted((K / args.dev).glob("cheap_c*.json")):
        rc.update(records(f))
        lc.update(jsonl(K / args.dev / f"{f.stem}_save" / "aeb_cheap_L0.jsonl", starts))
    PREV = {}
    if args.preventable:
        prow = []
        for f in sorted((K / args.dev).glob(f"{args.priv}_c*.json")):
            for r in json.load(open(f))["_checkpoint"]["records"]:
                rid = r["route_id"].split("_")[1]
                cl = preventable(r, lp.get(rid))
                PREV[rid] = any(x[4] for x in cl)
                for x in cl:
                    prow.append({"route": rid, "actor": x[0], "type": x[1], "lead_s": x[2], "v_onset": x[3], "preventable": x[4]})
        pd.DataFrame(prow).to_csv(OUT / f"{pre}collisions_preventable.csv", index=False)
    # ---- collision typing (amendment A): leaderboard collisions matched to the agent's contact log by actor id
    crow = []
    for br, R_, L_ in (("privileged", rp, lp), ("cheap", rc, lc)):
        for f in sorted((K / args.dev).glob(f"{args.priv if br == 'privileged' else 'cheap'}_c*.json")):
            for r in json.load(open(f))["_checkpoint"]["records"]:
                rid = r["route_id"].split("_")[1]
                cons = {c["id"]: c for c in L_.get(rid, {}).get("contacts", [])}
                for k_ in ("collisions_vehicle", "collisions_pedestrian"):
                    for txt in r.get("infractions", {}).get(k_, []):
                        m = re.search(r"id=(\d+)", txt)
                        c = cons.get(int(m.group(1))) if m else None
                        crow.append({"branch": br, "route": rid, "kind": k_, "actor": int(m.group(1)) if m else None,
                                     "type": c["type"] if c else "unmatched", "v_ego": c["v"] if c else np.nan,
                                     "hold": c["hold"] if c else np.nan, "text": txt[:120]})
    if crow:
        pd.DataFrame(crow).to_csv(OUT / f"{pre}collisions.csv", index=False)
    # ---- (a) lift error
    rows = []
    for rid, d in lp.items():
        for t in d["ticks"]:
            for br, lst in (t.get("err") or {}).items():
                for aid, near, ttc_t, z0, z1, tc1, tg, ff, refd in lst:
                    if near is None or near <= 0.5 or near > 80.0:
                        continue
                    rows.append({"route": rid, "t": t["t"], "branch": br, "actor": aid, "near": near, "ttc_true": ttc_t,
                                 "z_noref": z0, "z_ref": z1, "ttc_c1": tc1, "ttc_growth": tg, "from_full": ff, "refined": refd})
    E = pd.DataFrame(rows)
    lift = []
    gate = {}
    if len(E):
        E["lre_noref"] = np.abs(np.log(np.maximum(E.z_noref, 0.1) / E.near))
        E["lre_ref"] = np.abs(np.log(np.maximum(E.z_ref, 0.1) / E.near))
        cap = lambda x: np.minimum(pd.to_numeric(x, errors="coerce").fillna(np.inf), 20.0)          # noqa: E731
        tt = pd.to_numeric(E.ttc_true, errors="coerce")
        E["ttc_ok"] = E.from_full.astype(bool) & tt.notna() & (tt < 10.0)
        E["tte_c1"] = np.abs(cap(E.ttc_c1) - np.minimum(tt, 20.0))
        E["tte_growth"] = np.abs(cap(E.ttc_growth) - np.minimum(tt, 20.0))
        for br, g in list(E.groupby("branch")) + [("pooled (u960, native, v6S, v6Sb2, hybrid)", E[E.branch.isin(COMP_BRANCHES)])]:
            r_ = g[g.refined.astype(bool)]
            t_ = g[g.ttc_ok]
            lift.append({"branch": br, "matched": len(g), "median_lre_all_noref": g.lre_noref.median(), "median_lre_all_ref": g.lre_ref.median(),
                         "refined": len(r_), "median_lre_refined_noref": r_.lre_noref.median() if len(r_) else np.nan,
                         "median_lre_refined_ref": r_.lre_ref.median() if len(r_) else np.nan,
                         "ttc_n": len(t_), "median_tte_c1": t_.tte_c1.median() if len(t_) else np.nan,
                         "median_tte_growth": t_.tte_growth.median() if len(t_) else np.nan})
        pl = lift[-1]
        gate["refiner_on"] = bool(not (pl["median_lre_refined_ref"] > pl["median_lre_refined_noref"])) if pl["refined"] else True
        gate["c1_on"] = bool(not (pl["median_tte_c1"] > pl["median_tte_growth"])) if pl["ttc_n"] else True
        gate["a_table_produced"] = True
    else:
        gate["a_table_produced"] = False
    if args.no_a:
        gate = {"refiner_on": False, "c1_on": False, "a_table_produced": True, "a_note": "decided by the first gate (dev_gate.csv)"}
    else:
        pd.DataFrame(lift).to_csv(OUT / "dev_lift_error.csv", index=False)
    # ---- (b), (c)
    common = sorted(set(rp) & set(rc))
    mt = lambda d: min([t["min_ttc"] for t in d["ticks"] if t.get("min_ttc") is not None] or [10.0])   # noqa: E731
    out = []
    for rid in sorted(set(rp) | set(rc)):
        out.append({"route": rid, "priv_collision": rp.get(rid, {}).get("collision"), "cheap_collision": rc.get(rid, {}).get("collision"),
                    "priv_RC": rp.get(rid, {}).get("RC"), "cheap_RC": rc.get(rid, {}).get("RC"),
                    "priv_min_ttc": min(mt(lp[rid]), 10.0) if rid in lp else np.nan, "cheap_min_ttc": min(mt(lc[rid]), 10.0) if rid in lc else np.nan})
    O = pd.DataFrame(out)
    O.to_csv(OUT / f"{pre or 'dev_'}routes_outcome.csv", index=False)
    gate["dev_routes_priv"] = len(rp)
    gate["priv_collision_share"] = float(np.mean([v["collision"] for v in rp.values()])) if rp else np.nan
    if args.preventable:
        gate["priv_preventable_collision_share"] = float(np.mean([PREV.get(r_, False) for r_ in rp])) if rp else np.nan
        gate["b_pass"] = bool(gate["priv_preventable_collision_share"] <= 0.05) if rp else False
    else:
        gate["b_pass"] = bool(gate["priv_collision_share"] <= 0.05) if rp else False
    c = O[O.route.isin(common)].dropna(subset=["priv_min_ttc", "cheap_min_ttc"])
    if len(c):
        d1 = boot_diff(c.cheap_collision.astype(float), c.priv_collision.astype(float))
        d2 = boot_diff(c.cheap_min_ttc, c.priv_min_ttc)
        gate.update({"paired_routes": len(c), "coll_diff": d1[0], "coll_lo": d1[1], "coll_hi": d1[2],
                     "minttc_diff": d2[0], "minttc_lo": d2[1], "minttc_hi": d2[2]})
        gate["c_pass"] = bool(d1[1] > 0 or d1[2] < 0 or d2[1] > 0 or d2[2] < 0)
    else:
        gate["c_pass"] = False
    gate["GATE_PASS"] = bool(gate["a_table_produced"] and gate["b_pass"] and gate["c_pass"])
    pd.DataFrame([gate]).to_csv(OUT / f"{pre or 'dev_'}gate.csv", index=False)
    print(json.dumps(gate, indent=1, default=str))
    print("K32 DEV GATE EVALUATED")


if __name__ == "__main__":
    main()
