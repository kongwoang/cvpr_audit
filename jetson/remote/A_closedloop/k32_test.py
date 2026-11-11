#!/usr/bin/env python
"""K32 test tables (pre-registered metrics, handoff/K32/STATUS.md §4 + amendments D, E (E1 / E2'), F) from
results/k32/test/<branch>_<setting>_s<seed>_c<k>.json (+ _save/aeb_<branch>_<setting>.jsonl). Writes results/k32/tables/:
  test_per_route.csv   branch x setting x seed x route: §4 metrics (collision, per km, collision speed, min TTC, onset
                       delay vs privileged-AEB, unnecessary hard-brake episodes, RC, route time, flicker, px, latency),
                       D (preventable / non-preventable collisions), F (missed required episodes, unnecessary hard onsets),
                       shadow agreement with u960 + comp (v6-S)
  test_collisions.csv  every leaderboard collision: contact type, ego speed at contact, demand lead, preventable (D)
  test_summary.csv     per branch and setting (unit = route, seeds averaged within a route)
  test_intervals.csv   v6-S and v6-S <= 2 minus every branch, per setting, paired route bootstrap (2,000, seed 0);
                       privileged-AEB (L0 only) is compared in both settings
  e1_events.csv        E1: decision-change events (frame action != the draft's shadow action), correct / wrong vs a_gt,
                       and the outcome over H = 1, 2, 3 s after correct changes, wrong changes, and frames without a change
                       but with a_gt >= DECELERATE (min corridor TTC, front contact and preventable front contact within H,
                       max a_gt, deceleration achieved, stop without front contact)
  e1_episodes.csv      E1 cross-branch: the k-th required episode (a_gt >= DECELERATE run) of a route, per branch: onset
                       delay of the applied action, outcome over H after the episode onset; e1_intervals.csv: paired
                       route bootstrap of v6-S (and v6-S <= 2) minus each branch on the episodes present in both
  e2_forks.csv         E2' model-based forks at every decision-change event: arm X (X's applied actions) vs arm D (the
                       draft's shadow actions) over 3 s from the logged true states; per branch and setting: share of
                       events where X avoids a contact D has / causes one D avoids / no difference, mean change in
                       minimum gap and braking spent; route bootstrap (2,000, seed 0) in e2_intervals.csv
  g3_primary.csv       amendment G3 primary: onset delay relative to privileged-AEB on avoidable events, paired by route
                       and seed (k-th required episode aligned; avoidable = privileged-AEB's episode has true corridor min
                       TTC >= v / 6 + 0.4 s at its onset; a miss is censored at the episode end), per stratum (all, ego
                       speed at the event >= 8 / < 8 m/s); unit = (route, seed); g3_primary_intervals.csv: v6-S (and
                       v6-S <= 2) minus each branch, paired route x seed bootstrap; g3_events.csv every paired event
  <tag>_intervals_route_seed.csv  secondary metrics with the paired route x seed bootstrap
  <tag>_min_ttc_quantiles.csv     per-route-run min TTC quantiles (5 / 25 / 50 %) per branch and setting
usage: k32_test.py [--test test] (another directory under results/k32 for a check run)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(os.environ.get("B", "/vol/dl-nguyenb5-solar/users/hoangpc/CVPR"))
K = B / "results" / "k32"
OUT = K / "tables"
DT = 0.05
LOC = re.compile(r"x=(-?[\d.]+), y=(-?[\d.]+)")
MAIN = ("v6S", "v6Sb2")
HS = (1, 2, 3)
A_CMD = (0.0, 2.5, 6.0)            # commanded deceleration of KEEP / DECELERATE / HARD-BRAKE (rap.planner)
HALF_W, FRONT = 1.2, 2.45          # corridor half-width; E2' contact gap (the front bumper, amendment B)


def routes_info(xml):
    out = {}
    for r in ET.parse(xml).getroot().findall("route"):
        p = r.find("waypoints").find("position")
        sc = r.find("scenarios").find("scenario")
        tp = sc.find("trigger_point")
        out[r.get("id")] = {"start": (float(p.get("x")), float(p.get("y"))), "type": sc.get("type"),
                            "trigger": (float(tp.get("x")), float(tp.get("y")))}
    return out


def route_of(start, info):
    if not start:
        return None
    best = min(info.items(), key=lambda kv: math.hypot(kv[1]["start"][0] - start["x"], kv[1]["start"][1] - start["y"]))
    return best[0] if math.hypot(best[1]["start"][0] - start["x"], best[1]["start"][1] - start["y"]) < 30.0 else None


def route_list(xml):
    """the chunk's routes in run order: [(id, (x, y))]"""
    out = []
    for r in ET.parse(xml).getroot().findall("route"):
        p = r.find("waypoints").find("position")
        out.append((r.get("id"), (float(p.get("x")), float(p.get("y")))))
    return out


def match_lines(sv, xml, tol=30.0):
    """route id -> logged route line of one job. Lines are written in the chunk's run order, so each line is matched to
    the next route (from the current position) whose start lies within tol m; route starts alone are ambiguous (30 test
    route pairs start < 30 m apart). A line that matches no later route but the previous one is a re-logged route (a
    resumed job): the longer line is kept."""
    rl, out, j = route_list(xml), {}, 0
    for ln in open(sv):
        d = json.loads(ln)
        s = d.get("start")
        if not s:
            continue
        near = lambda k: math.hypot(rl[k][1][0] - s["x"], rl[k][1][1] - s["y"]) < tol  # noqa: E731
        k = next((k for k in range(j, len(rl)) if near(k)), None)
        if k is not None:
            out[rl[k][0]] = d
            j = k + 1
        elif j > 0 and near(j - 1) and len(d.get("ticks", [])) > len(out[rl[j - 1][0]].get("ticks", [])):
            out[rl[j - 1][0]] = d
    return out


def episodes(mask, gap):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return 0
    return 1 + int((np.diff(idx) > gap).sum())


def runs(mask):
    """(start, end) index pairs of the maximal True runs, end exclusive"""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def front_contacts(line, T):
    """front contacts of a route: (tick, preventable by D, ego speed); T = ticks with a_gt"""
    out = []
    tt = np.array([t["t"] for t in T]) if T else np.zeros(0)
    ag = np.array([t["a_gt"] for t in T]) if T else np.zeros(0)
    vv = np.array([t["v"] for t in T]) if T else np.zeros(0)
    for c in (line or {}).get("contacts", []):
        if c["type"] in ("side", "rear"):
            continue
        k = int(np.searchsorted(tt, c["t"], side="right"))           # ticks up to the contact
        on = None
        for j in range(k - 1, -1, -1):
            if ag[j] < 1:
                break
            on = j
        prev = bool(on is not None and (c["t"] - tt[on]) * DT >= vv[on] / 6.0 + 0.4)
        out.append((c["t"], prev, c["v"]))
    return out


def collisions(rec, line, T):
    """amendment D, every leaderboard collision: (actor, type, speed at contact, lead, preventable)"""
    cons = {c["id"]: c for c in (line or {}).get("contacts", [])}
    tt = np.array([t["t"] for t in T]) if T else np.zeros(0)
    ag = np.array([t["a_gt"] for t in T]) if T else np.zeros(0)
    vv = np.array([t["v"] for t in T]) if T else np.zeros(0)
    out = []
    for k in ("collisions_vehicle", "collisions_pedestrian", "collisions_layout"):
        for txt in rec.get("infractions", {}).get(k, []):
            m = re.search(r"id=(\d+)", txt)
            c = cons.get(int(m.group(1))) if m else None
            if c is None:
                out.append({"infraction": k, "actor": None, "contact_type": "unmatched", "v_contact": np.nan, "lead_s": np.nan,
                            "preventable": True})
                continue
            if c["type"] in ("side", "rear"):
                out.append({"infraction": k, "actor": c["id"], "contact_type": c["type"], "v_contact": c["v"], "lead_s": np.nan,
                            "preventable": False})
                continue
            j0 = int(np.searchsorted(tt, c["t"], side="right"))
            on = None
            for j in range(j0 - 1, -1, -1):
                if ag[j] < 1:
                    break
                on = j
            lead = (c["t"] - tt[on]) * DT if on is not None else 0.0
            out.append({"infraction": k, "actor": c["id"], "contact_type": c["type"], "v_contact": c["v"], "lead_s": lead,
                        "preventable": bool(on is not None and lead >= vv[on] / 6.0 + 0.4)})
    return out


def outcome(T, i, fc, h):
    """E1 outcome over h s after tick index i (true states)"""
    j = min(i + int(round(h / DT)), len(T) - 1)
    w = T[i:j + 1]
    t0, t1 = T[i]["t"], T[j]["t"]
    mt = [x["min_ttc"] for x in w if x.get("min_ttc") is not None]
    fcw = [c for c in fc if t0 < c[0] <= t1]
    v = np.array([x["v"] for x in w], float)
    return {"min_ttc": min(min(mt) if mt else 10.0, 10.0), "front_contact": float(bool(fcw)),
            "preventable_front_contact": float(any(c[1] for c in fcw)), "max_a_gt": max(x["a_gt"] for x in w),
            "decel": (v[0] - v[-1]) / max((t1 - t0) * DT, DT), "stop": float(v.min() < 0.5 and not fcw)}


def fork(act, v0, ax, ad):
    """E2': both arms from the true states at the event tick; act rows [id, near, far, lmin, lmax, closing, lat_rate]"""
    A = np.asarray(act, float).reshape(-1, 7)
    out = []
    for acts in (ax, ad):
        u = np.array([A_CMD[int(a)] for a in acts], float)
        n = len(u)
        v = np.empty(n + 1); v[0] = v0
        for k in range(n):
            v[k + 1] = max(v[k] - u[k] * DT, 0.0)
        s = np.r_[0.0, np.cumsum((v[:-1] + v[1:]) / 2 * DT)]
        res = {"contact": 0.0, "min_gap": 60.0, "min_ttc": 10.0, "v_final": float(v[-1]), "braking": float((u * DT).sum())}
        if len(A):
            k_ = np.arange(n + 1)[:, None] * DT
            va = v0 - A[:, 5]
            near, far = A[:, 1] + va * k_ - s[:, None], A[:, 2] + va * k_ - s[:, None]
            lmin, lmax = A[:, 3] + A[:, 6] * k_, A[:, 4] + A[:, 6] * k_
            inc = (lmin < HALF_W) & (lmax > -HALF_W) & (far > 0)
            gap = np.where(inc, near - FRONT, np.inf)
            res["contact"] = float((gap <= 0).any())
            res["min_gap"] = float(min(gap.min(), 60.0))
            cl = v[:, None] - va[None, :]
            ttc = np.where(inc & (cl > 0.1) & (gap > 0), gap / np.maximum(cl, 1e-6), np.inf)
            res["min_ttc"] = float(min(ttc.min(), 10.0))
        out.append(res)
    return out


def per_route(rec, line, info_r, keys):
    T = [t for t in line["ticks"] if "a_gt" in t]
    t = pd.DataFrame(T)
    v = t["v"].to_numpy(float)
    km = float(np.sum(v) * DT / 1000.0)
    xy = np.array(t["xy"].tolist(), float)
    a = t["a_applied"].to_numpy(int)
    g = t["a_gt"].to_numpy(int)
    out = {"km_driven": km, "ticks": len(t), "route_time_s": len(t) * DT}
    cl = collisions(rec, line, T)
    for c in cl:
        c.update(keys)
    out.update({"collision": bool(cl), "n_collisions": len(cl),
                "coll_speed_mean": float(np.nanmean([c["v_contact"] for c in cl])) if cl and not all(np.isnan(c["v_contact"]) for c in cl) else np.nan,
                "preventable_collision": float(any(c["preventable"] for c in cl)),
                "nonpreventable_collision": float(any(not c["preventable"] for c in cl)),
                "n_preventable": sum(c["preventable"] for c in cl),
                "RC": rec["scores"]["score_route"], "DS": rec["scores"]["score_composed"], "status": rec["status"]})
    mt = [x for x in t["min_ttc"] if x is not None and not (isinstance(x, float) and math.isnan(x))]
    out["min_ttc"] = float(min(min(mt), 10.0)) if mt else 10.0
    out["min_ttc_lt1"] = out["min_ttc"] < 1.0
    trig = info_r["trigger"]
    dtr = np.hypot(xy[:, 0] - trig[0], xy[:, 1] - trig[1]) if len(xy) else np.array([])
    ti = int(np.argmax(dtr < 10.0)) if (dtr < 10.0).any() else None
    if ti is None and len(dtr):
        ti = int(np.argmin(dtr))
    out["trigger_tick"] = ti
    on = np.flatnonzero(a[ti:] >= 1) if ti is not None else []
    out["onset_ticks_after_trigger"] = int(on[0]) if len(on) else np.nan
    out["unnecessary_hb_episodes"] = episodes((a == 2) & (g < 2), int(1.0 / DT))
    out["unnecessary_hb_per_km"] = out["unnecessary_hb_episodes"] / km if km > 0 else np.nan
    out["flicker_per_100"] = 100.0 * float((a[1:] != a[:-1]).sum()) / max(len(a), 1)
    # F
    out["missed_required_episodes"] = sum(1 for s_, e_ in runs(g >= 1) if not (a[s_:e_] >= 1).any())
    out["unnecessary_hard_onsets"] = int(((a[1:] == 2) & (a[:-1] < 2) & (g[1:] == 0)).sum())
    for c in ("px", "lat_ms", "rounds"):
        if c in t:
            out[f"{c}_mean"] = float(pd.to_numeric(t[c], errors="coerce").mean())
    if "a_shadow_u960" in t:
        ok = t["a_shadow_u960"].notna().to_numpy()
        af, sh = t["a_frame"].to_numpy(int)[ok], t["a_shadow_u960"].to_numpy(float)[ok].astype(int)
        dis = af != sh
        rl = np.array([e_ - s_ for s_, e_ in runs(dis)])
        out.update({"shadow_agreement": float((~dis).mean()) if len(dis) else np.nan, "shadow_dis_runs": len(rl),
                    "shadow_dis_run_mean_ticks": float(rl.mean()) if len(rl) else 0.0,
                    "shadow_dis_run_max_ticks": int(rl.max()) if len(rl) else 0,
                    "shadow_v6S_higher": float((af > sh).mean()) if len(af) else np.nan,
                    "shadow_v6S_lower": float((af < sh).mean()) if len(af) else np.nan})
    fc = front_contacts(line, T)
    # E1 events and E2' forks
    ev, fk = [], []
    if "a_shadow_cheap" in t and t["a_shadow_cheap"].notna().any():
        dr = t["a_shadow_cheap"].to_numpy(float)
        af = t["a_frame"].to_numpy(int)
        for i in range(len(T)):
            if np.isnan(dr[i]):
                continue
            d_ = int(dr[i])
            if af[i] != d_:
                kind = "correct" if abs(af[i] - g[i]) < abs(d_ - g[i]) else ("wrong" if abs(af[i] - g[i]) > abs(d_ - g[i]) else "tie")
            elif g[i] >= 1:
                kind = "no_change_required"
            else:
                continue
            r_ = {"kind": kind, "tick": int(T[i]["t"])}
            for h in HS:
                r_.update({f"{k_}_H{h}": v_ for k_, v_ in outcome(T, i, fc, h).items()})
            ev.append(r_)
            if kind in ("correct", "wrong", "tie"):
                j = min(i + int(round(3.0 / DT)), len(T) - 1)
                dd = [int(x) if not np.isnan(x) else int(a[i + k_]) for k_, x in enumerate(dr[i:j + 1])]
                X, D = fork(T[i].get("act", []), float(v[i]), a[i:j + 1], dd)
                fk.append({"kind": kind, "tick": int(T[i]["t"]), **{f"{k_}_X": v_ for k_, v_ in X.items()},
                           **{f"{k_}_D": v_ for k_, v_ in D.items()}})
    # E1 cross-branch episodes
    eps = []
    for k_, (s_, e_) in enumerate(runs(g >= 1)):
        on_ = np.flatnonzero(a[s_:e_] >= 1)
        r_ = {"episode": k_, "start_tick": int(T[s_]["t"]), "length_s": (e_ - s_) * DT, "missed": float(not len(on_)),
              "onset_delay_s": float(on_[0] * DT) if len(on_) else np.nan, "v_onset": float(v[s_]),
              "ttc_onset": float(T[s_]["min_ttc"]) if T[s_].get("min_ttc") is not None else np.nan}
        for h in HS:
            r_.update({f"{q}_H{h}": w for q, w in outcome(T, s_, fc, h).items()})
        eps.append(r_)
    return out, cl, ev, fk, eps


def boot(R, a_br, b_br, st, met, rng_seed=0, key="route", priv_l0=True):
    """paired route bootstrap of a_br - b_br on routes present in both (privileged-AEB is read at L0 when priv_l0)"""
    a = R[(R.branch == a_br) & (R.setting == st)].set_index(key)
    b = R[(R.branch == b_br) & (R.setting == ("L0" if priv_l0 and b_br == "privileged" else st))].set_index(key)
    com = sorted(set(a.index) & set(b.index))
    out = []
    if not com:
        return out
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(com), (2000, len(com)))
    for mt in met:
        if mt not in a or mt not in b:
            continue
        x, y = a.loc[com, mt].to_numpy(float), b.loc[com, mt].to_numpy(float)
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < 3:
            continue
        dd = np.where(ok, x - y, np.nan)
        v = np.nanmean(dd[idx], 1)
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        out.append({"setting": st, "main": a_br, "branch": b_br, "metric": mt, "routes": int(ok.sum()), "main_value": float(np.nanmean(x[ok])),
                    "branch_value": float(np.nanmean(y[ok])), "diff": float(np.nanmean(dd)), "ci_lo": float(lo), "ci_hi": float(hi),
                    "excludes_0": bool(lo > 0 or hi < 0)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="test")
    ap.add_argument("--routes", default="test_routes.xml")
    ap.add_argument("--tag", default="test")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    info = routes_info(K / args.routes)
    rows, CL, EV, FK, EP = [], [], [], [], []
    for f in sorted((K / args.test).glob("*_s[0-9]_c[0-9]*.json")):
        m = re.match(r"(.+)_(L[01])_s(\d)_c(\d+)$", f.stem)
        if not m:
            continue
        br, st, seed = m.group(1), m.group(2), int(m.group(3))
        recs = {r["route_id"].split("_")[1]: r for r in json.load(open(f))["_checkpoint"]["records"]}
        sv = K / args.test / f"{f.stem}_save" / f"aeb_{br}_{st}.jsonl"
        if not sv.exists():
            continue
        lines = match_lines(sv, K / f"test_routes_c{m.group(4)}.xml")
        for rid, d in lines.items():
            if rid not in recs:
                continue
            keys = {"branch": br, "setting": st, "seed": seed, "route": rid, "type": info[rid]["type"]}
            pr, cl, ev, fk, eps = per_route(recs[rid], d, info[rid], keys)
            rows.append({**keys, **pr})
            CL += cl
            EV += [{**keys, **e} for e in ev]
            FK += [{**keys, **e} for e in fk]
            EP += [{**keys, **e} for e in eps]
    pre = "" if args.tag == "test" else f"{args.tag}_"
    T = pd.DataFrame(rows)
    pr = T[T.branch == "privileged"][["route", "seed", "onset_ticks_after_trigger"]].rename(columns={"onset_ticks_after_trigger": "priv_onset"})
    T = T.merge(pr, on=["route", "seed"], how="left")
    T["onset_delay_s"] = (T["onset_ticks_after_trigger"] - T["priv_onset"]) * DT
    T["n_coll_per_km"] = T["n_collisions"] / T["km_driven"].replace(0, np.nan)
    T.to_csv(OUT / f"{args.tag}_per_route.csv", index=False)
    pd.DataFrame(CL).to_csv(OUT / f"{args.tag}_collisions.csv", index=False)
    met = ["collision", "preventable_collision", "nonpreventable_collision", "n_collisions", "n_coll_per_km", "coll_speed_mean",
           "min_ttc_lt1", "min_ttc", "onset_delay_s", "unnecessary_hb_per_km", "missed_required_episodes", "unnecessary_hard_onsets",
           "RC", "route_time_s", "flicker_per_100", "px_mean", "lat_ms_mean", "km_driven"]
    R = T.groupby(["branch", "setting", "route"])[met].mean(numeric_only=True).reset_index()
    S = R.groupby(["branch", "setting"])[met].mean(numeric_only=True).reset_index()
    S["routes"] = R.groupby(["branch", "setting"]).route.nunique().values
    S["collisions_per_km_pooled"] = (T.groupby(["branch", "setting"]).n_collisions.sum() / T.groupby(["branch", "setting"]).km_driven.sum()).values
    S.to_csv(OUT / f"{args.tag}_summary.csv", index=False)
    ints = []
    for st in ("L0", "L1"):
        for mb in MAIN:
            for br in sorted(R.branch.unique()):
                if br != mb:
                    ints += boot(R, mb, br, st, met)
    pd.DataFrame(ints).to_csv(OUT / f"{args.tag}_intervals.csv", index=False)
    sh = T[T.branch == "v6S"]
    if "shadow_agreement" in sh:
        sh.groupby("setting")[["shadow_agreement", "shadow_dis_run_mean_ticks", "shadow_dis_run_max_ticks", "shadow_v6S_higher",
                               "shadow_v6S_lower"]].mean().reset_index().to_csv(OUT / f"{args.tag}_shadow.csv", index=False)
    # E1 events
    E = pd.DataFrame(EV)
    if len(E):
        oc = [c for c in E.columns if c.endswith(tuple(f"_H{h}" for h in HS))]
        cnt = E.groupby(["branch", "setting", "kind"]).size().unstack(fill_value=0).add_prefix("n_")
        Om = E.groupby(["branch", "setting", "kind"])[oc].mean().reset_index()
        Om.merge(cnt.reset_index(), on=["branch", "setting"], how="left").to_csv(OUT / f"{pre}e1_events.csv", index=False)
    # E1 episodes, aligned by (route, seed, setting, k); privileged-AEB is L0 only
    P_ = pd.DataFrame(EP)
    if len(P_):
        P_.to_csv(OUT / f"{pre}e1_episodes.csv", index=False)
        em = ["missed", "onset_delay_s"] + [f"{q}_H{h}" for h in HS for q in ("min_ttc", "front_contact", "preventable_front_contact",
                                                                                "decel", "stop")]
        ei = []
        for st in ("L0", "L1"):
            for mb in MAIN:
                a_ = P_[(P_.branch == mb) & (P_.setting == st)]
                for br in sorted(P_.branch.unique()):
                    if br == mb:
                        continue
                    b_ = P_[(P_.branch == br) & (P_.setting == (st if br != "privileged" else "L0"))]
                    j_ = a_.merge(b_, on=["route", "seed", "episode"], suffixes=("_a", "_b"))
                    if not len(j_):
                        continue
                    Ra = j_.groupby("route")[[f"{x}_a" for x in em]].mean()
                    Rb = j_.groupby("route")[[f"{x}_b" for x in em]].mean()
                    Ra.columns, Rb.columns = em, em
                    RR = pd.concat([Ra.assign(branch=mb), Rb.assign(branch=br)]).reset_index().assign(setting=st)
                    ei += [dict(r, episodes=len(j_)) for r in boot(RR, mb, br, st, em, priv_l0=False)]
        pd.DataFrame(ei).to_csv(OUT / f"{pre}e1_intervals.csv", index=False)
    # E2' forks
    F_ = pd.DataFrame(FK)
    if len(F_):
        F_["avoids"] = ((F_.contact_D > 0) & (F_.contact_X == 0)).astype(float)
        F_["causes"] = ((F_.contact_X > 0) & (F_.contact_D == 0)).astype(float)
        F_["same"] = 1.0 - F_.avoids - F_.causes
        F_["d_min_gap"] = F_.min_gap_X - F_.min_gap_D
        F_["d_min_ttc"] = F_.min_ttc_X - F_.min_ttc_D
        F_["d_braking"] = F_.braking_X - F_.braking_D
        fm = ["avoids", "causes", "same", "d_min_gap", "d_min_ttc", "d_braking"]
        Sm = F_.groupby(["branch", "setting"])[fm].mean().reset_index()
        Sm["events"] = F_.groupby(["branch", "setting"]).size().values
        Sm.to_csv(OUT / f"{pre}e2_forks.csv", index=False)
        e2 = []
        for (br, st), g_ in F_.groupby(["branch", "setting"]):
            Rr = g_.groupby("route")[fm].mean()
            idx = np.random.default_rng(0).integers(0, len(Rr), (2000, len(Rr)))
            for mt in fm:
                x = Rr[mt].to_numpy(float)
                lo, hi = np.percentile(x[idx].mean(1), [2.5, 97.5])
                e2.append({"branch": br, "setting": st, "metric": mt, "routes": len(Rr), "events": len(g_), "mean_over_routes": float(x.mean()),
                           "ci_lo": float(lo), "ci_hi": float(hi), "excludes_0": bool(lo > 0 or hi < 0)})
        pd.DataFrame(e2).to_csv(OUT / f"{pre}e2_intervals.csv", index=False)
    # G3 (amendment G): primary = onset delay relative to privileged-AEB on avoidable events, paired by route and seed
    if len(P_) and (P_.branch == "privileged").any():
        P_["delay_c"] = P_.onset_delay_s.fillna(P_.length_s)                 # a miss is censored at the episode end
        pv = P_[P_.branch == "privileged"][["route", "seed", "episode", "delay_c", "v_onset", "ttc_onset"]].rename(
            columns={"delay_c": "delay_priv", "v_onset": "v_priv", "ttc_onset": "ttc_priv"})
        J = P_[P_.branch != "privileged"].merge(pv, on=["route", "seed", "episode"])
        J = J[J.ttc_priv.isna() | (J.ttc_priv >= J.v_priv / 6.0 + 0.4)].copy()     # avoidable on privileged-AEB's episode
        J["delay_rel"] = J.delay_c - J.delay_priv
        J["unit"] = J.route.astype(str) + "_s" + J.seed.astype(str)
        J.to_csv(OUT / f"{pre}g3_events.csv", index=False)
        strata = (("all", lambda x: np.ones(len(x), bool)), (">=8 m/s", lambda x: (x.v_priv >= 8.0).to_numpy()),
                  ("<8 m/s", lambda x: (x.v_priv < 8.0).to_numpy()))
        gs, gi = [], []
        for sn, f_ in strata:
            Js = J[f_(J)]
            U = Js.groupby(["branch", "setting", "unit"]).agg(delay_rel=("delay_rel", "mean"), missed=("missed", "mean")).reset_index()
            for (br, st), g_ in U.groupby(["branch", "setting"]):
                x = g_.delay_rel.to_numpy(float)
                idx = np.random.default_rng(0).integers(0, len(x), (2000, len(x)))
                lo, hi = np.percentile(x[idx].mean(1), [2.5, 97.5])
                gs.append({"stratum": sn, "branch": br, "setting": st, "units": len(g_), "events": int(((Js.branch == br) & (Js.setting == st)).sum()),
                           "delay_rel_mean": float(x.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "miss_rate": float(g_.missed.mean())})
            for st in ("L0", "L1"):
                for mb in MAIN:
                    for br in sorted(U.branch.unique()):
                        if br != mb:
                            gi += [dict(r, stratum=sn) for r in boot(U.rename(columns={"unit": "u"}), mb, br, st, ["delay_rel", "missed"], key="u",
                                                                     priv_l0=False)]
        pd.DataFrame(gs).to_csv(OUT / f"{pre}g3_primary.csv", index=False)
        pd.DataFrame(gi).to_csv(OUT / f"{pre}g3_primary_intervals.csv", index=False)
    # G3 secondary: paired route x seed bootstrap; min-TTC distribution
    T["unit"] = T.route.astype(str) + "_s" + T.seed.astype(str)
    sec = ["preventable_collision", "unnecessary_hb_per_km", "min_ttc", "collision", "nonpreventable_collision"]
    gi2 = []
    for st in ("L0", "L1"):
        for mb in MAIN:
            for br in sorted(T.branch.unique()):
                if br != mb:
                    gi2 += boot(T, mb, br, st, sec, key="unit")
    pd.DataFrame(gi2).to_csv(OUT / f"{args.tag}_intervals_route_seed.csv", index=False)
    T.groupby(["branch", "setting"]).min_ttc.quantile([0.05, 0.25, 0.5]).unstack().add_prefix("min_ttc_q").reset_index().to_csv(
        OUT / f"{args.tag}_min_ttc_quantiles.csv", index=False)
    print("K32 TEST TABLES DONE", len(T), "route runs;", len(E), "E1 rows;", len(F_), "E2' forks;", len(P_), "episodes")


if __name__ == "__main__":
    main()
