#!/usr/bin/env python
"""K32 amendment G4 (handoff/K32/STATUS.md): decision fidelity in shadow mode. Every perception branch runs on the frames
of the v6-S and u960 rollouts results/k32/shadow/<rollout>_L0_s0_c<k>.json (+ _save/aeb_<rollout>_logall_L0.jsonl, AEB_LOGALL = 1:
per tick rec["all"][branch] = [action, px, rounds]); the applied action is the rollout branch's. Per rollout source and
branch (unit = route):
  agreement with u960 + comp. (per-frame share of equal actions) and with the privileged requirement a_gt
  flicker (action changes per 100 frames)
  disagreement runs with u960 + comp. (count per 100 frames, mean / p95 / max length in frames)
  onset delay vs u960 + comp.: at every u960 braking episode (run of action >= DECELERATE; runs closer than 1 s merged),
    the branch's first action >= DECELERATE from 1 s before the episode start (not before the previous episode's end),
    minus the start (s; censored at the episode end; miss rate)
Route bootstrap (2,000, seed 0) of each branch's agreement and onset delay, and of v6-S minus each branch.
-> results/k32/tables/shadow_fidelity.csv, shadow_intervals.csv
usage: k32_shadow.py [--dir shadow] [--routes test_routes.xml]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

_s = importlib.util.spec_from_file_location("k32t", Path(__file__).with_name("k32_test.py"))
KT = importlib.util.module_from_spec(_s); _s.loader.exec_module(KT)
DT = KT.DT
REF = "u960"


def runs(mask):
    return KT.runs(np.asarray(mask, bool))


def per_route(T):
    """per branch metrics on one rollout route; T = ticks with rec["all"]"""
    T = [t for t in T if "all" in t and "a_gt" in t]
    if not T:
        return {}
    names = sorted(set().union(*[t["all"].keys() for t in T]))
    A = {nm: np.array([t["all"].get(nm, [np.nan])[0] for t in T], float) for nm in names}
    g = np.array([t["a_gt"] for t in T], float)
    ref = A[REF]
    eps = []                                                    # u960 braking runs, gaps <= 1 s merged (§4's episode rule)
    for s_, e_ in runs(ref >= 1):
        if eps and s_ - eps[-1][1] <= int(round(1.0 / DT)):
            eps[-1] = (eps[-1][0], e_)
        else:
            eps.append((s_, e_))
    out = {}
    for nm in names:
        a = A[nm]
        dis = a != ref
        rl = np.array([e - s for s, e in runs(dis)])
        dl, miss = [], 0
        for q, (s, e) in enumerate(eps):
            lo = max(s - int(round(1.0 / DT)), eps[q - 1][1] if q else 0)      # not before the previous episode's end
            on = np.flatnonzero(a[lo:e] >= 1)
            if len(on):
                dl.append((lo + on[0] - s) * DT)
            else:
                dl.append((e - s) * DT); miss += 1
        out[nm] = {"frames": len(a), "agree_u960": float((~dis).mean()), "agree_gt": float((a == g).mean()),
                   "flicker_per_100": 100.0 * float((a[1:] != a[:-1]).sum()) / max(len(a), 1),
                   "dis_runs_per_100": 100.0 * len(rl) / max(len(a), 1), "dis_run_mean": float(rl.mean()) if len(rl) else 0.0,
                   "dis_run_p95": float(np.percentile(rl, 95)) if len(rl) else 0.0, "dis_run_max": int(rl.max()) if len(rl) else 0,
                   "u960_episodes": len(eps), "onset_delay_vs_u960_s": float(np.mean(dl)) if dl else np.nan,
                   "miss_rate_vs_u960": miss / len(eps) if eps else np.nan}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="shadow")
    ap.add_argument("--routes", default="test_routes.xml")
    a = ap.parse_args()
    info = KT.routes_info(KT.K / a.routes)
    rows = []
    for f in sorted((KT.K / a.dir).glob("*_L0_s0_c[0-9]*.json")):
        m = re.match(r"(.+)_(L[01])_s(\d)_c(\d+)$", f.stem)
        src, st = m.group(1), m.group(2)
        sv = KT.K / a.dir / f"{f.stem}_save" / f"aeb_{src}_logall_{st}.jsonl"           # the pilot's name under AEB_LOGALL
        if not sv.exists():
            continue
        lines = KT.match_lines(sv, KT.K / f"test_routes_c{m.group(4)}.xml")     # run order + start check (k32_test.match_lines)
        for rid, d in lines.items():
            for nm, r in per_route(d["ticks"]).items():
                rows.append({"rollout": src, "route": rid, "branch": nm, **r})
    D = pd.DataFrame(rows)
    KT.OUT.mkdir(parents=True, exist_ok=True)
    met = ["agree_u960", "agree_gt", "flicker_per_100", "dis_runs_per_100", "dis_run_mean", "dis_run_p95", "dis_run_max",
           "onset_delay_vs_u960_s", "miss_rate_vs_u960"]
    S = D.groupby(["rollout", "branch"])[met].mean().reset_index()
    S["routes"] = D.groupby(["rollout", "branch"]).route.nunique().values
    S.to_csv(KT.OUT / "shadow_fidelity.csv", index=False)
    out = []
    for src, g in D.groupby("rollout"):
        P_ = g.pivot_table(index="route", columns="branch", values=["agree_u960", "onset_delay_vs_u960_s"])
        rt = P_.index.to_numpy()
        idx = np.random.default_rng(0).integers(0, len(rt), (2000, len(rt)))
        for mt in ("agree_u960", "onset_delay_vs_u960_s"):
            for br in P_[mt].columns:
                x = P_[mt][br].to_numpy(float)
                v = np.nanmean(x[idx], 1)
                out.append({"rollout": src, "metric": mt, "branch": br, "contrast": "value", "mean": float(np.nanmean(x)),
                            "ci_lo": float(np.nanpercentile(v, 2.5)), "ci_hi": float(np.nanpercentile(v, 97.5))})
                if br != "v6S" and "v6S" in P_[mt].columns:
                    dd = P_[mt]["v6S"].to_numpy(float) - x
                    v = np.nanmean(dd[idx], 1)
                    lo, hi = np.nanpercentile(v, [2.5, 97.5])
                    out.append({"rollout": src, "metric": mt, "branch": br, "contrast": "v6S minus branch", "mean": float(np.nanmean(dd)),
                                "ci_lo": float(lo), "ci_hi": float(hi)})
    pd.DataFrame(out).to_csv(KT.OUT / "shadow_intervals.csv", index=False)
    print("K32 SHADOW TABLES DONE", len(D), "route x branch rows")


if __name__ == "__main__":
    main()
