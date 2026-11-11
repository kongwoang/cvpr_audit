#!/usr/bin/env python
"""K31 item 1 (post-hoc): episode-level safety and temporal consistency on STORED per-frame actions (K28 phase-3
instrumented run results/k28/p3/<ds>_test_instr, setting base, AV2 with the ego mask; no re-run). Definitions as
pre-registered in handoff/K31/STATUS.md (2026-10-19). Writes results/k31/tables/:
  ep_summary.csv     per dataset, level (1 = >= DECELERATE, 2 = HARD-BRAKE) and pipeline: episodes, missed rate, median /
                     mean onset delay (frames, s; all episodes with capped delays, and detected episodes only), false
                     episodes per 1,000 frames, action changes and A -> B -> A reversals per 100 frames (act_gt as reference)
  ep_intervals.csv   v6-S minus each pipeline for every statistic, paired unit bootstrap (2,000, seed 0)
  ep_top_delays.csv  the 10 longest onset delays of v6-S per dataset (level 1), cause at the episode's first frame with the
                     K30 V3 rules (k30_analyze.v3c) on the K30 part-match records results/k30/match
usage: k31_episodes.py
"""
from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from rap import planner as P

B = Path(os.environ.get("B_REAL", os.environ["B"]))
OUT = B / "results" / "k31" / "tables"
NWS = {"nusc": 1600, "av2": 2048, "ahd": 1920}
DT = {"nusc": 0.5, "av2": 0.2, "ahd": 1 / 30}
DSL = {"nusc": "nuScenes new test", "av2": "AV2 val (ego mask)", "ahd": "Argoverse-HD val"}
REF = "v6-S"


def pipes(ds):
    n = NWS[ds]
    return {"cheap": "a_cheap", "u960 + C1 + v3a": "a_wf960+C1|v3a", "native + C1 + v3a": f"a_wf{n}+C1|v3a",
            "hybrid tsp8@S + C1 + v3a": "a_tsp8@S+C1|v3a", "uncertainty unc2@S + C1 + v3a": "a_unc2@S+C1|v3a",
            "v6-S": "a_v6S|v3a", "v6-S without refiner": "a_v6S|none"}


def load(ds):
    cols = ["unit", "frame", "gt", "act_gt", "v"] + list(pipes(ds).values())
    rows = []
    for f in sorted(glob.glob(str(B / "results" / "k28" / "p3" / f"{ds}_test_instr" / "*.pkl"))):
        rows += [{c: r.get(c) for c in cols} for r in pickle.load(open(f, "rb"))]
    d = pd.DataFrame(rows)
    d = d[d["gt"].astype(bool)].sort_values(["unit", "frame"]).reset_index(drop=True)
    return d


def runs(mask):
    """maximal runs of True after merging gaps of <= 1 frame: list of (start, length)"""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out, s, e = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - e <= 2:                     # gap of at most one frame
            e = i
        else:
            out.append((s, e - s + 1)); s = e = i
    out.append((s, e - s + 1))
    return out


def reversals(a):
    n, c = len(a), 0
    for t in range(1, n):
        if a[t] == a[t - 1]:
            continue
        A, Bv, u = a[t - 1], a[t], t + 1
        while u < n and a[u] == Bv:
            u += 1
        if u < n and u <= t + 2 and a[u] == A:
            c += 1
    return c


def unit_stats(g, col, L):
    """per unit: list of (delay, length, missed) per required episode; false episodes; frames; changes; reversals"""
    req = g["act_gt"].to_numpy(int)
    a = g[col].to_numpy(int) if col else req
    eps = []
    for s, n in runs(req >= L):
        hit = np.flatnonzero(a[s:s + n] >= L)
        eps.append((int(hit[0]) if len(hit) else n, n, not len(hit), s))
    false = sum(1 for s, n in runs(a >= L) if (req[s:s + n] < L).all())
    return eps, false, len(a), int((a[1:] != a[:-1]).sum()), reversals(a)


def wmedian(vals, w):
    o = np.argsort(vals, kind="stable")
    v, ww = vals[o], w[..., o]
    cw = np.cumsum(ww, axis=-1)
    tot = cw[..., -1:]
    k = (cw >= 0.5 * tot).argmax(axis=-1)
    return v[k]


def analyse(ds, d):
    units = np.unique(d["unit"].to_numpy())
    nu = len(units)
    rng = np.random.default_rng(0)
    bs = [rng.integers(0, nu, nu) for _ in range(2000)]
    C = np.stack([np.bincount(s, minlength=nu) for s in bs]).astype(float)       # (2000, nu) draw counts
    groups = {u: g for u, g in d.groupby("unit", sort=True)}
    summ, stats = [], {}
    pl = dict(pipes(ds)); pl["required action (act_gt)"] = None
    for L in (1, 2):
        for name, col in pl.items():
            ep_u, ep_d, ep_n, ep_m = [], [], [], []
            fa, fr, ch, rv = np.zeros(nu), np.zeros(nu), np.zeros(nu), np.zeros(nu)
            for i, u in enumerate(units):
                eps, f, nfr, c, r = unit_stats(groups[u], col, L)
                for dl, n, m, _ in eps:
                    ep_u.append(i); ep_d.append(dl); ep_n.append(n); ep_m.append(m)
                fa[i], fr[i], ch[i], rv[i] = f, nfr, c, r
            ep_u, ep_d, ep_m = np.array(ep_u, int), np.array(ep_d, float), np.array(ep_m, bool)
            ne = np.bincount(ep_u, minlength=nu).astype(float)
            nm = np.bincount(ep_u, ep_m.astype(float), minlength=nu)
            sd = np.bincount(ep_u, ep_d, minlength=nu)
            sdd = np.bincount(ep_u, ep_d * ~ep_m, minlength=nu)
            ndet = ne - nm
            det = ~ep_m
            # bootstrap distributions (2000,) of each statistic
            dist = {
                "missed_rate": (C @ nm) / np.maximum(C @ ne, 1e-12),
                "mean_delay_fr": (C @ sd) / np.maximum(C @ ne, 1e-12),
                "mean_delay_det_fr": (C @ sdd) / np.maximum(C @ ndet, 1e-12),
                "median_delay_fr": wmedian(ep_d, C[:, ep_u]) if len(ep_d) else np.full(2000, np.nan),
                "median_delay_det_fr": wmedian(ep_d[det], C[:, ep_u[det]]) if det.any() else np.full(2000, np.nan),
                "false_per_1000": 1000 * (C @ fa) / (C @ fr),
                "changes_per_100": 100 * (C @ ch) / (C @ fr),
                "reversals_per_100": 100 * (C @ rv) / (C @ fr),
            }
            point = {
                "missed_rate": nm.sum() / max(ne.sum(), 1e-12), "mean_delay_fr": sd.sum() / max(ne.sum(), 1e-12),
                "mean_delay_det_fr": sdd.sum() / max(ndet.sum(), 1e-12),
                "median_delay_fr": float(wmedian(ep_d, np.ones(len(ep_d)))) if len(ep_d) else np.nan,
                "median_delay_det_fr": float(wmedian(ep_d[det], np.ones(det.sum()))) if det.any() else np.nan,
                "false_per_1000": 1000 * fa.sum() / fr.sum(), "changes_per_100": 100 * ch.sum() / fr.sum(),
                "reversals_per_100": 100 * rv.sum() / fr.sum(),
            }
            stats[(L, name)] = (point, dist)
            row = {"dataset": DSL[ds], "level": "required (>= DECELERATE)" if L == 1 else "hard brake", "pipeline": name,
                   "units": nu, "frames": int(fr.sum()), "episodes": int(ne.sum()), "missed": int(nm.sum()),
                   "median_ep_len_fr": float(np.median(ep_n)) if ep_n else np.nan}
            for k, v in point.items():
                row[k] = v
                if k.endswith("_fr"):
                    row[k[:-3] + "_s"] = v * DT[ds]
            if col is None:                                          # reference row: only flicker is meaningful
                for k in ("missed_rate", "mean_delay_fr", "mean_delay_det_fr", "median_delay_fr", "median_delay_det_fr",
                          "mean_delay_s", "mean_delay_det_s", "median_delay_s", "median_delay_det_s", "false_per_1000", "missed"):
                    row[k] = np.nan
            summ.append(row)
    ints = []
    for L in (1, 2):
        pr, dr = stats[(L, REF)]
        for name in pipes(ds):
            if name == REF:
                continue
            px, dx = stats[(L, name)]
            for k in pr:
                diff = dr[k] - dx[k]
                lo, hi = np.nanpercentile(diff, [2.5, 97.5])
                sc = DT[ds] if k.endswith("_fr") else 1.0
                ints.append({"dataset": DSL[ds], "level": "required (>= DECELERATE)" if L == 1 else "hard brake",
                             "statistic": k[:-3] + "_s" if k.endswith("_fr") else k, "pipeline": name,
                             "v6S": pr[k] * sc, "pipeline_value": px[k] * sc, "v6S_minus_pipeline": (pr[k] - px[k]) * sc,
                             "ci_lo": lo * sc, "ci_hi": hi * sc,
                             "excludes_0": bool(lo > 0 or hi < 0)})
    return summ, ints, groups


def top_delays(ds, groups):
    pp = P.PlannerParams()
    col = pipes(ds)["v6-S"]
    rec = []
    for u, g in groups.items():
        g = g.reset_index(drop=True)
        eps, *_ = unit_stats(g, col, 1)
        for dl, n, m, s in eps:
            row = {"unit": u, "frame": int(g.frame[s]), "len_fr": n, "delay_fr": dl, "missed": m,
                   "len_s": n * DT[ds], "delay_s": dl * DT[ds], "act_gt_max": int(g.act_gt[s:s + n].max()),
                   "v": float(g.v[s])}
            for name, c in pipes(ds).items():
                if name == "v6-S":
                    continue
                hit = np.flatnonzero(g[c].to_numpy(int)[s:s + n] >= 1)
                row[f"delay_fr {name}"] = int(hit[0]) if len(hit) else n
            rec.append(row)
    t = pd.DataFrame(rec).sort_values(["delay_s", "len_fr", "unit", "frame"], ascending=[False, False, True, True]).head(10)
    m = []
    for f in sorted(glob.glob(str(B / "results" / "k30" / "match" / ds / "*.pkl"))):
        m += pickle.load(open(f, "rb"))
    x = pd.DataFrame(m).set_index(["unit", "frame"])
    gt = pd.read_pickle(B / "results" / "k30" / "gt" / f"{ds}.pkl").set_index(["unit", "frame"])
    out = []
    for rank, (_, r) in enumerate(t.iterrows(), 1):
        k = (r.unit, int(r.frame))
        cause, q = "frame not in the match records", None
        if k in x.index:
            q = x.loc[k]
            gq = gt.loc[k]
            if pd.isna(gq["dec_range"]):
                cause = "no decisive object at onset"
            else:
                found, inc = bool(q.get("dec_found_v6S", False)), bool(q.get("dec_in_crop_v6S", False))
                if not found:
                    cause = "decisive object inside a crop but missed" if inc else "decisive object outside every crop (not found)"
                else:
                    z, lmin, lmax, ttc = q["dec_z_v6S"], q["dec_lmin_v6S"], q["dec_lmax_v6S"], q["dec_ttc_v6S"]
                    a_h = P.discrete_action(P.required_decel([z], [lmin], [lmax], [ttc], r.v, pp)[0], pp)
                    a_r = P.discrete_action(P.required_decel([gq["dec_range"]], [lmin], [lmax], [ttc], r.v, pp)[0], pp)
                    cause = ("range error (decisive object found; its estimated range keeps the demand low)" if a_r > a_h
                             else "ttc / lateral error (found, range not the reason)")
        o = {"dataset": DSL[ds], "rank": rank, **r.to_dict(), "cause": cause, "panel_id": f"{ds}/{r.unit}/{int(r.frame)}"}
        if k in gt.index:
            o.update({"dec_class": gt.loc[k]["dec_class"], "dec_range_gt": gt.loc[k]["dec_range"], "dec_ttc_gt": gt.loc[k]["dec_ttc"]})
        if q is not None:
            o.update({"dec_found_v6S": q.get("dec_found_v6S"), "dec_in_crop_v6S": q.get("dec_in_crop_v6S"),
                      "dec_z_est": q.get("dec_z_v6S"), "dec_ttc_est": q.get("dec_ttc_v6S")})
        out.append(o)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S, I, T = [], [], []
    for ds in ("nusc", "av2", "ahd"):
        d = load(ds)
        s, i, groups = analyse(ds, d)
        S += s; I += i
        T += top_delays(ds, groups)
        print(ds, "units", d.unit.nunique(), "frames", len(d), flush=True)
    pd.DataFrame(S).to_csv(OUT / "ep_summary.csv", index=False)
    pd.DataFrame(I).to_csv(OUT / "ep_intervals.csv", index=False)
    pd.DataFrame(T).to_csv(OUT / "ep_top_delays.csv", index=False)
    print("K31 EPISODES DONE")


if __name__ == "__main__":
    main()
