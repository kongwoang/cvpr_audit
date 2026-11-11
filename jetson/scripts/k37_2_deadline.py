#!/usr/bin/env python
"""K37-2 (post-hoc, CPU, existing Jetson logs; an APPROXIMATION: the rows come from separate runs with independent
histories). Decision quality under deadlines D in {33.3, 50, 66.7, 83.3, 100} ms on the K29 frames (nuScenes test 20
scenes, AV2 8 logs with the ego mask; 3 repetitions).
Rows and sources (trt32 fast path):
  cheap, u960 + C1 + v3a, native + C1 + v3a   K29 runs r{1,2,3}b (columns cheap / u960+c1comp@trt32 / native+c1comp@trt32)
  hybrid tsp8@x1.0 + C1 + v3a                 K30-J1 runs j1x_r{1,2,3} (tsp8S+C1@trt32 in the x1.0-budget runs)
  v6-S, v6-S <= 2 batches, B2K-A               K34-3 jetson_frames.csv (reps 1-3)
Rules, per row, repetition and unit, over the timed frames in order:
  (a) hold: if latency > D, the frame's action is the row's last on-time action (KEEP before the first one).
  (b) anytime (DESP only): if v6-S is late, the v6-S <= 2 action of the same repetition if it is on time, else the
      cheap action of the frame (latency = the draft time: the cheap latency of the K29 run of that repetition; if even
      that is late, hold).
Cost: K28 decision_cost(action, a_gt, previous cheap action) with the stored GT demand a_gt (results/k30/gt); checked
first: recomputing every stored action's cost reproduces the stored J / decision_loss.
G = 1 - sum J / sum J_cheap (no deadline on the denominator). Missed rate: K31 required episodes (L = 1, runs of
act_gt >= DECELERATE over the timed frames, gaps <= 1 merged); false episodes per 1,000 frames. Paired unit bootstrap
(2,000, seed 0); each drawn unit carries all three repetitions.
usage: scripts/py scripts/k37_2_deadline.py <gt.csv>  -> docs/paper/handoff/K37/K37-2/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import planner as P                                                     # noqa: E402

RAW = ROOT / "results" / "raw"
OUT = ROOT / "docs/paper/handoff/K37/K37-2"
K29 = {"nusc_test20": {1: "20261015_144823_k29_timing_nusc_test20_v6S_r1b", 2: "20261015_162856_k29_timing_nusc_test20_v6S_r2b",
                       3: "20261018_040603_k29_timing_nusc_test20_v6S_r3b"},
       "av2": {1: "20261015_155641_k29_timing_av2_v6S_r1b", 2: "20261018_033322_k29_timing_av2_v6S_r2b",
               3: "20261018_045820_k29_timing_av2_v6S_r3b"}}
K30 = {"nusc_test20": {1: "20261018_064130_k30j_timing_nusc_test20_v6S_j1x_r1", 2: "20261018_065610_k30j_timing_nusc_test20_v6S_j1x_r2",
                       3: "20261018_071046_k30j_timing_nusc_test20_v6S_j1x_r3"},
       "av2": {1: "20261018_064849_k30j_timing_av2_v6S_j1x_r1", 2: "20261018_070327_k30j_timing_av2_v6S_j1x_r2",
               3: "20261018_071802_k30j_timing_av2_v6S_j1x_r3"}}
DEADLINES = (100 / 3, 50.0, 200 / 3, 250 / 3, 100.0)
PP, CP = P.PlannerParams(), P.CostParams()


def runs(mask):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out, s, e = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - e <= 2:
            e = i
        else:
            out.append((s, e - s + 1)); s = e = i
    out.append((s, e - s + 1))
    return out


def frames(ds, gt):
    """long table: rep, unit, frame, row, latency_ms, action, stored_J (+ a_gt, act_gt, a_cheap, prev_cheap)"""
    rows, base = [], None
    for rep in (1, 2, 3):
        d = pd.read_pickle(RAW / K29[ds][rep] / "frames.pkl")
        d["prev_cheap"] = d.groupby("unit").a_cheap.shift(1)          # over the whole run, warm-up frames included
        d = d[~d.warm].reset_index(drop=True)
        if base is None:
            base = d[["unit", "frame", "a_cheap", "act_gt", "prev_cheap"]].copy()
        assert d[["unit", "frame", "a_cheap"]].equals(base[["unit", "frame", "a_cheap"]]), (ds, rep)
        for row, k in (("cheap", "cheap"), ("u960 + C1 + v3a", "u960+c1comp@trt32"), ("native + C1 + v3a", "native+c1comp@trt32")):
            rows.append(pd.DataFrame({"rep": rep, "unit": d.unit, "frame": d.frame, "row": row, "latency_ms": d["ms_" + k],
                                      "action": d["a_cheap"] if k == "cheap" else d["a_" + k], "stored_J": d["J_" + k]}))
        j = pd.read_pickle(RAW / K30[ds][rep] / "frames.pkl")
        j = j[~j.warm].reset_index(drop=True)
        assert j[["unit", "frame"]].equals(base[["unit", "frame"]]), (ds, rep, "k30")
        rows.append(pd.DataFrame({"rep": rep, "unit": j.unit, "frame": j.frame, "row": "hybrid tsp8@x1.0 + C1 + v3a",
                                  "latency_ms": j["ms_tsp8S+C1@trt32"], "action": j["a_tsp8S+C1@trt32"], "stored_J": j["J_tsp8S+C1@trt32"]}))
    k = pd.read_csv(ROOT / "docs/paper/handoff/K34/jetson_frames.csv")
    k = k[k.dataset == ds]
    for pl in ("v6-S", "v6-S <= 2", "B2K-A"):
        x = k[k.pipeline == pl]
        rows.append(pd.DataFrame({"rep": x.rep, "unit": x.unit, "frame": x.frame, "row": pl, "latency_ms": x.latency_ms,
                                  "action": x.action, "stored_J": x.decision_loss}))
    L = pd.concat(rows, ignore_index=True)
    L["unit"] = L.unit.astype(str)
    base["unit"] = base.unit.astype(str)
    base["order"] = np.arange(len(base))
    g = gt[gt.dataset == ("nusc" if ds == "nusc_test20" else "av2")][["unit", "frame", "a_gt", "act_gt"]].rename(columns={"act_gt": "act_gt_k30"})
    g["unit"] = g.unit.astype(str)
    base = base.merge(g, on=["unit", "frame"], how="left", validate="1:1")
    L = L.merge(base[["unit", "frame", "order", "a_cheap", "act_gt", "prev_cheap", "a_gt", "act_gt_k30"]], on=["unit", "frame"],
                how="left", validate="m:1")
    assert L.order.notna().all(), "frames outside the K29 manifest"
    return L.sort_values(["row", "rep", "order"]).reset_index(drop=True)


def cost(a, a_gt, prev):
    if not np.isfinite(a_gt):
        return np.nan
    return P.decision_cost(int(a), float(a_gt), None if pd.isna(prev) else int(prev), PP, CP)["J"]


def apply_rules(L, D):
    out = {}
    for (row, rep), g in L.groupby(["row", "rep"], sort=False):
        a = g.action.to_numpy(int).copy()
        late = g.latency_ms.to_numpy(float) > D
        held = a.copy()
        last = {}
        for i, (u, lt) in enumerate(zip(g.unit.to_numpy(), late)):
            if lt:
                held[i] = last.get(u, P.KEEP)
            else:
                last[u] = a[i]
        out[(row, rep)] = (held, late)
    # anytime DESP
    for rep in (1, 2, 3):
        g = L[(L.row == "v6-S") & (L.rep == rep)]
        b = L[(L.row == "v6-S <= 2") & (L.rep == rep)]
        c = L[(L.row == "cheap") & (L.rep == rep)]
        assert g.unit.tolist() == b.unit.tolist() == c.unit.tolist() and g.frame.tolist() == b.frame.tolist() == c.frame.tolist()
        a6, a2, ac = g.action.to_numpy(int), b.action.to_numpy(int), c.action.to_numpy(int)
        l6, l2, lc = (x.latency_ms.to_numpy(float) > D for x in (g, b, c))
        res, last, src = np.empty(len(a6), int), {}, np.empty(len(a6), object)
        for i, u in enumerate(g.unit.to_numpy()):
            if not l6[i]:
                res[i], src[i] = a6[i], "v6-S"
            elif not l2[i]:
                res[i], src[i] = a2[i], "v6-S <= 2"
            elif not lc[i]:
                res[i], src[i] = ac[i], "cheap"
            else:
                res[i], src[i] = last.get(u, P.KEEP), "hold"
            last[u] = res[i]
        out[("DESP anytime (v6-S -> <= 2 -> cheap)", rep)] = (res, l6, src)
    return out


def main():
    gt = pd.read_csv(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    checks, res_rows, diffs = {}, [], []
    for ds in ("nusc_test20", "av2"):
        L = frames(ds, gt)
        ev = L.a_gt.notna()
        checks[f"{ds}_frames"] = int(L[(L.row == "cheap") & (L.rep == 1)].shape[0])
        checks[f"{ds}_frames_with_gt_demand"] = int(L[(L.row == "cheap") & (L.rep == 1)].a_gt.notna().sum())
        same = L[ev].act_gt == L[ev].act_gt_k30
        checks[f"{ds}_act_gt_equal_k30"] = bool(same.all())
        Jr = np.array([cost(a, g_, p_) for a, g_, p_ in zip(L.action, L.a_gt, L.prev_cheap)])
        dJ = np.abs(Jr - L.stored_J.to_numpy(float))
        checks[f"{ds}_stored_cost_reproduced_max_abs_diff"] = float(np.nanmax(dJ[ev.to_numpy()]))
        assert checks[f"{ds}_stored_cost_reproduced_max_abs_diff"] < 1e-6, checks
        units = np.array(sorted(L.unit.unique()))
        draws = np.random.default_rng(0).integers(0, len(units), (2000, len(units)))
        C = np.stack([np.bincount(x, minlength=len(units)) for x in draws]).astype(float)
        base = L[(L.row == "cheap") & (L.rep == 1)]
        evb = base.a_gt.notna().to_numpy()
        Jc_u = pd.Series([cost(a, g_, p_) for a, g_, p_ in zip(base.action, base.a_gt, base.prev_cheap)]).fillna(0).groupby(base.unit.to_numpy()).sum().reindex(units).to_numpy() * 3
        for D in DEADLINES + (np.inf,):
            R = apply_rules(L, D)
            per = {}
            for (row, rep), v in R.items():
                g = (L[(L.row == "v6-S") & (L.rep == rep)] if row.startswith("DESP anytime") else L[(L.row == row) & (L.rep == rep)])
                a, late = v[0], v[1]
                J = np.array([cost(x, y, z) for x, y, z in zip(a, g.a_gt, g.prev_cheap)])
                J = np.nan_to_num(J)
                evm = g.a_gt.notna().to_numpy()                 # evaluated (labelled) frames only
                uarr = g.unit.to_numpy()[evm]
                J, late, a, gt_ = J[evm], late[evm], a[evm], g.act_gt.to_numpy(int)[evm]
                stat = per.setdefault(row, {k: pd.Series(0.0, index=units) for k in ("J", "late", "n", "eps", "miss", "false")})
                df = pd.DataFrame({"u": uarr, "J": J, "late": late.astype(float), "n": 1.0})
                for k in ("J", "late", "n"):
                    stat[k] = stat[k].add(df.groupby("u")[k].sum().reindex(units).fillna(0))
                for u in units:
                    m = uarr == u
                    q, aa = gt_[m], a[m]
                    ep = runs(q >= 1)
                    stat["eps"][u] += len(ep)
                    stat["miss"][u] += sum(1 for s, n in ep if not (aa[s:s + n] >= 1).any())
                    stat["false"][u] += sum(1 for s, n in runs(aa >= 1) if (q[s:s + n] < 1).all())
                if len(v) > 2:
                    for s_ in ("v6-S", "v6-S <= 2", "cheap", "hold"):
                        stat.setdefault("src_" + s_, 0.0)
                        stat["src_" + s_] += float((v[2][evm] == s_).sum())
            for row, s in per.items():
                Ju, n, eps, miss, fa, late = (s[k].to_numpy() for k in ("J", "n", "eps", "miss", "false", "late"))
                dist = {"G": 1 - (C @ Ju) / (C @ Jc_u), "missed_rate": (C @ miss) / np.maximum(C @ eps, 1e-12),
                        "false_per_1000": 1e3 * (C @ fa) / (C @ n), "deadline_miss_share": (C @ late) / (C @ n)}
                r = {"dataset": ds, "deadline_ms": D, "row": row, "frames_x_reps": int(n.sum()), "G": 1 - Ju.sum() / Jc_u.sum(),
                     "deadline_miss_share": late.sum() / n.sum(), "episodes_x_reps": int(eps.sum()), "missed_rate": miss.sum() / max(eps.sum(), 1),
                     "false_per_1000": 1e3 * fa.sum() / n.sum()}
                for k, x in dist.items():
                    r[k + "_ci_lo"], r[k + "_ci_hi"] = np.nanpercentile(x, [2.5, 97.5])
                if "src_v6-S" in s:
                    tot = n.sum()
                    r.update({k: s[k] / tot for k in s if k.startswith("src_")})
                res_rows.append(r)
                s["_dist"] = dist
            ref = per["v6-S"]["_dist"]
            for row, s in per.items():
                if row == "v6-S":
                    continue
                for k in ("G", "missed_rate", "false_per_1000"):
                    x = s["_dist"][k] - ref[k]
                    lo, hi = np.nanpercentile(x, [2.5, 97.5])
                    pt = [r for r in res_rows if r["dataset"] == ds and r["deadline_ms"] == D]
                    pr = next(r for r in pt if r["row"] == row)[k] - next(r for r in pt if r["row"] == "v6-S")[k]
                    diffs.append({"dataset": ds, "deadline_ms": D, "row": row, "statistic": k, "row_minus_v6S_hold": pr, "ci_lo": lo, "ci_hi": hi,
                                  "excludes_0": bool(lo > 0 or hi < 0)})
    R = pd.DataFrame(res_rows)
    R.to_csv(OUT / "k37_2_rows.csv", index=False)
    pd.DataFrame(diffs).to_csv(OUT / "k37_2_diffs.csv", index=False)
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    pd.set_option("display.width", 250)
    print(R[["dataset", "deadline_ms", "row", "G", "deadline_miss_share", "missed_rate", "false_per_1000"]].round(3).to_string())
    print(checks)


if __name__ == "__main__":
    main()
