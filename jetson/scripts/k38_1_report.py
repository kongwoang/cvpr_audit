#!/usr/bin/env python
"""K38-1 report: in-loop deadline execution on the Jetson (experiments/k38_deadline_20261026, logs/state.json).
Per dataset (nusc_test20, av2t40) x D in {50, 66.7, inf} x row, pooled over the three staged repetitions (all rows of
a repetition ran in one process, each with its own history):
  DESP@D / B2K-A@D      the in-loop deadline rows (ours_dl<D>, ours_b2A_dl<D>)
  v6-S hold@D / B2K-A hold@D / u960 + C1 + v3a hold@D   the D = inf rows of the same runs with the hold rule
Applied action: if the row's latency > D, the last on-time action of the row in that unit (KEEP before the first one;
K37-2 rule (a)); otherwise its own action. Cost: K28 decision_cost(action, a_gt, previous cheap action of the run,
warm-up frames included) with the stored GT demand (results/k30/gt), checked against the stored J of every row.
Statistics: G, deadline-miss share (latency > D), missed / false episodes (K31 L = 1), mean / p95 latency; paired
unit bootstrap (2,000, seed 0; repetitions nested in units) of row - DESP@D. D = inf parity of v6-S / B2K-A with
K34-3 jetson_frames.csv on the K29 frames.
usage: scripts/py scripts/k38_1_report.py <gt.csv>  -> docs/paper/handoff/K38/K38-1/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import planner as P                                                     # noqa: E402

HERE = ROOT / "experiments/k38_deadline_20261026"
OUT = ROOT / "docs/paper/handoff/K38/K38-1"
PP, CP = P.PlannerParams(), P.CostParams()
DS = {"nusc_test20": "nusc", "av2t40": "av2"}


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


def cost(a, g, p):
    return P.decision_cost(int(a), float(g), None if pd.isna(p) else int(p), PP, CP)["J"]


def hold(units, act, late):
    out, last = act.copy(), {}
    for i, (u, lt) in enumerate(zip(units, late)):
        if lt:
            out[i] = last.get(u, P.KEEP)
        else:
            last[u] = act[i]
    return out


def main():
    gt = pd.read_csv(sys.argv[1])
    st = json.loads((HERE / "logs/state.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rows_out, diffs, checks = [], [], {"medians_sha256": st.get("medians_sha256")}
    k34 = pd.read_csv(ROOT / "docs/paper/handoff/K34/jetson_frames.csv")
    for ds, gds in DS.items():
        R = [r for r in st["runs"] if r["dataset"] == ds and not r["smoke"] and r["runtime"] == "staged"]
        assert len(R) == 3, (ds, len(R))
        frames = []
        for r in R:
            d = pd.read_pickle(Path(r["run"]) / "frames.pkl")
            d["prev_cheap"] = d.groupby("unit").a_cheap.shift(1)
            d = d[~d.warm].reset_index(drop=True)
            d["rep"] = r["rep"]
            frames.append(d)
        g = gt[gt.dataset == gds][["unit", "frame", "a_gt"]]
        F = []
        for d in frames:
            d["unit"] = d.unit.astype(str)
            m = d.merge(g.assign(unit=g.unit.astype(str)).rename(columns={"a_gt": "a_gt_k30"}), on=["unit", "frame"], how="left", validate="1:1")
            assert m.a_gt_k30.notna().all(), ds
            if "a_gt" in m:
                checks[f"{ds}_rep{int(m.rep.iloc[0])}_a_gt_vs_k30_max_abs_diff"] = float(np.max(np.abs(m.a_gt.to_numpy(float) - m.a_gt_k30.to_numpy(float))))
            m["a_gt"] = m["a_gt_k30"]
            F.append(m)
        # cost convention check on the stored J of the D = inf rows
        mx = 0.0
        for m in F:
            for c in ("ours@trt32", "ours_b2A@trt32", "u960+c1comp@trt32", "ours_dl50@trt32"):
                Jr = np.array([cost(a, g_, p_) for a, g_, p_ in zip(m["a_" + c], m.a_gt, m.prev_cheap)])
                mx = max(mx, float(np.max(np.abs(Jr - m["J_" + c].to_numpy(float)))))
        checks[f"{ds}_stored_cost_max_abs_diff"] = mx
        assert mx < 1e-6, checks
        # D = inf parity with K34-3
        kk = k34[k34.dataset == ("nusc_test20" if ds == "nusc_test20" else "av2")]
        par = {}
        for m in F:
            for pl, c in (("v6-S", "ours@trt32"), ("B2K-A", "ours_b2A@trt32")):
                x = kk[(kk.pipeline == pl) & (kk.rep == int(m.rep.iloc[0]))][["unit", "frame", "action"]]
                j = m[["unit", "frame", "a_" + c]].merge(x.assign(unit=x.unit.astype(str)), on=["unit", "frame"], how="inner")
                par[f"{pl} rep{int(m.rep.iloc[0])}"] = {"common_frames": len(j), "differing": int((j["a_" + c] != j.action).sum())}
        checks[f"{ds}_Dinf_parity_vs_K34_3"] = par
        units = np.array(sorted(F[0].unit.unique()))
        draws = np.random.default_rng(0).integers(0, len(units), (2000, len(units)))
        C = np.stack([np.bincount(x, minlength=len(units)) for x in draws]).astype(float)
        Jc = sum(pd.Series([cost(a, g_, p_) for a, g_, p_ in zip(m.a_cheap, m.a_gt, m.prev_cheap)]).groupby(m.unit.to_numpy()).sum()
                 .reindex(units).to_numpy() for m in F)
        for D in (50.0, 200 / 3, np.inf):
            tag = {50.0: "50", 200 / 3: "66.7"}.get(D)
            spec = {}
            if tag:
                spec[f"DESP@{tag} (in-loop)"] = f"ours_dl{tag}@trt32"
                spec[f"B2K-A@{tag} (in-loop)"] = f"ours_b2A_dl{tag}@trt32"
            spec["v6-S (hold)"] = "ours@trt32"
            spec["B2K-A (hold)"] = "ours_b2A@trt32"
            spec["u960 + C1 + v3a (hold)"] = "u960+c1comp@trt32"
            spec["cheap"] = "cheap"
            stat = {}
            for name, c in spec.items():
                Ju = np.zeros(len(units)); late_u = np.zeros(len(units)); n_u = np.zeros(len(units))
                ep_u = np.zeros(len(units)); mi_u = np.zeros(len(units)); fa_u = np.zeros(len(units)); stop_u = np.zeros(len(units))
                lat_all = []
                for m in F:
                    a = (m.a_cheap if c == "cheap" else m["a_" + c]).to_numpy(int)
                    lat = m["ms_" + c].to_numpy(float)
                    late = lat > D
                    ap = hold(m.unit.to_numpy(), a, late)
                    J = np.array([cost(x, g_, p_) for x, g_, p_ in zip(ap, m.a_gt, m.prev_cheap)])
                    ui = np.searchsorted(units, m.unit.to_numpy())
                    np.add.at(Ju, ui, J); np.add.at(late_u, ui, late.astype(float)); np.add.at(n_u, ui, 1.0)
                    if f"dl_stopped_{c}" in m:
                        np.add.at(stop_u, ui, m[f"dl_stopped_{c}"].astype(float).to_numpy())
                    lat_all.append(lat)
                    gtv = m.act_gt.to_numpy(int)
                    for k, u in enumerate(units):
                        q = np.flatnonzero(m.unit.to_numpy() == u)
                        if not len(q):
                            continue
                        gg, aa = gtv[q], ap[q]
                        ep = runs(gg >= 1)
                        ep_u[k] += len(ep); mi_u[k] += sum(1 for s, n in ep if not (aa[s:s + n] >= 1).any())
                        fa_u[k] += sum(1 for s, n in runs(aa >= 1) if (gg[s:s + n] < 1).all())
                lat_all = np.concatenate(lat_all)
                dist = {"G": 1 - (C @ Ju) / (C @ Jc), "missed_rate": (C @ mi_u) / np.maximum(C @ ep_u, 1e-12),
                        "false_per_1000": 1e3 * (C @ fa_u) / (C @ n_u), "deadline_miss_share": (C @ late_u) / (C @ n_u)}
                r = {"dataset": ds, "deadline_ms": D, "row": name, "frames_x_reps": int(n_u.sum()), "units": len(units),
                     "G": 1 - Ju.sum() / Jc.sum(), "deadline_miss_share": late_u.sum() / n_u.sum(),
                     "stopped_early_share": stop_u.sum() / n_u.sum(), "episodes_x_reps": int(ep_u.sum()), "missed_x_reps": int(mi_u.sum()),
                     "missed_rate": mi_u.sum() / max(ep_u.sum(), 1), "false_per_1000": 1e3 * fa_u.sum() / n_u.sum(),
                     "mean_ms": float(lat_all.mean()), "p95_ms": float(np.percentile(lat_all, 95)), "p99_ms": float(np.percentile(lat_all, 99))}
                for k, x in dist.items():
                    r[k + "_ci_lo"], r[k + "_ci_hi"] = np.nanpercentile(x, [2.5, 97.5])
                rows_out.append(r); stat[name] = (r, dist)
            ref = next((n for n in stat if n.startswith("DESP@")), "v6-S (hold)")
            for name, (r, dist) in stat.items():
                if name == ref:
                    continue
                for k in ("G", "missed_rate", "false_per_1000", "deadline_miss_share"):
                    lo, hi = np.nanpercentile(dist[k] - stat[ref][1][k], [2.5, 97.5])
                    diffs.append({"dataset": ds, "deadline_ms": D, "row": name, "vs": ref, "statistic": k,
                                  "row_minus_ref": r[k] - stat[ref][0][k], "ci_lo": lo, "ci_hi": hi, "excludes_0": bool(lo > 0 or hi < 0)})
    pd.DataFrame(rows_out).to_csv(OUT / "k38_1_rows.csv", index=False)
    pd.DataFrame(diffs).to_csv(OUT / "k38_1_diffs.csv", index=False)
    (OUT / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows_out)[["dataset", "deadline_ms", "row", "G", "deadline_miss_share", "stopped_early_share", "missed_x_reps",
                                  "episodes_x_reps", "false_per_1000", "mean_ms", "p95_ms"]].round(3).to_string())
    print(json.dumps(checks, indent=1)[:3000])


if __name__ == "__main__":
    main()
