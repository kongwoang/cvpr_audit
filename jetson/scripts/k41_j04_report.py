#!/usr/bin/env python
"""K41-5 J04 report (handoff/K41/PREREG.md, 2026-11-01 J04 addendum), adapted from scripts/k39_1_report.py:
adaptive / deadline benchmark on the Jetson AGX Xavier (experiments/k41_jetson_20261030, logs/state.json; three
repetitions, all rows in one process per repetition). Rows: cheap; whole frame + C1 + v3a at u512 / u640 / u960 / u1280 /
native; capped hybrid tsp8S+C1 and uncertainty unc2S+C1 at v6-S's crop px; the dev-selected K33 rows (AdaScale, AnA
one-pass / iterated, counterfactual one-pass / iterated, FOVEA KDE_S / KDE_I, DVR / BAR routers, Chanakya); DESP v6-S,
B2K-A, B1K-A at D = inf and in-loop under D in {33.3, 50, 66.7, 100} ms (K38-1 rule); iterated AnA / CF in-loop too.
Applied action under D: the row's last on-time action of the unit if its latency > D (KEEP before the first one).
Cost: K28 decision_cost with K30 GT (gt_k30.csv) and the run's previous cheap action (warm-up included), checked
against the stored J. Paired unit bootstrap (2,000, seed 0; repetitions nested in units) of DESP variant@D - row@D.
Parity gate: v6-S / B2K-A per-frame action (with and without refiner), px, crop / round / call counts vs the stored
K38-1 staged records (every K38 repetition) and K34-3 jetson_frames.csv.
Holm over the compared rows per dataset, D and reference (G; two-sided bootstrap p = 2 min(P(d <= 0), P(d >= 0))).
usage: scripts/py scripts/k41_j04_report.py  -> docs/paper/handoff/K41/J04/"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import planner as P                                                     # noqa: E402

HERE = ROOT / "experiments/k41_jetson_20261030"
K39 = ROOT / "experiments/k39_jetson_20261027"
K38 = ROOT / "experiments/k38_deadline_20261026"
OUT = ROOT / "docs/paper/handoff/K41/J04"
PP, CP = P.PlannerParams(), P.CostParams()
DS = {"nusc_test20": "nusc", "av2t40": "av2"}
DL = (33.3, 50.0, 66.7, 100.0, np.inf)
WF = {"u512 + comp.": "u512+c1comp@trt32", "u640 + comp.": "u640+c1comp@trt32",
      "u960 + comp.": "u960+c1comp@trt32", "u1280 + comp.": "u1280+c1comp@trt32", "native + comp.": "native+c1comp@trt32"}
CONTROL = {}
HYB = {"hybrid tsp8 + comp. @ DESP px": "tsp8S+C1@trt32", "uncertainty unc2 + comp. @ DESP px": "unc2S+C1@trt32",
       "AdaScale ada+1t1280": "ada+1t1280@trt32", "AnA one-pass ana05hinfo": "ana05hinfo@trt32",
       "CF one-pass cf40q20o": "cf40q20o@trt32", "FOVEA KDE_S fovs1280b64a1": "fovs1280b64a1@trt32",
       "FOVEA KDE_I fovi1280b64a2": "fovi1280b64a2@trt32", "DVR router dvr1280_0.5": "dvr1280_0.5@trt32",
       "BAR router bar1280_0.5": "bar1280_0.5@trt32", "Chanakya ch128B_csc": "ch128B_csc@trt32"}
ITER = {"AnA iterated ana05hinfi": "ana05hinfi", "CF iterated cf40q20i": "cf40q20i"}
DESP = {"DESP v6-S": "ours", "DESP B2K-A": "ours_b2A", "DESP B1K-A (post-hoc)": "ours_b1A"}


def dtag(D):
    return f"{D:g}"


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


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - k) * p[i])); adj[i] = run
    return adj


def parity(ds, F):
    st = json.loads((K38 / "logs/state.json").read_text())
    ref = [pd.read_pickle(Path(r["run"]) / "frames.pkl") for r in st["runs"] if r["dataset"] == ds and not r["smoke"] and r["runtime"] == "staged"]
    cols = [f"{x}_{nm}@trt32" for nm in ("ours", "ours_b2A") for x in ("a", "px")] + ["a_ours@trt32_noref", "a_ours_b2A@trt32_noref",
            "n_crops_ours@trt32", "rounds_ours@trt32", "calls_loop_ours_b2A@trt32", "bound_hit_ours_b2A@trt32", "a_cheap", "px_cheap"]
    out = {}
    for i, m in enumerate(F):
        for j, r in enumerate(ref):
            r = r[~r.warm].assign(unit=lambda x: x.unit.astype(str))
            x = m[["unit", "frame"] + cols].merge(r[["unit", "frame"] + cols], on=["unit", "frame"], suffixes=("", "_k38"), validate="1:1")
            out[f"K41 rep{i + 1} vs K38 rep{j + 1}"] = {"frames": len(x), "timed_frames_k41": len(m),
                                                     **{c: int((~np.isclose(x[c].astype(float), x[c + "_k38"].astype(float), rtol=0, atol=0)).sum()) for c in cols}}
    k34 = pd.read_csv(ROOT / "docs/paper/handoff/K34/jetson_frames.csv")
    kk = k34[k34.dataset == ("nusc_test20" if ds == "nusc_test20" else "av2")]
    for i, m in enumerate(F):
        for pl, c in (("v6-S", "a_ours@trt32"), ("B2K-A", "a_ours_b2A@trt32")):
            x = kk[(kk.pipeline == pl) & (kk.rep == 1)][["unit", "frame", "action"]]
            j = m[["unit", "frame", c]].merge(x.assign(unit=x.unit.astype(str)), on=["unit", "frame"], how="inner")
            out[f"K41 rep{i + 1} {pl} vs K34-3 rep1"] = {"common_frames": len(j), "differing": int((j[c] != j.action).sum())}
    return out


def main():
    gt = pd.read_csv(K39 / "gt_k30.csv")
    st = json.loads((HERE / "logs/state.json").read_text())
    rows_out, dl_out, diffs, checks = [], [], [], {"medians_sha256": st.get("medians_sha256")}
    for ds, gds in DS.items():
        R = [r for r in st["runs"] if r["dataset"] == ds and not r["smoke"] and r["tag"].startswith("r")]
        assert len(R) == 3, (ds, len(R))
        F = []
        for r in R:
            d = pd.read_pickle(Path(r["run"]) / "frames.pkl")
            d["prev_cheap"] = d.groupby("unit").a_cheap.shift(1)
            d = d[~d.warm].reset_index(drop=True)
            d["unit"] = d.unit.astype(str)
            g = gt[gt.dataset == gds][["unit", "frame", "a_gt"]].assign(unit=lambda x: x.unit.astype(str)).rename(columns={"a_gt": "a_gt_k30"})
            m = d.merge(g, on=["unit", "frame"], how="left", validate="1:1")
            assert m.a_gt_k30.notna().all(), ds
            checks[f"{ds}_rep{r['rep']}_a_gt_vs_k30_max_abs_diff"] = float(np.max(np.abs(m.a_gt.to_numpy(float) - m.a_gt_k30.to_numpy(float))))
            m["a_gt"] = m.a_gt_k30
            m["rep"] = r["rep"]
            F.append(m)
        checks[f"{ds}_parity"] = parity(ds, F)
        mx = 0.0
        for m in F:
            for c in sorted(x[2:] for x in m.columns if x.startswith("J_")):
                Jr = np.array([cost(a, g_, p_) for a, g_, p_ in zip(m["a_" + c], m.a_gt, m.prev_cheap)])
                mx = max(mx, float(np.max(np.abs(Jr - m["J_" + c].to_numpy(float)))))
        checks[f"{ds}_stored_cost_max_abs_diff"] = mx
        assert mx < 1e-6, checks
        units = np.array(sorted(F[0].unit.unique()))
        draws = np.random.default_rng(0).integers(0, len(units), (2000, len(units)))
        C = np.stack([np.bincount(x, minlength=len(units)) for x in draws]).astype(float)
        Jc = sum(pd.Series([cost(a, g_, p_) for a, g_, p_ in zip(m.a_cheap, m.a_gt, m.prev_cheap)]).groupby(m.unit.to_numpy()).sum()
                 .reindex(units).to_numpy() for m in F)
        for D in DL:
            spec = {"cheap": "cheap", **WF, **CONTROL, **HYB}
            for lab, b in {**DESP, **ITER}.items():
                spec[lab + (f"@{dtag(D)} (in-loop)" if np.isfinite(D) else "")] = (f"{b}_dl{dtag(D)}@trt32" if np.isfinite(D) else f"{b}@trt32")
                if np.isfinite(D):
                    spec[lab + " (hold)"] = f"{b}@trt32"
            stat = {}
            for name, c in spec.items():
                Ju, late_u, n_u, ep_u, mi_u, fa_u, stop_u = (np.zeros(len(units)) for _ in range(7))
                lat_all, ex_all, px_all = [], [], []
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
                    ex_all.append(m[f"execs_{c}"].to_numpy(float) if f"execs_{c}" in m else np.full(len(m), np.nan))
                    px_all.append(m[f"px_{c}"].to_numpy(float) if f"px_{c}" in m else np.full(len(m), np.nan))
                    gtv = m.act_gt.to_numpy(int)
                    for k, u in enumerate(units):
                        q = np.flatnonzero(m.unit.to_numpy() == u)
                        if not len(q):
                            continue
                        gg, aa = gtv[q], ap[q]
                        ep = runs(gg >= 1)
                        ep_u[k] += len(ep); mi_u[k] += sum(1 for s, n in ep if not (aa[s:s + n] >= 1).any())
                        fa_u[k] += sum(1 for s, n in runs(aa >= 1) if (gg[s:s + n] < 1).all())
                lat_all, ex_all, px_all = (np.concatenate(x) for x in (lat_all, ex_all, px_all))
                dist = {"G": 1 - (C @ Ju) / (C @ Jc), "missed_rate": (C @ mi_u) / np.maximum(C @ ep_u, 1e-12),
                        "false_per_1000": 1e3 * (C @ fa_u) / (C @ n_u), "late_share": (C @ late_u) / (C @ n_u)}
                r = {"dataset": ds, "deadline_ms": D, "row": name, "key": c, "frames_x_reps": int(n_u.sum()), "units": len(units),
                     "G": 1 - Ju.sum() / Jc.sum(), "late_share": late_u.sum() / n_u.sum(), "stopped_early_share": stop_u.sum() / n_u.sum(),
                     "episodes_x_reps": int(ep_u.sum()), "missed_x_reps": int(mi_u.sum()), "missed_rate": mi_u.sum() / max(ep_u.sum(), 1),
                     "false_per_1000": 1e3 * fa_u.sum() / n_u.sum(), "mean_ms": float(lat_all.mean()), "p50_ms": float(np.percentile(lat_all, 50)),
                     "p95_ms": float(np.percentile(lat_all, 95)), "p99_ms": float(np.percentile(lat_all, 99)),
                     "max_ms": float(lat_all.max()), "engine_execs_mean": float(np.nanmean(ex_all)) if np.isfinite(ex_all).any() else np.nan,
                     "px_mean": float(np.nanmean(px_all)) if np.isfinite(px_all).any() else np.nan}
                for k, x in dist.items():
                    r[k + "_ci_lo"], r[k + "_ci_hi"] = np.nanpercentile(x, [2.5, 97.5])
                (rows_out if not np.isfinite(D) else dl_out).append(r)
                if not np.isfinite(D):
                    dl_out.append(r)
                stat[name] = (r, dist)
            refs = [n for n in stat if n.startswith(("DESP v6-S", "DESP B2K-A")) and "(hold)" not in n]
            others = [n for n in stat if n in WF or n in HYB or n.startswith(tuple(ITER)) and "(hold)" not in n]
            for ref in refs:
                block = []
                for name in others:
                    for k in ("G", "missed_rate", "false_per_1000", "late_share"):
                        dd = stat[ref][1][k] - stat[name][1][k]
                        lo, hi = np.nanpercentile(dd, [2.5, 97.5])
                        p = float(min(1.0, 2 * min(np.nanmean(dd <= 0), np.nanmean(dd >= 0))))
                        block.append({"dataset": ds, "deadline_ms": D, "desp": ref, "row": name, "statistic": k,
                                      "desp_minus_row": stat[ref][0][k] - stat[name][0][k], "ci_lo": lo, "ci_hi": hi,
                                      "excludes_0": bool(lo > 0 or hi < 0), "p_boot": p})
                for k in ("G", "missed_rate", "false_per_1000", "late_share"):
                    sub = [b for b in block if b["statistic"] == k]
                    for b, q in zip(sub, holm([b["p_boot"] for b in sub])):
                        b["p_holm"] = float(q); b["holm_sig_05"] = bool(q < 0.05)
                diffs += block
    rows, dlt, dif = pd.DataFrame(rows_out), pd.DataFrame(dl_out), pd.DataFrame(diffs)
    OUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT / "k41_5_rows.csv", index=False)
    dlt.to_csv(OUT / "k41_5_deadline.csv", index=False)
    dif.to_csv(OUT / "k41_5_diffs.csv", index=False)
    (OUT / "parity.json").write_text(json.dumps(checks, indent=1) + "\n")
    figure(rows, dlt)
    pd.set_option("display.width", 250)
    print(rows[["dataset", "row", "G", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "engine_execs_mean", "px_mean"]].round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in checks.items() if "parity" not in k}, indent=1))


def figure(rows, dlt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.2))
    for j, ds in enumerate(DS):
        a = ax[0, j]
        r = rows[rows.dataset == ds]
        wf = r[r.row.isin(WF)]
        a.plot(wf.p95_ms, wf.G, "-o", color="#4c72b0", label="whole frame + comp.")
        for _, x in wf.iterrows():
            a.annotate(x.row.split(" ")[0], (x.p95_ms, x.G), textcoords="offset points", xytext=(4, -10), fontsize=8)
        for lab, mk, col in (("DESP v6-S", "*", "#c44e52"), ("DESP B2K-A", "D", "#dd8452"), ("DESP B1K-A (post-hoc)", "s", "#8172b3"),
                             ("hybrid tsp8 + comp. @ DESP px", "^", "#55a868"), ("cheap", "x", "0.4"),
                             ("u960 + comp. (dyn engine, control)", "+", "#4c72b0")):
            x = r[r.row == lab]
            if len(x):
                a.scatter(x.p95_ms, x.G, marker=mk, s=80, color=col, label=lab, zorder=3)
        a.set_xlabel("p95 latency (ms), Jetson AGX Xavier, TensorRT FP16")
        a.set_ylabel("G (timed frames, 3 repetitions)")
        a.set_title({"nusc_test20": "nuScenes test, 20 scenes", "av2t40": "AV2 test, 40 logs"}[ds])
        a.grid(alpha=0.3)
        b = ax[1, j]
        d = dlt[dlt.dataset == ds]
        Ds = [x for x in DL if np.isfinite(x)]
        for lab, col, ls in (("DESP v6-S", "#c44e52", "-"), ("DESP B2K-A", "#dd8452", "-"), ("DESP B1K-A (post-hoc)", "#8172b3", "-"),
                             ("u640 + comp.", "#9ecae1", "--"), ("u960 + comp.", "#4c72b0", "--"), ("u1280 + comp.", "#08306b", "--"),
                             ("hybrid tsp8 + comp. @ DESP px", "#55a868", ":")):
            ys = []
            for D in Ds:
                nm = lab + f"@{dtag(D)} (in-loop)" if lab.startswith("DESP") else lab
                x = d[(d.deadline_ms == D) & (d.row == nm)]
                ys.append(float(x.G.iloc[0]) if len(x) else np.nan)
            b.plot(Ds, ys, ls, marker="o", color=col, label=lab)
        b.set_xlabel("deadline D (ms)")
        b.set_ylabel("G(D) (late frames hold)")
        b.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7, loc="lower right")
    ax[1, 0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"k41_5_frontier.{ext}", dpi=200)


if __name__ == "__main__":
    main()
