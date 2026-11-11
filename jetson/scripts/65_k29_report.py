#!/usr/bin/env python
"""K29 — tables for handoff/K29/REPORT.md from the runs of 64_k29_latency.py (results/raw/*_k29_*):
  selection_dev.csv      dev timing of every candidate path (v6-S and u960 + T4 / + C1 + ref; nuScenes val 20 scenes, AV2 dev 10 logs)
  parity_dev.csv         R2 on the full dev splits: per pipeline and path, identical actions, differing frames / units, dev G both paths
  c1stats_dev.csv        chains, pruned, crops encoded per path (dev parity runs)
  latency_runs.csv       every timed pipeline x repetition (test-frame window), mean / median / p95 / p99 / max, G, px
  latency_fast.csv       the paper's latency table with the selected fast path: G, median over repetitions of mean / p95 ...
  latency_ref.csv        the same with the K28 reference path (re-timed in the same window)
  tail.csv               mean / p95 / p99 / max for cheap, u960 / u1280 / native + C1 + ref, v6-S, v6-S <= 2
  parity_timed.csv       R2 on every timed frame, per pipeline, dataset and repetition
  breakdown.csv          per-component time of v6-S (draft, field, crops per round, guard, C1, refiner, rest) at mean / p95
  front.csv              latency-G front (mean, p95) per dataset and path set, with the dominating rows
  front_intervals.csv    G(v6-S) - G(row) for every row (paired unit bootstrap, 2,000, seed 0) on the timed frames
  front_{fast,ref}.{pdf,png}
usage: 65_k29_report.py --sel trt16 [--since 20261015]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper" / "handoff" / "K29"
CFGL = {"v6S": "v6-S", "v6M": "v6-M", "v6L": "v6-L", "v51S": "v5.1-S"}
DSL = {"nusc_test20": "nuScenes new test, 20 scenes", "nusc": "nuScenes, 20 old DEEP scenes (K28 set)",
       "av2": "AV2 val, 8 logs, ego mask (K28 set)", "nusc_val": "nuScenes new val (dev)", "av2_dev": "AV2 CFG-DEV (dev), ego mask"}
HEURL = {"tsp2S+C1": "heuristic tsp2@S + C1 + ref (YOLOv8s)", "tsp8S-yolo11m+C1": "heuristic tsp8@S + C1 + own ref (YOLO11m)"}


def runs(kind, since):
    out = []
    for r in sorted(glob.glob(str(ROOT / "results" / "raw" / f"*_k29_{kind}_*"))):
        if Path(r).name[:8] < since or not (Path(r) / "frames.pkl").exists():
            continue
        out.append((Path(r), json.loads((Path(r) / "config.json").read_text())))
    return out


def split_path(nm):
    """pipeline column name -> (base name, path)"""
    if "@" not in nm:
        return nm, "ref"
    base, p = nm.split("@")
    p, _, suf = p.partition("_")
    return base + (f"_{suf}" if suf else ""), p


def label(base, cfg):
    if base == "cheap":
        return "cheap (u320)", "cheap"
    noref = base.endswith("_noref")
    b = base[:-6] if noref else base
    if b.startswith("ours"):
        n = b.split("_b")[1] if "_b" in b else None
        s = CFGL[cfg] + (f", rounds <= {n}" if n else "") + (", no refiner" if noref else "")
        return s, ("ours" if cfg != "v51S" else "previous ours")
    if b in HEURL:
        return HEURL[b] + (", no refiner" if noref else ""), "heuristic + C1"
    side, _, kind = b.partition("+")
    s = f"whole frame {side}" + {"": "", "comp": " + T4 + ref", "c1comp": " + C1 + ref"}[kind]
    return s, "whole frame" + {"": "", "comp": " + T4 + ref", "c1comp": " + C1 + ref"}[kind]


def lat_stats(ms):
    return {"mean_ms": ms.mean(), "median_ms": ms.median(), "p95_ms": ms.quantile(.95), "p99_ms": ms.quantile(.99), "max_ms": ms.max()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", required=True)
    ap.add_argument("--since", default="20261015")
    ap.add_argument("--tagsuf", default="", help="window C clock: '' = MAXN 1377 MHz tags, '_g905' = GPU locked at 905 MHz")
    ap.add_argument("--out_sub", default="", help="sub-directory of handoff/K29 for the test-window tables")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- dev selection timing
    sel = []
    for r, c in runs("timing", args.since):
        if c.get("tag") not in ("select", "select2"):         # select2 = the fixed t16g, with the reference again
            continue
        d = pd.read_pickle(r / "frames.pkl"); d = d[~d.warm]
        L = d.J_cheap.to_numpy(float)
        for col in [x for x in d.columns if x.startswith("ms_")]:
            base, p = split_path(col[3:])
            lab, _ = label(base, c["config"])
            if c["tag"] == "select" and p == "t16g":
                p = "t16g (lifetime bug, not reported as FP16)"
            sel.append({"dataset": DSL[c["dataset"]], "pipeline": lab, "path": p, "run": c["tag"], "frames": len(d), "units": d.unit.nunique(),
                        "G": (L.sum() - d[f"J_{col[3:]}"].sum()) / L.sum() if col != "ms_cheap" else 0.0, **lat_stats(d[col])})
    if sel:
        pd.DataFrame(sel).to_csv(OUT / "selection_dev.csv", index=False)
    # ---- dev parity
    par, c1s = [], []
    for r, c in runs("parity", args.since):
        if c["dataset"] not in ("nusc_val", "av2_dev") or c.get("tag", "") not in ("a", "b", "c", "d", "e"):   # no smoke / memtest
            continue
        meta = json.loads((r / "meta.json").read_text())
        p = pd.read_csv(r / "parity.csv")
        p["pipeline"] = [label(x, c["config"])[0] for x in p.pipeline]
        p.insert(0, "dataset", DSL[c["dataset"]]); p["run"] = r.name
        p["crop64_checked"], p["crop64_mismatch"], p["missing_images"] = meta["crop64_checked"], meta["crop64_mismatch"], meta["missing_images"]
        par.append(p)
        s = pd.read_csv(r / "c1stats.csv"); s.insert(0, "dataset", DSL[c["dataset"]]); s.insert(1, "config", c["config"]); s["run"] = r.name
        c1s.append(s)
    if par:
        pd.concat(par).to_csv(OUT / "parity_dev.csv", index=False)
        pd.concat(c1s).to_csv(OUT / "c1stats_dev.csv", index=False)
    # ---- test-frame window: every timed pipeline x repetition
    rows, pt, frames, md5 = [], [], {}, {}
    for r, c in runs("timing", args.since):
        tag = c.get("tag", "")
        if not (tag.startswith("r") and tag[1:2].isdigit()) or ("_g" in tag) != bool(args.tagsuf) or not tag.endswith(args.tagsuf):
            continue
        rep = int(tag[1])
        d = pd.read_pickle(r / "frames.pkl"); d = d[~d.warm].reset_index(drop=True)
        md5.update(json.loads((r / "meta.json").read_text())["engines_md5"])
        L = d.J_cheap.to_numpy(float)
        for col in [x for x in d.columns if x.startswith("ms_")]:
            nm = col[3:]
            base, p = split_path(nm)
            if p not in ("ref", args.sel):
                continue
            lab, grp = label(base, c["config"])
            if base == "cheap" and c["config"] != "v6S":
                continue
            if base.startswith("u") and "+" not in base and p != "ref":
                continue
            key = (c["dataset"], lab, p, rep)
            if key in frames:
                continue
            frames[key] = pd.DataFrame({"unit": d.unit, "frame": d.frame, "J": d[f"J_{nm}"].astype(float), "J0": L})
            rows.append({"dataset": c["dataset"], "pipeline": lab, "group": grp, "path": p, "rep": rep, "run": r.name, "frames": len(d),
                         "units": d.unit.nunique(), "G": (L.sum() - d[f"J_{nm}"].sum()) / L.sum() if nm != "cheap" else 0.0,
                         **lat_stats(d[col]), "px_mean": d[f"px_{nm}"].mean() if f"px_{nm}" in d else np.nan})
        q = pd.read_csv(r / "parity.csv")
        if len(q):
            q = q[q.path == args.sel].copy()
            q["pipeline"] = [label(x, c["config"])[0] for x in q.pipeline]
            q.insert(0, "dataset", c["dataset"]); q.insert(1, "rep", rep); q["run"] = r.name
            pt.append(q)
    if not rows:
        print("no test-window runs yet"); return
    TOUT = OUT / args.out_sub if args.out_sub else OUT              # test-window tables (one GPU clock per directory)
    TOUT.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(TOUT / "latency_runs.csv", index=False)
    pd.concat(pt).to_csv(TOUT / "parity_timed.csv", index=False)
    agg = t.groupby(["dataset", "pipeline", "group", "path"]).agg(
        reps=("rep", "nunique"), frames=("frames", "first"), units=("units", "first"), G=("G", "first"), G_spread=("G", lambda x: x.max() - x.min()),
        mean_ms=("mean_ms", "median"), median_ms=("median_ms", "median"), p95_ms=("p95_ms", "median"), p99_ms=("p99_ms", "median"),
        max_ms=("max_ms", "median"), px_mean=("px_mean", "first")).reset_index()
    # plain whole frame and cheap have one path; they belong to both tables
    shared = agg[agg.group.isin(["cheap", "whole frame"])]
    fast = pd.concat([shared.assign(path=args.sel), agg[(agg.path == args.sel)]]).drop_duplicates(["dataset", "pipeline"])
    ref = agg[agg.path == "ref"]
    fast.to_csv(TOUT / "latency_fast.csv", index=False)
    ref.to_csv(TOUT / "latency_ref.csv", index=False)
    tails = ["cheap (u320)", "whole frame u960 + C1 + ref", "whole frame u1280 + C1 + ref", "whole frame native + C1 + ref", "v6-S",
             "v6-S, rounds <= 2"]
    tl = pd.concat([fast.assign(table="fast"), ref.assign(table="ref")])
    tl = tl[tl.pipeline.isin(tails)][["table", "dataset", "pipeline", "G", "mean_ms", "p95_ms", "p99_ms", "max_ms"]]
    tl.to_csv(TOUT / "tail.csv", index=False)
    # ---- front + intervals (rep 1 per-frame J; every rep has the same frames)
    fr, iv = [], []
    rng = np.random.default_rng(0)
    for setname, tab in (("fast", fast), ("ref", ref)):
        tab = tab.copy()
        for lat in ("mean_ms", "p95_ms"):
            dom = []
            for _, a in tab.iterrows():
                o = tab[(tab.dataset == a.dataset) & (tab.pipeline != a.pipeline)]
                o = o[(o[lat] <= a[lat]) & (o.G >= a.G) & ((o[lat] < a[lat]) | (o.G > a.G))]
                dom.append("; ".join(o.pipeline))
            tab[f"on_front_{lat[:-3]}"] = [x == "" for x in dom]
            tab[f"dominated_by_{lat[:-3]}"] = dom
        tab.insert(0, "path_set", setname)
        fr.append(tab)
        for ds in tab.dataset.unique():
            pth = args.sel if setname == "fast" else "ref"
            base = frames.get((ds, "cheap (u320)", "ref", 1))
            o = frames.get((ds, "v6-S", pth, 1))
            if base is None or o is None:
                continue
            uq = base.unit.unique(); idx = [np.flatnonzero(base.unit.to_numpy() == u) for u in uq]
            boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
            L = base.J0.to_numpy(float)
            Vo = L - base[["unit", "frame"]].merge(o, on=["unit", "frame"], how="left").J.to_numpy(float)
            for _, b in tab[(tab.dataset == ds) & (tab.pipeline != "v6-S")].iterrows():
                fb = frames.get((ds, b.pipeline, b.path if b.group not in ("cheap", "whole frame") else "ref", 1))
                if fb is None:
                    continue
                Vb = L - base[["unit", "frame"]].merge(fb, on=["unit", "frame"], how="left").J.to_numpy(float)
                dr = [(Vo[i].sum() - Vb[i].sum()) / L[i].sum() for i in boots]
                a = tab[(tab.dataset == ds) & (tab.pipeline == "v6-S")].iloc[0]
                iv.append({"path_set": setname, "dataset": ds, "row": b.pipeline, "G_v6S": a.G, "G_row": b.G, "diff": a.G - b.G,
                           "ci_lo": np.percentile(dr, 2.5), "ci_hi": np.percentile(dr, 97.5), "mean_ms_v6S": a.mean_ms, "mean_ms_row": b.mean_ms,
                           "p95_ms_v6S": a.p95_ms, "p95_ms_row": b.p95_ms, "row_faster_mean": b.mean_ms <= a.mean_ms,
                           "row_faster_p95": b.p95_ms <= a.p95_ms})
    front = pd.concat(fr)
    front.to_csv(TOUT / "front.csv", index=False)
    pd.DataFrame(iv).to_csv(TOUT / "front_intervals.csv", index=False)
    (TOUT / "engines_md5.json").write_text(json.dumps(md5, indent=1))
    # ---- breakdown
    bd = []
    for r, c in runs("breakdown", args.since):
        if c.get("tag") != "bd" + args.tagsuf:
            continue
        d = pd.read_pickle(r / "frames.pkl"); d = d[~d.warm].reset_index(drop=True)
        for nm in ("ours", f"ours@{args.sel}"):
            if f"ms_{nm}" not in d:
                continue
            tot = d[f"ms_{nm}"]
            comps = {}
            for k in ("draft", "field", "crops_r1", "crops_r2", "crops_r3", "guard", "c1", "refiner"):
                col = f"bd_{k}_{nm}"
                comps[k] = d[col].fillna(0.0) if col in d else pd.Series(0.0, index=d.index)
            comps["rest (support, merge, lift)"] = tot - sum(comps.values())
            comps["total"] = tot
            for k, s in comps.items():
                bd.append({"dataset": c["dataset"], "path": "ref" if nm == "ours" else args.sel, "component": k, "mean_ms": s.mean(),
                           "p95_ms": s.quantile(.95), "share_of_mean": s.mean() / tot.mean(), "frames": len(d)})
    if bd:
        pd.DataFrame(bd).to_csv(TOUT / "breakdown.csv", index=False)
    # ---- figure
    sty = {"cheap": ("k", "s"), "whole frame": ("tab:gray", "o"), "whole frame + T4 + ref": ("tab:blue", "^"),
           "whole frame + C1 + ref": ("tab:purple", "v"), "heuristic + C1": ("tab:green", "P"), "previous ours": ("tab:orange", "D"),
           "ours": ("tab:red", "*")}
    for setname in ("fast", "ref"):
        tab = front[front.path_set == setname]
        dss = [x for x in ("nusc_test20", "nusc", "av2") if x in set(tab.dataset)]
        fig, ax = plt.subplots(len(dss), 2, figsize=(11, 4.2 * len(dss)), squeeze=False)
        for i, ds in enumerate(dss):
            for j, lat in enumerate(("mean_ms", "p95_ms")):
                a = ax[i, j]
                tt = tab[tab.dataset == ds]
                for g, (col, mk) in sty.items():
                    s = tt[tt.group == g]
                    a.scatter(s[lat], s.G, c=col, marker=mk, s=110 if g == "ours" else 40, label=g, zorder=3 if g == "ours" else 2)
                f = tt[tt[f"on_front_{lat[:-3]}"]].sort_values(lat)
                a.step(f[lat], f.G, where="post", c="0.6", lw=1, zorder=1)
                for _, r in tt[tt.pipeline.isin(["v6-S", "v6-S, rounds <= 2", "v5.1-S", "v6-M", "v6-L"])].iterrows():
                    a.annotate(r.pipeline, (r[lat], r.G), fontsize=7, xytext=(5, -10), textcoords="offset points")
                a.set_xscale("log"); a.set_xlabel(f"{'mean' if j == 0 else 'p95'} latency per frame, Jetson AGX Xavier (ms)")
                a.set_ylabel("G"); a.set_title(f"{DSL[ds]} — {'fast path' if setname == 'fast' else 'K28 reference path'}", fontsize=9)
                a.grid(alpha=.3)
        ax[0, 0].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(TOUT / f"front_{setname}.{ext}", dpi=150)
        plt.close(fig)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    print(front[["path_set", "dataset", "pipeline", "G", "mean_ms", "p95_ms", "on_front_mean", "on_front_p95"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
