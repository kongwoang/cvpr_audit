#!/usr/bin/env python
"""K28 PHASE 3 (post-hoc) — S3: is ours on the Jetson latency-G front, in mean and in p95, on both timed datasets
(nuScenes: the 20 local DEEP scenes, 797 timed frames; AV2: 8 local test logs with the ego mask, 586 timed frames)?
Inputs: the final-window runs of 61_k28_v6_latency.py (results/raw/20261014_*_k28_v6_latency_{nusc,av2}_*_final*), all on the
same frames (warm-up dropped). Every timed pipeline enters the front: cheap, whole frame (YOLOv8s at every timed side; YOLOv8m /
YOLO11m at 448 / 640 on nuScenes), whole frame + T4 + refiner v3a ("+comp"), whole frame + C1 + refiner v3a ("+c1comp"),
v5.1-S, and ours v6-S / M / L (with / without refiner, round-bounded). A row is dominated if another row has latency <= and
G >= with one strict. G intervals: paired unit bootstrap of (ours - row), 2,000 resamples, seed 0.
Outputs: results/k28_jetson/s3_front.csv, s3_dominance.csv, cost_model.csv, s3_front.{pdf,png}
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "k28_jetson"
CFG = {"v6S": "v6-S", "v6M": "v6-M", "v6L": "v6-L", "v51S": "v5.1-S"}
WFN = {"yolov8s": "YOLOv8s", "yolov8m": "YOLOv8m", "yolo11m": "YOLO11m"}


def label(nm, cfg, wf_det):
    if nm == "cheap":
        return "cheap (YOLOv8s 192x320)", "cheap"
    if nm.startswith("ours"):
        b = nm.split("_b")[1].split("_")[0] if "_b" in nm else None
        s = CFG[cfg] + (f", rounds <= {b}" if b else "") + (", no refiner" if nm.endswith("noref") else "")
        return s, ("ours" if cfg != "v51S" else "previous ours")
    side, _, kind = nm.partition("+")
    s = f"whole frame {side} ({WFN[wf_det]})" + {"": "", "comp": " + T4 + refiner", "c1comp": " + C1 + refiner"}[kind]
    return s, "whole frame" + {"": "", "comp": " + components", "c1comp": " + C1 + components"}[kind]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, frames, cm = [], {}, []
    for r in sorted(glob.glob(str(ROOT / "results" / "raw" / "20261014_*_k28_v6_latency_*_final*"))):
        c = json.loads((Path(r) / "config.json").read_text())
        if c["dataset"] not in ("nusc", "av2") or not (Path(r) / "k28_latency.pkl").exists():
            continue
        d = pd.read_pickle(Path(r) / "k28_latency.pkl")
        d = d[~d.warm].reset_index(drop=True)
        ds = c["dataset"]
        for col in [x for x in d.columns if x.startswith("ms_")]:
            nm = col[3:]
            wf = c["wf_det"] if not nm.startswith("ours") else c["verifier"]
            lab, grp = label(nm, c["config"], wf)
            key = (ds, lab)
            if key in frames:                                   # same pipeline timed again in a later run: keep the first
                continue
            frames[key] = pd.DataFrame({"unit": d.unit, "frame": d.frame, "J": d[f"J_{nm}"].astype(float)})
            L = d.J_cheap.to_numpy(float)
            rows.append({"dataset": ds, "pipeline": lab, "group": grp, "run": Path(r).name, "frames": len(d), "units": d.unit.nunique(),
                         "G": float((L - d[f"J_{nm}"].to_numpy(float)).sum() / L.sum()), "mean_ms": d[col].mean(),
                         "median_ms": d[col].median(), "p95_ms": d[col].quantile(.95), "p99_ms": d[col].quantile(.99),
                         "px_mean": d[f"px_{nm}"].mean()})
            if nm in ("ours", "ours_noref") and c["config"] != "v51S":
                X = np.c_[np.ones(len(d)), d[f"calls_ours"], d[f"px_{nm}"]]
                y = d[col].to_numpy(float)
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                res = y - X @ beta
                cm.append({"dataset": ds, "pipeline": lab, "frames": len(d), "ms_intercept": beta[0], "ms_per_call": beta[1],
                           "ms_per_native_px_frac": beta[2], "R2": 1 - res.var() / y.var(), "resid_sd_ms": res.std(),
                           "calls_mean": d.calls_ours.mean(), "calls_p95": d.calls_ours.quantile(.95)})
    t = pd.DataFrame(rows)
    for lat in ("mean_ms", "p95_ms"):
        dom = []
        for _, a in t.iterrows():
            o = t[(t.dataset == a.dataset) & (t.pipeline != a.pipeline)]
            o = o[(o[lat] <= a[lat]) & (o.G >= a.G) & ((o[lat] < a[lat]) | (o.G > a.G))]
            dom.append("; ".join(o.pipeline))
        t[f"on_front_{lat[:-3]}"] = [x == "" for x in dom]
        t[f"dominated_by_{lat[:-3]}"] = dom
    t.to_csv(OUT / "s3_front.csv", index=False)
    # paired unit bootstrap of G(ours) - G(row) for every ours row against every row that dominates it or sits next to it
    rng = np.random.default_rng(0)
    out = []
    for ds in ("nusc", "av2"):
        base = frames[(ds, "cheap (YOLOv8s 192x320)")]
        uq = base.unit.unique(); idx = [np.flatnonzero(base.unit.to_numpy() == u) for u in uq]
        boots = [np.concatenate([idx[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(2000)]
        al = {k[1]: base[["unit", "frame"]].merge(v, on=["unit", "frame"], how="left") for k, v in frames.items() if k[0] == ds}
        L = base.J.to_numpy(float)
        tt = t[t.dataset == ds]
        for o in tt[tt.group == "ours"].pipeline:
            Vo = L - al[o].J.to_numpy(float)
            for _, b in tt[tt.pipeline != o].iterrows():
                Vb = L - al[b.pipeline].J.to_numpy(float)
                dr = [(Vo[i].sum() - Vb[i].sum()) / L[i].sum() for i in boots]
                a = tt[tt.pipeline == o].iloc[0]
                out.append({"dataset": ds, "ours": o, "row": b.pipeline, "row_group": b.group, "G_ours": a.G, "G_row": b.G,
                            "diff": a.G - b.G, "ci_lo": np.percentile(dr, 2.5), "ci_hi": np.percentile(dr, 97.5),
                            "mean_ms_ours": a.mean_ms, "mean_ms_row": b.mean_ms, "p95_ms_ours": a.p95_ms, "p95_ms_row": b.p95_ms,
                            "row_faster_mean": b.mean_ms <= a.mean_ms, "row_faster_p95": b.p95_ms <= a.p95_ms})
    pd.DataFrame(out).to_csv(OUT / "s3_dominance.csv", index=False)
    pd.DataFrame(cm).to_csv(OUT / "cost_model.csv", index=False)
    # figure: G vs latency (mean, p95) for both datasets
    sty = {"cheap": ("k", "s"), "whole frame": ("tab:gray", "o"), "whole frame + components": ("tab:blue", "^"),
           "whole frame + C1 + components": ("tab:purple", "v"), "previous ours": ("tab:orange", "D"), "ours": ("tab:red", "*")}
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    for i, ds in enumerate(("nusc", "av2")):
        for j, lat in enumerate(("mean_ms", "p95_ms")):
            a = ax[i, j]
            tt = t[t.dataset == ds]
            main = tt.pipeline.isin(["v6-S", "v6-M", "v6-L", "v5.1-S"])
            for g, (col, mk) in sty.items():
                s = tt[(tt.group == g) & (main | ~tt.group.str.contains("ours"))]
                a.scatter(s[lat], s.G, c=col, marker=mk, s=110 if g == "ours" else 40, label=g, zorder=3 if g == "ours" else 2)
                v = tt[(tt.group == g) & ~main & tt.group.str.contains("ours")]
                if len(v):
                    a.scatter(v[lat], v.G, facecolors="none", edgecolors=col, marker=mk, s=40, lw=.8, zorder=2,
                              label=f"{g}: no refiner / rounds <= 2, 3")
            f = tt[tt[f"on_front_{lat[:-3]}"]].sort_values(lat)
            a.step(f[lat], f.G, where="post", c="0.6", lw=1, zorder=1)
            for k, (_, r) in enumerate(tt[main].sort_values("G").iterrows()):
                a.annotate(r.pipeline, (r[lat], r.G), fontsize=8, xytext=(6, -12 + 8 * (k % 2)), textcoords="offset points")
            a.set_xscale("log"); a.set_xlabel(f"{'mean' if j == 0 else 'p95'} latency per frame on Jetson AGX Xavier (ms)")
            a.set_ylabel("G (decision value captured)"); a.set_title({"nusc": "nuScenes (20 scenes, 797 frames)",
                                                                      "av2": "AV2, ego mask (8 logs, 586 frames)"}[ds], fontsize=10)
            a.grid(alpha=.3)
    ax[0, 0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"s3_front.{ext}", dpi=150)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(t[["dataset", "pipeline", "G", "mean_ms", "p95_ms", "px_mean", "on_front_mean", "on_front_p95", "dominated_by_mean",
             "dominated_by_p95"]].round(3).to_string(index=False))
    print(pd.DataFrame(cm).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
