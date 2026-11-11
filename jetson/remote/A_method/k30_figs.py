#!/usr/bin/env python
"""K30 figures (PDF + PNG) from results/k30/tables (and results/navsim_exp/k30_probe_time): post-hoc analyses of v6-S.
  k30_v1a_matched   G vs realised px (ratio to v6-S) of the 8 S heuristics + C1 + v3a at m x v6-S's px cap, per dataset
  k30_v1b_axis      v6-S over crop long side x support context: G - v6-S with 95% intervals, per dataset
  k30_v8d_speed     nuScenes, ego speed x {0.5, 1, 1.5}: G per pipeline, and v6-S's field (farthest sensitive range, area)
  k30_v8c_kitti     KITTI: 2D AP (moderate; Car, Pedestrian) and the mean of DEEP's four cells vs px, flag 0.0
  k30_v6ii_probe    NAVSIM probing: measured wall clock relative to 7x7 with intervals, and calls per frame
usage: k30_figs.py [--out results/k30/figs]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

B = Path(os.environ.get("B_REAL", os.environ["B"]))
T = B / "results" / "k30" / "tables"
DSN = {"nusc": "nuScenes (new test)", "av2": "AV2 val (ego mask)", "ahd": "Argoverse-HD val"}
FAMC = {"uncertainty": "tab:orange", "temporal-crit": "tab:green", "ego corridor": "tab:purple", "hybrid (support + tube)": "tab:red"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7, "figure.dpi": 150})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def v1a(out):
    d = pd.read_csv(T / "v1a_matched.csv")
    d = d[d.refiner == "v3a"]
    dss = [x for x in ("nusc", "av2", "ahd") if x in set(d.dataset)]
    fig, axs = plt.subplots(1, len(dss), figsize=(3.2 * len(dss), 2.8), squeeze=False)
    for ax, ds in zip(axs[0], dss):
        x = d[d.dataset == ds]
        for row, g in x[x.row != "v6-S"].groupby("row"):
            g = g.sort_values("m")
            fam = g.family.iloc[0]
            ax.plot(g.px_ratio_to_v6S, g.G, "-o", ms=2.5, lw=0.8, color=FAMC.get(fam, "grey"), alpha=0.8,
                    label=fam if fam not in ax.get_legend_handles_labels()[1] else None)
        s = x[x.row == "v6-S"].iloc[0]
        ax.plot([1.0], [s.G], "k*", ms=9, label="v6-S")
        ax.axhline(s.G, color="k", lw=0.5, ls=":")
        ax.set_title(DSN[ds]); ax.set_xlabel("realised px / v6-S px"); ax.set_ylabel("G (refiner v3a)")
        ax.grid(alpha=0.3)
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("V1a: S heuristics + C1 + v3a at a per-frame cap of m x v6-S's px (m = 0.75 ... 1.5)", fontsize=9)
    save(fig, out, "k30_v1a_matched")


def v1b(out):
    d = pd.read_csv(T / "v1b_axis.csv")
    d = d[d.crop_long_side.notna()]
    dss = [x for x in ("nusc", "av2", "ahd") if x in set(d.dataset)]
    fig, axs = plt.subplots(1, len(dss), figsize=(3.2 * len(dss), 2.6), squeeze=False)
    for ax, ds in zip(axs[0], dss):
        x = d[d.dataset == ds]
        for i, (ctx, g) in enumerate(x.groupby("support_context")):
            g = g.sort_values("crop_long_side")
            xs = g.crop_long_side + (i - 1) * 25
            ax.errorbar(xs, g.minus_v6S, yerr=[g.minus_v6S - g.ci_lo, g.ci_hi - g.minus_v6S], fmt="o", ms=3, capsize=2,
                        label=f"support context {ctx:g}")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(DSN[ds]); ax.set_xlabel("crop long side (px)"); ax.set_ylabel("G(point) - G(v6-S)")
        ax.grid(alpha=0.3)
    axs[0][0].legend(loc="best")
    fig.suptitle("V1b: v6-S's axis (crop side x support context); v6-S = (960, 2)", fontsize=9)
    save(fig, out, "k30_v1b_axis")


def v8d(out):
    d = pd.read_csv(T / "v8d_speed.csv")
    p = d[~d.pipeline.str.startswith("(")]
    f = d[d.pipeline.str.startswith("(")].sort_values("speed_scale")
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for lab, g in p.groupby("pipeline"):
        g = g.sort_values("speed_scale")
        axs[0].plot(g.speed_scale, g.G, "-o", ms=3, lw=1.2 if lab == "v6-S" else 0.8, color="k" if lab == "v6-S" else None, label=lab)
    axs[0].set_xlabel("ego speed scale s"); axs[0].set_ylabel("G (refiner v3a)"); axs[0].set_xticks([0.5, 1, 1.5]); axs[0].grid(alpha=0.3)
    axs[0].legend(fontsize=6); axs[0].set_title("nuScenes new test, G")
    axs[1].plot(f.speed_scale, f.field_rmax_mean_nonempty, "-o", color="tab:blue", ms=3)
    axs[1].set_ylabel("farthest sensitive range (m)", color="tab:blue"); axs[1].set_xlabel("ego speed scale s"); axs[1].set_xticks([0.5, 1, 1.5])
    a2 = axs[1].twinx()
    a2.plot(f.speed_scale, f.field_area_v6S_mean, "-s", color="tab:red", ms=3, label="v6-S field crop area")
    a2.plot(f.speed_scale, f.crop_area_tsp8_mean, "--s", color="tab:gray", ms=3, label="hybrid crop area")
    a2.set_ylabel("area / image"); a2.legend(fontsize=6, loc="upper left"); axs[1].set_title("v6-S's field")
    fig.suptitle("V8d: ego speed scaled in the controller, field, support and GT closing speeds", fontsize=9)
    fig.tight_layout()
    save(fig, out, "k30_v8d_speed")


def v8c(out):
    a = pd.read_csv(T / "v8c_kitti_ap.csv")
    c = pd.read_csv(T / "v8c_kitti_cells.csv")
    c = c[c.frames_set.str.startswith("all")]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for cls, mk in (("Car", "o"), ("Pedestrian", "s")):
        x = a[(a["class"] == cls) & (a.difficulty == "moderate")]
        axs[0].plot(x.px_mean, x.AP_R40, mk, ms=4, label=cls)
        for _, r in x.iterrows():
            axs[0].annotate(r.pipeline, (r.px_mean, r.AP_R40), fontsize=6, xytext=(3, 2), textcoords="offset points")
    axs[0].set_xlabel("px / native"); axs[0].set_ylabel("2D AP R40, moderate"); axs[0].grid(alpha=0.3); axs[0].set_xlim(0, 1.25)
    axs[0].legend(); axs[0].set_title("KITTI test (6 seq), 2D AP")
    px = {"u960": 288 * 960 / (375 * 1242), "native": 384 * 1248 / (375 * 1242)}
    for _, r in c.iterrows():
        nm = r.pipeline
        if "ref_f1" in nm or nm.endswith("_f1"):
            continue
        base = nm.split("+")[0]
        xx = r.px_mean if nm.startswith("v6S") else px.get(base, np.nan)
        axs[1].plot(xx, r.mean4, "k*" if nm == "v6S_f0" else "o", ms=7 if nm == "v6S_f0" else 3)
        axs[1].annotate(nm.replace("_f0", ""), (xx, r.mean4), fontsize=5.5, xytext=(3, 1), textcoords="offset points")
    axs[1].set_xlabel("px / native"); axs[1].set_ylabel("G, mean of DEEP's 4 cells"); axs[1].grid(alpha=0.3); axs[1].set_xlim(0, 1.35)
    axs[1].set_title("KITTI decision cells, refiner flag 0.0")
    fig.tight_layout()
    save(fig, out, "k30_v8c_kitti")


def v6ii(out):
    p = B / "results" / "navsim_exp" / "k30_probe_time" / "v6ii_probe_time.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.4))
    axs[0].bar(d.variant, d.ratio_to_7x7, yerr=[d.ratio_to_7x7 - d.ratio_ci_lo, d.ratio_ci_hi - d.ratio_to_7x7], capsize=3, color="tab:blue")
    axs[0].set_ylabel("wall clock / 7x7 (same tokens)"); axs[0].set_title("measured probing time"); axs[0].grid(alpha=0.3, axis="y")
    axs[1].bar(d.variant, d.calls_mean, color="tab:gray")
    axs[1].set_ylabel("PDM-Closed calls per frame"); axs[1].set_title("calls"); axs[1].grid(alpha=0.3, axis="y")
    fig.suptitle("V6 (ii): NAVSIM planner probing, 1,000 navtest tokens", fontsize=9)
    fig.tight_layout()
    save(fig, out, "k30_v6ii_probe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(B / "results" / "k30" / "figs"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (v1a, v1b, v8d, v8c, v6ii):
        try:
            f(out)
        except Exception as e:                                                        # noqa: BLE001
            print(f.__name__, "failed:", type(e).__name__, e)


if __name__ == "__main__":
    main()
