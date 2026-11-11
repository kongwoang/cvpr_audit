#!/usr/bin/env python
"""K28 figure: decision value G (with refiner / own refiner) against compute (GFLOPs per frame relative to v5.1-S, log
scale) on the test splits, one panel per dataset, from the phase-2 tables (table_main_<ds>_v6.csv). Rows: v6-S / M / L,
v5.1-S, whole frame (YOLOv8s + C1, YOLO11m + C1, YOLOv8m + C1), crop heuristics + C1 (YOLOv8s), routers; the S / M / L
caps are drawn as vertical lines. PDF + PNG.
usage: k28_fig_pareto.py <tables_v6 dir> <out stem>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DS = (("nusc", "nuScenes new test"), ("av2", "AV2 val (ego mask)"), ("ahd", "Argoverse-HD val"))
STYLE = [  # (label, selector, marker, colour)
    ("v6-S / M / L (ours)", lambda t: t.group == "ours v6", "*", "#d62728"),
    ("v5.1-S (K24)", lambda t: t.method == "v5.1-S", "P", "#ff7f0e"),
    ("whole frame YOLOv8s + C1", lambda t: t.group.eq("K28 whole frame + C1") & t.crop_detector.eq("yolov8s"), "o", "#1f77b4"),
    ("whole frame YOLO11m + C1", lambda t: t.group.eq("K28 whole frame + C1") & t.crop_detector.eq("yolo11m"), "s", "#17becf"),
    ("whole frame YOLOv8m + C1", lambda t: t.group.eq("K28 whole frame + C1") & t.crop_detector.eq("yolov8m"), "D", "#9467bd"),
    ("crop heuristics YOLOv8s + C1", lambda t: t.group.eq("K28 heuristic+C1") & t.crop_detector.eq("yolov8s"), "^", "#2ca02c"),
    ("routers (K24, + components)", lambda t: t.group.eq("router + components") & ~t.method.str.contains("oracle"), "v", "#7f7f7f"),
    ("oracle router, true V (K24, + components)", lambda t: t.group.eq("router + components") & t.method.str.contains("oracle"), "X", "#000000"),
]


def main():
    src, stem = Path(sys.argv[1]), Path(sys.argv[2])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (ds, title) in zip(axs, DS):
        f = src / f"table_main_{ds}_v6.csv"
        if not f.exists():
            ax.set_title(f"{title}: no table"); continue
        t = pd.read_csv(f)
        for lab, sel, mk, col in STYLE:
            x = t[sel(t)]
            if len(x):
                ax.scatter(x.GFLOPs_ratio, x.G_refiner, marker=mk, s=110 if mk == "*" else 32, color=col, label=lab,
                           edgecolor="black" if mk == "*" else "none", linewidth=0.5, zorder=3 if mk == "*" else 2)
        for c in (1.0, 1.5, 2.5):
            ax.axvline(c * 1.02, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xscale("log"); ax.set_xlabel("GFLOPs per frame / v5.1-S (log)"); ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("G (with refiner)")
    h, lb = axs[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(h, lb, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    print("written", stem)


if __name__ == "__main__":
    main()
