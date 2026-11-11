#!/usr/bin/env python
"""K41-3 D05 figure: G vs pixels / v6-S for the grid points, one panel per braking split (refiner v3a).
Range spacing is a no-op (bit-identical crops for factors 0.5 / 1 / 2; K41/STATUS 2026-11-02 18:10), so each panel shows
the 27 distinct (lateral factor, probe, v_close) points at range factor 1. Colour = probe, marker = lateral factor,
filled = Holm-significant difference from v6-S (either sign), open = n.s.; star = frozen v6-S.
usage: k41_d05_figure.py <k41_3_grid.csv> <out_prefix>   -> <out_prefix>.pdf / .png"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

COL = {"ped": "#2a78d6", "car": "#eb6834", "truck": "#1baf7a"}     # categorical slots 1-3 (validated default palette)
MRK = {0.5: "o", 1.0: "s", 2.0: "^"}
TITLE = {"nusc": "nuScenes", "av2": "AV2", "ahd": "Argoverse-HD"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def main():
    t = pd.read_csv(sys.argv[1]); out = sys.argv[2]
    t = t[(t.refiner == "v3a") & (t.range_factor == 1.0)]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.7))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.27, wspace=0.22)
    for ax, ds in zip(axs, ("nusc", "av2", "ahd")):
        d = t[t.dataset == ds]
        for _, r in d.iterrows():
            sig = r.p_holm < 0.05
            ax.scatter(r.px_ratio, r.G, s=34, marker=MRK[r.lateral_factor], linewidths=1.4, zorder=3,
                       facecolors=COL[r.probe] if sig else "white", edgecolors=COL[r.probe])
        ax.scatter([1.0], [d.G_v6S.iloc[0]], s=120, marker="*", color=INK, zorder=4)
        ax.axhline(d.G_v6S.iloc[0], color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.set_title(TITLE[ds], fontsize=10, color=INK)
        ax.set_xlabel("pixels / v6-S", fontsize=9, color=INK)
        ax.grid(color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    axs[0].set_ylabel("G (refiner v3a)", fontsize=9, color=INK)
    h = [Line2D([], [], ls="", marker="o", mfc=c, mec=c, label=p) for p, c in COL.items()]
    h += [Line2D([], [], ls="", marker=m, mfc="white", mec=MUTED, label=f"lateral ×{k:g}") for k, m in MRK.items()]
    h += [Line2D([], [], ls="", marker="*", color=INK, ms=11, label="v6-S (frozen)"),
          Line2D([], [], ls="", marker="o", mfc=MUTED, mec=MUTED, label="Holm-significant vs v6-S"),
          Line2D([], [], ls="", marker="o", mfc="white", mec=MUTED, label="n.s.")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=7.5, frameon=False, handletextpad=0.3, columnspacing=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)


if __name__ == "__main__":
    main()
