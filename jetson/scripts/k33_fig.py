#!/usr/bin/env python
"""K33 figure: decision value G vs tensor px (log) per test split, every Table 1 v7 row (filled) and its pixel-cap form
(hollow, connected), v6-S starred; 95% paired-bootstrap interval of G(v6-S) - G(row) shown as the bar colour (grey = n.s.).
-> docs/paper/handoff/K33/fig_table1_v7.{pdf,png}"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

D = Path(__file__).resolve().parents[1] / "docs" / "paper" / "handoff" / "K33"
T = pd.read_csv(D / "table1_v7.csv")
NAME = {"ada+1t1280": "AdaScale", "sahi640o1s10": "SAHI", "fovi1280b64a2": "FOVEA KDE_I", "fovs1280b64a1": "FOVEA KDE_S",
        "bar1280_0.5": "BAR", "dvr1280_0.5": "DVR", "ufp20s512": "UFP", "ch128B_csc": "Chanakya", "ana05hinfo": "AnA",
        "ana05hinfi": "AnA iter.", "cf40q20o": "CF", "cf40q20i": "CF iter.", "wf640": "u640", "wf768": "u768", "wf960": "u960",
        "wf1280": "u1280", "cheap": "cheap", "v6S": "v6-S"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
for a, (ds, title) in zip(ax, (("nusc", "nuScenes test"), ("av2", "AV2 val (ego mask)"), ("ahd", "Argoverse-HD val"))):
    t = T[(T.dataset == ds) & T.row.isin(NAME)]
    for _, r in t.iterrows():
        if r.row == "v6S":
            a.scatter(r.px, r.G, marker="*", s=260, c="crimson", zorder=5)
            a.annotate("v6-S", (r.px, r.G), xytext=(5, 6), textcoords="offset points", color="crimson", fontsize=9, weight="bold")
            continue
        sig = (r.delta_lo > 0) or (r.delta_hi < 0)
        col = ("tab:blue" if r.delta_v6S_minus_row > 0 else "tab:orange") if sig else "grey"
        a.scatter(r.px, r.G, c=col, s=36, zorder=4)
        a.annotate(NAME[r.row], (r.px, r.G), xytext=(4, -10), textcoords="offset points", fontsize=7, color=col)
        if pd.notna(r.get("G_cap")):
            sc = (r.delta_cap_lo > 0) or (r.delta_cap_hi < 0)
            cc = ("tab:blue" if r.delta_cap > 0 else "tab:orange") if sc else "grey"
            a.scatter(r.px_cap, r.G_cap, facecolors="none", edgecolors=cc, s=36, zorder=3)
            a.plot([r.px, r.px_cap], [r.G, r.G_cap], c=cc, lw=0.5, alpha=0.5)
    a.set_xscale("log"); a.set_title(title); a.set_xlabel("tensor px / native (log)"); a.axhline(0, c="k", lw=0.5)
    a.grid(alpha=0.3)
ax[0].set_ylabel("decision value G")
fig.text(0.5, -0.02, "filled: row as selected on dev; hollow: the same row under v6-S's per-frame pixel cap. Blue: v6-S significantly "
         "better; orange: the row significantly better; grey: n.s. (paired unit bootstrap, 95%).", ha="center", fontsize=9)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(D / f"fig_table1_v7.{ext}", dpi=170, bbox_inches="tight")
print("saved")
