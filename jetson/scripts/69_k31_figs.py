#!/usr/bin/env python
"""K31 figures from the handoff CSVs (docs/paper/handoff/K31/): PDF + PNG next to them.
  k31_episodes      item 1: missed required episodes, mean onset delay (s), false episodes / 1,000 frames, reversals / 100
                    frames per pipeline and dataset (required level)
  k31_consumers     item 2: DESP's TTC loop / ours-IDM / frozen v6-S minus each row, 95% intervals
  k31_strata        item 5: v6-S minus each pipeline per stratum, 95% intervals
  k31_curv          item 3 (when curv_test.csv exists): curved v6-S minus each row, overall / terciles / outside the tube
  k31_bounded       item 4 (when bounded.csv exists): v6-S <= 2 minus each row
usage: scripts/py scripts/69_k31_figs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                   # noqa: E402
import pandas as pd                                                               # noqa: E402

D = Path(__file__).resolve().parents[1] / "docs" / "paper" / "handoff" / "K31"
DS = ["nuScenes new test", "AV2 val (ego mask)", "Argoverse-HD val"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(D / f"{name}.pdf"); fig.savefig(D / f"{name}.png", dpi=150)
    plt.close(fig)


def forest(ax, x, lab, title):
    y = range(len(x))
    ax.errorbar(x["diff"], list(y), xerr=[x["diff"] - x["lo"], x["hi"] - x["diff"]], fmt="o", ms=3, capsize=2,
                color=["tab:blue" if (lo > 0 or hi < 0) else "0.5" for lo, hi in zip(x["lo"], x["hi"])][0] if len(x) == 1 else "k")
    for i, (d, lo, hi) in enumerate(zip(x["diff"], x["lo"], x["hi"])):
        ax.plot([lo, hi], [i, i], color="tab:blue" if (lo > 0 or hi < 0) else "0.6", lw=2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(list(y)); ax.set_yticklabels(lab, fontsize=7)
    ax.set_title(title, fontsize=8)


def episodes():
    s = pd.read_csv(D / "ep_summary.csv")
    s = s[s.level.str.startswith("required") & ~s.pipeline.str.startswith("required")]
    stats = [("missed_rate", "missed required episodes (share)"), ("mean_delay_s", "mean onset delay (s, capped)"),
             ("false_per_1000", "false episodes / 1,000 frames"), ("reversals_per_100", "A->B->A reversals / 100 frames")]
    fig, axs = plt.subplots(len(DS), len(stats), figsize=(13, 8))
    for i, ds in enumerate(DS):
        x = s[s.dataset == ds]
        for j, (c, t) in enumerate(stats):
            ax = axs[i, j]
            col = ["tab:red" if p == "v6-S" else "tab:gray" for p in x.pipeline]
            ax.barh(range(len(x)), x[c], color=col)
            ax.set_yticks(range(len(x))); ax.set_yticklabels(x.pipeline, fontsize=6) if j == 0 else ax.set_yticklabels([])
            ax.set_title(f"{ds}: {t}" if j == 0 else t, fontsize=7)
            ax.tick_params(labelsize=6)
    save(fig, "k31_episodes")


def consumers():
    t = pd.read_csv(D / "consumer_ttc.csv").dropna(subset=["loop_minus_row"])
    i = pd.read_csv(D / "consumer_idm.csv").dropna(subset=["loop_minus_row"])
    fig, axs = plt.subplots(3, 3, figsize=(13, 9))
    for k, ds in enumerate(DS):
        x = t[t.dataset == ds].rename(columns={"loop_minus_row": "diff", "ci_lo": "lo", "ci_hi": "hi"})
        forest(axs[k, 0], x, x.row.str.slice(0, 48), f"{ds}: TTC consumer, DESP TTC loop minus row")
        for c, lp in ((1, "ours-IDM"), (2, "frozen v6-S")):
            y = i[(i.dataset == ds) & (i.loop == lp)].rename(columns={"loop_minus_row": "diff", "ci_lo": "lo", "ci_hi": "hi"})
            forest(axs[k, c], y, y.row.str.slice(0, 48), f"IDM consumer, {lp} minus row")
    save(fig, "k31_consumers")


def strata():
    s = pd.read_csv(D / "strata.csv").dropna(subset=["v6S_minus_pipeline"])
    s = s[s.pipeline.isin(["u960 + C1 + v3a", "native + C1 + v3a", "hybrid tsp8@S + C1 + v3a", "uncertainty unc2@S + C1 + v3a"])]
    fig, axs = plt.subplots(1, 4, figsize=(15, 7), sharey=True)
    lab = (s.dataset.str.split(" ").str[0] + " " + s.axis + "=" + s.stratum.astype(str) + " (" + s.units.astype(str) + ")").unique()
    for j, p in enumerate(["u960 + C1 + v3a", "native + C1 + v3a", "hybrid tsp8@S + C1 + v3a", "uncertainty unc2@S + C1 + v3a"]):
        x = s[s.pipeline == p].rename(columns={"v6S_minus_pipeline": "diff", "ci_lo": "lo", "ci_hi": "hi"})
        forest(axs[j], x, (x.dataset.str.split(" ").str[0] + " " + x.axis + "=" + x.stratum.astype(str) + " (" + x.units.astype(str) + ")").tolist(),
               f"v6-S minus {p}")
    save(fig, "k31_strata")


def curv():
    f = D / "curv_test.csv"
    if not f.exists():
        return
    c = pd.read_csv(f).dropna(subset=["v6Sc_minus_row"])
    st = list(dict.fromkeys(c.stratum))
    fig, axs = plt.subplots(len(DS), 1, figsize=(10, 12))
    for k, ds in enumerate(DS):
        x = c[c.dataset == ds].rename(columns={"v6Sc_minus_row": "diff", "ci_lo": "lo", "ci_hi": "hi"})
        forest(axs[k], x, (x.stratum.str.slice(0, 30) + " | " + x.row.str.slice(0, 40)).tolist(), f"{ds}: curved v6-S minus row")
    save(fig, "k31_curv")


def bounded():
    f = D / "bounded.csv"
    if not f.exists():
        return
    b = pd.read_csv(f).dropna(subset=["b2_minus_row"])
    fig, ax = plt.subplots(figsize=(8, 4))
    x = b.rename(columns={"b2_minus_row": "diff", "ci_lo": "lo", "ci_hi": "hi"})
    forest(ax, x, (x.dataset.str.split(" ").str[0] + " | " + x.row).tolist(), "v6-S <= 2 calls minus row")
    save(fig, "k31_bounded")


if __name__ == "__main__":
    episodes(); consumers(); strata(); curv(); bounded()
    print("K31 FIGS DONE")
