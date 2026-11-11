"""K39-6 figure: repair rate A | cheap != pi(F) vs mean realised px (log), per dataset (handoff/K39/k39_6_fidelity.csv)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

D = Path(__file__).resolve().parents[1] / "docs/paper/handoff/K39"
R = pd.read_csv(D / "k39_6_fidelity.csv")
R = R[(R.row != "always KEEP (reference)") & ~R.row.str.contains("= F") & (R.row != "cheap")]
STY = {"ours / ablation": ("#c44e52", "o"), "whole frame": ("#4c72b0", "s"), "heuristic": ("#55a868", "^"),
       "published / object-level": ("#8172b3", "D"), "object-level (AnA / counterfactual)": ("#dd8452", "v"), "published (router)": ("#937860", "P")}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
for a, ds, t in zip(ax, ("nusc", "av2", "ahd"), ("nuScenes test", "AV2 val (ego mask)", "Argoverse-HD val")):
    x = R[R.dataset == ds]
    for fam, (c, m) in STY.items():
        y = x[x.family == fam]
        a.scatter(y.px_mean, y["A|cheap!=F"], color=c, marker=m, s=34, label=fam, alpha=0.85)
    w = x[x.family == "whole frame"].sort_values("px_mean")
    a.plot(w.px_mean, w["A|cheap!=F"], color="#4c72b0", lw=1)
    v = x[x.row == "DESP v6-S"]
    a.scatter(v.px_mean, v["A|cheap!=F"], marker="*", s=320, color="#c44e52", edgecolor="k", zorder=5, label="DESP v6-S")
    for _, r in x.iterrows():
        if r.row in ("DESP v6-S", "AnA one pass", "AdaScale", "whole frame u640 + C1", "whole frame u1280 + C1") or r.row.startswith("hybrid tsp8 [tsp8] strict"):
            a.annotate(r.row.replace("whole frame ", "").replace(" [tsp8]", "").replace(" + C1", ""), (r.px_mean, r["A|cheap!=F"]),
                       fontsize=7, xytext=(4, -9), textcoords="offset points")
    a.set_xscale("log"); a.set_xlabel("mean realised detector px / native (log)"); a.set_title(t); a.grid(alpha=0.3, which="both")
ax[0].set_ylabel("repair rate: A | cheap action ≠ π(F)  (no refiner)")
ax[0].legend(fontsize=7, loc="lower right")
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(D / f"k39_6_fidelity_px.{e}", dpi=200)
