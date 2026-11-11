"""K5 handoff: copy latency / breakdown tables, add slowest-frame rows, draw the latency-G scatter (PDF+PNG+CSV)."""
import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = {"nusc": "results/raw/20261004_031842_large_latency_nusc", "av2": "results/raw/20261004_033101_large_latency_av2"}
O = "docs/paper/handoff/K5/"
bd, worst, sc = [], [], []
for ds, run in R.items():
    lat = pd.read_csv(f"{run}/latency.csv")
    lat.insert(0, "dataset", ds)
    lat = lat.rename(columns={"ci_lo_ours_minus": "G_ci_lo_ours_minus_it", "ci_hi_ours_minus": "G_ci_hi_ours_minus_it"})
    lat.round(4).to_csv(f"{O}latency_{ds}.csv", index=False)
    b = pd.read_csv(f"{run}/breakdown.csv"); b.insert(0, "dataset", ds)
    b = b.rename(columns={"mean": "mean_ms_or_calls", "p95": "p95_ms_or_calls"}); bd.append(b)
    d = pd.read_pickle(f"{run}/large_latency.pkl"); d = d[~d.warm]
    for tag, r in (("most_crops", d.loc[d.n_crops.idxmax()]), ("slowest", d.loc[d.ms_ours.idxmax()])):
        worst.append({"dataset": ds, "which": tag, "unit": r.unit, "frame": int(r.frame), "n_crops": int(r.n_crops),
                      "rounds": int(r.rounds), "engine_calls": int(r.calls_ours), "ms_ours": round(r.ms_ours, 2),
                      "ms_ours_noref": round(r.ms_ours_noref, 2), "network_px_ours": round(r.px_ours, 3),
                      **{c: round(r[c], 2) for c in d.columns if c.startswith("bd_")}})
    d[["unit", "frame", "ms_ours", "ms_ours_noref", "n_crops", "rounds", "calls_ours", "px_ours"]].round(4).to_csv(
        f"{O}per_frame_ours_{ds}.csv", index=False)
    lat["dataset"] = ds; sc.append(lat[["dataset", "pipeline", "mean_ms", "p95_ms", "G", "network_px_mean"]])
pd.concat(bd).round(3).to_csv(f"{O}breakdown.csv", index=False)
pd.DataFrame(worst).to_csv(f"{O}worst_frame.csv", index=False)
S = pd.concat(sc).round(4); S.to_csv(f"{O}scatter_latency_G.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
for ax, ds in zip(axs, R):
    s = S[S.dataset == ds]
    w = s[~s.pipeline.str.startswith("ours")]
    ax.plot(w.mean_ms, w.G, "-o", color="0.4", ms=4)
    for _, r in w.iterrows():
        ax.annotate(r.pipeline.replace(" (native)", "*"), (r.mean_ms, r.G), fontsize=7, xytext=(3, -9), textcoords="offset points")
    for nm, mk, c in (("ours_noref", "s", "tab:blue"), ("ours", "*", "tab:red")):
        r = s[s.pipeline == nm].iloc[0]
        ax.errorbar(r.mean_ms, r.G, xerr=[[0], [r.p95_ms - r.mean_ms]], fmt=mk, color=c, ms=9 if mk == "*" else 6, capsize=2)
        ax.annotate(nm, (r.mean_ms, r.G), fontsize=7, xytext=(4, 4), textcoords="offset points", color=c)
    ax.set_xlabel("mean latency per frame [ms], Jetson AGX Xavier"); ax.set_ylabel("G"); ax.grid(alpha=.3)
    ax.set_title({"nusc": "nuScenes CAM_FRONT (local DEEP-85 scenes)", "av2": "AV2 ring_front_center (val logs)"}[ds], fontsize=8)
fig.tight_layout(); fig.savefig(f"{O}scatter_latency_G.pdf"); fig.savefig(f"{O}scatter_latency_G.png", dpi=200)
print(pd.DataFrame(worst).T)
