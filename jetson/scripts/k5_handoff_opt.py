"""K5 follow-up handoff: latency tables of the optimised implementation (scripts/57: --gpupost --dyncrops [--verifier])
and the latency-G scatter with the first measurement (scripts/55) for reference."""
import sys, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
O = "docs/paper/handoff/K5/"
RUNS = {("nusc", "v8s_dyn"): "results/raw/20261004_143529_large_latency_opt_nusc",
        ("av2", "v8s_dyn"): "results/raw/20261004_144842_large_latency_opt_av2",
        ("nusc", "v8s_static"): "results/raw/20261004_031842_large_latency_nusc",
        ("av2", "v8s_static"): "results/raw/20261004_033101_large_latency_av2"}
for a in sys.argv[1:]:                                   # extra runs: ds:tag:path
    ds, tag, p = a.split(":", 2); RUNS[(ds, tag)] = p
rows = []
for (ds, tag), run in RUNS.items():
    l = pd.read_csv(f"{run}/latency.csv"); l.insert(0, "implementation", tag); l.insert(0, "dataset", ds); l["run"] = run
    rows.append(l)
L = pd.concat(rows).rename(columns={"ci_lo_ours_minus": "G_ci_lo_ours_minus_it", "ci_hi_ours_minus": "G_ci_hi_ours_minus_it"})
for (ds, tag), _ in RUNS.items():
    if tag != "v8s_static":
        L[(L.dataset == ds) & (L.implementation == tag)].round(4).to_csv(f"{O}latency_{ds}_{tag}.csv", index=False)
L.round(4).to_csv(f"{O}scatter_latency_G_all.csv", index=False)
mk = {"v8s_static": ("o", "0.6", "YOLOv8s, static canvases (first run)"), "v8s_dyn": ("o", "0.2", "YOLOv8s, ceil32 canvases"),
      "v8l_dyn": ("s", "tab:green", "YOLOv8l verifier, ceil32 canvases")}
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, ds in zip(axs, ("nusc", "av2")):
    for tag in [t for (d, t) in RUNS if d == ds]:
        s = L[(L.dataset == ds) & (L.implementation == tag)]
        m, c, lab = mk[tag]
        w = s[~s.pipeline.str.startswith("ours")]
        ax.plot(w.mean_ms, w.G, "-" + m, color=c, ms=3, lw=1, label="whole frame: " + lab)
        r = s[s.pipeline == "ours"].iloc[0]
        ax.plot(r.mean_ms, r.G, "*", color="tab:red" if tag == "v8s_dyn" else c, ms=12 if tag == "v8s_dyn" else 8,
                label="ours: " + lab)
        if tag == "v8s_dyn":
            for _, q in w.iterrows():
                ax.annotate(q.pipeline.replace(" (native)", "*"), (q.mean_ms, q.G), fontsize=6, xytext=(2, -8), textcoords="offset points")
    ax.set_xlabel("mean latency per frame [ms], Jetson AGX Xavier"); ax.set_ylabel("G"); ax.grid(alpha=.3)
    ax.set_title({"nusc": "nuScenes CAM_FRONT (local DEEP-85 scenes)", "av2": "AV2 ring_front_center (val logs)"}[ds], fontsize=8)
axs[1].legend(fontsize=6, loc="lower right")
fig.tight_layout(); fig.savefig(f"{O}scatter_latency_G_v2.pdf"); fig.savefig(f"{O}scatter_latency_G_v2.png", dpi=200)
print(L[L.pipeline.isin(["ours", "u960", "u1280", "u1536", "u640"])][["dataset", "implementation", "pipeline", "G", "mean_ms"]].round(3).to_string())
