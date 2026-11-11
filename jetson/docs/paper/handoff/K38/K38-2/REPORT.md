# K38-2 — the cost of covering fast-closing objects (post-hoc, test splits)

**Row.** v6-S with the deployable probe changes of tab:locality applied together (K16 chain steps c1 + d + e):
- closing probe 10 m/s;
- extended grid: ranges 8, 12.5, …, 70, 80 m; lateral −9…9 m in 1.5 m steps;
- three probe sizes: pedestrian 0.6 × 1.8 m, car, truck 2.6 × 3.6 m; the footprint is their union.

Real crop detections, no oracle. Round cap, context, guard, C1 and v3a are unchanged. This is not a method change.
- **Run.** VUW experiments/k38_probe_20261026/k30_probe.py (sha256 e7cdb7b3…), an isolated copy of the frozen
  k30_eval.py with one part. v6-S and u960 + C1 (π(F)) run on the same frames. AV2 with the K10 ego mask.
- **Report script.** scripts/A_method/k38_k2_report.py.
- **Checks** (checks.json; all 0 differences):
  - v6-S recomputed = the stored K28 test records (a, J, px; also π(F)'s action);
  - its decisive-object coverage = K34-2's covered flag on every frame.
- **Definitions.**
  - Coverage (K34-2): the decisive object's box centre lies in the executed field / support / guard crops. The
    population and the closing-speed terciles are those of K34-2.
  - Agreement with π(F): a_row|none = a_u960+C1|none.
  - Paired unit bootstrap: 2,000 draws, seed 0.

| split | row | G [95% CI] | px mean / p95 | agreement with π(F) | coverage all | coverage, high closing tercile |
|---|---|---|---|---|---|---|
| nuScenes test | v6-S | 0.211 [0.104, 0.348] | 0.078 / 0.176 | 0.976 | 0.918 (928) | **0.803** (309) |
| nuScenes test | + probe changes | 0.242 [0.131, 0.376] | 0.188 / 0.395 | 0.984 | 0.999 | **0.997** |
| AV2 test | v6-S | 0.294 [0.232, 0.356] | 0.054 / 0.129 | 0.968 | 0.954 (2,854) | 0.940 (951) |
| AV2 test | + probe changes | 0.325 [0.252, 0.392] | 0.121 / 0.238 | 0.977 | 0.976 | 0.984 |
| Argoverse-HD test | v6-S | 0.418 [0.314, 0.517] | 0.071 / 0.175 | 0.940 | 0.959 (3,593) | 0.963 (1,198) |
| Argoverse-HD test | + probe changes | 0.326 [0.141, 0.480] | 0.139 / 0.360 | 0.970 | 0.962 | 0.969 |

Differences, probe changes − v6-S [95% CI] (bold = excludes 0):

| split | ΔG | Δpx (mean) | Δagreement | Δcoverage all | Δcoverage high tercile |
|---|---|---|---|---|---|
| nuScenes | +0.031 [−0.007, +0.074] | **+0.111** (×2.4) | **+0.008** | **+0.081** | **+0.194 [0.126, 0.269]** |
| AV2 | +0.031 [−0.007, +0.081] | **+0.068** (×2.3) | **+0.009** | **+0.022** | **+0.044** |
| Argoverse-HD | **−0.092 [−0.278, −0.003]** | **+0.069** (×2.0) | **+0.030** | **+0.004** | **+0.006** |

## Reading
- **What closing the coverage gap buys.**
  - The probe changes close the nuScenes high-closing-speed gap: 0.80 → 0.997, overall 0.92 → 0.999. On AV2 the high
    tercile goes 0.94 → 0.98.
  - Decisions move toward π(F) on every split (+0.8 to +3.0 points).
  - G gains +0.03 on nuScenes and AV2, but the intervals include 0.
- **What it costs.** About 2.0–2.4× the network px (mean), with the p95 px also roughly doubling.
- **Argoverse-HD gets worse:** −0.092 G, significant. The extra probes trigger more braking on a split where coverage
  was already 0.96; agreement with π(F) rises, yet G falls.
- **Implication.** The deployed probe set is a reasonable operating point. Covering fast-closing objects is available
  at about twice the compute, with a small, not significant G gain on nuScenes / AV2 and a loss on Argoverse-HD.
  - The nuScenes coverage gap is therefore a deliberate compute trade-off, not a missing capability.
