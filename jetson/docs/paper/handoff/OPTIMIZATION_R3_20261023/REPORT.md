# Round 3 — final handoff (2026-10-24)

**Done:** 87 supervised jobs, 228 full-development rows across 8 suites, both Jetson windows (3 repetitions each). Main source/weights/thresholds/K34 unchanged; no test selection or core promotion. Full analysis, caveats and commands: [TONG_HOP.md](TONG_HOP.md).

Development: 152 nuScenes scenes (6,117 frames, 99 episodes) and 300 AV2 CFG-DEV logs (23,665 frames, 366 episodes). Training/holdout regeneration 785 units. Paired unit bootstrap 2,000, seed 0, pointwise 95%; repeated adaptive exploration, not confirmation. Comparison baseline: round1 B2K-A+recovery.

| Finding | nuScenes | AV2 | Assessment |
|---|---|---|---|
| Extent persistence: loss delta [95% CI] | −0.00969 [−0.02748,−0.00036] | −0.00744 [−0.01776,−0.00114] | Misses unchanged; false episodes essentially unchanged |
| Extent persistence: extra mean ms [95% CI] | +0.90 [0.01,2.27] | +1.18 [0.14,3.00] | Smallest measured overhead; tails still increase |
| Point+extent: missed episodes | 38→35/99 | 49→43/366 | 3/3 and 5/6 rescues have decisive-object evidence |
| Point+extent: loss delta [95% CI] | −0.03145 [−0.05458,−0.01249] | −0.03362 [−0.05357,−0.01825] | Better loss; combined Jetson cost unmeasured |
| Point+extent: extra false episodes/1,000 [95% CI] | +1.308 [0.327,2.452] | +0.803 [0.381,1.267] | Measurable tradeoff |
| Compact: missed episodes / extra mean ms | 33/99; +1.86 [1.06,2.78] | 48/366; +3.72 [1.63,6.16] | Weak AV2 benefit; G intervals include zero |

**Negative evidence:** joint decision-aware lifting raises AV2 loss +0.04668 [0.02101,0.07606] despite higher G; calibrated upper fallback raises false alarms substantially; learned identity gate rescues no extra episode. AV2 uncertainty query rescues 12 episodes, but no decisive hard-brake evidence in its 5 hard-peak rescues. Severe-case weakness remains. Point+extent still misses 43 AV2 episodes versus cheap 30, and its loss 0.33489 remains above cheap 0.31396 (at much lower false-alarm rate).

**All rows and evidence:** [overview](overview/REPORT.md), [key CSV](key_results.csv), [all intervals](all_intervals.csv), [paired effects](key_contrasts.csv), [detector mean/p95](detector_cost/REPORT.md), [Jetson](jetson_tradeoff/REPORT.md), [cross-backend differences](jetson_tradeoff_backend/REPORT.md), [case timelines](cases/REPORT.md), [completion audit](completion_audit.json). The calibrated/raw distinction is explicit in suite names.

Exact worker commands/code hashes: experiments/optimization_r3_20261023/job_status/; local 243-source snapshot unchanged. Failed attempts retained. Summary CSVs/PDF/PNG mirrors: results/vuw_summaries/OPTIMIZATION_R3_20261023/. No claim of closed-loop safety, independent calibration, equal total budget superiority, or negligible latency tradeoff.
