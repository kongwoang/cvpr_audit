# R4 — component simplification study (complete)

8 supervised GPU jobs exited 0; 58 suite-labelled development rows (42 unique dataset/policy/refiner rows), all 452 units; two completed Jetson windows, three repetitions/dataset each. No core, weight, threshold or K34 change; no test selection or promotion. [Vietnamese synthesis](TONG_HOP.md), [component review](COMPONENT_REVIEW.md), [exact simplification notes](SIMPLIFICATION_NOTES.md), [all rows](all_rows.csv), [all paired effects](all_contrasts.csv).

Paired scene/log bootstrap 2,000 seed0, pointwise 95%; adaptive exploratory development. Main comparator is round1 B2K-A+recovery. nuScenes 152 scenes/6,117 frames/99 episodes; masked AV2 CFG-DEV 300 logs/23,665 frames/366 episodes.

**Exact runtime improvement:** K31 bottleneck selector caches the base object scores and batches probe-only columns. All action/field/support outputs match in 6,000 paired comparisons on K34’s fixed 1,000 navtest tokens, three repetitions. Post-hoc timing only; no quality selection. Same-host selector-stage speedups:

| grid | speedup [95% CI] |
| --- | --- |
| 3x3 | 3.77× [3.53, 3.99] |
| 7x7 | 9.21× [8.25, 10.18] |

Details and absolute costs: [nav_selector/REPORT.md](nav_selector/REPORT.md). This is not a general PDM-Closed or end-to-end perception speedup.

- **Simple objectwise median:** AV2 misses 49→37, loss delta −0.05336 [−0.07145,−0.03851], false episodes +2.578 [1.269,3.976]/1,000; false frames 53.29→108.94/1,000. 9/12 rescues have decisive-object evidence, none is a new hard-peak rescue. nuScenes misses unchanged, loss +0.00360 [−0.00192,0.01197]. A tradeoff, not common dominance.
- **One-round extent:** fewer requested pixels (−13.4%/−16.5%), but nuScenes loss +0.03222 [0.00325,0.08471] and G −0.06494 [−0.13344,−0.01727]. New misses and weak AV2 rescue attribution; reject as a common simplification.
- **Containment / union / semantic checks:** no additional episode rescue; union pixel saving only ~0.08%/0.27%. Hard-brake corroboration worsens AV2 G −0.04606 [−0.06679,−0.02322] despite fewer false episodes.
- **Exact abstention within median policy:** 74.53% [65.90,80.86] / 67.09% [63.68,70.60] refiner objects avoidable; all 29,782 stored-scene actions match. Direct implementation overhead motivated removal of duplicate matching/demand work. Both runtime implementations retained; do not confuse median-policy parity with original-method parity.

Optimized implementation mean-latency paired effects:

| window | dataset | comparator | delta ms [95% CI] |
| --- | --- | --- | --- |
| jetson_fast | av2_dev | round1 combined | -0.425 [-0.919, -0.052] |
| jetson_fast | av2_dev | objectwise median full | -0.573 [-1.296, -0.149] |
| jetson_fast | nusc_val | round1 combined | -0.277 [-0.410, -0.149] |
| jetson_fast | nusc_val | objectwise median full | -0.138 [-0.306, +0.008] |

All latency tails and three repetitions: [direct](jetson/REPORT.md), [optimized](jetson_fast/REPORT.md); [backend differences](jetson_backend/REPORT.md). R3 extent persistence remains the simplest supported quality addition; R4 does not remove the cross-dataset tradeoff. Severe misses remain.

Exact commands/code hashes: experiments/optimization_r4_20261024/job_status/ and two R4 Jetson window logs. Reproduce synthesis: `scripts/py experiments/optimization_r4_20261024/consolidate.py`. [Completion audit](completion_audit.json), [execution notes](execution_notes.md), [quality figure](quality_tradeoff.pdf), [cases](case_timelines.pdf). Local mirror: results/vuw_summaries/OPTIMIZATION_R4_20261024/.
