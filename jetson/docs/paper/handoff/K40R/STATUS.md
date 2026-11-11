# K40R status — final 2026-11-03

**All K40 tasks closed (2026-11-03).** D01 (K40-3), D02 (K40-6) and D06 (K40-5) ran with exact frozen parity. Final answers are at the top of REPORT.md. The 88 protected hashes were re-checked: 0 changes. The 2026-10-30 / 31 paragraphs below are history.

Inventory, PREREG and available CPU/cached analyses complete. Remaining RUNPLAN jobs authorized on 2026-10-31. D01/D02 adapters and CPU checks complete; detached queues are active on VUW/VUW1, waiting for eligible GPUs. No new detector result yet. Jetson JP01–JP03 feasibility is complete; full device/adaptive comparisons remain pending.

CPU first attempt stopped before output on absent namespaced u768 column. Schema-only fallback added for the K33-only column; exact stored-loss parity remains mandatory. No method/config change.

CPU loss/safety milestone complete: K40-1(a,b) and K40-2 reweighting, deployed frame/episode safety, original dev thresholds and full descriptive sweep executed on all three allowed splits. Loss parity passes 123 row/dataset checks. See CPU_LOSS_SAFETY.md and source CSVs. Adverse result: AV2 DESP G becomes negative at insufficient ×10; AV2 no-refiner DESP has lower G than its target. K40-1(c), K40-3, K40-4/7 remain in CPU analysis; K40-5/6 await RUNPLAN.

Latest CPU milestone supersedes the preceding progress line: K40-1(c) completed from final `objects_v4` on all 36,005 frames, no source-action/pixel parity failures; K40-4/7 executed on all frames with identical oracle traces, but 4/48 floating-point boundary fixtures show reordered algebra is not universally bit-exact. See FIELD_EQUIVALENCE.md and K41/OBJECT_ANALYSIS.md. K40-3 cached diagnostics completed; dev-selected AdaScale test row absent and router cached histories are not deployment-exact, so corrected36-family remains unresolved (MEAN_BUDGET.md). Registered protected-file check:88 files, zero changes.

| ID | Status | Evidence / outstanding |
|---|---|---|
| K40-1 | Completed CPU endpoints | target/decomposition/taxonomy CSVs; causal spuriousness claim unsupported |
| K40-2 | Completed CPU endpoints; device frontier via J04 | all losses / safety / thresholds / Pareto CSVs. Device G, p95 and deadline frontier measured in K41 J04. Energy at window level only (per-row energy not identifiable) |
| K40-3 | Completed (D01, 2026-11-01) | own-history mean-budget rows: 28/36 raw, 23 Holm (historical 29/25); none favour a baseline; achieved budgets reported, AdaScale 1.8–2.6× |
| K40-4 | Completed numerical audit | all observed frames match; boundary caveat reported |
| K40-5 | Completed (D06, 2026-11-02; post-hoc, test-only) | v6S_lc G 0.231 vs 0.211 (CI excludes 0), repair 0.781 vs 0.734, natural recovery 0.43 vs 0.31, at 1.9× px; not a replacement for the frozen config |
| K40-6 | Completed (D02, 2026-11-01) | both tubes × 3 caps × 3 splits: v6-S significantly better on nuScenes and mostly AV2; A-HD differences ≤0.016, K39-strict n.s. |
| K40-7 | Completed CPU/oracle representation audit | no observed difference; keep reference arithmetic for universal bit equality; no detector superiority claim |

2026-10-31 execution authorization: user requested all remaining tasks after the Jetson pilot. Completing D01/D02 adapters and checks, then following the consolidated RUNPLAN order. GPU inventory: no device currently satisfies MIN_FREE_MB=12000 on VUW/VUW1; ready jobs will wait via the host-local wrapper. No processes outside B touched.

2026-10-31 D02 CPU milestone: 8 literal-K39 cap fixtures, explicit denied-region counterexample, 818 tube geometry cases, 60 uncapped reference-loop fixtures, 180 cap checks and pending-guard retention passed. Tests use synthetic detections, not a neural network. D02_SOURCE_MANIFEST.json registers the detector, helpers and queue controller before first GPU execution. Real detector parity is still required.

2026-10-31 D01/D02 dispatch milestone: D01 controller PID2611404 on VUW; D02 PID1645466 on VUW1. Each queues three dataset smokes followed by complete N/A/H, with 10 frozen endpoints per frame, exact unit/frame validation, atomic unit checkpoints, source hashes and aggregate occupancy ceilings of40h/60h. Host wrappers require12GB free, no SHARE. D01_ROUTING_MANIFEST.json covers36,061 causal/history frames and36,005 evaluated frames; only locked dev thresholds and saved routing scores used. No new G/repair/CI result is available while queued.

2026-11-01: D01 and D02 complete on VUW / VUW1 (frozen parity exact on all 36,005 frames). Results in REPORT.md §2026-11-01 and k40_3_*/k40_6_* CSVs. Remaining K40R: K40-5 (optional D06).

2026-11-02 05:30 D06 / K40-5 (optional P3) launched on user instruction: `remote/A_method/k40r_lowconf_support.py` → results/k40r/lowconf/nusc. Definitions are in K41/STATUS.md 2026-11-02 05:30. The smoke passed with frozen parity.

2026-11-02 07:00 **D06 / K40-5 DONE and analysed** (REPORT section; k40_5_lowconf.csv): v6S_lc G 0.231 vs 0.211 (CI excludes 0), repair +4.7 points, natural recovery 0.43 vs 0.31, at 1.9× px. Post-hoc, test-only; no frozen change.

- 2026-11-02 17:20: analysis fix in `k40r_d0102_analyze.py`. The no-refiner rows of `k40_3_mean_budget.csv` / `k40_6_speed_tubes.csv` had compared row|none with v6-S|v3a; each refiner now uses its own control. CSVs regenerated: the v3a rows are bit-identical (max |diff| 0), so every number quoted in REPORT is unchanged. Same fix as `k41_d03_analyze.py`; D06 was already correct.
