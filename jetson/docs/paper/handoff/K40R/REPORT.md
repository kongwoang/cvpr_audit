# K40R — CPU milestone report, 2026-10-30

## Final answers to the six questions (2026-11-03; supersedes the interim answers further down)

**Scope.**
- All test-set results are pre-specified post-hoc analyses, not independent preregistration.
- Frozen v6-S is unchanged and no holdout was used.
- The 88 protected paper / frozen-file hashes were re-checked on 2026-11-02: 0 changes (`K40R/protected_files_final_20261102.json`).
- Every detector job reproduces the released v6-S records exactly (max |ΔJ| = 0).
- "Measured" = run in this task. "Estimated" figures are labelled where they occur.

1. **Is the braking field genuinely different from an action-adaptive stopping-distance tube?**
   - **Not for this braking consumer.** With the same discrete cells, projection and scheduling, the field and the action-adaptive tube are identical on all 36,005 frames (K40-4/7). Numerical boundary handling has to stay identical: 4 / 48 fixtures differ if the algebra is reordered.
   - **The action dependence matters.** Measured with action-*independent* speed tubes (K40-6 / D02), v6-S is significantly better on nuScenes in every form (Δ +0.057 … +0.071) and on AV2 in 5 / 6 forms. On A-HD the gap is small (≤ 0.016; K39-strict n.s.).
   - The full 81-point field grid (D05) shows the frozen field is not a test-selected peak:
     - no setting improves on it across all three splits;
     - range spacing provably does not change the request;
     - one AV2 rule (lateral ×2, truck probe) gains +0.02 at 1.16× px.
   - **Claim:** novelty lies in the consumer-derived construction, support, iteration and conditional guarantee, not in the braking corridor's shape.

2. **Why does DESP's G differ from its own π(F)?**
   - G scores annotated braking demand, whereas π(F) agreement compares against a detector-derived target.
   - Target G = 0.085 / 0.291 / 0.296, and no-refiner DESP − target = +0.096 / −0.049 / +0.126 (N / A / H; K40-1).
   - On agreeing frames the contribution is exactly zero (shared previous cheap action).
   - The signed difference comes from frames where they disagree: state / history and refiner effects, plus a few decisive objects that only DESP finds.
   - General "spurious-suppression" causality is not supported.
   - D06 (post-hoc): seeding the loop with low-confidence corridor drafts raises nuScenes G 0.211 → 0.231 and natural-miss recovery 0.31 → 0.43, at 1.9× px. Part of the natural-miss gap is therefore a draft-threshold effect.

3. **Does DESP remain competitive under safety-sensitive losses and matched-mean budgets?**
   - **Mean budgets: yes.** In the own-history mean-budget family (D01), v6-S is significantly better in 28 / 36 comparisons (23 after Holm) and no baseline is better. Historical per-frame cap: 29 / 25.
     - AdaScale on nuScenes is n.s., but it uses 1.77× v6-S's pixels; an exact budget match is impossible on its ladder.
   - **Safety-weighted losses: no, on AV2.** Full DESP falls to G −0.039 at insufficient × 10 and is significantly below u960 at weights 2 / 5 / 10 (K40-2, CPU).
   - **Claim:** report this as an adverse trade-off; G is not a safety guarantee.

4. **How much p95/p99 improvement survives optimized-baseline parity?**
   - **None new.** Both exact optimization candidates fail the pre-registered tail screens:
     - JP01: p95 58.75 → 59.02 / 111.69 → 111.36 ms;
     - JP03: −1.19% / −0.23%, CIs include 0.
   - On the full adaptive-deadline benchmark (J04, Jetson, 3 repetitions):
     - **p95:** DESP v6-S 56.4 / 75.4 ms vs u960 29.2 / 45.4 ms; the bounded B2K-A gives 47.5 / 64.2 ms.
     - **Holm:** no row is significantly better than DESP after Holm at any deadline.
     - **Adverse but n.s. after Holm:**
       - AV2 at D ≥ 66.7 ms, where whole frame / AdaScale / AnA lead by about 0.08–0.11 G;
       - AV2 at D = 50 ms vs u640 (−0.22);
       - nuScenes at D = 33.3 ms, where DESP collapses to −0.19 (16% late).
   - J02: the tail is crop rounds plus per-crop C1, so it scales with the number of rounds.
   - Energy is reported at window level only: 18.6 W mean.

5. **Is there any nontrivial scene-dependent planner regime with real-crop downstream gains?**
   - **None found.**
   - **NAVSIM:**
     - the bottleneck selector has no PDMS gain (P01, 12,146 tokens; all CIs contain 0);
     - real-crop field repair beats planner-agnostic PiP (a) / (b) at fewer pixels, but PDMS is n.s. (D04);
     - the k = 2 field equals the singleton field (D07).
   - **Interaction orders:** joint effects are rare (P02, 1,000 tokens):
     - 0 / 58,398 adjacent pairs;
     - 27 / 14,395 removal triples;
     - 4 / 100k insertion pairs and 7 / 100k insertion triples.
   - **Braking splits (D03):** no PiP-style trajectory-anchored row beats v6-S under the natural or deletion draft. Only PiP + DESP support approaches v6-S. v6-S is significantly ahead on AV2 at strict caps and of plan-selected PiP on A-HD.
   - **P03 (all remaining 11,146 navtest tokens, 337 CPU h)** confirms this at scale:
     - adjacent k = 2 pairs move the plan 0 / 651,247 times;
     - insertion pairs / triples: 1.8e-5 / 4.9e-5;
     - removal triples move it more often than in P02: 3.2e-3 per combination, 2.2% of eligible tokens;
     - when joint effects occur they mostly lower PDMS (mean −0.32).
     - So the singleton guarantee misses a small but consequential set of multi-object cases.

6. **Which review criticisms still stand?**
   - **W3 (safety)** stands: AV2 is adverse under safety weights.
   - **W4 (embedded latency)** stands:
     - whole-frame rows have much shorter tails;
     - DESP is fragile at very tight deadlines on nuScenes;
     - DESP is below whole frame on AV2 at loose deadlines.
   - **W7 (planner generalization)** stands in part: the guarantee is shown for the stated planners; no real-crop downstream gain is demonstrated; interactions are empirically rare.
   - **Answered by narrowing the claims:**
     - **W1:** state the equivalence; the action dependence is shown to matter on N / A.
     - **W2:** distinguish target fidelity from annotated G.
     - **W5:** report the strong whole-frame rows and the J04 trade-offs.
     - **W6:** the mean-budget family confirms the per-frame-cap result.
   - Field-parameter sensitivity is now answered (D05). Trajectory-anchored PiP-style baselines do not beat DESP (D03 / D04).

Per-experiment sections (D01, D02, D06) follow; the 2026-10-30 milestone text is kept as history. Device / planner evidence: [K41/REPORT.md](../K41/REPORT.md).


Pre-specified post-hoc analysis, not independent preregistration. Frozen v6-S: YOLOv8s320/960, C1, v3a, static probe, three rounds/two guard growths, AV2 ego mask. Three allowed splits: nuScenes228 scenes/9,181 frames; AV2 val150 logs/11,818; A-HD val24 sequences/15,006. Primary records `$B/tmp/desp_k40/artifacts/records/{nuscenes,av2,argoverse_hd}.parquet`; missing object data from original K30 records. No holdout. Protected paper/frozen hashes:88/88 unchanged.

Executed results and exact commands: [loss/safety](CPU_LOSS_SAFETY.md), [attribution](../K41/OBJECT_ANALYSIS.md), [mean-budget diagnostic](MEAN_BUDGET.md), [field equivalence](FIELD_EQUIVALENCE.md). Scripts `remote/A_method/{k40r_cpu,k40r_matched,k40r_field,k41_cached_objects,k41_object_tables}.py`; md5 in code_manifest.csv and individual output run JSONs. Heavy CPU ran on VUW/VUW1 with CUDA hidden. Raw outputs `$B/results/{k40r,k41}/`, summaries KIT `results/vuw_summaries/`; handoff CSVs and PDF/PNG figures accompany each endpoint. Intervals use original release paired unit bootstrap,2,000 draws,seed0; Δ=DESP−row. Full status by original ID is in STATUS.md.

The six required conclusions at this milestone:

1. **Field versus action-adaptive tube:** identical on all36,005 observed frames and oracle traces. Independent algebra differs at4/48 floating boundary fixtures; exact representation must keep frozen membership arithmetic. No braking-shape novelty claim. Both action-independent K40-6 tubes remain distinct and unrun.
2. **G versus target:** target G=.0854/.2905/.2960; no-ref DESP minus target=.0957/−.0494/.1258. Shared `prev_cheap` gives exactly zero loss difference on agree frames. State differences and few DESP-only decisive objects explain much of the signed difference; general spurious-suppression causality is unsupported.
3. **Safety and mean budget:** not uniformly favorable. AV2 full DESP becomes negative at insufficient×10 and loses to u960 at weights2/5/10. Cached router diagnostics exist, but own routed histories and selected AdaScale test policy are missing; corrected36-family is unresolved. Historical29/36 raw,25/36 Holm is kept separate.
4. **New latency gain:** none established. Historical K39 DESP p95 remains55.1/74.3ms versus u96029.1/45.2ms; CPU optimization tests do not measure speedup.
5. **Scene-dependent real-crop downstream gain:** not established for the NAVSIM bottleneck selector; historical PDM scores do not substitute for it.
6. **Remaining criticisms:** W3/W4/W6/W7 remain material; W1/W2 need narrower claims, W5 needs explicit strong-baseline trade-offs. See [review matrix](../REVIEW_RESPONSE_MATRIX.md).

No paper files edited; all manuscript changes above are recommendations. K40-5/6 and missing deployment-exact policy comparisons await the single consolidated K41/RUNPLAN gate. No detector/GPU/Jetson job has run in this task.

## 2026-11-01 — D01 (K40-3) and D02 (K40-6) detector jobs: results (post-hoc test analysis)

Analysis: `remote/A_method/k40r_d0102_analyze.py` (md5 e31405a55313351c0ea442a361ebd0ea), run on VUW CPU:
`python scripts/A_method/k40r_d0102_analyze.py results/k40r/d0102_analysis`. Inputs `$B/results/k40r/{mean_history,tubes}/<ds>/*.pkl`
(D01 VUW, complete 2026-11-01 08:18; D02 VUW1, complete 2026-11-01 00:18). Outputs here: `k40_3_mean_budget.csv`,
`k40_3_family.csv`, `k40_3_family_summary.csv`, `k40_6_speed_tubes.csv`, `k40r_d0102_checks.json`.
**Frozen parity:** both jobs reproduce J_v6S|v3a and J_cheap of the released records on every one of the 9,181 /
11,818 / 15,006 evaluated frames (max |ΔJ| = 0); G(v6-S) = 0.211 / 0.294 / 0.418.

**K40-3 (mean budget, own routed history; refiner v3a; Δ = v6-S − row, 95% paired unit CI).** Achieved mean pixels
relative to v6-S (no exact match is possible for AdaScale's ladder): AdaScale 1.77 / 2.59 / 2.23×, BAR 0.99 / 1.22 /
0.79×, Router-G 0.95 / 1.23 / 0.46×, Chanakya 1.03 / 0.95 / 1.04× (nuScenes / AV2 / A-HD).
- AdaScale G 0.197 / 0.120 / 0.319: Δ +0.014 [−0.050, +0.076] (n.s.), +0.174 [+0.075, +0.292], +0.099 [+0.027, +0.220].
- BAR 0.009 / 0.026 / 0.026: Δ significant on all three (+0.202, +0.268, +0.391).
- Router-G 0.126 / 0.014 / 0.001: nuScenes +0.085 [−0.010, +0.188] n.s.; AV2, A-HD significant.
- Chanakya 0.115 / 0.195 / 0.321: nuScenes +0.096 [−0.018, +0.216] n.s.; AV2 +0.099, A-HD +0.096 significant.
- **Family (12 × 3, the four replaced by these rows, other eight keep their per-frame-cap records):** v6-S significantly
  better in **28 / 36** (historical per-frame cap: 29 / 36), **23** after Holm (historical 25); no comparison favours a
  baseline in either family. The changed comparisons: nuScenes AdaScale (now n.s. at 1.77× v6-S's pixels), Router-G and
  Chanakya on nuScenes n.s.; AdaScale on AV2 / A-HD becomes significant.
- Repair rate (K39-6): v6-S higher than every mean-budget row except AdaScale on A-HD (−0.028 [−0.128, +0.046]).

**K40-6 (action-independent speed tubes, length 2 + 0.4v + v²/2τ, τ1 = 1, τ2 = 3.5; support, C1, v3a, crop scale,
guard, merge unchanged).** G, pixels relative to v6-S, Δ CI:
| tube / cap | nuScenes | AV2 | A-HD |
|---|---|---|---|
| τ1 nominal | 0.154 (1.03×), +0.057 [+0.004, +0.117] | 0.285 (1.10×), +0.009 [−0.008, +0.026] n.s. | 0.411 (1.06×), +0.007 [+0.002, +0.013] |
| τ1 strict (K39 literal) | 0.148 (0.98×), +0.062 [+0.006, +0.123] | 0.248 (0.96×), +0.046 [+0.012, +0.080] | 0.429 (0.93×), −0.011 [−0.039, +0.016] n.s. |
| τ1 strict (observed) | 0.148, +0.063 [+0.006, +0.124] | 0.256, +0.037 [+0.011, +0.068] | 0.402, +0.016 [+0.009, +0.022] |
| τ2 nominal | 0.140 (0.80×), +0.071 [+0.016, +0.131] | 0.271 (0.93×), +0.022 [+0.003, +0.042] | 0.408 (0.90×), +0.009 [+0.003, +0.015] |
| τ2 strict (K39 literal) | 0.141 (0.79×), +0.070 [+0.016, +0.131] | 0.272 (0.90×), +0.022 [+0.001, +0.042] | 0.417 (0.88×), +0.000 [−0.011, +0.011] n.s. |
| τ2 strict (observed) | 0.141, +0.070 [+0.016, +0.131] | 0.267, +0.027 [+0.007, +0.048] | 0.407, +0.011 [+0.004, +0.018] |

Repair: v6-S 0.734 / 0.870 / 0.803; τ1 ties v6-S on nuScenes (0.734, Δ [−0.011, +0.011]) and is slightly lower
elsewhere; τ2 is lower on all three. Natural-miss recovery: v6-S 0.311 / 0.511 / 0.692; τ1 0.300 / 0.527 / 0.736, τ2
0.252 / 0.489 / 0.702 (the speed tubes recover more natural misses on AV2 and A-HD).
**Reading.** An action-independent speed tube is significantly below v6-S on nuScenes in every form and on AV2 in five of
six forms, but on A-HD the differences are small (≤ 0.016) and the K39-literal strict τ1 / τ2 tubes are not
significantly different (τ1 point estimate above v6-S). The action dependence of the field therefore matters clearly on
nuScenes, modestly on AV2 and little on A-HD; the paper should not claim a large advantage over a speed-adaptive tube
on A-HD. (K40-7: the action-adaptive tube is equivalent to the field by construction; no detector comparison.)

## 2026-11-02 K40-5 / D06: low-confidence corridor support (optional P3, post-hoc variant; measured)

nuScenes test, 9,181 gt frames / 228 scenes. Driver `remote/A_method/k40r_lowconf_support.py`; analysis `remote/A_method/k40r_d06_analyze.py`; tables `k40_5_lowconf.csv` + checks.
- **Definition.** One extra cheap pass at conf 0.05. Draft-band boxes (0.05 ≤ conf < 0.25) that are not duplicates (IoU < 0.5 with a draft box) and lie in the planner corridor seed the first round of the frozen v6-S loop as context-2 box crops. The extra pass is charged in px. Nothing is tuned; the frozen v6-S is the control.
- **Parity.** The v6-S control matches the released records exactly (max |ΔJ| = 0, 0 unit failures).
- **Coverage.** 54,607 band boxes; 7,377 corridor seeds on 3,857 frames (42%).

| | v6-S (frozen) | v6S_lc |
|---|---|---|
| G (v3a) | 0.2107 | **0.2309** (v6-S − row −0.020 [−0.038, −0.002]) |
| G (no refiner) | 0.1810 | 0.2031 (−0.022 [−0.040, −0.004]) |
| px (fraction of native) | 0.078 | 0.148 (crops only 0.105; the extra cheap pass adds 0.043) |
| repair vs π(F) | 0.734 | 0.781 (Δ CI [−0.075, −0.023], row better) |
| natural-miss recovery (270 misses) | 0.311 | 0.433 |

- **Reading.** On nuScenes, verifying low-confidence corridor drafts significantly helps G, repair and natural recovery, at 1.9× v6-S's pixels (1.36× without the extra cheap pass). 47 frames change action.
- This is a **post-hoc variant evaluated on test.** It does not replace the frozen configuration and cannot be claimed as the method without a dev-only selection.
- Suggested paper use: a supplementary note that the support set can be widened to low-confidence corridor drafts. The natural-miss limitation is partly a draft-threshold effect, which is a direction for future work.
