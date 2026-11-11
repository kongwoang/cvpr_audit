# LATENCY_OPT_20261030 — audit and Jetson feasibility, 2026-10-31

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

Measured device evidence for Q4: [K41/REPORT.md](../K41/REPORT.md) §J04 / J02 / J03 and `K41/J04/`. The 2026-10-31 text below is kept as history. LAT-1 / LAT-5 / LAT-8 status: see STATUS.md (2026-11-03 entry).


Primary baseline: frozen K39 static graphs on K38 staged runtime, TensorRT detector, C1 trt32, v3a; live release PyTorch is audited separately. [AUDIT.md](AUDIT.md) covers every Appendix C stage/opportunity, ranked concrete changes, dependency and algorithmic trade-off matrices. K39 raw traces reproduce p95 DESP55.1/74.3ms versus u960+C1+refiner29.1/45.2ms on20 nuScenes scenes/805 frames and40 AV2 logs/2,968 frames, three repetitions. BENCHMARK.csv has all tails/intervals/deadlines/pixels; PROFILE.csv is a separately labelled instrumented K38 dev prefix, whose crop-call labels are not algorithm rounds. New JP01 dev measurements are reported separately below; no meaningful DESP tail or energy gain is established.

The separate original Tab.6 reference also reproduces from allowed release per-frame records: DESP p95=59.173613/93.803421ms versus u960+C1+refiner30.917981/49.212624ms,20 nuScenes scenes/805 frames and8 AV2 logs/586 frames. `REFERENCE_TAB6.csv` retains the original median-over-three-repetitions convention; `REFERENCE_TAB6_REPETITIONS.csv` retains each source run. Script `remote/A_method/k41_tab6_cached.py`, md5/source in `REFERENCE_TAB6_RUN.json`, command `CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/A_method/k41_tab6_cached.py` after `source env.sh`. These are historical measurements, not a new device run or improvement.

Implemented isolated candidate: `experiments/latency_opt_20261030/columns_only.py` removes an unused all-class raw-candidate diagnostic from GP.host_columns, applying equally to whole-frame/crop paths. No frozen source change. Built K39-derived `scripts/91_latency_opt.py` with baseline/candidate/audit switches, separate debug history recording and explicit timing-valid flag. CPU `compare.py` checks per-frame actions/crops/pixels/rounds/history; audit mode separately checks selected raw columns. JP01 device execution is complete: see the pilot results below. Four CPU fixture groups pass on VUW; PARITY_CPU.csv identifies scope. Existing ROI/embedding/staging/static-graph improvements remain credited to K29/K38/K39. Jetson cv2 native-library import fails with CUDA hidden (cuInit100); Subsequent GPU-visible OpenCV/Torch/TensorRT/CUDA preflight passed.

Reproduce CPU outputs after `source $B/env.sh`: `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/A_method/k41_latency_cached.py`; candidate CPU tests use `LATENCY_TEST_OUTPUT=results/latency_opt/cpu_tests/PARITY_CPU.csv python scripts/A_method/k41_latency_cpu_tests.py` with the same environment. Build-only command on KIT: `python3 experiments/latency_opt_20261030/build.py`. Exact md5s: source_manifest.json, PARITY_CPU.json, K40R/code_manifest.csv; input hashes: `raw_input_manifest.json` and KIT `results/vuw_summaries/latency_opt/cached/sources.csv`. CHANGELOG.md records changes; STATUS.md preserves LAT-1…LAT-8.

Six required answers: **(1)** braking field and action-adaptive tube are equivalent under identical discrete arithmetic/scheduling; **(2)** G measures annotated action loss and can differ from detector-target fidelity; **(3)** safety-weighted AV2 is adverse and own-history mean-budget fairness remains open; **(4)** **no new p95/p99 reduction established**, CPU parity is not speed evidence; **(5)** scene-dependent real-crop selector downstream benefit is unestablished; **(6)** embedded tail-latency and several safety/generalization criticisms remain. The [review matrix](../REVIEW_RESPONSE_MATRIX.md) states scope changes explicitly.

The copied public-release CPU regression suite also passed102 tests, with3 CUDA tests explicitly skipped (`release_cpu_stdout.txt`, JUnit and source manifest). `PARITY.csv` combines the scoped CPU checks and an explicit **not_run** full-dev/timed-test row alongside the passed prefix-device checks; blank device mismatch counts are not zero mismatches. The debug harness now captures detector/NMS, requests, geometry/TTC, refinement and own history; JP01 prefix checks passed, while full-dev/timed-test parity remains unrun. A CPU check cannot establish a device speedup.

The consolidated [RUNPLAN](../K41/RUNPLAN.md) received partial authorization for Jetson optimization feasibility first. JP01–JP03 use separate parity/debug and clean timing with equally optimized baselines, exclusive MAXN/clocks/fan and restoration. Other detector/planner/full adaptive jobs remain pending.

## Measured Jetson pilot outcomes

[JP01 detailed report](PILOT_REPORT.md): omitting the unused raw-count reduction preserves all47 recorded debug endpoints on712 frames per dev dataset and66 scalar endpoints across all clean repetitions. DESP p95 changes58.75→59.02ms on nuScenes and111.69→111.36ms on AV2; paired95% intervals include0. Both pre-registered feasibility screens fail. u960 mean improves by0.19/0.27ms, but this does not remove the large DESP tail gap. These are512-frame dev-prefix measurements repeated three times, not test-set results or a replacement for historical Tab.6. No promotion of this candidate is supported.

[JP02 C1 encoder probe](JP02_REPORT.md): fixed-context FP32 graph replay saves0.123ms (6.42%) at batch1, less at larger buckets, with120 exact output comparisons. This is an isolated measurement. Conditional JP03 is complete: p95 reductions1.19%/0.23%, both paired intervals include0; both pre-registered screens fail. Existing detector graphs, crop staging and embedding cache remain credited to K29/K38/K39.

JP01 recovery: all20 result directories were written, but the original wrapper did not record its final exit/restoration; a machine restart followed. Original state is preserved; recovery verified artifacts/source/engine/protected hashes, restored controls and CPU-validated completed timing data. The last instrumented profile exit code is unknown. Raw logs and recovery metadata are retained; this limitation is not concealed as normal completion.

[Optimization progress](OPTIMIZATION_PROGRESS.md) distinguishes existing K29/K38/K39 optimizations from new changes and untested proposals. [Decision/latency pilot](DECISION_LATENCY_PILOT.md) reports existing bounded variants separately: their shorter tails come with different actions/history, and AV2 point G remains below refined u960 on this prefix. No live deadline or energy claim is made.

## Final authorized feasibility result

[JP03 full-pipeline report](JP03_REPORT.md) confirms the isolated encoder gain does not yield a meaningful DESP tail reduction. Three repetitions/arm/dataset give p9561.886→61.149ms nuScenes and115.241→114.971ms AV2, paired Δ95% CIs[−1.515,+0.847]/[−1.871,+2.777]ms. Mean changes−0.076/−0.122ms; nuScenes p99 is adverse(+0.596ms), both p99 intervals include0. No significant mean/p95 slowdown flag in the included whole-frame rows, but neither primary≥5%/CI screen passes.112 clean endpoint pairs and3,167 real encoder checks pass exactly. All repetitions and diagnostic cheap controls are retained.

This answers the prioritized feasibility request: safe opt-in candidates work correctly on the tested prefixes but do not materially address W4. A large exact tail reduction is **not demonstrated**, not proved impossible. [Measured summary](FEASIBILITY_SUMMARY.md) records remaining bottlenecks and limits. No candidate replaces frozen v6-S. Full-dev/test/adaptive/live-deadline/energy work remains outside the completed pilot scope. JP03 completed after a disclosed controller interruption; all18 run folders and source/engine/protected hashes validate, clock/fan restored.
