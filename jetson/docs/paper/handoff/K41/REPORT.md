# K41 — CPU milestone report, 2026-10-30

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

Per-experiment sections (P01, P02, D04, D07, J04/J02/J03, D03, D05) follow; the 2026-10-30 milestone text is kept as history.


Pre-specified post-hoc, frozen v6-S unchanged, no holdout. All36,005 labelled frames on228/150/24 allowed scene/log/sequence units reconstructed with zero source action/pixel parity failures. Original release records plus original K30 boxes/states/crops; final `objects_v4` supersedes diagnostics. Paired unit bootstrap2,000 draws,seed0,95% percentile. Exact sources/commands/md5 are in [OBJECT_ANALYSIS.md](OBJECT_ANALYSIS.md), [CACHED_NAVSIM.md](CACHED_NAVSIM.md), [SCENE_DIAGNOSTIC.md](SCENE_DIAGNOSTIC.md), [AVAILABILITY.md](AVAILABILITY.md), PREREG and K40R/code_manifest.csv. Scripts `remote/A_method/k41_*`, `remote/A_navsim/k41_*`; VUW outputs `$B/results/k41/`, KIT mirror `results/vuw_summaries/k41/` and handoff CSV/PDF/PNG.

All requested descriptive strata ran for five pipelines, with ref/no-ref separation. Crop-origin locality at IoU≥.5 is64.68%/75.72%/75.07%; coverage misses strongly associate with A-HD disagreement. First covered round and originating canvas are not recoverable from final unions. The81-setting grid ran only for CPU geometry/cell counts/time, not detector G/pixels. NAVSIM's12,146-token field reconstruction matches saved counts;6,041 matched-bin pairs remain confounded by ego/candidate/camera changes. Prior interaction counts6/9,449 and11/139,675 reproduce; finite tests cannot prove a global guarantee. Real-crop selector agreement96.76%, but selector downstream scores are missing. Official VeteranAD modesK=20; availability only, no end-to-end reproduction. No local learned object-input planner found.

Six required answers: **(1)** braking field is the action-adaptive stopping tube under identical discrete semantics; **(2)** target fidelity and annotated-loss G differ, with state/history/refiner effects and no general causal spuriousness conclusion; **(3)** AV2 safety-heavy losses are adverse and exact mean-budget comparison remains unresolved; **(4)** no new fair p95/p99 improvement measured; **(5)** no decisive scene-controlled real-crop selector downstream gain yet, while historical PDM EPDMS versus strong rows includes zero; **(6)** W3/W4/W6/W7 remain, and W1/W2/W5 need precise scope and baselines. [Review matrix](../REVIEW_RESPONSE_MATRIX.md) maps evidence/counterevidence and proposed manuscript changes.

Controlled follow-up supersedes the matched-bin limitation for the narrow synthetic question: with ego/camera/candidates/probes fixed, 357/2,644 same-action pairs change field masks (13.50%, CI11.02–16.05%) and311 change requested crops (11.76%, CI9.60–13.98%). This establishes scene-dependent geometry, not a real-crop/downstream benefit. See SCENE_DIAGNOSTIC.md. Train-only K20 anchors and CPU loop tests are complete; CPU_PREPARATION.md documents a strict-cap adapter discrepancy that must be resolved transparently before new runs. FEASIBILITY_MEMO.md retains all five P3 directions without launching a new research program.

STATUS.md preserves every K41 ID. PiP(a/b/c), triples/insertion/k2 calls, missing grid detector points, full adaptive deadline benchmark and true guard profile remain unfinished; the consolidated RUNPLAN specifies execution and remaining adapter preparation. No new planner or remote detector sweep has run. Authorized Jetson feasibility pilots JP01–JP03 are now complete; paper and frozen harnesses remain unchanged.

2026-10-31 latency feasibility update: [JP03](../LATENCY_OPT_20261030/JP03_REPORT.md) preserves recorded outputs/history/counters on the measured dev prefixes but yields p95 reductions only1.19%/0.23% (paired CIs include0), failing the pre-registered screen on both datasets. JP01 columns-only also fails. This does not complete the full K41-5/6 adaptive/deadline/guard-depth scope or resolve reviewer W4. Clock/fan restored, interruption provenance disclosed, no frozen promotion. See [feasibility summary](../LATENCY_OPT_20261030/FEASIBILITY_SUMMARY.md).

## 2026-11-01 P02 bounded-order interactions (K41-2, measured, CPU, saved K20 seed-0 subset, 1,000 navtest tokens, 0 errors)

Driver `remote/A_navsim/k41_p02_interactions.py`; PDMS of joint effects `remote/A_navsim/k41_p02_score.py`. Tables: `k41_2_p02_families.csv`, `k41_2_p02_joint_pdms.csv`, `k41_2_p02_cost.csv`. Moved = PDM-Closed plan change > 0.5 m over 4 s, every query from the original draft scene. Wilson 95% CIs are per combination and treat combinations as independent; they are not clustered by token, so they are descriptive.

| family | combinations tested / available | moving | rate [95% CI] | tokens with ≥ 1 / tokens tested |
|---|---|---|---|---|
| removal triples of free objects | 14,395 / 19,482 | 27 | 1.9e-3 [1.3e-3, 2.7e-3] | 7 / 546 |
| insertion pairs of quiet cells | 100,000 / 721,907 | 4 | 4.0e-5 [1.6e-5, 1.0e-4] | 4 / 1,000 |
| insertion triples of quiet cells | 100,000 / 8,895,511 | 7 | 7.0e-5 [3.4e-5, 1.4e-4] | 6 / 1,000 |
| k = 2 adjacent quiet cells (Manhattan 1) | 58,398 / 58,398 (all) | 0 | 0 [0, 6.6e-5] | 0 / 1,000 |

- Joint effects are rare but present at every order above one except the adjacent-cell k = 2 family. Therefore the k = 2 field adds nothing on this subset, and D07 k2 ≡ singleton. D07 field stage: 0 changed tokens, as pre-specified.
- Removal triples cluster in a few tokens. 5 tokens have 4–6 moving triples, and the triples on each of these tokens share a common pair, for example (10, 121) on 0497454a. This is consistent with a pair-level effect that a singleton test cannot see; it is not a spread-out triple effect.
- **This is a finite diagnostic, not a guarantee.** Joint effects occur outside every tested order, so no k-wise statement can be made. These results stay outside Theorem 1, and the threshold-cardinality counterexample still applies.
- **PDMS of the 38 joint-effect trajectories, scored on the real scene.** These are counterfactual plans (objects removed or phantom cells inserted), so the numbers show that the changed plans matter. They are not the quality of any method.
  - mean ΔPDMS vs the draft plan −0.315; 24 lower, 14 higher.
  - removal triples −0.399 (16 lower, 11 higher);
  - insertion triples −0.160 (5/2);
  - insertion pairs −0.017 (3/1).
  - Shifts are 1.0–23.5 m; 10 joint effects drop PDMS to 0.
- **Cost (measured).** 277,841 planner calls, 29.1 CPU h, about 0.38 s per call. The full k = 2 adjacent family costs 58 calls/token (6.1 CPU h per 1,000 tokens) and found no effect. On the full 12,146 tokens this would be about 74 CPU h before scoring (estimate).
- **Caveat found 2026-11-02: NAVSIM v1.1 PDM-Closed state leak.** Red-light collided ids are never reset between scenes when one planner instance is reused; see STATUS. P02 reused one instance, as K20 did, so each token's tests are consistent with each other, but on a few tokens the reference plan differs from a fresh planner (for example bef3d546: 4.4 m). The joint-effect PDMS is being recomputed with a fresh instance. The number of affected tokens will be reported from the D07 rescore.

## 2026-11-02 D04: PiP on NAVSIM with real crops (K41-1; measured, 12,146 navtest tokens / 136 logs)

Driver `remote/A_navsim/k41_pip_nav.py`; summary `remote/A_navsim/k41_d04_summary.py`; tables `D04/k41_1_d04_summary.csv` and `D04/run.json`.
- **Protocol.** Official NAVSIM v1.1 PDMS (P01 protocol, not EPDMS). Paired log bootstrap, 2,000 draws.
- **Validity checks.** K31 field parity (pi / px / rounds) held on 12,146/12,146 tokens. The adapter's selector choice equals the P01-scored choice on 60,730/60,730 token-rows.
- **Repair** is measured on the 724 tokens where the cheap and full selector choices differ.

| row | PDMS [95% CI] | agreement with full | repair (n = 724) | field − row repair, paired | px | crops | rounds |
|---|---|---|---|---|---|---|---|
| field (DESP K31) | 0.5108 [0.4764, 0.5462] | 0.9676 | 0.689 [0.650, 0.725] | — | 0.827 | 5.53 | 1.57 |
| PiP (a) all | 0.5102 | 0.9645 | 0.638 [0.596, 0.680] | +0.051 [+0.019, +0.083] | 1.256 | 5.65 | 1.00 |
| PiP (b) selected (M = 3) | 0.5112 | 0.9648 | 0.601 [0.554, 0.644] | +0.088 [+0.057, +0.124] | 0.977 | 3.76 | 2.78 |
| PiP (c) selected + support | 0.5108 | 0.9673 | 0.663 [0.619, 0.703] | +0.026 [−0.004, +0.058] | 0.840 | 5.39 | 3.16 |
| geometric (support first) | 0.5112 | 0.9680 | 0.674 [0.633, 0.712] | +0.015 [−0.012, +0.044] | 0.907 | 5.92 | 1.13 |
| cheap / full / hybrid / oracle (P01) | 0.5123 / 0.5105 / 0.5109 / 0.5103 | 0.940 / 1 / 0.973 / 0.997 | 0 / 1 / 0.671 / 0.952 | | | | |

px is in the field's unit (crop native area / budget, K31 convention).
- **PDMS does not separate any row.** Every field − row PDMS CI contains 0, as in P01.
- **Field vs PiP (a) and (b).** The field repairs significantly more selector disagreements than PiP (a) (+5.1 points) and (b) (+8.8 points). It also spends fewer pixels than (a) (0.83 vs 1.26) and than (b) (0.83 vs 0.98).
- **Field vs PiP (c) and geometric.** PiP (c) and the support-first geometric row are not distinguishable from the field on repair or agreement. Both spend slightly more pixels (0.84 and 0.91).
- **Hybrid** has significantly lower agreement than the field (−0.52 points, CI [−0.76, −0.27]).
- **Reading.** Planner-agnostic PiP without decision support loses repair, so the support set is what drives repair. Once support is added, the field's advantage is within noise on NAVSIM.

## 2026-11-02 D07: k = 2 field of PDM-Closed (K41-2; measured, 1,000-token K20 seed-0 subset)

Tables `D07/k41_2_d07_rows.csv` and `D07/k41_2_d07_leak_audit.csv`.
- **The k = 2 field adds nothing here.** No moving adjacent pair exists (P02), so k2 and k2all have the same field as the singleton field on every token: the detect stage ran 0 GPU tokens and the scores are identical.
- **PDMS:**
  - cheap 0.8844;
  - full960 0.8882;
  - singleton = k2 = k2all 0.8892, row − cheap +0.0048 [−0.0005, +0.0109], row − full960 +0.0010 [−0.0038, +0.0058];
  - geometric 0.8867.
- **Agreement with full960:** cheap 0.906, singleton 0.958, geometric 0.965. Crop px: singleton 0.627, geometric 0.641.

**NAVSIM state-leak impact (measured).** Every D07 plan and metric cache used a fresh PDM-Closed instance.
- Draft plan vs the P02 draft (one shared instance per shard): differs on 1/1,000 tokens (bef3d546, 4.40 m).
- Rebuilt PDMS vs the stored original `pdms_final`: identical on 1,000/1,000 tokens for cheap, full960 and singleton.
- Shared-instance vs fresh metric cache: 0/873 token score differences.
- P02 joint-effect PDMS: unchanged after the fresh-instance rescore (38 trajectories, mean Δ −0.3147).

The leak is real and reproducible, but on these runs it changes 1 reference plan in 1,000 and no stored score. It should be noted as a NAVSIM v1.1 reproducibility caveat, not as a result change.

## 2026-11-02 J04 / J02 / J03: adaptive-deadline benchmark on the Jetson AGX Xavier (K41-5/6; measured)

- **Window.** 2026-11-01 13:17–18:20 Jetson clock: exclusive, MAXN, jetson_clocks, fan 255, hardware restored.
  - Dev calibration, then 2 smokes, 3 repetitions × (nuScenes test 20 scenes / 805 frames, AV2 40 logs / 2,968 frames), and 2 breakdown runs. All 12 runs DONE.
  - Source and engine hashes were unchanged at the end. The median files were hashed before any test run (state.json).
- **Scripts and tables.** `scripts/k41_j04_report.py` (adapted from k39_1_report.py), `scripts/k41_j02_breakdown.py`, and `remote/A_method/k41_j04_device_parity.py`. Tables are in `K41/J04/`: rows, deadline, diffs (raw CI + Holm), breakdown, device_parity, frontier.
- **Validity.**
  - v6-S / B2K-A actions, px and crop / round / call counts match every K38-1 staged repetition: 0 mismatches on all 805 / 2,968 frames × 3 × 3. They also match K34-3 (0 / 805 and 0 / 586).
  - Stored J is reproduced (max |ΔJ| < 5e-15) and a_gt matches K30.
  - **Device vs VUW K33 test records**, reported only (not a gate):
    - action agreement for v6-S, AnA, CF, AdaScale and FOVEA: 0.992–0.999;
    - chosen size for DVR / BAR / Chanakya: 0.968–1.000;
    - identical across repetitions.
    - Differences come from the device TensorRT FP16 draft vs the cached torch draft.

**G at D = ∞ and p95** (timed frames, 3 repetitions). Selected rows; all rows are in k41_5_rows.csv.

| row | nuScenes G | p95 ms | AV2 G | p95 ms |
|---|---|---|---|---|
| DESP v6-S | 0.380 | 56.4 | 0.297 | 75.4 |
| DESP B2K-A | 0.382 | 47.5 | 0.304 | 64.2 |
| DESP B1K-A (post-hoc) | 0.391 | 33.9 | 0.269 | 45.4 |
| u640 / u960 / u1280 + comp. | 0.330 / 0.386 / 0.395 | 23.5 / 29.2 / 35.0 | 0.373 / 0.380 / 0.400 | 39.0 / 45.4 / 53.5 |
| hybrid tsp8 / uncertainty unc2 @ DESP px | 0.373 / 0.198 | 74.1 / 109.3 | 0.296 / 0.191 | 98.4 / 160.3 |
| AdaScale ada+1t1280 | 0.367 | 51.0 | 0.405 | 81.5 |
| AnA one-pass / iterated | 0.326 / 0.326 | 57.1 / 67.0 | 0.400 / 0.398 | 111.1 / 128.5 |
| CF one-pass / iterated | 0.071 / 0.071 | 37.8 / 48.5 | 0.276 / 0.320 | 68.0 / 104.6 |
| FOVEA KDE_S / KDE_I | −0.199 / 0.402 | 38.4 / 46.9 | 0.331 / 0.370 | 58.1 / 66.8 |
| DVR / BAR router, Chanakya | 0.228 / 0.158 / 0.232 | 43.9 / 65.6 / 32.1 | 0.197 / 0.165 / 0.234 | 58.6 / 96.6 / 47.3 |

**Deadlines.** Late frames hold the last on-time action; DESP and iterated AnA / CF use in-loop stopping. G(D) for DESP v6-S on nuScenes / AV2:

| D | nuScenes G | AV2 G |
|---|---|---|
| 33.3 ms | −0.194 | 0.000 |
| 50 ms | 0.354 | 0.147 |
| 66.7 ms | 0.378 | 0.314 |
| 100 ms | 0.380 | 0.297 |

- **No row is significantly better than DESP v6-S or B2K-A after Holm,** at any D, on either dataset.
- **Where DESP is significantly better after Holm (v6-S):**
  - nuScenes: 2 / 5 / 4 / 3 rows at D = 50 / 66.7 / 100 / ∞, for example CF iterated +0.31 and Chanakya +0.15;
  - AV2: 8 / 5 / 5 / 1 / 2 rows at D = 33.3 / 50 / 66.7 / 100 / ∞, mainly the heavy rows AdaScale, FOVEA, u1280 and native, which miss tight deadlines.
- **Adverse results:**
  - **AV2 D ≥ 66.7 ms.** Whole frame u960 / u1280, AdaScale and iterated AnA reach 0.38–0.41 vs DESP 0.30–0.31 (DESP − row about −0.08 to −0.11, raw CIs include 0, n.s. after Holm).
  - **AV2 D = 50 ms.** u640 beats DESP v6-S by 0.22 (raw CI [−0.40, −0.01], n.s. after Holm), because DESP is late or stopped on 11.5% of frames.
  - **nuScenes D = 33.3 ms.** DESP v6-S collapses to −0.194 (16% late), while u960 / u1280 / B1K-A keep 0.42–0.43. DESP − u960 = −0.62, raw CI [−1.49, −0.02], n.s. after Holm.
  - **The bounded B1K-A** is the best DESP variant under tight deadlines on nuScenes (0.43 at 33.3 ms) but is worse on AV2.
- **J03 equal-protocol comparison.** Under one protocol, DESP is in the whole-frame band on nuScenes. On AV2 it is about 0.08 G below whole-frame u960 at no deadline, with p95 of 75 vs 45 ms. The iterated AnA tail is longer than DESP's (p95 67 / 128 ms vs 56 / 75 ms).

**J02 breakdown** (one `--breakdown` repetition; per-stage CUDA syncs add about 4% to the mean). Values are v6-S stage means, all frames → frames ≥ p95:

| stage | nuScenes ms | AV2 ms |
|---|---|---|
| total | 21.5 → 70.2 | 33.3 → 97.6 |
| draft | 7.0 → 7.3 | 9.5 → 9.7 |
| crop rounds | 8.8 → 45.0 | 13.4 → 55.3 |
| C1 | 1.4 → 5.3 | 4.0 → 15.8 |
| refiner | 0.8 → 2.6 | 1.3 → 3.9 |
| rest (field, guard, lift, merge) | 3.6 → 10.0 | 5.2 → 12.8 |

Tail latency is crop rounds plus the per-crop C1 work, so it scales with the number of rounds. Bounded rows clock the crops inside "rest".

**Energy, window level only.** Mean sum of rails 18.6 W (p95 22.8 W), GPU rail 8.3 W, 90.8 Wh over 4.9 h, GR3D mean 41.7%, GPU ≤ 57 °C. Rows interleave per frame, so per-row energy is not identifiable and is not claimed.

## 2026-11-02 D03 PiP rows on the braking splits, K28 deletion setting (del_all; K41-1; measured)

- **Scripts and tables.** Driver `remote/A_method/k41_d03_detector.py` (md5 ece23bec…), analysis `remote/A_method/k41_d03_analyze.py`; tables `K41/D03/k41_1_d03_del_all.csv` + checks.
- **Parity.** The v6-S control reproduces the K28 phase-3 deletion records on all frames: 0 unit failures; 91,810 / 118,180 / 150,620 comparisons on N / A / H (A-HD includes its 56 history frames).
- **Statistics.** Δ = v6-S − row with the K40 release paired unit bootstrap; Holm over the 9 rows per dataset; refiner v3a.
- **Recovery** = deleted relevant objects found again.

| row (cap) | N: G (Δ [CI]) | A: G (Δ [CI]) | H: G (Δ [CI]) | px / v6-S N / A / H |
|---|---|---|---|---|
| v6-S (control) | 0.630 | 0.599 | 0.508 | 1 |
| (a) all anchors, nominal | 0.122 (+0.51 [+0.35, +0.63]*) | 0.159 (+0.44 [+0.35, +0.54]*) | 0.091 (+0.42 [+0.24, +0.58]*) | 1.07 / 0.59 / 1.12 |
| (b) plan-selected, nominal | 0.650 (−0.020 [−0.040, +0.006]) | 0.593 (+0.006 [−0.013, +0.024]) | 0.498 (+0.010 [−0.027, +0.043]) | 1.39 / 1.32 / 1.49 |
| (b) K39 strict | 0.664 (−0.034 [−0.073, +0.012]) | 0.568 (+0.031 [+0.007, +0.056]) | 0.476 (+0.031 [−0.015, +0.076]) | 0.94 / 0.92 / 0.93 |
| (c) (b) + support, nominal | 0.648 (−0.018 [−0.037, +0.002]) | 0.597 (+0.002 [−0.012, +0.018]) | 0.500 (+0.007 [−0.026, +0.035]) | 1.38 / 1.34 / 1.45 |
| (c) K39 strict | 0.659 (−0.029 [−0.061, +0.011]) | 0.567 (+0.032 [+0.010, +0.054]*) | 0.486 (+0.022 [−0.002, +0.045]) | 0.95 / 0.95 / 0.96 |

\* Holm-significant. Observed-strict rows are within 0.003 of the K39-strict rows.
- **Repair (v6-S → best row):** N 0.797 → 0.799 (c nominal); A 0.881 → 0.836; H 0.822 → 0.781. The PiP rows are at or below v6-S everywhere.
- **Deletion recovery:**
  - v6-S: 0.847 / 0.876 / 0.846;
  - (b) nominal: 0.848 / 0.889 / 0.899;
  - (c) strict: 0.837 / 0.833 / 0.807;
  - (a): about 0.2.
- **Reading:**
  - Planner-agnostic PiP over all 20 anchors (a) fails badly: G is 0.4–0.5 lower, because the near-ego segment boxes exhaust the budget.
  - Plan-selected PiP, with or without support, is close to v6-S. At about 1.3–1.5× v6-S's pixels (nominal cap) the differences are n.s.
  - At matched (strict) pixels, (b) / (c) are slightly above v6-S on nuScenes (n.s.), slightly below on AV2 (significant for (c) after Holm), and n.s. on A-HD.
  - This matches D04 on NAVSIM: the decision-adaptive selection matters; trajectory-anchored requests are not better at equal pixels.
- **Base setting:** done, see the next section.
- **Correction 2026-11-02.** In `k41_d03_analyze.py`, the no-refiner rows of the CSVs compared row|none against v6-S|v3a. Fixed so each refiner uses its own control, and both CSVs were regenerated. The v3a numbers above are unchanged.

## 2026-11-02 D03 PiP rows on the braking splits, natural draft (base; K41-1; measured)

- **Run and tables.** Same driver and analysis as del_all. Base ran on VUW1 (`K41_D03_HOST=cuda13`). Tables: `K41/D03/k41_1_d03_base.csv` + checks.
- **Parity:**
  - the v6-S control equals the released `J_v6S|v3a` on every frame: max |ΔJ| = 0 on 9,181 / 11,818 / 15,006 gt frames (N / A / H);
  - 0 unit failures;
  - 10 comparisons per row against the K30 match records.
- **Statistics.** As del_all: Δ = v6-S − row, Holm over the 9 rows, refiner v3a. Recovery here = natural misses of the cheap draft found by the row.

| row (cap) | N: G (Δ [CI]) | A: G (Δ [CI]) | H: G (Δ [CI]) | px / v6-S N / A / H |
|---|---|---|---|---|
| v6-S (control) | 0.211 | 0.294 | 0.418 | 1 |
| (a) all anchors, nominal | 0.024 (+0.19 [+0.08, +0.33]*) | 0.028 (+0.27 [+0.20, +0.33]*) | 0.031 (+0.39 [+0.29, +0.48]*) | 1.07 / 0.58 / 1.12 |
| (b) plan-selected, nominal | 0.137 (+0.074 [−0.001, +0.163]) | 0.265 (+0.028 [+0.004, +0.050]) | 0.390 (+0.028 [+0.007, +0.052]*) | 1.42 / 1.36 / 1.50 |
| (b) K39 strict | 0.182 (+0.029 [−0.002, +0.064]) | 0.217 (+0.076 [+0.019, +0.136]) | 0.361 (+0.057 [+0.017, +0.108]*) | 0.94 / 0.93 / 0.92 |
| (c) (b) + support, nominal | 0.190 (+0.020 [−0.008, +0.045]) | 0.289 (+0.004 [−0.008, +0.017]) | 0.401 (+0.017 [+0.007, +0.029]*) | 1.38 / 1.37 / 1.45 |
| (c) K39 strict | 0.216 (−0.005 [−0.053, +0.051]) | 0.267 (+0.026 [+0.009, +0.045]*) | 0.408 (+0.010 [−0.003, +0.027]) | 0.95 / 0.96 / 0.96 |

\* Holm-significant (p_Holm < 0.05). On AV2, (b) has raw CIs that exclude 0 but p_Holm = 0.060. Observed-strict rows are within 0.002 of the K39-strict rows.
- **No-refiner rows** have the same sign pattern. v6-S is ahead of (b) everywhere (Holm-significant on N and H), and of (c) on A (strict) and H. No row is above v6-S.
- **Repair** (agreement with π(F) on frames where the draft differs), v6-S → rows:
  - N 0.734 → (b) 0.645–0.695, (c) 0.698–0.732;
  - A 0.870 → (b) 0.768–0.803, (c) 0.822–0.834;
  - H 0.803 → (b) 0.717–0.749, (c) 0.768–0.782;
  - (a) 0.03–0.09.
  - The repair CI excludes 0 for every (b) row and for (c) on A, and for (c) strict on H. It includes 0 for (c) nominal on N and H, and its lower bound is exactly 0.000 for (c) strict on N.
- **Natural-miss recovery** (v6-S N / A / H 0.311 / 0.511 / 0.692; 270 / 713 / 1,378 objects):
  - (c) strict 0.344 / 0.492 / 0.660;
  - (b) strict 0.326 / 0.512 / 0.687;
  - (a) 0.02–0.03.
  - PiP's corridor anchors find a few more natural misses on nuScenes (+9 objects for (c) strict), but this does not translate into G.
- **Reading:**
  - With the natural draft, no PiP-style row beats v6-S on any split, at any cap, with either refiner.
  - (a) collapses (G ≈ 0.02–0.04).
  - Plan-selected (b) is below v6-S on all three splits, and significantly so after Holm on A-HD even at 1.4–1.5× pixels.
  - Adding DESP support (c) closes most of the gap: at matched pixels it is n.s. on N and H, but still significantly below v6-S on AV2.
  - Together with del_all and D04: trajectory-anchored requests need DESP's decision-adaptive support to approach v6-S, and do not exceed it at equal pixels.
- **Proposed paper text** (not applied to `$KIT/paper/`): "A PiP-style trajectory-anchored request, with or without the planner's own plan, does not match the decision field at equal pixels on any braking split (natural or deletion draft); only when DESP's support is added does it approach v6-S." The tables can go to the appendix.

## 2026-11-02 D05: complete 81-point field grid on the braking splits (K41-3; measured)

### Setup

- **Runs.** 243 jobs (81 points × N / A / H), each run with `k40r_detector.py --part grid --grid_index g`. The work was split across VUW (main chain plus workers) and VUW1, using atomic claims. Outcome: 0 failed jobs. Every job's reference units have `.done.json` with no failures and the registered detector md5.
- **Analysis and outputs:**
  - analysis: `remote/A_method/k41_d05_analyze.py` (md5 ffff714b…);
  - figure: `scripts/k41_d05_figure.py`;
  - outputs in `K41/D05/`: `k41_3_grid.csv` (every point × refiner), `k41_3_grid_summary.csv`, `k41_3_grid_checks.json`, `k41_3_grid_G_px.{pdf,png}`.
- **Parity.** The v6-S control equals the released `J_v6S|v3a` in all 243 jobs (max |ΔJ| = 0), with 10 comparisons per row. Internal check: the grid points (car, v_close 0, lateral ×1 or ×0.5) reproduce v6-S exactly (Δ = 0, CI [0, 0]).
- **Statistics.**
  - Δ = G(v6-S) − G(point) under the same refiner, using the K40 release paired unit bootstrap.
  - Holm is applied over the 81 registered points per dataset × refiner.
  - Also reported: agreement with act_gt, repair vs π(F), decisive coverage (decisive-box centre inside the executed crops), px / v6-S, and mean field time.
  - Cost: 178 GPU-wall h in total. This is job wall time, including the v6-S control pass and waits for a shared GPU.

### Range spacing is a no-op (structural; measured and code-verified)

- **Measurement.** Range factors 0.5 / 1 / 2 give the same G and px at every one of the 27 (lateral, probe, v_close) settings, on all three datasets and both refiners. The executed crops are bit-identical on all 1,775 frames compared (grid 00 vs 54, N + A).
- **Cause.**
  - The insertion demand (`vtf/dsf.insertion_grid`) decreases monotonically with range, so the sensitive cells are always {r ≤ r*}.
  - All three range grids contain the nearest cell (12.5 m).
  - `field_crop_g` takes the bounding box of the projected probe boxes. The 12.5 m cell supplies the bottom edge and lateral extent, and also the top edge whenever the probe is at least the camera height.
- **Consequences.**
  - The grid has 27 distinct request rules. Below, "points" means these 27, quoted at range factor 1.
  - Finer range spacing only costs field time: mean 414–502 / 535–715 / 499–684 µs for N / A / H, from range ×2 up to ×0.5.

### Results (refiner v3a; all points kept; figure `k41_3_grid_G_px`)

| | nuScenes | AV2 | A-HD |
|---|---|---|---|
| v6-S G | 0.211 | 0.294 | 0.418 |
| points with G above v6-S | 9 / 27, all at 1.25–1.77× px | 17 / 27 | 4 / 27, by ≤ 0.003, at 0.88–0.98× px |
| points above v6-S at ≤ 1× px | 0 | 3 (≤ +0.007, n.s.) | 4 (n.s.) |
| best point | lateral ×0.5, truck, v_close 10: 0.268 at 1.77× px (Δ −0.058 [−0.102, −0.003], p_Holm 1.0) | lateral ×2, truck, v_close 5: 0.315 at 1.16× px (Δ −0.022 [−0.041, −0.007], p_Holm 0.081) | lateral ×2, truck, v_close 0: 0.420 at 0.95× px (n.s.) |
| worst point | pedestrian probe, lateral ≥ ×1: −0.075 at 0.71× px (Δ +0.286, Holm-significant) | pedestrian probe, lateral ≥ ×1: 0.254 at 0.80× px | lateral ×0.5, truck, v_close 10: 0.334 at 1.66× px |
| Holm (81): point > v6-S / v6-S > point | 0 / 6 | 0 / 0 | 0 / 0 |

- **Not favourable to DESP:**
  - Without the refiner, two AV2 points (lateral ×2, truck, v_close 5 and 10) are Holm-significantly above v6-S over 81 points: Δ −0.024 and −0.023, at 1.16× and 1.25× px.
  - Sensitivity check, with Holm over the 27 distinct points: the same two points are also significant with refiner v3a (p_Holm 0.027). Without the refiner, two more AV2 points (lateral ×0.5, pedestrian, v_close 5 / 10, at 1.03–1.12× px) become significant.
  - The registered family is the 81 points. Because range is a no-op, those 81 contain each distinct test three times, so 81-point Holm is conservative here. Both counts are reported.
- **No point improves on v6-S on all three datasets.** The AV2-best rule (lateral ×2, truck, v_close 5) gives N 0.206 (−0.005) and H 0.396 (−0.022), and uses 1.05–1.16× px.
  - The nuScenes gains need ≥ 1.25× px.
  - Every A-HD point above v6-S is within 0.003 of it.
- **Decisive coverage is not the bottleneck.**
  - Larger probes and v_close > 0 raise decisive coverage to 0.93–1.00, against v6-S's 0.65 / 0.67 / 0.83.
  - G does not follow: on A-HD the lateral ×0.5 truck points cover 0.95 of decisive boxes, yet G is 0.334 against 0.418.
  - Interpretation (not tested here): larger crops at the same canvas lower the effective resolution.
- **The pedestrian probe at lateral ≥ ×1 fails on nuScenes** (G −0.07, repair 0.62–0.64 vs 0.73). At lateral ×0.5 it recovers to 0.115–0.129. The mechanism was not analysed.
- **Agreement with act_gt** moves by at most 0.03 across the grid (N 0.943–0.948, A 0.903–0.911, H 0.791–0.819).

### Reading

- The frozen v6-S field is not a test-selected peak. It sits at or near the G-per-pixel frontier on all three splits; it is the exact (car, v_close 0) grid point.
- Some points buy more G with more pixels on N and A. A larger probe on AV2 is a real, though small, gain: Δ ≈ 0.02 at 1.16× px, significant without the refiner (and with v3a under 27-point Holm). The same rule loses on N and H.
- This supports the paper's claim that the field parameters are not finely tuned. It also adds an honest caveat: one dataset prefers a larger probe at more pixels.

### Proposed paper text (not applied to `$KIT/paper/`)

"Over the full 81-point field grid (range spacing, lateral spacing, probe size, closing speed), no setting improves on the frozen field across all three braking splits; points above it on one split use more pixels and lose on another, and range spacing provably does not change the request (the nearest sensitive cell bounds the crop)."

Add an appendix table / figure from `K41/D05`, together with the AV2 caveat above.

## 2026-11-03 P03: P02 interaction diagnostics on the remaining 11,146 navtest tokens (K41-2; measured, CPU)

### Run

- **Scripts.** `k41_p02_interactions.py --subset remaining --fresh_per_token` (md5 06322b9a…) ran as 24 CPU shards (VUW 0–11, VUW1 12–23). The last three queued logs ran in parallel through `k41_p03_one_log.py`: the same code plus a one-log filter, so the computation is identical and deterministic. Then `k41_p02_score.py` (env `K41_P02_IN=p03`, `K41_P02_SCORE=p03_score`): stage score, then stage summary.
- **Tables.** `K41/P03/k41_2_p02_{families,joint_pdms,cost}.csv`.
- **Coverage.** Every field_test navtest token outside the P02 1,000-token subset: 11,146 tokens in 136 logs, 0 errors. PDM-Closed is instantiated fresh per token (no state leak).
- **Sampling.** Same rules as P02: up to 100 removal triples, 100 insertion pairs and 100 insertion triples per token; every Manhattan-adjacent k = 2 pair.
- **Cost.**
  - Measured: 337 CPU h for 3.10 M planner calls (insertion pairs 121 h, insertion triples 121 h, k2 71 h, removal triples 18 h). The pre-run estimate was 324 CPU h.
  - Wall time: about 26 h, dominated by one 498-token log.

### Results

| family | tested / available | moving (> 0.5 m / 4 s) | rate [Wilson 95%] | tokens with ≥ 1 | P02 (1,000 tokens) rate |
|---|---|---|---|---|---|
| removal triple | 166,866 / 220,276 | 528 | 3.16e-3 [2.91e-3, 3.45e-3] | 135 / 6,205 (2.2%) | 1.88e-3 [1.29e-3, 2.73e-3]; 7 / 546 tokens |
| insertion pair | 1,114,600 / 8.05 M | 20 | 1.79e-5 [1.16e-5, 2.77e-5] | 18 / 11,146 (0.16%) | 4.0e-5 [1.6e-5, 1.0e-4] |
| insertion triple | 1,114,600 / 99.2 M | 55 | 4.93e-5 [3.79e-5, 6.42e-5] | 36 / 11,146 (0.32%) | 7.0e-5 [3.4e-5, 1.4e-4] |
| k = 2 adjacent pair | 651,247 / 651,247 | **0** | 0 [0, 5.9e-6] | 0 / 11,146 (≤ 0.035%) | 0 / 58,398 |

- **Joint effects scored with official NAVSIM v1.1 PDMS:** 603 effects on 182 tokens, all scored. Base-plan PDMS on these tokens averages 0.856.
- **Effect on PDMS** (joint − base):

  | family | mean | lower PDMS | equal | higher PDMS | median plan shift |
  |---|---|---|---|---|---|
  | removal triples | −0.328 | 252 | 101 | 175 | 1.3 m |
  | insertion pairs | −0.258 | 19 | 0 | 1 | 17 m |
  | insertion triples | −0.227 | 50 | 0 | 5 | 22 m |

  - Overall mean −0.316 (P02: −0.315).
  - 114 / 182 tokens have at least one joint effect that lowers PDMS.

### Reading

- **Replication of P02.** P03 confirms P02 on 11× the data:
  - adjacent k = 2 interactions never move PDM-Closed: 0 of 709,645 pairs over both runs;
  - insertion interactions are very rare (about 2–5 per 100k combinations).
- **Higher removal-triple rate.** Removal triples move the plan more often than in P02: 3.2e-3 vs 1.9e-3 per combination, with non-overlapping Wilson intervals. At token level that is 2.2% vs 1.3%, with overlapping intervals. The 1,000-token subset therefore slightly under-estimated this family. It remains a finite-order empirical diagnostic, not a certificate.
- **When joint effects occur they mostly hurt PDMS.** So the singleton (pairwise-free) guarantee misses a small but real set of consequential cases: about 2% of tokens through removal triples, and well under 1% through insertions.
- **Proposed paper text** (not applied): "On all 12,146 navtest tokens, joint effects of up to three objects that no single object explains are rare (removal triples 3.0 × 10⁻³ per combination, 2.1% of eligible tokens; insertion pairs / triples ≤ 5 × 10⁻⁵; adjacent cell pairs 0 / 709,645), but when present they mostly lower PDMS."
  - Combined counts over P02 + P03: removal triples 555 / 181,261 = 3.06e-3, on 142 / 6,751 tokens (2.1%); insertion pairs 24 / 1,214,600; insertion triples 62 / 1,214,600.
