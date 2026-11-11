# K39 — answers to the mock review (post-hoc; frozen v6-S; test splits already used; holdout untouched)

- **Protocol.** PREREG.md was written before any run (md5 at registration 109152a7). Amendment §A1 (Jetson engineering
  outcome, hybrid budget wording, medians) was written after the dev checks and before any test run.
- **Statistics.** Paired unit bootstrap, 2,000 draws, seed 0, 95% percentile. Δ = DESP − row; Δ > 0 means DESP is better.
- **Status.** All five items are DONE on 2026-10-27, ahead of the 10-30 / 11-02 dates.

## K39-1 — latency–decision frontier on the Jetson AGX Xavier (k39_1_rows.csv, k39_1_deadline.csv, k39_1_diffs.csv, parity.json, k39_1_frontier.pdf/png)

**Setup**
- Exclusive window 10:15–13:04 (kit time): MAXN, clocks locked, fan 255. Hardware was restored afterwards, and the
  source / engine hashes were unchanged. No engine was built.
- Harness: experiments/k39_jetson_20261027/scripts/89_k39.py, built from the frozen K38-1 harness by build.py.
- Frames are exactly K38-1's: nuScenes test, 20 scenes (805 timed frames); AV2 test, 40 logs (2,968 timed frames); a
  200-frame warm-up.
- Three repetitions. All rows ran in one process per repetition, in rotated order, each with its own history.

**A. Output-identical engineering** (decided on dev only; PREREG §A1):
- **A1, CUDA-graph replay of the fixed-shape TensorRT engines** (cheap draft and static whole-frame engines): kept.
  - On the dev prefix, outputs were identical and mean latency dropped on every row: DESP 15.27 → 15.04 ms; u960 + comp.
    18.74 → 17.69 ms; u1280 + comp. 24.83 → 23.77 ms; cheap 6.76 → 6.40 ms.
  - The change applies to every row.
- **A1 for the dynamic crop engine:** dropped. Graph replay crashed with a CUDA illegal memory access.
- **A2 / A3 / A4:** no room.
  - Host copies are 0.15–0.4 ms per call.
  - Every engine call in a round depends on the previous one.
  - Crops are already batched per canvas, and the guard per growth step.
- **Gate on test: PASSED, 100%.** For v6-S and B2K-A, the action (with and without refiner), px, crop / round / call
  counts and bound hits equal the stored K38-1 staged records.
  - This holds on all 805 + 2,968 timed frames, for every K39 × K38 repetition pair.
  - The actions also equal K34-3 jetson_frames.csv (805 / 586 common frames).
- **Other checks:**
  - stored J recomputed to 4e-15;
  - a_gt equals K30 GT to 9e-16;
  - medians_k39.json sha256 8fa995f7…, measured on dev in the window.

**Rows at D = ∞** (G on the timed frames, 3 repetitions pooled; latency in ms; execs = TensorRT engine executions per
frame; px includes the guard):

| nuScenes test 20 | G | mean | p50 | p95 | p99 | execs | px |
|---|---|---|---|---|---|---|---|
| cheap | 0 | 7.0 | 7.2 | 7.7 | 8.1 | 1.0 | 0.043 |
| u512 + comp. | 0.336 | 13.3 | 10.2 | 21.4 | 32.2 | 1.50 | 0.102 |
| u640 + comp. | 0.330 | 15.3 | 12.3 | 23.5 | 33.1 | 1.53 | 0.171 |
| u800 + comp. (dyn engine) | 0.369 | 20.7 | 17.6 | 28.6 | 36.0 | 1.54 | 0.267 |
| u960 + comp. | 0.386 | 21.2 | 18.1 | 29.1 | 38.9 | 1.56 | 0.384 |
| u1280 + comp. | 0.395 | 27.2 | 24.2 | 34.8 | 40.6 | 1.56 | 0.654 |
| native + comp. | 0.364 | 36.5 | 33.6 | 43.7 | 49.3 | 1.58 | 1.031 |
| hybrid tsp8 + comp. @ DESP px | 0.373 | 22.9 | 12.0 | 73.1 | 97.6 | 3.12 | 0.103 |
| **DESP v6-S** | **0.380** | 20.4 | 13.4 | 55.1 | 77.6 | 2.74 | **0.074** |
| **DESP B2K-A** | **0.382** | 19.4 | 13.5 | 46.8 | 56.7 | 2.24 | 0.078 |
| **DESP B1K-A** (post-hoc) | **0.391** | 15.7 | 13.7 | **32.8** | 38.8 | 1.92 | **0.062** |

| AV2 test 40 | G | mean | p50 | p95 | p99 | execs | px |
|---|---|---|---|---|---|---|---|
| cheap | 0 | 9.6 | 9.6 | 10.1 | 10.5 | 1.0 | 0.026 |
| u512 + comp. | 0.358 | 21.1 | 20.8 | 36.3 | 44.5 | 1.95 | 0.067 |
| u640 + comp. | 0.373 | 24.0 | 23.6 | 38.9 | 47.6 | 1.97 | 0.103 |
| u800 + comp. (dyn engine) | 0.381 | 28.2 | 27.9 | 43.3 | 51.0 | 1.98 | 0.153 |
| u960 + comp. | 0.380 | 30.5 | 30.1 | 45.2 | 53.3 | 1.98 | 0.232 |
| u1280 + comp. | 0.400 | 39.6 | 39.4 | 53.1 | 63.5 | 1.99 | 0.400 |
| native + comp. | 0.147 | 72.0 | 72.1 | 83.3 | 92.4 | 1.97 | 1.012 |
| hybrid tsp8 + comp. @ DESP px | 0.296 | 37.1 | 30.1 | 96.8 | 138.8 | 4.44 | 0.073 |
| DESP v6-S | 0.297 | 31.5 | 27.7 | 74.3 | 105.0 | 3.64 | 0.053 |
| DESP B2K-A | 0.304 | 31.0 | 30.2 | 63.8 | 89.5 | 2.86 | 0.059 |
| DESP B1K-A (post-hoc) | 0.269 | 23.8 | 23.1 | 44.6 | 59.7 | 2.29 | 0.042 |

The u960 control on the dynamic engine gives G 0.380 / 0.403 at p95 31.8 / 47.0 ms (+2.7 / +1.8 ms over the static
engine). u800 therefore pays this overhead too.

**Deadline sweep** (G(D) / late share; in-loop rule for DESP, hold for every row; full table in k39_1_deadline.csv):

| nuScenes | 33.3 | 50 | 66.7 | 83.3 | 100 |
|---|---|---|---|---|---|
| u512 / u640 + comp. | 0.351 / 0.345 | 0.336 / 0.330 | same | same | same |
| u960 + comp. | 0.423 (3%) | 0.386 | 0.386 | 0.386 | 0.386 |
| u1280 + comp. | 0.434 (9%) | 0.410 | 0.395 | 0.395 | 0.395 |
| native + comp. | −8.02 (57%) | 0.364 | 0.364 | 0.364 | 0.364 |
| hybrid @ DESP px | −0.280 (23%) | 0.153 (13%) | 0.196 (7%) | 0.335 | 0.345 |
| DESP v6-S | −0.204 (15%) | 0.354 (5%) | 0.378 | 0.374 | 0.380 |
| DESP B2K-A | 0.290 (13%) | 0.383 (4%) | 0.391 | 0.382 | 0.382 |
| **DESP B1K-A** | **0.437 (4%)** | **0.400** | 0.391 | 0.391 | 0.391 |

| AV2 | 33.3 | 50 | 66.7 | 83.3 | 100 |
|---|---|---|---|---|---|
| u512 + comp. | −0.095 (7%) | 0.361 | 0.361 | 0.360 | 0.360 |
| u640 + comp. | 0.139 (9%) | 0.367 | 0.375 | 0.373 | 0.373 |
| u960 + comp. | −0.692 (27%) | 0.352 (2%) | 0.382 | 0.380 | 0.380 |
| u1280 + comp. | −1.649 (73%) | 0.119 (8%) | 0.408 | 0.401 | 0.400 |
| native + comp. | −1.649 (100%) | −1.649 (100%) | −1.647 (72%) | 0.165 | 0.148 |
| hybrid @ DESP px | −0.726 (47%) | −0.279 (27%) | −0.089 (16%) | 0.331 | 0.325 |
| DESP v6-S | 0.000 (0%) | 0.213 (11%) | 0.315 (4%) | 0.290 | 0.302 |
| DESP B2K-A | −0.241 (24%) | 0.130 (19%) | 0.336 (4%) | 0.336 | 0.305 |
| DESP B1K-A | −0.148 (23%) | 0.277 (3%) | 0.269 | 0.269 | 0.269 |

**Paired differences** (k39_1_diffs.csv; selected):
- **Against the hybrid at the same pixels, DESP is significantly better under tight deadlines.**
  - nuScenes D = 50: v6-S +0.201 [+0.008, +0.768]; B2K-A +0.229; B1K-A +0.247.
  - AV2 D = 33.3–66.7: v6-S +0.40 to +0.73, B2K-A +0.41 to +0.49, B1K-A +0.36 to +0.58. All CIs exclude 0.
- **Against whole frame, at D = ∞ and loose deadlines:**
  - nuScenes: DESP ties u800 – u1280 and native, and is above u640 (+0.050 [+0.001, +0.102]).
  - AV2, D ≥ 66.7: no DESP − whole-frame difference excludes 0, with two exceptions:
    - DESP is above native at 66.7 (+1.9);
    - B1K-A is below the u960 dynamic control (−0.134 [−0.274, −0.003]).
  - AV2 whole frame at 512–1280 is ahead in G at D = ∞ (by 0.06–0.10), but not significantly with 40 logs.
- **At D = 33.3:**
  - nuScenes: B1K-A is above u640 (+0.092) and the hybrid (+0.717); v6-S and B2K-A are below the 512–1280 whole frames.
  - AV2: every DESP form is above u1280 / native / hybrid.
- **AV2 D = 50:** v6-S is below u800 (dyn) by 0.170 [−0.312, −0.015].

**Reading (for the writing session):**
1. **The frontier claim the data support.**
   - On the Jetson, DESP reaches the whole-frame frontier with 4–6× fewer network pixels. nuScenes: B1K-A 0.391 at p95
     32.8 ms with 0.062 px, vs u960 0.386 at 29.1 ms with 0.384 px, and u1280 0.395 at 34.8 ms with 0.654 px.
   - The tail costs time: v6-S p95 55 ms, B2K-A 47 ms. Whole-frame inference at 512–960 px is a strong latency
     baseline on this device.
   - On AV2 (40 logs), whole frame is ahead in G at every scale up to 1280 (0.36–0.40 vs 0.27–0.30), as in K38-1.
2. **Pixels vs time.** DESP's advantage is decision value per pixel; it is not wall-clock time on the Xavier. The hybrid
   that uses the same pixels has a far worse tail (p95 73 / 97 ms) and loses under every deadline ≤ 66.7 ms.
3. **B1K-A** (post-hoc, labelled) is the best DESP point on nuScenes under tight deadlines: 0.437 at 33.3 ms,
   0.400 at 50 ms. On AV2 it gives up G (0.269).
4. **Native frames are unusable at ≤ 66.7 ms on AV2** (100% / 72% late) and at 33.3 ms on nuScenes.

## K39-2 — strict post-guard per-frame pixel cap (k39_2_rows.csv, k39_2_diffs.csv)

**Run and checks**
- One VUW run (experiments/k39_vuw_20261027/k39_eval.py, md5 b8608ba3) on the three test splits.
- Checks (k39_vuw_checks.json), all 0 differences:
  - v6-S and u960 + C1 vs K28 on all 9,181 / 11,818 / 15,062 frames;
  - the 8 heuristics at the nominal m = 1 cap vs K30 match (a / J / px);
  - strict machinery with an infinite cap = nominal (smoke).
- Under the strict cap, realised px / DESP px ≤ 1.000 on every frame of every row.
- Under the nominal cap the ratio averages 1.10–1.47 and reaches 5–13× on single frames (guard growth).

**v6-S − row (G, refiner v3a).** v6-S G = 0.211 / 0.294 / 0.418. Bold = primary row (PREREG dev rule).

| row | nuScenes strict | nuScenes nominal | AV2 strict | AV2 nominal | AHD strict | AHD nominal |
|---|---|---|---|---|---|---|
| **tsp8 (hybrid)** | **+0.032 [+0.012, +0.051]** | +0.037 [−0.012, +0.096] | **+0.019 [+0.010, +0.031]** | −0.003 [−0.014, +0.008] | **+0.026 [+0.011, +0.046]** | +0.009 [−0.002, +0.022] |
| tsp2 | +0.078 [+0.034, +0.125] | +0.097 n.s. | +0.013 [−0.003, +0.029] | +0.007 n.s. | +0.030 [+0.008, +0.054] | +0.014 n.s. |
| unc (primary: nuScenes unc4; AV2 / AHD unc2) | **+0.271 [+0.058, +0.435]** | +0.280 | **+0.278 [+0.186, +0.370]** | +0.072 | **+0.138 [+0.017, +0.263]** | +0.133 |
| tcr (primary: tcr3 / tcr0 / tcr0) | **+0.058 [+0.022, +0.105]** | +0.043 | **+0.087 [+0.052, +0.127]** | +0.058 | **+0.035 [+0.004, +0.073]** | +0.041 |
| cor (primary: cor0 / cor8 / cor8) | **+0.134 [+0.033, +0.233]** | +0.158 | **+0.179 [+0.103, +0.265]** | +0.032 | **+0.041 [−0.006, +0.099]** | +0.025 n.s. |

Secondary rows, strict cap:
- nuScenes: unc2 +0.327, tcr0 +0.117, cor8 +0.259 (all significant).
- AV2: unc4 +0.267, tcr3 +0.121, cor0 +0.143 (all significant).
- AHD: unc4 +0.129 (significant), tcr3 +0.034 (n.s.), cor0 −0.040 [−0.144, +0.061] (n.s.).

**Reading.** Under a strict cap, v6-S is significantly above every primary heuristic in 11 of 12 cells. The exception
is AHD ego corridor: +0.041, n.s. The hybrid, which tied v6-S under the nominal cap, is significantly below under the
strict cap on all three splits.
- With an equal pixel budget, guard included, the selection rule matters.
- The earlier ties came from the heuristics' extra guard pixels (1.2–1.5× on average).

## K39-3 — multiplicity (k39_3_pvalues.csv, k39_3_summary.csv)

- Published deltas and CIs reproduced exactly (max |diff| 9e-17).
- p = 2·min(P*(Δ ≤ 0), P*(Δ ≥ 0)); the +1-corrected version gives the same counts.

| family | comparisons | CI excludes 0 | Holm < 0.05 | BH < 0.05 |
|---|---|---|---|---|
| Tab. 1 @cap, 12 methods × 3 (primary) | 36 | 29 | 25 | 28 |
| Tab. 1 @cap, all 14 rows × 3 | 42 | 35 | 30 | 34 |
| ablation: field only / support only / pivots × 3 | 9 | 7 | 7 | 7 |

- **Significant after BH but not after Holm:**
  - AnA one pass, nuScenes (p_raw 0.024);
  - FOVEA KDE_I, AV2 (0.016);
  - counterfactual one pass, AHD (0.015).
- **Significant before correction but not after either:** AnA iterated, nuScenes (p_raw 0.046; p_BH 0.057).
- No comparison favours a baseline over v6-S in either family.

## K39-4 — exactness cross-tabulation (k39_4_crosstab.csv, k39_4_single.csv)

**Agreement with π(F) = u960 + C1, without / with refiner:**
- DESP v6-S: 0.976 / 0.980 (nuScenes), 0.968 / 0.969 (AV2), 0.940 / 0.942 (AHD).
- native + comp.: 0.974 / 0.977, 0.951 / 0.958, 0.922 / 0.926.

**On the frames where DESP ≠ π(F)** (without refiner), DESP equals the required action vs π(F) does:
- nuScenes: 128 vs 81 (9 both wrong);
- AV2: 219 vs 126 (37);
- AHD: 722 vs 130 (45).

The native row against π(F) shows the same pattern at a lower agreement.

**Single interventions, each applied ALONE** (deployed procedure without refiner; share of DESP's disagreement with π(F)
that the intervention removes; new disagreements in parentheses):

| intervention alone | nuScenes (D0 = 218) | AV2 (382) | AHD (897) |
|---|---|---|---|
| a exact locality | 42.7% (26) | 46.9% (39) | 30.0% (78) |
| b F's geometry / history | 30.3% (3) | 54.5% (4) | 30.8% (11) |
| c closing probe 10 m/s (= 20 m/s, identical on every frame) | 20.2% (4) | 19.4% (45) | 19.3% (15) |
| d extended grid | 23.9% (36) | 27.7% (78) | 15.8% (111) |
| e three probe templates | 22.9% (34) | 21.5% (74) | 12.8% (103) |
| f round cap 10 | 0% (0) | 0% (0) | 0% (0) |
| resolved by no single intervention | 26.6% | 19.9% | 42.4% |
| cumulative chain a → f (stored) | 78.4% | 90.8% | 52.6% |

- The check that the recomputed "a" equals the stored ch_a passed on every frame.
- The 10 and 20 m/s closing probes produce the same footprint on every frame, which is consistent with the stored chain:
  its c2 step adds nothing.

## K39-5 — NAVSIM planner-specific field (k39_5_summary.csv, k39_5_top5.csv, k39_5_tokens.csv.gz)

**Scope and checks**
- 12,146 navtest tokens, 136 logs.
- Replanning on the draft reproduces the stored base poses exactly.
- The braking footprint is empty exactly when v3 ran 0 rounds (0 mismatches).

**Footprints**
- The probed PDM-Closed footprint differs from the braking-field footprint on 99.8% of tokens (IoU < 0.9; median IoU
  0.33).
- 67% of the PDM footprint lies outside the braking field.

**Trajectories and ΔPDMS**
- The selected trajectory differs (> 0.5 m) on 4.3% [3.6, 4.9] of tokens (517).
- On those tokens, ΔPDMS (spec_pdm − braking field) = +0.045 [+0.005, +0.091]; over all tokens it is +0.0019
  [+0.0002, +0.0040].
- On the 517 tokens, spec_pdm is better on 186, worse on 265 and tied on 66. 35 tokens move by 1.0, 31 of them in
  spec_pdm's favour; the top 5 (|Δ|, then token) are all +1.

**Wall clock per frame** (stored K20 timings), p50 / p95:
- identical-output fast implementation, 1 worker: 6.0 / 10.1 s;
- 8 workers: 1.34 / 1.96 s;
- frozen implementation: 14.7 / 19.2 s.

## K39-6 — target-action fidelity vs compute, without the decision-value loss (k39_6_fidelity.csv, k39_6_diffs.csv, k39_6_checks.json, k39_6_fidelity_px.pdf/png)

**Setup**
- PREREG K39-6 was written before computing.
- Target: π(F) = u960 + C1, no refiner (K28 instr).
- Every row is taken without refiner, with the braking controller and its own stored history, on the GT frames of the
  three test splits.
- Script: remote/A_method/k39_6_fidelity.py (CPU, stored records only; no detector re-run).

**Checks, all 0 differing frames**
- K28 basewf u960 + C1 = F.
- K33 pub u960 (C1, no refiner) = F.
- BAR / router-G / Chanakya: their stored refined J is reproduced on every frame from the stored routing sizes. Their
  no-refiner action is therefore derived from the routed whole-frame row (a_wf<s>|none) or the draft.
- Whole frame at 800 has no full-split record and is reported missing; AV2 has no 448.

**Main numbers.** A = share of frames equal to π(F); repair = A | cheap ≠ π(F); px = mean realised detector px incl.
the guard.

| | nuScenes A / repair / px | AV2 A / repair / px | AHD A / repair / px |
|---|---|---|---|
| always KEEP (reference) | 0.899 / 0.534 / 0 | 0.662 / 0.417 / 0 | 0.592 / 0.293 / 0 |
| cheap | 0.930 / 0 / 0.043 | 0.840 / 0 / 0.026 | 0.770 / 0 / 0.031 |
| **DESP v6-S** | **0.976 / 0.734 / 0.078** | **0.968 / 0.870 / 0.054** | **0.940 / 0.803 / 0.070** |
| support only / field only / pivots | 0.964 / 0.944 / 0.975 | 0.929 / 0.880 / 0.960 | 0.888 / 0.822 / 0.924 |
| hybrid tsp8: @S / nominal x1 / strict | 0.974 / 0.974 / 0.969 | 0.967 / 0.968 / 0.962 | 0.939 / 0.939 / 0.924 |
| u448 / u512 / u640 + C1 | 0.970 / 0.971 / 0.979 | – / 0.953 / 0.964 | 0.933 / 0.942 / 0.958 |
| u1280 / u1536 / native + C1 | 0.982 / 0.975 / 0.974 | 0.974 / 0.966 / 0.951 | 0.964 / 0.944 / 0.922 |
| AnA one pass (uncapped / @cap) | 0.981 / 0.974 (px 0.100 / 0.057) | 0.963 / 0.955 | 0.959 / 0.932 |
| AdaScale (uncapped / @cap) | 0.981 / 0.967 (px 0.306 / 0.055) | 0.986 / 0.928 | 0.985 / 0.908 |
| SAHI / FOVEA KDE_I / UFP (uncapped) | 0.967 / 0.976 / 0.975 | 0.918 / 0.951 / 0.957 | 0.920 / 0.956 / 0.953 |
| BAR / router-G / Chanakya (uncapped) | 0.962 / 0.964 / 0.963 | 0.901 / 0.913 / 0.917 | 0.875 / 0.775 / 0.900 |

On frames where the row ≠ π(F), the share where the row equals the required action (π(F)'s share in parentheses):
- v6-S: 0.59 (0.37), 0.57 (0.33), 0.80 (0.14);
- whole frame u1280: 0.53 (0.42), 0.47 (0.43), 0.38 (0.46);
- AdaScale: 0.45 (0.48), 0.43 (0.50), 0.41 (0.49).

Most of v6-S's remaining disagreements with π(F) are frames where π(F) itself is wrong.

**Where v6-S is NOT best (stated plainly)**
1. **Raw A is dominated by KEEP.** Always-KEEP already scores 0.899 on nuScenes, and the draft 0.930. The conditional
   metrics (repair, transitions, braking) are the informative ones.
2. **More pixels buy more fidelity.** Every row that is significantly above v6-S on a fidelity metric uses more pixels:
   - **nuScenes:**
     - AnA one pass / iterated (1.3× px): A +0.004 / +0.005; repair +0.050 / +0.052; transitions +0.051 / +0.053;
       π(F) ≥ DECEL +0.048 / +0.050; braking required +0.023 (iterated only).
     - u640 + C1 (2.2×): repair +0.083, transitions +0.053, π(F) ≥ DECEL +0.070.
     - AdaScale (3.9×): A +0.004, repair +0.095.
     - u1280 (8.4×): A +0.006, repair +0.139.
     - FOVEA ×2 (8–9×) and SAHI (24×).
   - **AV2:** only AdaScale (5.0×; A +0.018, repair +0.075, braking required +0.032) and u1280 / u1536 (7.5–10.7×).
   - **AHD:** only AdaScale (4.2×; A +0.045, repair +0.161, transitions +0.207), u1280 (6.3×) and u1536 / native /
     SAHI on some metrics.
3. **At v6-S's pixel budget, nothing beats it significantly.** This holds for every capped published row, the
   object-level rows @cap and the hybrid under both caps; but there are near-ties and two weak spots:
   - The hybrid tsp8 in its Tab. 1 / nominal form ties v6-S on repair rate on all three splits (nuScenes +0.016
     [−0.003, +0.035], AV2 +0.002, AHD −0.001). It uses 1.3–1.5× px.
   - Under the strict cap it is significantly below v6-S on A, repair and braking on all three splits.
   - **AHD whole frame u512 + C1 at the same px** (0.071 vs 0.070) has a higher repair rate (0.831 vs 0.803, n.s.) and
     equal A (0.942 vs 0.940, n.s.). v6-S is ahead only on A | braking required (+0.056 [+0.008, +0.132]).
   - On nuScenes, u448 / u512 (1.0–1.3× px) are below v6-S on A, but u512 is higher on A | π(F) ≥ DECEL (0.827 vs
     0.799, n.s.).
4. **Transitions are v6-S's weakest conditional metric:** 0.77 / 0.82 / 0.64. On AHD, whole frame at 640 px and above
   and AnA uncapped are higher (0.66–0.85), and on nuScenes AnA uncapped is significantly higher. v6-S's same-fidelity
   history breaks when the crop set changes (as in the K33 AnA mechanism).

**Reading.** v6-S repairs 73 / 87 / 80% of the draft's disagreements with π(F) at 0.05–0.08 px. It is the best or tied
pipeline at its pixel budget on every fidelity metric, except a non-significant AHD u512 edge on repair. Pipelines with
higher fidelity spend 1.3× (AnA, nuScenes only) to 4–8× (AdaScale, u1280) the pixels.

## Files
- **Code:**
  - experiments/k39_jetson_20261027/ (build.py, graphs.py, bench.py, compare_runs.py, medians.py, window.py,
    medians_k39.json, gt_k30.csv);
  - experiments/k39_vuw_20261027/ (build.py → k39_eval.py);
  - remote/A_method/ (k39_vuw_chain.sh, k39_vuw_check.py, k39_2_report.py, k39_3_mult.py, k39_4_report.py);
  - remote/A_navsim/ (k39_5_navsim.py, k39_5_chain.sh);
  - scripts/k39_1_report.py.
- **Raw outputs:**
  - Jetson results/raw/*k39_static*;
  - VUW results/k39/{vuw, k3, k4, k5}.
