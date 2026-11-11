# K35-1 — missed-episode decomposition and attribution (POST-HOC, three test splits)

**Post-hoc analysis** of stored records (K28 p3 instrumented test rows, K30 match geometry, K34-1 replay annotations); no
pipeline was run. Frozen v6-S; K31 episode definitions; thresholds τ1 = 1.0, τ2 = 3.5 m/s² unchanged.
- Code: remote/A_method/k35_k1.py (md5 47db6171), run on VUW 2026-10-25.
- Inputs: hashed in <ds>/manifest.json.
- Episodes, severity strata, onset capping and insufficient loss are K34-1's.
  - Reproduction check (<ds>/k34_reproduction.json): every shared pipeline's episode-level missed flag equals K34's
    (0 mismatches), and the loss matches to 3e-14.
- Paired unit bootstrap, 2,000 draws, seed 0, shared draws across pipelines. **Bold** = the 95% interval excludes 0.
  Sign: A − B, so > 0 means A misses more.

## (a) Missed-episode rate per pipeline (all episodes)

| pipeline | nuScenes (147 ep.) | AV2 (190 ep.) | Argoverse-HD (71 ep.) |
|---|---|---|---|
| cheap | 0.347 | 0.111 | 0.169 |
| v6-S | 0.408 | 0.189 | 0.380 |
| v6-S without refiner | 0.388 | 0.142 | 0.211 |
| u960 + C1 + v3a | 0.347 | 0.163 | 0.366 |
| u960 + C1 without refiner | 0.299 | 0.100 | 0.183 |
| native + C1 + v3a | 0.361 | 0.153 | 0.380 |
| native + C1 without refiner | 0.320 | 0.105 | 0.169 |

## (a) Registered contrasts: missed-episode rate

| contrast | stratum | nuScenes | AV2 | Argoverse-HD |
|---|---|---|---|---|
| v6-S - v6-S without refiner | all | +0.020 [+0.000, +0.054] | **+0.047 [+0.021, +0.078]** | **+0.169 [+0.070, +0.271]** |
| v6-S - v6-S without refiner | below_hard | +0.018 [+0.000, +0.057] | **+0.059 [+0.023, +0.102]** | **+0.180 [+0.069, +0.320]** |
| v6-S - v6-S without refiner | at_or_above_hard | +0.026 [+0.000, +0.091] | +0.018 [+0.000, +0.065] | +0.100 [+0.000, +0.286] |
| v6-S without refiner - native + C1 without refiner | all | **+0.068 [+0.022, +0.120]** | **+0.037 [+0.005, +0.075]** | +0.042 [+0.000, +0.093] |
| v6-S without refiner - native + C1 without refiner | below_hard | **+0.073 [+0.015, +0.143]** | **+0.044 [+0.007, +0.088]** | +0.049 [+0.000, +0.103] |
| v6-S without refiner - native + C1 without refiner | at_or_above_hard | +0.053 [+0.000, +0.138] | +0.018 [+0.000, +0.064] | +0.000 [+0.000, +0.000] |
| v6-S without refiner - u960 + C1 without refiner | all | **+0.088 [+0.039, +0.142]** | **+0.042 [+0.015, +0.077]** | +0.028 [+0.000, +0.076] |
| v6-S without refiner - u960 + C1 without refiner | below_hard | **+0.092 [+0.032, +0.159]** | **+0.044 [+0.014, +0.083]** | +0.033 [+0.000, +0.091] |
| v6-S without refiner - u960 + C1 without refiner | at_or_above_hard | +0.079 [+0.000, +0.176] | +0.036 [+0.000, +0.094] | +0.000 [+0.000, +0.000] |
| native + C1 + v3a - native + C1 without refiner | all | **+0.041 [+0.007, +0.080]** | **+0.047 [+0.021, +0.078]** | **+0.211 [+0.104, +0.322]** |
| native + C1 + v3a - native + C1 without refiner | below_hard | +0.037 [+0.000, +0.083] | **+0.059 [+0.023, +0.102]** | **+0.230 [+0.105, +0.377]** |
| native + C1 + v3a - native + C1 without refiner | at_or_above_hard | +0.053 [+0.000, +0.138] | +0.018 [+0.000, +0.065] | +0.100 [+0.000, +0.286] |
| v6-S without refiner - cheap | all | +0.041 [-0.025, +0.113] | +0.032 [-0.015, +0.080] | +0.042 [-0.039, +0.127] |
| v6-S without refiner - cheap | below_hard | +0.055 [-0.027, +0.144] | +0.037 [-0.023, +0.099] | +0.033 [-0.056, +0.125] |
| v6-S without refiner - cheap | at_or_above_hard | +0.000 [-0.107, +0.107] | +0.018 [+0.000, +0.064] | +0.100 [+0.000, +0.333] |

Onset delay and insufficient-loss share for the same contrasts and strata: <ds>/k1a_contrasts.csv; per pipeline:
<ds>/k1a_summary.csv.

**Reading.**
- The refiner v3a is the largest part of v6-S's excess misses: v6-S − v6-S without refiner is +0.169 on Argoverse-HD and
  +0.047 on AV2 (both significant), +0.020 on nuScenes (n.s.).
- The refiner has the same effect inside the whole-frame row: native + C1 + v3a − native + C1 without refiner is +0.211 /
  +0.047 / +0.041, all significant.
- Without the refiner, v6-S is not distinguishable from cheap on any split (+0.041 / +0.032 / +0.042, all n.s.).
- The remaining selection gap vs the whole frame without refiner is significant on nuScenes and AV2:
  - vs native + C1: +0.068 and +0.037;
  - vs u960 + C1: +0.088 and +0.042;
  - on Argoverse-HD it is not significant: +0.042 [0.000, 0.093] vs native, +0.028 vs u960.

## (b) Attribution of episodes where exactly one of cheap / v6-S detects

**Direction cheap detects, v6-S misses (exclusive classes, in the registered order).**
- F = the episode frames where cheap fires (action ≥ DECELERATE).
- E1: the decisive object is outside the verified region on every frame of F.
- E2: it is inside on some frame of F, but on none of them does v6-S have it with a finite TTC.
  - Subclass "matched": v6-S's scene has the object, but C1 has no 3-observation chain.
  - Subclass "not matched": inside, but no v6-S box matches it.
- E3: inside with finite TTC on some frame of F, and v6-S without refiner would brake there.
- E6: cheap has no box matching the decisive object on any frame of F.
- E-other: everything else.

**Reverse direction (v6-S detects, cheap misses)**, analogous classes on v6-S's firing frames:
- R1: the decisive object is matched only in a verified crop, with no cheap box;
- R2: both see it, and v6-S's estimate brakes;
- R6: v6-S fires without a decisive-object match;
- R-other.

| dataset | direction | class | subclass | episodes | share of episodes | v6-S insufficient-loss share | cheap insufficient-loss share |
|---|---|---|---|---|---|---|---|
| nuScenes | cheap_detects_v6S_misses | E-other | inside_finite_ttc_refiner_not_decisive | 2 | 0.111 | 0.012 | 0.007 |
| nuScenes | cheap_detects_v6S_misses | E2 | inside_matched_no_finite_ttc | 15 | 0.833 | 0.469 | 0.377 |
| nuScenes | cheap_detects_v6S_misses | E2 | inside_not_matched | 1 | 0.056 | 0.520 | 0.616 |
| nuScenes | v6S_detects_cheap_misses | R1 | decisive_object_only_in_verified_crop | 5 | 0.556 | 0.521 | 0.433 |
| nuScenes | v6S_detects_cheap_misses | R2 | both_see_it_v6S_estimate_brakes | 4 | 0.444 | 0.479 | 0.567 |
| AV2 | cheap_detects_v6S_misses | E-other | inside_finite_ttc_refiner_not_decisive | 2 | 0.111 | 0.023 | 0.028 |
| AV2 | cheap_detects_v6S_misses | E2 | inside_matched_no_finite_ttc | 12 | 0.667 | 0.881 | 0.871 |
| AV2 | cheap_detects_v6S_misses | E2 | inside_not_matched | 1 | 0.056 | 0.019 | 0.029 |
| AV2 | cheap_detects_v6S_misses | E3 | refiner_lowers_demand_below_tau1 | 3 | 0.167 | 0.077 | 0.072 |
| AV2 | v6S_detects_cheap_misses | R1 | decisive_object_only_in_verified_crop | 3 | 1.000 | 1.000 | 1.000 |
| Argoverse-HD | cheap_detects_v6S_misses | E-other | inside_finite_ttc_refiner_not_decisive | 4 | 0.211 | 0.271 | 0.249 |
| Argoverse-HD | cheap_detects_v6S_misses | E1 | outside_verified_region | 1 | 0.053 | 0.004 | 0.000 |
| Argoverse-HD | cheap_detects_v6S_misses | E2 | inside_matched_no_finite_ttc | 3 | 0.158 | 0.049 | 0.071 |
| Argoverse-HD | cheap_detects_v6S_misses | E3 | refiner_lowers_demand_below_tau1 | 11 | 0.579 | 0.675 | 0.681 |
| Argoverse-HD | v6S_detects_cheap_misses | R1 | decisive_object_only_in_verified_crop | 4 | 1.000 | 1.000 | 1.000 |

By severity:

| dataset | direction | severity | class | episodes |
|---|---|---|---|---|
| Argoverse-HD | cheap_detects_v6S_misses | at_or_above_hard | E-other | 1 |
| Argoverse-HD | cheap_detects_v6S_misses | at_or_above_hard | E3 | 1 |
| Argoverse-HD | cheap_detects_v6S_misses | below_hard | E-other | 3 |
| Argoverse-HD | cheap_detects_v6S_misses | below_hard | E1 | 1 |
| Argoverse-HD | cheap_detects_v6S_misses | below_hard | E2 | 3 |
| Argoverse-HD | cheap_detects_v6S_misses | below_hard | E3 | 10 |
| Argoverse-HD | v6S_detects_cheap_misses | below_hard | R1 | 4 |
| AV2 | cheap_detects_v6S_misses | at_or_above_hard | E2 | 2 |
| AV2 | cheap_detects_v6S_misses | below_hard | E-other | 2 |
| AV2 | cheap_detects_v6S_misses | below_hard | E2 | 11 |
| AV2 | cheap_detects_v6S_misses | below_hard | E3 | 3 |
| AV2 | v6S_detects_cheap_misses | below_hard | R1 | 3 |
| nuScenes | cheap_detects_v6S_misses | at_or_above_hard | E2 | 2 |
| nuScenes | cheap_detects_v6S_misses | below_hard | E-other | 2 |
| nuScenes | cheap_detects_v6S_misses | below_hard | E2 | 14 |
| nuScenes | v6S_detects_cheap_misses | at_or_above_hard | R2 | 1 |
| nuScenes | v6S_detects_cheap_misses | below_hard | R1 | 5 |
| nuScenes | v6S_detects_cheap_misses | below_hard | R2 | 3 |

**Reading.**
- On nuScenes (16 of 18) and AV2 (13 of 18), the episodes v6-S misses while cheap brakes are mostly E2: the decisive
  object is inside the verified region and usually in v6-S's scene, but has no finite C1 TTC on the frames where cheap
  fires. They carry 99% (nuScenes) and 90% (AV2) of v6-S's insufficient loss in this direction.
- On Argoverse-HD, 11 of 19 are E3, the refiner lowering demand below τ1 (68% of the loss); 1 is E1, outside the region.
- E6 is empty as an exclusive class. As a non-exclusive flag, cheap fired without any box on the decisive object in
  0 / 2 / 1 of these episodes (nuScenes / AV2 / Argoverse-HD). Those episodes are classified earlier in the order.
- Reverse episodes (9 / 3 / 4): mostly R1, objects found only in verified crops.
- On every firing frame used, the match records' v6-S action equals the K28 action (0 differences).

## Files
- <ds>/k1a_summary.csv: per pipeline, stratum and metric (missed rate, mean onset delay s, insufficient-loss share of
  cheap's total loss), with intervals.
- <ds>/k1a_contrasts.csv: the five registered contrasts × strata × metrics.
- <ds>/k1b_episodes.csv: every attributed episode with its class and the per-condition firing-frame counts.
- <ds>/k1b_classes.csv: class counts and loss shares, all and per severity.
