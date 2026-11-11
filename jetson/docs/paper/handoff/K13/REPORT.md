**Result (post-hoc, ego mask):** most pre-K10 AV2 conclusions hold under the ego mask. Five do not:
1. **(a) Refiner.** The refiner now carries most of ours' AV2 G: 0.031 without it, 0.122 with it.
2. **(c) Fine-tuned detectors.** Every fine-tuned detector becomes NEGATIVE: ours-ft + ref −0.100, u960-ft −0.473, ZoomDet −0.384, ESOD −0.736. Their unmasked G (0.30–0.55) came almost entirely from not detecting the hood, which the COCO cheap reference does.
3. **(f) Ours without refiner vs heuristics.** Without the refiner, ours is no longer above the crop heuristics at equal crop area (ours − uncertainty [−0.108, +0.002]).
4. **(b, d) field_only.** field_only, which was crippled by the hood, now recovers 75% of deleted objects (was 24%).
5. **(g) Dev choice.** On masked AV2 train, v4 + ref v2 beats the reported v4-lite + ref v2. As a variant on masked AV2 val it reaches 0.150 vs 0.122 (+0.028 [+0.007, +0.048]), at 1.4× the crop pixels.

The Argoverse-HD ablation (no ego part visible) shows the same pattern as masked AV2.

# K13: AV2 analyses redone with the ego mask (2026-10-07)

## Setup
- **Mask.** AV2 rows ≥ 1830, containment ≥ 0.5. It is applied to every pipeline exactly as in K10: `k10_masked_run.py` (md5 4abc786b…) wraps `rap.mono.predicted_geometry` in the shadow tree `$B/k10_masked`.
- **Configurations.** Frozen; nothing was tuned.
- **Data.** AV2 val: 150 logs, 11,818 frames.
- **Statistics.** Log bootstrap, 2,000 resamples.
- **Labels.** All numbers are **post-hoc, ego mask**.
- **Scripts (md5 prefixes).** Unchanged scripts: `instr_eval.py` 2177d413, `instr_analyze.py` 73cacf72, `k12_locality.py` 841da74f, `k9_tune.py` 6cec0743, `av2_ft_eval.py`. New for K13: `k13_ahd_ablation.py` d120c0c5, `k13_ft_hood.py` dc11895d, `k13_px.py` c35fc05f, `k13_k9pairs.py` 41a892bd. Chain: `k13_chain.sh`.
- **Parity.** The masked instrumented run reproduces K10: ours 0.122 with refiner, 0.031 without. The Argoverse-HD ablation's "full" row reproduces K10 / K12: G 0.091 / 0.039, tensor px 0.070. The unmasked K9 pair reproduces +0.012 [−0.042, +0.065].

## Per item

**(a) Ablation** (`a_ablation_av2.csv`, `a_ablation_ahd.csv`). G with refiner. CI is full − variant.

| Variant | AV2 masked | CI | AV2 unmasked CI | Argoverse-HD | CI |
|---|---|---|---|---|---|
| full | **0.122** (0.031 without refiner) | | | **0.091** (0.039) | |
| field only | −0.035 | [+0.115, +0.204] | [+0.513, +0.614] | −0.066 | [+0.100, +0.224] |
| support only | 0.122 | [−0.050, +0.055] | [−0.012, +0.026] | 0.127 | [−0.159, +0.106] |
| pivots | 0.099 | [+0.011, +0.037] | [+0.036, +0.055] | 0.087 | [−0.013, +0.024] |
| one round | 0.121 | [−0.009, +0.012] | [−0.014, +0.023] | 0.106 | [−0.028, −0.002] |
| no guard | 0.097 | [+0.006, +0.045] | [+0.002, +0.017] | 0.043 | [+0.016, +0.084] |
| draft history | 0.067 | [+0.011, +0.103] | [+0.009, +0.042] | 0.057 | [−0.135, +0.239] |

- **Holds:** field alone is not enough; support carries G; guard and merged-scene history are needed (guard also on Argoverse-HD).
- **Changed:** on Argoverse-HD a single round is slightly better than three.
- **Changed:** the refiner's share of G (masked AV2 0.031 → 0.122; unmasked 0.537 → 0.561). Harmed share among affected frames rises to 0.43 (was 0.09).

**(b) Agreement with π(u960)** (`b_agreement_av2.csv`, `b_confusion_*`, `b_disagreement_av2.csv`, `b_locality_av2.csv`)
- **Agreement (no refiner / with refiner):** ours 0.861 / 0.842 (unmasked 0.810 / 0.790).
- **Best alternatives:** u640 0.845, temporal_crit 0.841, uncertainty 0.835, cheap 0.795.
- **Holds in rank, smaller margin:** ours still agrees most, but the gap to the best heuristic shrinks from 7 to 2 points and the CIs overlap.
- **Disagreement split, holds:** when ours brakes less, F carries the loss (2,429 vs ours 578). When ours brakes more, ours does (1,731 vs 366).
- **Locality, holds** (first 50 logs; real crops → exact locality → + F's history):
  - v_close 0: 0.854 → 0.932 → 0.952 (unmasked 0.802 → 0.923 → 0.945);
  - v_close 10: 0.859 → 0.976 → 0.985.

**(c) Fine-tuned detectors** (`c_ft_detectors_av2.csv`, `c_ft_hood.csv`)
- **Hood detection:** u960-ft, ESOD and ZoomDet put **no box in the hood band** (0 per frame). COCO u960 does on 16.8% of frames.
- **G masked (unmasked):**

| Detector | G masked | G unmasked |
|---|---|---|
| ours-ft + ref | −0.100 | 0.546 |
| ours-ft | −0.150 | 0.527 |
| u960-ft | −0.473 | 0.402 |
| ZoomDet | −0.384 | 0.437 |
| ESOD | −0.736 | 0.295 |
| COCO ours | **0.122** | 0.561 |

- **Holds:** the ranking inside the fine-tuned table. ours-ft + ref is above u960-ft [+0.286, +0.486], ZoomDet and ESOD.
- **Does NOT hold:**
  - the fine-tuned rows' positive G;
  - "ours-ft + ref ≈ COCO ours": COCO ours is now significantly higher, [+0.157, +0.294].

**(d) Missed objects and TTC consumer** (`d_missed_*.csv`, `d_k9_*.csv`)
- **Natural misses, holds:** recovery ours 0.838 (unmasked 0.765), temporal_crit 0.780, uncertainty 0.462, random 0.245.
- **Controlled deletion, del_all, holds:** ours recovers 0.870 (0.789) at px 0.085; temporal_crit 0.844 at px 0.154; uncertainty 0.476. del_half is similar.
- **Changed:** field_only recovery 0.754 (was 0.240). Unmasked, the hood exhausted the field.
- **K9 TTC consumer, numbers nearly unchanged:** ours 0.120 (0.139); ours − uncertainty at equal px (1.5×) **+0.006 [−0.048, +0.061]** (was +0.012 [−0.042, +0.065]). The static hood generates no TTC warnings.

**(e) Jetson** (`e_jetson_av2_masked.csv`, 10 logs, 586 frames)

| Pipeline | ms | G masked |
|---|---|---|
| **ours** | 37.8 | **0.173** |
| u1536 | 42.1 | 0.147 |
| u1280 | 32.9 | 0.131 |
| native | 66.0 | 0.125 |
| u960 | 23.5 | 0.083 |

- Ours has the highest G and is faster than u1536 and native.
- Every CI of ours − whole-frame includes 0.
- **Holds:** ours is on the latency–G front. Under the mask it is the top point.

**(f) Detector pixels** (`f_px_av2.csv`)
- **K10's 0.049 is AREA-based:** cheap canvas + crop area × scale².
- **Tensor px under the mask:** ours 0.0532 (p95 0.128); unmasked 0.0740.
- **Heuristics at ours' masked crop area:** uncertainty 0.081, temporal_crit 0.084, random 0.092 tensor px, i.e. **52–72% more than ours. Holds.**
- **G without refiner:** ours 0.031, uncertainty 0.084, temporal_crit 0.053, random 0.002.
- **Does NOT hold:** ours (no refiner) − uncertainty [−0.108, +0.002], − temporal_crit [−0.062, +0.014]. Unmasked, ours was significantly above all three.
- With the refiner, ours (0.122) stays significantly above uncertainty crops (K10: [+0.007, +0.124]).

**(g) Dev check on AV2 train, masked** (`g_dev_check_av2.csv`, `remote/A_method/k13_g.py`; train = 700 logs, 55,246 frames; px are area-based, as in K10)

| Candidate | train masked G (px) | − reported | train unmasked | val masked G (px) | − reported |
|---|---|---|---|---|---|
| v4-lite + ref v2 (reported) | 0.115 (0.051) | | 0.516 | 0.122 (0.049) | |
| v4 + ref v2 | **0.134** (0.071) | [+0.006, +0.032] | 0.510 | **0.150** (0.067) | **[+0.007, +0.048]** |
| v3 + ref v1 | 0.073 (0.071) | [−0.059, −0.027] | 0.501 | 0.077 (0.068) | [−0.080, −0.016] |

- **Changed:** with the mask, **v4 + ref v2 wins on AV2 train**. It is v4-lite with crops at 1280 instead of 960 per long side. Unmasked, v4-lite had won (v4 − v4-lite [−0.016, +0.003]).
- **Variant, reported once:** v4 + ref v2 on AV2 val, masked = **0.150**, vs 0.122 for the reported configuration: +0.028 [+0.007, +0.048], at 1.4× the crop pixels (0.067 vs 0.049).
  - This number comes from the single masked AV2 val pass that produced K10's 0.122, which evaluates all variants in one pass. No new test run was made.
- **Recommendation:** keep v4-lite + ref v2 as the frozen configuration. Report v4 + ref v2 as the mask-selected variant, with its higher pixel cost.
- v3 + ref v1 (the nuScenes configuration) stays clearly below on both splits.
