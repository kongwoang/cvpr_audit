**Result** (post-hoc analysis; existing per-frame results; no new inference; AV2 with the ego mask)
- **Where ours' decision value comes from.** It comes from fewer unnecessary decelerations / hard brakes and fewer delayed hard brakes, not from fewer missed decelerations. Against whole-frame u960 / native, ours MISSES MORE decelerations on all three datasets; these rows are reported as they are.
- **Missed decelerations per 1,000 frames, ours minus u960:**
  - nuScenes [+3.3, +8.5];
  - AV2 [+0.5, +4.0];
  - Argoverse-HD [+0.9, +4.7].
- **Collision term.** Rare:
  - nuScenes 8 frames in 3 scenes (cheap 11, u960 17, native 7);
  - AV2 5 frames in 3 logs (cheap 4, native 7);
  - Argoverse-HD 0 for every pipeline.
- **Per scene / log.** Ours has a lower decision loss than u960 in 82 vs 52 nuScenes scenes (sign test p = 0.012) and 97 vs 45 AV2 logs (p < 0.001); vs native: 73 vs 64 (p = 0.49) and 93 vs 50 (p < 0.001). On Argoverse-HD (24 sequences): vs cheap 12 vs 11 (p = 1.0); vs u960 / native 17 vs 6 (p = 0.035).

# K19: safety-relevant quantities

**Data and terms**
- **Frames.** Those of K14: nuScenes new test 9,181; AV2 val 11,818; Argoverse-HD val 15,006 (frames with 3D labels).
- **Loss terms** are recomputed with the consumer's cost (`rap.planner.decision_cost`):
  - insufficient braking λ_risk·shortfall²;
  - excessive braking λ_brake·excess²;
  - command changes λ_jerk·|Δcommand|;
  - collisions λ_coll·1[shortfall ≥ collision_a].

  Their sum reproduces the tables' J exactly (check = 0).
- **Previous action** = the draft's (as every table).
- **Required action a\*** = the consumer on the annotated scene.

**Sources of the actions**
- K14 (most pipelines);
- instr_eval / K13 (u640, support only, field only, with refiner as in the ablation);
- k10_hood (u640 / u1280 on Argoverse-HD, u1280 on nuScenes);
- AV2 u1280 from the K10 masked table. The action is recovered from its cost given a\* and the previous action; it is unique on 11,818/11,818 frames.
- Routers at 1× ours' tensor px (k18_routers): best Budget-Adaptive Routing picked on val (cond_moric on both datasets) and the oracle. Routers are not available on Argoverse-HD.

**Statistics.** Scene / log bootstrap, 2,000 resamples, of ours minus each pipeline. Scripts: `remote/A_method/k19_gt.py`, `k19_safety.py`.

**(1) Loss decomposition.** Share of the cheap draft's total loss (`safety_<ds>.csv`, columns `*_share_of_cheap`):

| | nuScenes cheap / u960 / native / ours | AV2 cheap / u960 / native / ours | A-HD cheap / u960 / native / ours |
|---|---|---|---|
| insufficient braking | 0.78 / 0.76 / 0.68 / **0.65** | 0.44 / 0.37 / 0.42 / **0.36** | 0.19 / 0.07 / 0.07 / **0.09** |
| excessive braking | 0.19 / 0.20 / 0.18 / **0.15** | 0.53 / 0.62 / 0.61 / **0.49** | 0.79 / 0.94 / 1.00 / **0.80** |
| command changes | 0.014 / 0.015 / 0.015 / 0.013 | 0.020 / 0.023 / 0.025 / 0.021 | 0.015 / 0.022 / 0.025 / 0.020 |
| collisions | 0.013 / 0.021 / 0.009 / 0.010 | 0.003 / 0.003 / 0.005 / 0.003 | 0 / 0 / 0 / 0 |

**(2) Event rates per 1,000 frames for ours** (all pipelines and every interval in `safety_<ds>.csv`; figure `event_rates.pdf/png`):

| | missed hard brake | delayed hard brake | missed decel. | unnecessary decel. | unnecessary hard brake |
|---|---|---|---|---|---|
| nuScenes | 3.2 | 3.2 | 28.5 | 17.0 | 14.2 |
| AV2 masked | 1.6 | 2.8 | 18.9 | 69.6 | 61.2 |
| Argoverse-HD | 0.3 | 0.6 | 15.5 | 49.0 | 151.8 |

- **Ours vs u960, nuScenes:**
  - delayed hard brake [−10.5, −0.8];
  - unnecessary decel. [−19.4, −10.2];
  - unnecessary hard brake [−6.3, −1.4];
  - missed decel. **[+3.3, +8.5]** (worse);
  - missed hard brake n.s.
- **Ours vs u960, AV2:**
  - unnecessary decel. [−36.6, −20.3];
  - unnecessary hard brake [−26.5, −12.9];
  - missed decel. **[+0.5, +4.0]** (worse).
- **Ours vs u960, Argoverse-HD:**
  - unnecessary decel. [−45.1, −4.9];
  - unnecessary hard brake [−46.6, −18.4];
  - missed decel. **[+0.9, +4.7]** (worse).
- **Ours vs crop heuristics.**
  - **Uncertainty crops:** they have fewer unnecessary hard brakes than ours on all three datasets (ours − it [+0.7, +5.2] / [+0.3, +9.8] / [+4.1, +33.5]); ours misses fewer decelerations (AV2 [−11.3, −3.7]).
  - **Support only + refiner:** fewer unnecessary events than ours, but many more missed decelerations.
  - **Oracle router:** below ours on most events.

**(3) Collision term** (`safety_<ds>_collision.csv`).
- **Counts.** Frames and scenes with a non-zero collision term are listed above.
- **What the consumer exposes:**
  - per-object required deceleration (from range, closing speed and ttc);
  - the shortfall of the commanded deceleration (9 m/s² on nearly every collision frame);
  - the annotated minimum range / ttc in the corridor.
- **Range and ttc on collision frames.** The minimum annotated range in the corridor has a median of 15 m. The annotated ttc is mostly infinite or large: a collision-term frame is a stopping-distance shortfall to a slow or static object, not a short-ttc event.
- **No clearance statistic.** The consumer exposes no clearance / time-to-collision statistic of its own beyond these.

**(4) Multi-obstacle frames** (≥ 2 annotated objects in the ego corridor within range; nuScenes 1,572, AV2 5,201, Argoverse-HD 5,162). G restricted to them:
- nuScenes: ours 0.002 (u1280 0.179, native 0.053, oracle 0.236, u960 −0.191);
- AV2: ours **0.103** (best realizable; oracle 0.299, support only 0.089, uncertainty 0.080);
- Argoverse-HD: ours −0.005 (support only 0.078, uncertainty 0.040, u960 −0.064).

Event rates on these frames are in the `multi …` columns.

**(5) Per scene / log** (`safety_<ds>_scenes.csv`): counts and sign tests in the first lines.
