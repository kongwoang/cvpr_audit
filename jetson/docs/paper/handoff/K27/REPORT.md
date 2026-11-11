# K27: temporal (looming) cue of crop objects (DEVELOPMENT data only)

**Result.**
- **Arm T4 passes the pre-registered rule** (method_freeze_draft §2g = §2e rule). T4 is robust looming: growth from 3 same-scale observations.
- **v5.1 is frozen pending the confirmatory run** (§2h). K24 is not started.

| v3a, dev | v5 | **v5.1-S** (T4+T2_1) | − v5 | **v5.1-L** (T3+T4+T2_1) | − v5 | px vs v5 (S / L) |
|---|---|---|---|---|---|---|
| nuScenes new val (152 scenes) | 0.276 | **0.302** | +0.026 [+0.011, +0.052] | 0.305 | +0.029 [+0.013, +0.059] | 1.00× / 1.26× |
| AV2 CFG-DEV (300 logs, mask) | 0.204 | **0.272** | +0.068 [+0.038, +0.096] | 0.271 | +0.067 [+0.044, +0.090] | 1.00× / 1.36× |

- **The gate adds nothing to T4.** T4 alone is within 0.0002 of T4+T2_1; the rule picks by mean G.
- **L does not add over S:** L − S is +0.004 [−0.006, +0.014] (nuScenes) and −0.000 [−0.014, +0.014] (AV2).
- **Halves:** v5.1-S is above v5 on both halves of each dev set (`halves.csv`).
- **nuScenes geometry decision:** the shared v_close 0 beats v_close 10 with the same arm, +0.007 [+0.001, +0.018]. Keep one shared configuration.

**(0) Diagnosis on v5** (`diag_*.csv` section A, `k27_diag.pdf`). Action-raising crop objects, on frames where only ours brakes (ours' action above u960's):

| | AV2 | nuScenes |
|---|---|---|
| frames per 1,000 | 49.8 | 14.7 |
| caused by the ttc | 88% | 78% |
| annotated object does not require the action | 95% | 77% |

- **Pairing source** of these objects:
  - AV2: a same-scale crop box 65%, a guard crop 7%, a DRAFT box 26%, none 2%;
  - nuScenes: 54% / 29% / 10% / 7%.
  - So mixed-fidelity pairing exists, but it is not the main cause.
- **Main cause: jitter of the crop box between frames.**
  - The crop canvas moves and is re-rounded every frame.
  - With crop–crop pairing, the box is 27.7 px high at the detector scale and changes by a median of 2.2 px (+8%): AV2 ttc 2.5 s, annotated about 9 s.
  - u960's box of the same object, at the same detector scale, changes by 0.5 px (+2%): ttc about 15 s.
  - nuScenes: 4.6 px vs 1.0 px.
  - When u960 also brakes, the two boxes jitter alike (AV2 1.1 vs 1.25 px).
  - The draft-paired cases have a median Δh of 33 px between scales. T1 removes them.

**Arms alone** (`temporal_*.csv`, `k27_arms.pdf`; v3a; vs v5):

| arm | nuScenes | AV2 |
|---|---|---|
| T1 same-scale history | +0.010 [+0.004, +0.021] | +0.031 [+0.011, +0.051] |
| T2_1 / T2_1.5 / T2_2 | +0.010 / +0.010 / +0.011 | +0.033 / +0.039 / +0.047, all > 0 |
| T3 retrospective re-detection | +0.005 n.s. (1.15× px) | +0.022 [+0.008, +0.037] (1.19× px) |
| **T4 robust looming** | **+0.026 [+0.011, +0.052]** | **+0.067 [+0.038, +0.096]** |

- Combinations with T2 / T3 add nothing significant over T4.
- **Events** (AV2, per 1,000 frames):
  - unnecessary 141 → 100;
  - missed 25.9 → 29.3.
  - nuScenes: 29.3 → 18.8 and 35.1 → 37.1.
- **Field-raise frames** (field-crop object raises the action above support only), per 1,000: AV2 25.2 → 18.4; nuScenes 13.7 → 9.6.
- **Cost on the K17(3) regime** (braking required, draft has no supporting object): AV2 −0.054 [−0.138, +0.010]; nuScenes −0.006 [−0.024, +0.000].
  - Not significant, but negative: T4 withholds the ttc of a newly found object for 2 frames.
  - v5.1-L recovers most of it on AV2: −0.016 [−0.072, +0.030].

**Fair controls** (`controls_*.csv`; same dev frames, v3a). The same fix helps every pipeline:
- **AV2:**
  - u960 + T4: +0.083;
  - uncertainty + T4: +0.049;
  - corridor + T4: +0.059.
- **nuScenes:** u960 + T4: +0.022.
- **v5.1-S vs each control with T4** (AV2):
  - u960 + T4 (4.2× px): +0.030 [+0.010, +0.053];
  - uncertainty + T4 (1.6× px): +0.049 [+0.025, +0.076];
  - corridor + T4 (1.27× px): +0.067 [+0.035, +0.114].
- **nuScenes:** vs u960 + T4, +0.073 [−0.026, +0.206], n.s. at 1/4.8 px; vs the heuristics + T4, > 0.
- So most of the AV2 gain is a better temporal cue that any pipeline can use. The part specific to ours is the jitter of the moving crop canvas. The remaining margin over u960 + T4 is +0.03 on AV2.

**Diagnostic only: changed reference** (`diag_*.csv` section B). The gate T2_k applied to every pipeline, including the draft, lowers G for every pipeline (AV2, v3a, k = 1):

| | u640 | u960 | native | ours |
|---|---|---|---|---|
| ungated | 0.142 | 0.158 | 0.101 | 0.204 |
| gated | 0.107 | 0.118 | 0.055 | 0.198 |

- The order native < u640 ≤ u960 does not change. "Higher resolution degrades the decisions" on AV2 is therefore NOT explained by a simple pixel gate on small boxes.
- The single-pair looming cue is noisy at every resolution: T4 helps u960 by +0.083.

**Caveats.**
- These are dev results, and v5.1 was selected on them; the test effect may shrink.
- The motivation came from a test-split diagnostic (K22(3)), which is stated as such.
- nuScenes' cached draft ttc matches a recomputation from consecutive cached boxes on only 73% of frames, presumably because the cache's prev set differs. This affects only the gated-draft diagnostic, never an arm.
