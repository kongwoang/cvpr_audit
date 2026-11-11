# Speculative perception — method freeze and confirmatory pre-registration

**FINAL for nuScenes as of 2026-09-27 17:17 NZT: method v3 (§2b).** `remote/confirm_chain.sh` runs it once on the new
test scenes after all scenes are detected, logging md5 of the code and refiner first (`logs/confirm_chain.log` on VUW).
KITTI and nuPlan-mini numbers remain development evidence.

*Drafted 2026-09-27 08:10. Becomes binding when the refiner (§2.4) is frozen and this header is changed to FINAL with a
date. Nothing below may be changed after that without a dated amendment.*

## 1. Why a freeze

About sixty exploratory analyses (directions_log D1–D76) used the KITTI and nuScenes test splits of DEEP. Every number
there is development evidence. The paper's headline numbers must come from one run of a frozen method on data that
the exploration did not touch.

## 2. The frozen method

1. **Draft.** YOLOv8s at 96×320 (DEEP's cheap mode), threshold 0.25, DEEP's ego-frame monocular lift.
2. **Decision-sensitivity field.** Closed form for max-composable consumers (`vtf/dsf.py`, verified equal to probing
   on all 8,008 KITTI frames): ground grid r ∈ {12.5, 17.5, 22.5, 27.5, 35, 45, 55, 70} m, y ∈ {−6, −4.5, …, 6} m;
   probe object 1.8 m × 1.6 m, static; discovery crop = bounding box of the projected sensitive cells.
   Non-max-composable consumers: vectorised probing of the same grid.
3. **Pivots.** Objects whose removal changes the draft decision (closed form for max-composable consumers); crop =
   2× the pivot box.
4. **Refiner.** One image-level range refiner (crop + box features), trained on training units only, epoch chosen on
   validation units; its architecture and recipe are fixed by B1 before this document becomes FINAL.
5. **Verification detector.** YOLOv8s on each crop at the 960-px scale (0.773 × KITTI width; nuScenes 0.6), threshold
   0.25; detections whose centre lies in a crop replace the draft's there.
6. **One round** (no second speculative round), no pre-escalation gate (every frame runs the loop; the D62b gate is a
   secondary analysis).

## 2b. Amendment 2026-09-27 17:15 NZT — v3 (supersedes §2 items 2, 3, 5, 6; FINAL)

Development evidence on nuScenes old train+val (directions_log "nuScenes gap diagnosed") changed the method before any
confirmatory run. Frozen candidate **v3**:

1. Draft: unchanged.
2. Field: closed form, grid as §2.2, probe 1.8 m × 1.6 m **closing at v_close = 10 m/s**.
3. **Support, not pivots**: crops around every object whose demand alone reaches the current action's threshold
   (2× box); **the loop iterates** (≤ 3 rounds) until no new crop is needed (docs/paper_theory.md).
4. Refiner: ResNet-18 image refiner (`out/resnet18/image_refiner.pt`, trained on KITTI + nuScenes train units of DEEP's
   split, epoch on validation units), applied to support objects detected in crops.
5. Verification detector: YOLOv8s per crop at scale 0.6 of the native image on nuScenes (chosen on dev: 0.6 > 0.4);
   KITTI deployed: the 96×288 TensorRT canvas. **Truncation guard (all truncated boxes)**: a crop detection touching
   an interior crop edge is re-detected in a crop grown around it (≤ 2 growths). (The corridor-filtered guard was
   worse on nuScenes dev, 0.154 vs 0.194, reason open; not used.)
6. **ttc history** of crop detections = the previous frame's merged scene (not the previous cheap boxes).
7. No pre-escalation gate.

Dev numbers of v3 (not confirmatory): nuScenes old train+val spec 0.194 / + refiner 0.195 vs uniform 640 0.181,
uniform 960 0.183 at 11.8% pixels; KITTI test (Jetson, TRT) spec 0.213 / + refiner 0.240 vs 0.123 / 0.129.

## 3. Confirmatory data (not used in the exploration)

- **nuScenes**: scenes outside the 85-scene blob used so far (DEEP's scene split rule applied to the new scenes), CAM_FRONT,
  same consumers; requires downloading further trainval blobs.
- **nuPlan** with PDM-Closed and IDM, front camera, real detections (DEEP's Task 5 pipeline), logs not in DEEP's split,
  plus the benchmark's own planning scores.
- KITTI has no unused labelled sequences; KITTI results stay development evidence.

## 4. Primary endpoints

- **E1** decision value (share of the all-cheap loss) of speculative v1 vs uniform escalation to 640 and to 960, paired
  unit bootstrap (2,000 draws), per cell.
- **E2** end-to-end latency and energy per frame on the Jetson (TensorRT FP16, same process, image decode excluded)
  vs uniform 640, uniform 960 and DEEP's cascade.
- **E3** on nuPlan: the benchmark's planning score of PDM-Closed fed by speculative v1 vs uniform 640/960 perception.

Secondary: whole-image recall/precision (reported as the trade-off), decision agreement with the full mode, harmed
share of affected inputs, crops per frame.

## 5. Decision rule

- **Claim "matches uniform escalation"** in a cell if the E1 interval of (speculative − uniform 960) lies above −0.02 of
  the all-cheap loss; **"beats"** if it lies above 0.
- **Claim "cheaper"** only if E2's median and p95 are below uniform 960's.
- The paper is written around E1–E3 as they come out, including cells where the method loses.

## 2c. Pre-registration of CONFIRMATORY 2 — Argoverse 2 sensor val (written 2026-09-28 02:30 NZT, before any AV2 result)

- **Data:** the 150 logs of the Argoverse 2 sensor dataset *val* split, ring_front_center camera (1550×2048), never
  used in development. Frames: every 2nd annotation timestamp (5 Hz); GT and ego speed via `src/vtf/av2.py` (DEEP's
  nuScenes conventions: footprints, projected 2D boxes, range rate; LiDAR→camera motion compensation).
- **Method:** frozen v3 exactly as §2b (same code md5 as the nuScenes confirmatory run for `vtf/speculative.py` and
  `vtf/dsf.py`, same frozen refiner `out/resnet18/image_refiner.pt`, v_close 10, guard, merged-scene history, ≤ 3
  rounds). The only dataset adaptation, fixed now: detector input sizes by the long image side — cheap 320×256, uniform
  640×512, uniform 960×768 — and the crop scale = 960 / long side = 0.469 (on nuScenes this rule gives 0.6).
- **Endpoints and rule:** as §4–§5 (E1 decision value vs uniform 640 and uniform 960, braking, monocular lift, cheap
  action history, paired bootstrap over logs, 2,000 draws; "matches" if the lower bound of spec(+ref) − uniform 960 >
  −0.02, "beats" if > 0). One run; the result is reported whatever it is.
- The *tuned* method (after baselines) will need its own untouched set (AV2 sensor *train* logs, to be downloaded).

## 2d. Pre-registration of CONFIRMATORY 3 — tuned v4 on Argoverse 2 sensor TRAIN logs (written 2026-09-28, before any AV2-train result)

- **Why:** v4 was tuned on nuScenes new VAL scenes after both confirmatory runs; it needs data never used.
- **Data:** Argoverse 2 sensor *train* split logs (all fully downloaded logs at run time, up to 700), front-centre
  camera, same adapter and frame rule as §2c (5 Hz). None of these logs was used anywhere.
- **Methods (primary v4, secondary v4-lite), everything else as frozen v3:** v_close 0 (static probe), ≤ 3 rounds,
  truncation guard, merged-scene history, refiner v2 (`out/resnet18_v2/image_refiner.pt`); crop scale = 1280 / long
  side (0.625 on AV2) for v4 and 960 / long side (0.469) for v4-lite. Frozen v3 is re-run on the same logs as a third
  confirmatory replication of v3.
- **Endpoints/rule:** as §4–§5 (vs uniform 640 / 960 by the long-side rule; log bootstrap, 2,000). Also reported:
  v4 − v3. One run.

## §3 Per-benchmark configuration: NAVSIM (frozen 2026-10-01 on the DEV split = OpenScene mini, before navtest)
Consumer = PDM-Closed (the planner NAVSIM scores), so the method probes PDM-Closed itself (same algorithm, planner
swapped): on the cheap draft scene, an unseen static vehicle probe (class-median size, ego heading) at each cell of a
7 x 7 grid (forward x = 8, 12, 17, 23, 30, 40, 55 m; lateral y = −10.5 … +10.5 m in 3.5 m lanes) whose box projects
into CAM_F0; a cell is sensitive if PDM-Closed's 4 s plan moves by > 0.5 m. Support = draft objects inside CAM_F0 whose
removal moves the plan by > 0.5 m. Crops: one per horizontal image column of sensitive cells (union of their boxes,
160-px bins) + one per supporting object (its cheap detection box, v3's box_crop); v3's truncation guard and merge
rule; crop scale 960 / long side = 0.5; one round. No ground truth enters the field or the crops.
Selection evidence (dev, 6,300 tokens): PDMS spec_pdm 0.9475 vs full 0.9448, u960 0.9447, v3 (braking field) 0.9444,
cheap 0.9415; spec_pdm − full [+0.0012, +0.0044], − u960 [+0.0013, +0.0046], − v3 [+0.0005, +0.0066].
Rejected on dev: merge-rule variants (keep conf ≥ 0.25 / 0.5) and crop scales 0.6 / 0.75 / 1.0 (all within ±0.0004).
Navtest outcome (run once, 2026-10-01; no NAVSIM retuning after it): spec_pdm 0.8945 ≈ u960 0.8949 / full 0.8944 /
FOVEA 0.8945 / SAHI 0.8947 (all CIs contain 0), > v3 0.8926 ([+0.0002, +0.0040]). The dev gain did not transfer.

## §3 Per-benchmark configuration: AV2 (fixed 2026-10-01 on the DEV split = AV2 train, confirmatory 3; before the val run)
v4-lite + refiner v2: braking field with v_close 0, ≤ 3 rounds, crops at 960 / long side, truncation guard, merged-scene
history, ResNet-18 refiner v2. Evidence (AV2 train, 700 logs): G 0.516 vs v3 + ref 0.501, u960 0.398; cheapest (px 0.045).
Run once on AV2 val (av2_eval_v4.py --split val --tag _v4_final; v3 / v4 also computed in the same pass for reference).

## §3 Per-benchmark configuration: nuScenes
Pre-registered v3 + refiner stays the reported configuration; the new-val-tuned v4 (v_close 0, scale 0.8, refiner v2)
did not transfer to new test (0.147 vs 0.171) and is reported as the tuned variant.

## §3 Per-benchmark configuration: KITTI (fixed 2026-10-01 on the DEV split = DEEP KITTI train + val)
Selection rule declared before results: highest mean G over the four DEEP KITTI cells (+ refiner); "matched"
(braking field on brake cells, Planner B's field on traj cells) evaluated as one candidate; then one KITTI test run.
Selected: union of the braking field (v_close 0) and Planner B's own field + pivots per round (vtf.spec_general,
Brake ∪ PlannerB), ≤ 3 rounds, truncation guard, crop canvas 96x288, refiner on the union's support objects.
Dev mean4 0.795 (v3 + ref 0.792, u960 0.655, SAHI 0.635).
Test outcome (run once): mean4 0.401 vs v3 + ref 0.389, SAHI 0.327, u960 0.231 (− u960 [+0.091, +0.229], − SAHI [−0.003, +0.160]).

## 2e. Amendment 2026-10-10 (K23) — selection of a v5 configuration on DEVELOPMENT data only (written before any K23 run)

**Scope.** K23 runs nothing on nuScenes new test, AV2 val or Argoverse-HD val. No earlier test-split result is used to
choose between candidates (the components below were motivated by test-split diagnostics K14 / K17 / K18 / K22, which
is stated as such; the CHOICE among them uses development data only). K23 stops after freezing; the confirmatory run
is K24.

**Development data.**
- nuScenes: new val, 152 scenes (`results/nusc_full/split.csv`, set = new, split = val). Note: refiner v2's epoch
  selection used 151 of these scenes; the frozen nuScenes configuration (v3 + refiner v1) did not.
- AV2: the 700 sensor-train logs (`datasets/av2/lists/train/*.done`), ego mask of K10 (rows >= 1830, containment
  >= 0.5) on every pipeline. Split by log: sorted log names, permutation `numpy.random.default_rng(20261010)`;
  the first 300 = **CFG-DEV** (configuration selection; never seen by any refiner), the next 340 = **REF-TRAIN**
  (refiner v3 training), the last 60 = **REF-VAL** (refiner v3 epoch / model selection). The lists are written to
  `results/k23/av2_split.csv` before any run. No refiner (v1, v2, v3) has seen an AV2 log in training or selection.
- Argoverse-HD: no tuning; it receives the AV2 / shared configuration untouched.

**Reference.** The frozen configurations on the same frames: v3 + refiner v1 (nuScenes), v4-lite + refiner v2 (AV2).
Every dev number: G without / with refiner, tensor px mean / p95, paired unit bootstrap (2,000, seed 0) of
(candidate − frozen).

**Components** (each tested ALONE on top of the frozen configuration, then the combinations of those that helped):
- **C1 refiner v3.** v2's architecture and inputs (ResNet-18 on the 64x64 crop + the 13 box features) with up to three
  heads, trained on v2's training data (KITTI train, nuScenesAll train scenes; nuScenesAll's validation scenes are
  NOT used because they include nuScenes new val) PLUS AV2 REF-TRAIN boxes (crop detections of the frozen v4-lite
  loop and u960 detections, ego mask applied, matched to annotations with class-compatible IoU >= 0.3). Epoch /
  model selection on KITTI val + AV2 REF-VAL only. Variants: (a) log-range correction only; (b) + lateral-offset
  correction (target: annotated minus estimated lateral centre, AV2 boxes only); (c) + a validity score (probability
  that a matching annotated object lies inside the ego corridor; all AV2 detections, matched or not), used only to
  veto an object that would otherwise RAISE the action when the score is below a threshold chosen on dev from
  {0.1, 0.2, 0.3, 0.5}. v3 is also applied on dev to whole frame u960 and to the uncertainty heuristic.
- **C2 evidence threshold for discovered objects.** A crop detection with no draft counterpart (IoU < 0.5 with every
  draft box) that would raise the action is kept only if its confidence is >= c_new, c_new in {0.25 (current), 0.35,
  0.45, 0.55}; variant "persist": a DECELERATE raised only by such a detection needs a matching detection (IoU >= 0.3)
  in the previous frame's merged scene; HARD_BRAKE is never delayed. Applied to the loop's final scene.
- **C3 temporal support.** An object in the support of the previous frame's merged scene with no match (IoU >= 0.3)
  in the current draft is re-verified with one crop at its previous box (context 2), added to the first round.
- **C4 geometry** (car probe only): crop long side {960, 1280, 1536} px on AV2 and scale {0.6, 0.8, 1.0} on
  nuScenes; support context {2, 2.5, 3}; field margin {0, 0.15}; v_close {0, 5, 10} m/s.
- **C5 bounded loop** of K18(B): N in {2, 3, unbounded} detector calls with batched / deferred guard growth (G and
  tensor px only).

**Selection rule (fixed now).**
- Two operating points per dataset: S = tensor px <= 1.1 x the frozen configuration's dev px; L = <= 2 x.
- A component enters v5 only if, on dev, the interval of (with − without) is above zero on at least one of the two
  datasets and its point estimate is not negative on the other.
- Prefer ONE shared configuration for nuScenes and AV2 (same v_close, crop long side in pixels, context, margin,
  c_new, refiner v3); chosen by the mean of the two dev G values (with refiner); per-dataset settings only for a
  parameter whose shared value is significantly worse than the per-dataset value on dev.
- If nothing passes the rule on a dataset, the frozen configuration stays for that dataset.
- Stability: the 5 best candidates per operating point with intervals; G of the chosen configuration on two disjoint
  halves (by unit, fixed split) of each dev set.

**Code.** `remote/A_method/k23_*.py` (split, refiner data, refiner training, dev evaluation), loop switches in
`remote/A_method/k_loop.py`; md5s of code and weights are appended when v5 is frozen.

## 2f. v5 — FROZEN PENDING CONFIRMATORY RUN (2026-10-11; selected on development data only by the rule of §2e)

**Configuration (one shared configuration for nuScenes, AV2 and, untuned, Argoverse-HD; v5-S = v5-L).**
- Frozen loop (`vtf.speculative.run`, equivalently `k_loop.run_g` with defaults): cheap draft at 320 px long side.
  - Braking field: car probe 1.8 × 1.6 m, frozen grid, **v_close 0 m/s on every dataset** (was 10 on nuScenes).
  - Support crops: context 2, no field margin.
  - Loop: ≤ 3 rounds, truncation guard, merged-scene history.
  - Crops at **960 px per long side**: nuScenes scale 0.6, AV2 960 / 2048, Argoverse-HD 960 / 1920.
- **Refiner v3a** (`out/resnet18_v3a/image_refiner.pt`; range head only) on the support objects of the final scene that come from a crop. This is the rule of refiners v1 / v2. The features and crop are those of v2, and the camera flag is 1.0.
- Every other component tested was rejected (§2e rule):
  - v3b / v3c (lateral head; validity veto): worse on both datasets.
  - C2 c_new 0.35–0.55: helps AV2 [+0.008 … +0.056] but is negative on nuScenes. C2 persistence: negative on both.
  - C3 temporal support: n.s. on both.
  - C4 crop scale, context, margin: n.s. or negative alone. v_close 5 / 10 on AV2: negative.
  - C5 bounded loop: n.s. on both.
- No component passed at the L operating point (≤ 2 × px), so **v5-L = v5-S**.

**Development evidence** (paired unit bootstrap, 2,000, vs the frozen configuration on the same frames):

| dataset | frozen | v5 | v5 − frozen | tensor px |
|---|---|---|---|---|
| nuScenes new val (152 scenes) | v3 + ref v1: G 0.263 | 0.276 | [+0.003, +0.031] | 0.080 vs 0.104 (0.77×) |
| AV2 CFG-DEV (300 logs, ego mask) | v4-lite + ref v2: G 0.113 | 0.204 | [+0.061, +0.120] | 0.055 vs 0.055 (1.00×) |

- **Halves of each dev set** (v5 / frozen):
  - nuScenes: 0.311 / 0.305 and 0.209 / 0.179;
  - AV2: 0.237 / 0.112 and 0.176 / 0.114.
- **Baselines with the same refiner v3a on dev:**
  - u960: 0.207 (nuScenes), 0.158 (AV2);
  - uncertainty crops: −0.060, 0.178.

**md5.**

| file | md5 |
|---|---|
| refiner v3a weights | 6b29d05e8f06d121ec9e11aeb9dc2730 |
| models/yolov8s.pt | 0a1d5d0619d5a24848a267ec1e3a3118 |
| src/vtf/speculative.py | 651da631134f7ef40c1c674d0e377ea4 |
| remote/A_method/k_loop.py | 480cf13a34e260a74d29b07858727f7f |
| remote/A_method/k23_dev.py (refiner application) | 01b57e712c1536dfbbeef020223f21a8 |
| k23_train_refiner.py | 957b54a5d211c043a0288195b1e7b66c |
| k23_refdata.py | d5f4c331feb6204f7c4d50ef9e978815 |
| instr_eval.py | 2177d4131830cd9c56a262dd47b19d52 |
| results/k23/av2_split.csv | 39860f4d23a372c7d065338cba83fc40 |

**Confirmatory run (K24)** follows the criterion fixed for v3: "matches" a baseline if the lower bound of (v5 − baseline) > −0.02, "outperforms" it if > 0. Test splits: nuScenes new test, AV2 val with the ego mask, Argoverse-HD val; run once.

## 2g. Amendment 2026-10-11 (K27) — temporal (looming) cue of crop objects, DEVELOPMENT data only (written before any K27 run)

**Scope.** Same data, refiners and selection rule as §2e: nuScenes new val (152 scenes) and AV2 CFG-DEV (300 logs,
ego mask); Argoverse-HD untouched; no test split is run. The motivation comes from a test-split diagnostic (K22(3):
far field-crop objects whose lifted ttc is ~1–3 s while the annotated ttc is ~9 s or infinite), stated as such; the
CHOICE uses dev data only. K27 stops after freezing (or not); K24 is not started.

**Base.** v5 (§2f): nuScenes scale 0.6 / AV2 960 px long side, v_close 0, context 2, no margin, ≤ 3 rounds, guard,
refiner v3a. Reference also reported: the previous frozen configuration (nuScenes v_close 10 + refiner v1; AV2 the
same loop + refiner v2). Every arm is scored without refiner, with the previous refiner and with v3a; selection uses v3a.

**What the base does now** (stated from the code before running): the lift of the crop detections pairs each box with
the previous frame's MERGED scene of the same pipeline (draft boxes at 320 px + crop / guard boxes) by greedy IoU
>= 0.3; ttc = dt / (h_t / h_{t-1} − 1) on full-image box heights (dt 0.5 s nuScenes, 0.2 s AV2). Draft boxes pair
with the previous draft.

**Arms** (they change only the ttc given to crop detections, i.e. the lift's history; the draft is unchanged):
- **T1 same-geometry history:** the previous-frame set is the previous final scene's crop and guard-crop detections
  only (same detector scale); a crop box without such a match gets ttc = inf (range term only).
- **T2_k noise gate (k in {1, 1.5, 2}):** T1, and ttc is kept only if (h_t − h_{t−1}) x crop scale > k pixels.
- **T3 retrospective verification:** T1; a crop box with no same-scale match at t−1 is paired with the detections of
  the SAME crop region (same canvas) re-detected in the buffered frame t−1 (batched with the round: no extra round;
  extra tensor px and extra crop detections are counted and reported).
- **T4 robust looming:** growth from the last 3 consecutive same-scale observations (t−2, t−1, t, chained by greedy
  IoU >= 0.3): median of the three pairwise rates (h_b / h_a − 1) / ((b − a) dt); ttc = 1 / rate (inf if rate <= 0);
  no ttc with fewer than 3 observations.
- **Combinations** (computed in the same run; considered ONLY if every component passes alone): T3+T2_k, T4+T2_k
  (gate on h_t − h_{t−2}), T3+T4 (T3 re-detects in t−1 and t−2 when the chain is missing), T3+T4+T2_k.
- **nuScenes geometry decision only:** every arm also with v_close 10 (the pre-registered nuScenes geometry).

**Controls (same dev frames):** u960 with T2_k and T4 (T1 is its own history already); uncertainty crops and the
ego-corridor tube with guard, history and the refiners, per-frame canvas tensor px <= v5's crop tensor px on that
frame (K18 fill), with base / T1 / T2_k / T4.
**Diagnostic only (changed reference):** the gate T2_k applied in the consumer to EVERY pipeline including the cheap
draft (k in {1, 1.5, 2}, at each pipeline's detector scale); G of u640 / u960 / native / ours relative to the gated
draft.

**Per arm:** G (3 refiner settings), paired unit bootstrap (2,000, seed 0) of (arm − v5) at the same refiner,
tensor px; frames per 1,000 where a field-crop object raises the action above support only (v3a), with the median
estimated and annotated ttc of that object; G on the K17(3) regime (a* >= DECELERATE, a_cheap < a*); unnecessary /
missed events (action above / below a*) per 1,000 frames. Diagnosis (0) on the base: pairing source of the
action-raising crop objects (draft / crop / guard / none), box height at the detector scale, relative change, and the
same for u960's box of the object.

**Selection rule (= §2e).** Operating points S (px <= 1.1 x v5's dev px) and L (<= 2 x). An arm enters only if the
interval of (arm − v5) with v3a is above zero on at least one dataset and its point estimate is not negative on the
other; one shared arm preferred, chosen by the mean of the two dev G; if one passes, "v5.1" = v5 + the arm is frozen
with code md5; otherwise v5 stays. Also reported for nuScenes only: v_close 10 + the passing arm.
**Code:** `remote/A_method/k27_dev.py`, `k27_analyze.py`.

## 2h. v5.1 — FROZEN PENDING CONFIRMATORY RUN (2026-10-11; selected on development data only by the rule of §2g / §2e)

**Change to v5:** only the ttc (looming cue) given to the crop detections. The loop, the crops, the draft and refiner v3a are unchanged.
- **v5.1-S = v5 + T4 + T2_1:**
  - history = the crop and guard-crop boxes of the previous two final scenes (same detector scale; never draft boxes);
  - growth = median of the three pairwise rates over t−2, t−1, t (chained by greedy IoU >= 0.3); no ttc with fewer than 3 observations;
  - the ttc is kept only if h_t − h_{t−2} > 1 px at the crop's detector scale.
  - T4 alone gives the same dev G within 0.0002 (nuScenes identical; AV2 +0.0002 [−0.0000, +0.0004]); the gate is kept because the rule picks by mean G.
- **v5.1-L = v5.1-S + T3:** a crop box missing from the same-scale history is paired with the same crop canvas re-detected in the buffered frames t−1 / t−2, batched with the round.
  - Extra crops per frame: 0.8 nuScenes, 1.2 AV2.
  - Tensor px: 1.26× v5 nuScenes, 1.36× AV2.
  - v5.1-L − v5.1-S is n.s. on both datasets: +0.004 [−0.006, +0.014] and −0.000 [−0.014, +0.014].

**Development evidence** (refiner v3a; paired unit bootstrap 2,000; vs v5 on the same frames):

| dataset | v5 | v5.1-S | v5.1-S − v5 | v5.1-L | v5.1-L − v5 |
|---|---|---|---|---|---|
| nuScenes new val | 0.276 | 0.302 | +0.026 [+0.011, +0.052] | 0.305 | +0.029 [+0.013, +0.059] |
| AV2 CFG-DEV | 0.204 | 0.272 | +0.068 [+0.038, +0.096] | 0.271 | +0.067 [+0.044, +0.090] |

- **Halves.** S is above v5 on both halves of each dev set:
  - nuScenes: 0.332 / 0.311 and 0.242 / 0.209;
  - AV2: 0.324 / 0.237 and 0.227 / 0.176.
- **Cost on the K17(3) regime** (braking required, draft has no supporting object): AV2 −0.054 [−0.138, +0.010]; nuScenes −0.006 [−0.024, +0.000]. Not significant, but negative.
- **Shared geometry kept.** nuScenes with v_close 10 + T4+T2_1 is 0.295, below the shared v_close 0: +0.007 [+0.001, +0.018] for v_close 0.

**md5:**
- remote/A_method/k27_dev.py (ttc_ss, the lift policy): 33e0b3c6b94460e6a31d1e16532c308e
- k27_analyze.py: 3c3d49d7476c75f945afc8f51fba3635
- k_loop.py: 480cf13a34e260a74d29b07858727f7f (unchanged)
- speculative.py: 651da631134f7ef40c1c674d0e377ea4 (unchanged)
- refiner v3a: 6b29d05e8f06d121ec9e11aeb9dc2730 (unchanged)

The K24 criterion is unchanged and now applies to v5.1, with v5 as an extra reference arm. The baselines are run with the same temporal policy where it applies: u960 + T4 and the heuristics + T4, because T4 also helps them on dev.

## 2i. Pre-registration of the CONFIRMATORY run K24 (written 2026-10-11, before any K24 result)

**Primary configuration: v5.1** (§2h; decided by the user before the run): v5.1-S and v5.1-L.
**Secondary reference:** v5 (§2f; v5-S = v5-L), run in the same pass on the same frames. v5 is not used to choose anything.
**Rules.** Each test split is run ONCE: nuScenes new test (228 scenes), AV2 val with the K10 ego mask (150 logs),
Argoverse-HD val (24 sequences; the shared configuration, untuned, at its own frame interval). Nothing is changed after
seeing a result; a crashed shard is re-run identically and reported.
**Criterion** (as for v3): a configuration "matches" a baseline if the lower bound of (configuration − baseline) is
above −0.02, and "outperforms" it if the lower bound is above 0. Statistic: G with refiner v3a (and without
refiner), paired unit bootstrap (2,000, seed 0) by scene / log / sequence.
**Baselines for v5.1** (same frames):
1. Whole frame u640 / u960 / u1280 / u1536 / native, without and with refiner v3a.
2. Crop heuristics: uncertainty, temporal_crit, ego corridor, traj_support, random.
   - Each with guard, history and refiner v3a (K18 protocol).
   - Per-frame canvas budgets b ∈ {0.5, 1, 2} × the crop tensor px of v5.1-S, and likewise of v5.1-L (L includes the retrospective px).
3. Routers and the oracle router (K18 protocol), with the mean tensor px re-matched within 2% to v5.1-S and to v5.1-L.
4. The previous frozen configuration with its own refiner and with v3a:
   - nuScenes: v3 + refiner v1;
   - AV2 / Argoverse-HD: v4-lite + refiner v2.

**Fair temporal controls** (labelled as such): the whole-frame pipelines and the heuristics, also with the robust looming
policy T4 + T2_1 of v5.1 (same-detector history). The "does NOT outperform" list in the report covers every baseline,
including these.
**Code:** `remote/A_method/k24_test.py` (lift policy shared with k27_dev.ttc_ss) and `k24_analyze.py`.

**§2i revision (2026-10-11 ~21:40 VUW, received and written while the main pass was running, before ANY K24 result
was looked at; the smoke units were checked for parity only).** The revised K24 spec ("baselines for v5.1") replaces the
lists above where they differ:
- **Primary: v5.1-S.** v5.1-L and v5 are references from the same pass.
- **"+ components"** means refiner v3a AND the temporal rule of v5.1-S, i.e. T4 with the T2_1 gate, exactly the lift of v5.1-S.
- **(1) Whole frame:** plain (without / with v3a), and + components.
- **(2) Heuristics:** at b ∈ {0.5, 1, 2} × v5.1-S crop tensor px, plain and + components. The v5.1-L-matched rows are kept as labelled extras.
- **(3) Routers and oracle:** re-matched to v5.1-S (v5.1-L as extra). Escalated frames run u960 plain, or u960 + components. The oracle takes the top-k frames by true V of u960 with v3a, and its single mask is used for both settings.
- **(4) Previous frozen configuration:**
  - "As reported": nuScenes v3 + refiner v1; AV2 / A-HD v4-lite + refiner v2.
  - "+ components": nuScenes v3 loop with the v5.1-S lift + v3a. On AV2 / A-HD the v4-lite loop is the v5 loop, so this row is v5.1-S by construction.
- **(5) Ablations:**
  - support only + components and field only + components;
  - the 2 × 2 of v5.1-S {with / without v3a} × {with / without T4} (without T4 = v5).
  - Rows (5) also report G on the K17(3) regime (act_gt ≥ 1 and draft action below it) and unnecessary / missed actions per 1,000 frames.
  - field only = support crops disabled (K13 "support_mode none"); support only = k_loop field=False.

The configurations not in the main pass are run ONCE in a supplementary pass on the same units and frames:
- nuScenes v3 + T4;
- support only and field only with the v5.1-S lift.

Code is `remote/A_method/k24_supp.py` with `k24_supp_chain.sh`. k_loop.py is unchanged: the field-only loop replaces k_loop's reference to SP.support for the duration of the call.

Tables (`k24_analyze.py`, revised): table_main_{ds}_v51.csv, ablation2x2_{ds}_v51.csv, pareto_{ds}_v51.csv,
headline_v51.csv, not_outperformed_v51.csv. Analysis code is fixed before any result. The not-outperformed list covers
every row of (1)–(5), including the extras.
md5 (deployed before the supplementary launch, 21:37 VUW): k24_supp.py cc4b714f11b231e374470548a7a380f8,
k24_analyze.py (revised) 3ee3290b6ab32e2b8aa9dbbc4b81d301, k24_supp_chain.sh 520517682f8db1caaf85cd3ff541144b;
k24_test.py d2ae37cc6dc31046f9d94fdb0d4bf9ab (main pass, unchanged). The main chain still calls k24_analyze.py at its
end; with the supplementary pass incomplete it writes no table ("NOT READY").

## 2j. Amendment 2026-10-11 (K28 round 1) — DEVELOPMENT data only (written before any K28 run)

**Status of v5.1-S.** K24 (the confirmatory run of §2i) is complete. v5.1-S is the final configuration unless K28 adopts something under the rules below.

**K28 hard rules (task text; they override the goal).**
- **R1:** selection uses development data only:
  - nuScenes new val, 152 scenes;
  - AV2 CFG-DEV, 300 logs, with the ego mask;
  - refiner data from REF-TRAIN / REF-VAL as in K23.

  Argoverse-HD is never used. No decision uses a test number, including K24's.
- **R2:** exactly one further confirmatory test pass, and only if something is adopted.
- **R3:** the rule below, written before each round; at most 3 rounds.
- **R4:** equal tuning effort for the baselines.
- **R5:** everything is reported.
- **R6:** a rule wins over the goal.

**Reference.** v5.1-S (§2h). K27 dev, with v3a:
- nuScenes: G 0.3019, tensor px 0.0803;
- AV2: G 0.2717, tensor px 0.0554.

It is re-run in the same pass. Parity with K27 is a check, not a selection input.

**Round-1 candidates (each alone; everything else = v5.1-S).**
- **A — operating points:**
  - A1 v_close 5; A2 v_close 10;
  - A3 / A4 / A5: v_close = α × ego speed with α = 0.25 / 0.5 / 1;
  - A6 retrospective re-detection on (T3 + T4 + T2_1 = v5.1-L);
  - A7 / A8: support context 2.5 / 3;
  - A9: crop long side 1280 (nuScenes scale 0.8; AV2 1280 / 2048);
  - A10 / A11: bounded loop with N = 2 / 3 detector rounds (k_loop.run_bounded, the guard's grown crops deferred to the next round).
- **B — stronger verifier on the crops only.** The draft stays YOLOv8s at 320. Candidates: B1 YOLOv8m, B2 YOLOv8l, B3 YOLO11m.
  - Each B candidate uses its own refiner, trained with the v3a recipe (k23_train_refiner, variant a, seed 0) on data regenerated with that detector:
    - AV2 REF-TRAIN / REF-VAL: the k23_refdata recipe, with the detector in the crops and u960;
    - nuScenesAll TRAIN scenes: the refiner_data_v2 recipe, whole frame 384×640 with the detector.
  - KITTI images are not on VUW, so the KITTI part stays the YOLOv8s boxes of v2/v3a (a disclosed deviation).
  - Selection uses the refiner trained on REF-TRAIN, selected on REF-VAL.
  - Also reported: GFLOPs per frame (each canvas × the detector's GFLOPs at 640² / 640²) and one Jetson timing window per candidate.
  - Fair counterpart: whole frame with the same detector, at the resolutions whose latency brackets the candidate's.
- **C — learned temporal cue.** A head on the last three same-scale crops of an object (t−2, t−1, t), with:
  - inputs: the three 64×64 crops through a shared ImageNet ResNet-18, plus the 13 box features at t, the height ratios h_t/h_{t−1} and h_{t−1}/h_{t−2}, and dt;
  - output: the relative closing rate r = 1 / ttc (annotated ttc of the matched object; 0 if not closing);
  - training on AV2 REF-TRAIN and selection on REF-VAL; the chains are the same-scale chains of T4.
  - Candidates:
    - **C1 (replace):** the ttc of a crop object with a 3-observation chain = 1 / r̂ if r̂ > 1e-3 / dt, else ∞; no ttc before 3 observations, as T4.
    - **C2 (in addition):** r = mean(r̂, r_T4), with the T2_1 gate kept.
  - If C passes, the baselines receive it too (R4).
- **D — refiner seeds.** The refiner of the selected configuration is retrained with seeds 1 and 2; seed 0 stays the confirmatory weight. Mean / std of dev G are reported. No ensembling unless it passes as its own candidate.

**Rule (R3, as K23 / K27).**
- **Pass:** a candidate passes if, on dev with its refiner, the interval of (candidate − v5.1-S) is above 0 on one dataset and the point estimate is ≥ 0 on the other. Statistics: unit bootstrap, 2,000 resamples, seed 0, by scene / log.
- **Shared values:** shared values are preferred, chosen by the mean of the two dev G. A per-dataset value is kept only if the shared value is significantly worse on dev.
- **Budgets:**
  - S: tensor px ≤ 1.0 × v5.1-S dev px on each dataset;
  - M: ≤ 1.5 ×;
  - L: ≤ 2.5 ×;
  - each with the 2% tolerance of the K18 matching protocol.
  - B candidates are placed by GFLOPs per frame relative to v5.1-S instead of px.
- **Choosing a budget's configuration:** among the passing candidates within the cap (alone in round 1; in round 2 also combinations of passing ones), the highest mean dev G is chosen. If nothing passes at a cap, that budget's configuration is the next lower one, and S is v5.1-S.
- **When phase 1 stops:** when a round adopts nothing, after round 3, or on 2026-10-18.

**Reported per candidate:**
- dev G without / with refiner, and the interval vs v5.1-S;
- tensor px mean / p95, and GFLOPs where relevant;
- G on the K17(3) regime (act_gt ≥ 1 and draft action below it);
- unnecessary / missed actions per 1,000 frames;
- the same on two disjoint halves of each dev set (units sorted, alternate).

**E — baseline tuning (R4).**
- **Budgets.** Every family is tuned on the same dev data at the same budgets.
  - Per-frame crop budget at cap c = c × (draft px + v5.1-S crop px of the frame) − draft px. At S this equals v5.1-S's crop px, the K24 form.
  - Whole frame: dev-best resolution under each cap, and unconstrained.
- **Components.** Every row has history and guard, and is scored plain and "+ components" (T4 + T2_1 lift on its own same-detector history, refiner v3a).
- **12 configurations per family** (= ours' round-1 family A, 11 alternatives + base), the K24 form included:
  - **uncertainty:** context {1.5, 2*, 2.5, 3, 4, 5} × ordering {confidence ascending*, entropy descending};
  - **temporal_crit:** box growth per side {0, 0.25*, 0.5, 1.0} × previous objects considered {all*, top 3, top 1} by braking demand;
  - **ego-corridor:** tube near range {5, 10*} m × far range {40, 70*, 100} m × width {widest that fits the budget*, half-width capped at 2.5 m};
  - **traj_support:** the support crops of the draft (context 2) + the same 12-point tube grid for the remaining budget;
  - **whole frame:** 12 long sides, nuScenes {448, 512, 576, 640, 768, 896, 960, 1088, 1280, 1408, 1536, 1600}, AV2 {512, 640, 768, 896, 960, 1088, 1280, 1408, 1536, 1664, 1792, 2048}.

  (* = K24 form.)
- **Random crops** stay in their K24 form; they are not listed for tuning.
- **Choice.** Per family and budget, the dev-best configuration with components (mean of the two dev G) is the one run in phase 2, next to its K24 form. A baseline is never run weaker than in K24.

**Code.** k28_dev.py / k28_base.py / k28_refdata.py / k28_train.py / k28_analyze.py (md5 recorded at launch). k_loop.py is unchanged. Pivots and the field-only switch use the call-scoped SP.support replacement of k24_supp.
**§2j launch record (2026-10-11 19:30 +07 / VUW 01:30).**
- md5:
  - k28_dev.py 6a64c53b…;
  - k28_refdata.py 98bcbf64…;
  - k28_train.py e0fb2401…;
  - k28_r1_chain.sh 208aa360…;
  - k28_bdata_chain.sh 5561c650…;
  - k_loop.py 480cf13a… (unchanged);
  - k27_dev.py 33e0b3c6… (unchanged).
- Smoke parity: on nuScenes scene-0108, k28_dev's v5.1-S and T3 reproduce K27 exactly (J, actions, px; 100%).
- The B refiners use k28_train.py with --seed 0 (v3a recipe). The D seeds of v3a are out/resnet18_v3a_s1 / _s2.

## 2k. Amendment 2026-10-12 (K28) — end of round 1 and pre-registration of round 2 (DEVELOPMENT data only; written before any round-2 result)

**Round-1 outcome.** The §2j rule was applied as written (k28_analyze.py --tag r1; results/k28/tables/candidates_r1.csv, selection_r1.csv). Dev G with the candidate's refiner; (candidate − v5.1-S) [95% CI].

| Candidate | nuScenes | AV2 | Ratio | Class | Passes |
|---|---|---|---|---|---|
| v5.1-S | 0.302 | 0.272 | 1.00 | S | ref |
| C1 learned temporal cue, replacing T4 | 0.321, +0.019 [−0.000, +0.059] | 0.309, **+0.037 [+0.024, +0.051]** | px 1.003 | S | **yes** |
| C2 mean of C1 and T4 | 0.311, **+0.009 [+0.002, +0.021]** | 0.295, **+0.024 [+0.014, +0.035]** | px 1.002 | S | yes |
| B3 YOLO11m on the crops, own refiner | 0.409, +0.107 [−0.003, +0.220] | 0.293, **+0.022 [+0.008, +0.037]** | GFLOPs 1.78 | L | **yes** |
| B1 YOLOv8m, own refiner | 0.384, +0.082 [−0.013, +0.183] | 0.285, **+0.013 [+0.001, +0.028]** | GFLOPs 1.97 | L | yes |
| B2 YOLOv8l, own refiner | 0.391, +0.090 [−0.018, +0.205] | 0.292, **+0.020 [+0.006, +0.036]** | GFLOPs 3.68 | over | yes, over budget |
| A1–A11 | see selection_r1.csv | | | | none |

**Adopted in round 1:**
- S: C1;
- M: C1, since no M-class candidate passes and C1 has the highest mean dev G among the passing candidates within M;
- L: B3, YOLO11m on the crops with its own refiner.

**Rejected in round 1:**
- C2 and B1 pass but have lower mean dev G than C1 / B3.
- B2 passes but is over the L budget.
- None of A1–A11 passes:
  - A3–A5 (v_close = α × ego speed) are identical to v5.1-S by construction: dsf._insert_demand takes max(v²/(2·gap), v_close²/(2(r − standoff))), so v_close ≤ ego speed never adds demand;
  - A1 / A2 are significantly worse on AV2;
  - the closest is A7 (context 2.5): nuScenes +0.033 [−0.003, +0.075], AV2 +0.004 [−0.003, +0.012].

**Deviations and operational events in round 1 (disclosed):**
- The C head's training data also include nuScenes new TRAIN scenes (development data; a hold-out of them is in its validation split), not AV2 REF-TRAIN alone as §2j says. Head md5 7bfdecec…; val MAE 0.065 vs T4 0.099.
- B2's first evaluation group died at start-up with CUDA OOM and was re-run (k28_bdev_fix.sh); finished units are skipped.
- Two nuScenes E shards ended without an error; they were re-run (k28_efix.sh).
- k28_analyze.py gained the B same-detector whole-frame table (bwholeframe_r1.csv; reporting only).

**Not selection inputs, reported:**
- **Whole frame benefits more from C1 on AV2:** u960 + C1 reaches 0.328 vs v5.1-S + C1 0.309. On nuScenes it is 0.272 vs 0.321. R4: every baseline receives C1 (below).
- **Same-detector whole frame for B3:**
  - nuScenes: 0.337 at 448 (2.4× GFLOPs) and 0.413 at 768 (7.1×);
  - AV2: ≤ 0.279 at every long side.
- **Jetson timing** (phase1/latency_B_jetson.csv, nuScenes, 797 frames) — ours with YOLO11m: mean 28.6 ms, median 20.7 ms, p95 71 ms. Whole frame YOLO11m + components: 14.4 ms at 448, 20.4 ms at 640.

**Round-2 candidates.** Each is a combination of round-1 passing items; everything else is v5.1-S. All use YOLO11m on the crops with its own refiner, so the class is set by GFLOPs.
- **R2-1 B11C1:** B3 + C1.
- **R2-2 B11C2:** B3 + C2.
- **R2-3 / R2-4 B11C1s768 / B11C1s640:** R2-1 with crops at long side 768 / 640 (nuScenes scale 0.48 / 0.40; AV2 768 / 2048, 640 / 2048).
  - This is the crop-side operating parameter of A9, here so that the YOLO11m verifier can reach the M / S classes.
  - The values 768 / 640 are new; A9 tested 1280 only.
- **Reference rows in the same pass:**
  - v5.1-S (parity with r1);
  - C1 (parity with r1's temporal part);
  - B11 = B3 (parity with r1's ver-yolo11m).
- The C head is the round-1 head; it is not retrained on YOLO11m chains.

**Rule (unchanged from §2j).**
- **Pass:** vs v5.1-S on dev with the candidate's seed-0 refiner (unit bootstrap 2,000, seed 0): the interval is above 0 on one dataset and the point estimate is ≥ 0 on the other.
- **Choice per budget:** the highest mean dev G among the passing candidates of rounds 1 and 2 whose class is within the cap.
- **Adoption:** a round-2 candidate is adopted if it changes a budget's choice.
- **Round 3:** only if round 2 adopts something and an untested combination of passing items remains. Otherwise phase 1 ends after round 2, and in any case on 2026-10-18.

**D (refiner seeds), reported.**
- Dev G over refiner seeds 0 / 1 / 2:
  - C1 with v3a s0 / s1 / s2 (md5 6b29d05e / 7a05f69a / 9daa5e43);
  - every YOLO11m row with own s0 / s1 / s2 (md5 eec4782b / 5f1a82bc / eebc629f).
- Mean and std are reported. Seed 0 stays the confirmatory weight, and the seeds are not a selection input.

**E with C1 (R4).** C1 was adopted, so every baseline receives it.
- **Whole frame (YOLOv8s), the 12 E long sides with + C1 (refiner v3a):**
  - the dev-best long side per cap is chosen with + C1 (mean of the two dev G) and reported next to the + T4 choice;
  - phase 2 runs both choices, plus the K24 forms.
- **Heuristic families.** The dev-best configuration per family and budget stays the one chosen on the E grid with + T4 (§2j); a C1 grid over all 48 configurations × 3 caps is not run.
  - The chosen configurations and the K24 forms are re-run on dev with + C1 (k28_r2dev.py --part baseheur).
  - In phase 2 they are scored plain / + T4 / + C1.
- **Same detector for L:**
  - whole frame with YOLO11m at the 6 B long sides with + C1 (own refiner);
  - the heuristic families with YOLO11m on their crops at the S crop budget (own refiner, + T4 / + C1), using their dev-best S configurations.

  This gives the L configuration a same-detector, same-px counterpart in every family.
- **Reporting:** a baseline row with a higher G than ours is reported as such (R5).

**Code.** k28_r2dev.py (parts ours / basewf / baseheur; functions copied from k28_tdev.py and the E construction from k28_dev.py; k_loop unchanged). Its md5 is recorded at launch. A one-scene smoke run is checked for parity only.
**§2k launch record (2026-10-12 07:45 +07 / VUW 13:45).**
- md5:
  - k28_r2dev.py 102b7ec8…;
  - k28_r2_chain.sh 013df282…;
  - k28_analyze_r2.py 4ccd2777…;
  - k28_analyze.py 18af0ccc… (the K24_form label of the corridor families corrected to cor8 / tsp8; selection logic unchanged);
  - k_loop.py 480cf13a… (unchanged).
- Smoke parity on nuScenes scene-0108 (39 frames): identical J / px on 100% of frames for:
  - r2 v5.1-S = r1;
  - r2 C1 (none / v3a) = r1 temporal C1;
  - r2 YOLO11m (none / own, px) = r1 ver-yolo11m;
  - r2 whole frame 960 + C1 = r1 u960 + C1;
  - r2 whole frame 960 + T4 = r1 E wf960 + T4.

  No candidate value of the smoke run was looked at.

## 2l. Amendment 2026-10-12 (K28) — end of round 2 and pre-registration of round 3 (DEVELOPMENT data only; written before any round-3 result)

**Round-2 outcome.** The §2k rule was applied as written.
- Source: k28_analyze_r2.py (md5 4ccd2777…), run at 2026-10-12 15:45 +07 with its output folder redirected to $B/tmp/k28_r2prelim/. Unreadable *.pkl files being written were skipped.
- The round-2 ours parts were complete at that time (nuScenes 152, AV2 300), and the selection reads nothing else. The chain's own run on the complete data writes the official tables, and it is checked to give the same candidates_r2 / selection_r2.
- Dev G with the own (YOLO11m) seed-0 refiner; (candidate − v5.1-S) [95% CI].

| Candidate | nuScenes | AV2 | GFLOPs ratio | Class | Passes | Mean G |
|---|---|---|---|---|---|---|
| R2-1 YOLO11m + C1 | 0.408, +0.106 [−0.007, +0.224] | 0.311, **+0.039 [+0.022, +0.057]** | 1.78 | L | yes | **0.3592** |
| R2-2 YOLO11m + C2 | 0.409, +0.107 [−0.003, +0.222] | 0.307, **+0.036 [+0.021, +0.052]** | 1.78 | L | yes | 0.3580 |
| R2-3 YOLO11m + C1, crops 768 | 0.391, **+0.089 [+0.004, +0.179]** | 0.311, **+0.040 [+0.024, +0.057]** | 1.32 | M | yes | 0.3510 |
| R2-4 YOLO11m + C1, crops 640 | 0.396, **+0.094 [+0.005, +0.190]** | 0.306, **+0.035 [+0.018, +0.052]** | 1.09 | M | yes | **0.3512** |

**Adopted in round 2.**
- S: C1, unchanged.
- M: R2-4, YOLO11m + C1 with crops at long side 640, replacing C1.
  - R2-3 is within 0.0002 of it in mean dev G. The rule's order is kept and the near tie is disclosed.
- L: R2-1, YOLO11m + C1, replacing B3.

**Rejected in round 2.** R2-2 has a lower mean G than R2-1. R2-3 has a lower mean G than R2-4 within M, and than R2-1 within L.

**Checks and reported rows.**
- Parity: 100% identical frames (nuScenes 6,117; AV2 23,665) for v5.1-S, C1, B3 (J and px), wf960 + T4 vs E, and wf960 + C1 vs r1 u960 + C1.
- D (refiner seeds, dev G seed 0 / 1 / 2):
  - C1 0.321 / 0.324 / 0.321 (nuScenes) and 0.309 / 0.309 / 0.312 (AV2);
  - R2-1 0.408 / 0.407 / 0.413 and 0.311 / 0.308 / 0.308;
  - R2-4 0.396 / 0.395 / 0.397 and 0.306 / 0.304 / 0.304;
  - std ≤ 0.006 for every row (seeds_r2.csv).

**R4 baselines with C1 (reported, not selection inputs; AV2 heuristics were partial at the time of writing):**
- **nuScenes:** ours leads at S / M. At L the YOLO11m heuristic tsp2@S + C1 reaches 0.414 vs R2-1 0.408.
- **AV2:**
  - whole frame + C1 reaches 0.316 at 512 (1.21×) and 0.322 at 640 (1.86×), above R2-4 0.306 and R2-1 0.311;
  - corridor heuristics + C1 are above ours vs v5.1-S on the same units (cor0@M +0.080, cor0@L +0.106; ours M +0.035, L +0.039).
- These rows are reported as such (R5), whatever round 3 gives.

**Round-3 candidates.** These are the untested combinations of passing items: B1 (YOLOv8m) and B2 (YOLOv8l) with C1, and with the crop long sides of R2-3 / R2-4. Everything else is v5.1-S; each uses its detector's own seed-0 refiner from round 1.
- **R3-1 B8mC1:** B1 + C1. Expected ≈ 2.0× GFLOPs, class L.
- **R3-2 B8mC1s768:** B1 + C1, crops 768. Expected ≈ 1.3–1.5×, class M.
- **R3-3 B8mC1s640:** B1 + C1, crops 640. Expected ≈ 1.2×, class M.
- **R3-4 B8lC1s768:** B2 + C1, crops 768. Expected ≈ 2.2–2.6×, L or over.
- **R3-5 B8lC1s640:** B2 + C1, crops 640. Expected ≈ 1.8–2.0×, class L.

The class is set by the measured GFLOPs ratio, as before.

**Not included:**
- **C2 with any detector or crop side:** C2 was below C1 in both rounds (YOLOv8s in round 1, YOLO11m in round 2).
- **B2 + C1 at native crops:** B2 alone is 3.68× and C1 adds no pixels.
- **Combinations with A items:** none passed.
- **Retraining the C head on YOLO11m / v8m / v8l chains:** a new item, not a combination.

**Reference rows in the same pass:**
- v5.1-S, for parity with r1;
- B8m = round-1 B1, for parity with r1 ver-yolov8m (J and px).

**Rule (unchanged from §2j / §2k).**
- **Pass:** the interval is above 0 on one dataset and the point estimate is ≥ 0 on the other. The comparison is vs v5.1-S on dev, with the candidate's seed-0 refiner (unit bootstrap 2,000, seed 0).
- **Choice per budget:** the highest mean dev G among the passing candidates of rounds 1–3 whose class is within the cap.
- **Adoption:** a round-3 candidate is adopted if it changes a budget's choice.
- **Phase 1 ends after round 3** (the ≤ 3 rounds of R3).

**If a round-3 candidate is adopted** (before phase 2, dev only, not selection inputs):
- (a) own refiners of seeds 1 / 2 for its detector are trained with the round-1 recipe, and D is reported;
- (b) the same-detector baselines of §2k (R4) are run with that detector:
  - whole frame at the B long sides + C1;
  - the heuristic families at the S crop budget + T4 / + C1.

**Phase-2 baselines (direction from the user, 2026-10-12).**
- The published prior works (AdaScale, FOVEA, LZU, SAHI, UFP / UFPMP-Det, ESOD, ZoomDet, Budget-Adaptive Routing) are run with their official code and settings as in K24. They do not receive C1.
- Whole frame and the heuristic families are our own controls and receive C1 under R4, as in §2k.
- Which table is the paper's main table is left to the user.

**Phase-3 additions (reporting only, after the freeze).** For the frozen v6 configurations on Argoverse-HD val and nuScenes:
- whole-image AP against tensor px, next to the whole-frame curve;
- streaming AP with Jetson per-frame latencies (Li et al. simulation).

**Code.** k28_r3dev.py, a copy of k28_r2dev.py plus part ours3: the parts of §2k are unchanged and not used, and k_loop is unchanged. Also k28_analyze_r3.py and k28_r3_chain.sh. md5s are recorded at launch. A one-scene smoke run is checked for parity only.
**§2l launch record (2026-10-12 16:35 +07 / VUW 22:35).**
- md5:
  - k28_r3dev.py f5f74b28…;
  - k28_analyze_r3.py a56758cd…;
  - k28_r3_chain.sh 5c4d0c37… (6 groups of 3 shard processes, MIN_FREE_MB 9000; a completeness check re-runs groups with ended shards; the analysis waits for the round-2 official tables);
  - k_loop.py 480cf13a… (unchanged).
- Smoke parity on nuScenes scene-0108 (39 frames): identical J / px on 100% of frames for:
  - r3 v5.1-S (YOLOv8m and YOLOv8l processes) = r1;
  - r3 B8m (none / own, px) = r1 ver-yolov8m.

  No candidate value of the smoke run was looked at.

## 2m. Amendment 2026-10-12 (K28) — end of round 3 and of phase 1; follow-ups of the adoption (DEVELOPMENT data only)

**Round-3 outcome.** The §2l rule was applied as written.
- Source: k28_analyze_r3.py (md5 a56758cd…), run at 2026-10-12 19:50 +07 with its output folder redirected to $B/tmp/k28_r3prelim/. It read the preliminary selection_r2.csv of §2l.
- The round-3 parts were complete (nuScenes 152 and AV2 300 for both detectors; r3_done.log "K28 R3 DEV DONE" at VUW 23:39).
- The chain's own run writes the official tables after the round-2 chain finishes. It is checked to give the same candidates_r3 / selection_r3.
- Parity (parity_r3): 100% identical frames (nuScenes 6,117; AV2 23,665) for v5.1-S (YOLOv8m and YOLOv8l processes) = r1, and for B8m (none / own, px) = r1 ver-yolov8m.
- Dev G with the detector's own seed-0 refiner; (candidate − v5.1-S) [95% CI].

| Candidate | nuScenes | AV2 | GFLOPs ratio | Class | Passes | Mean G |
|---|---|---|---|---|---|---|
| R3-1 YOLOv8m + C1 | 0.401, **+0.099 [+0.000, +0.203]** | 0.314, **+0.042 [+0.027, +0.058]** | 1.98 | L | yes | 0.3572 |
| R3-2 YOLOv8m + C1, crops 768 | 0.394, +0.092 [−0.011, +0.197] | 0.311, **+0.039 [+0.024, +0.056]** | 1.46 | M | yes | **0.3523** |
| R3-3 YOLOv8m + C1, crops 640 | 0.335, +0.034 [−0.019, +0.090] | 0.312, **+0.040 [+0.024, +0.057]** | 1.18 | M | yes | 0.3237 |
| R3-4 YOLOv8l + C1, crops 768 | 0.411, +0.109 [−0.000, +0.224] | 0.310, **+0.038 [+0.022, +0.056]** | 2.58 | over | yes | 0.3603 |
| R3-5 YOLOv8l + C1, crops 640 | 0.405, +0.103 [−0.010, +0.220] | 0.311, **+0.039 [+0.023, +0.056]** | 2.01 | L | yes | 0.3580 |
| (reference) B8m = round-1 B1 | 0.384, +0.082 [−0.013, +0.183] | 0.285, +0.013 [+0.001, +0.028] | 1.97 | L | (r1) | 0.3344 |

**Adopted in round 3.**
- S: C1, unchanged.
- M: **R3-2, YOLOv8m + C1 with crops at long side 768**, replacing R2-4 (mean dev G 0.3523 vs 0.3512).
  - The margin (0.0011) is far below the seed spread of the refiner (std up to 0.006). The rule's order is kept and the near tie is disclosed, as for R2-3 / R2-4 in §2l.
- L: R2-1 (YOLO11m + C1), unchanged. R3-5 (0.3580) and R3-1 (0.3572) are below it.
  - R3-4 has the highest mean G of all candidates (0.3603), but its GFLOPs ratio 2.58 exceeds the L cap (2.5 × 1.02 = 2.55). It is reported, not chosen.

**Phase 1 ends here** (round 3 is the last round allowed by R3).

**v6 after phase 1** (to be frozen with md5s in §2n, before the single test pass):
- v6-S = C1: YOLOv8s crops at 960 + C1, refiner v3a.
- v6-M = R3-2: YOLOv8m crops at 768 + C1, own YOLOv8m refiner (seed 0).
- v6-L = R2-1: YOLO11m crops at 960 + C1, own YOLO11m refiner (seed 0).

**Follow-ups pre-registered in §2l (dev only; not selection inputs; nothing above changes whatever they give).**
- (a) Own YOLOv8m refiners of seeds 1 / 2 (k28_train.py --det yolov8m --seed 1 / 2, the round-1 recipe). D is reported for R3-2 (dev G with seeds 0 / 1 / 2).
- (b) The same-detector baselines (R4) with YOLOv8m, each with the own YOLOv8m refiner (seed 0):
  - whole frame at the B long sides (nuScenes 448 / 640 / 768 / 960 / 1280 / 1600; AV2 512 / 640 / 896 / 1280 / 1536 / 2048): plain, + T4, + C1;
  - the heuristic families at the S crop budget, with the nine S configurations used for YOLO11m in §2k (unc2, unc4, tcr0, tcr3, cor0, cor8, tsp2, tsp8, rnd): + T4 and + C1.
- Parity checks: v5.1-S = r1; R3-2 with the seed-0 refiner = r3; YOLOv8m whole frame plain / + T4 = r1 vwf-yolov8m.
- Code (md5s at writing):
  - k28_r3fu.py cfc60d54…: k28_r3dev.py with parts oursd and basewf3 and its own output folders; k_loop.py 480cf13a… unchanged;
  - k28_analyze_r3fu.py 6b1ec433…: parity_fu, seeds_fu, basewf_fu, baseheur_fu;
  - k28_r3fu_chain.sh dc14dc7c…; k28_train.py e0fb2401…;
  - a one-scene smoke run (whole frame and heuristics parts) is checked for parity only before the launch.

**§2m launch record (2026-10-12 20:05 +07 / VUW 02:05 on 2026-10-13).**
- Smoke parity on nuScenes scene-0108 (39 frames): identical J / px on 100% of frames for v5.1-S = r1 (whole-frame and heuristic processes) and for YOLOv8m whole frame plain / + T4 at the six long sides (none / own, px) = r1 vwf-yolov8m. No other smoke value was looked at.
- k28_r3fu_chain.sh launched (PID 2812983). Seeds 1 / 2 are trained first and then family D runs; the whole-frame and heuristic groups run at the same time.

**§2m confirmation (2026-10-12 21:20 +07 / VUW 03:10 on 2026-10-13).**
- The round-2 chain finished ("K28 R2 ANALYZE DONE", VUW 03:05) and the round-3 chain ran its analysis ("K28 R3 ANALYZE DONE", VUW 03:10).
- Official vs preliminary tables:
  - candidates_r2 / r3, selection_r2 / r3, seeds_r2 and parity_r2 / r3 are identical in every G, interval, pass flag, class and choice;
  - the GFLOPs ratios differ by ≤ 0.0027, because the official runs read the measured phase0/gflops_640.csv. Official ratios: R2-1 1.787, R2-3 1.325, R2-4 1.086, R3-2 1.457;
  - parity_r2 and parity_r3 are 100% on all 14 rows each.
- Choices confirmed: S = C1, M = R3-2, L = R2-1.

**R4 baselines with C1, final (round-2 tables complete; these replace the partial AV2 figures of §2l).** Dev G with refiner; (row − v5.1-S) [95% CI]; px ratio vs v5.1-S.
- AV2, YOLOv8s heuristics + C1:
  - tsp0@L 0.346, +0.074 [+0.050, +0.100] (px 2.41);
  - tsp8@M 0.325, +0.054 (px 2.08);
  - cor0@M 0.315, +0.044 (px 1.42);
  - tsp8@S 0.311, +0.040 (px 1.37).
  - The partial figures of §2l (cor0@M +0.080, cor0@L +0.106) were from 104 of 300 logs. Final: cor0@M +0.044, cor0@L +0.066.
- AV2, other controls with C1:
  - YOLO11m tsp8@S 0.319;
  - YOLOv8s whole frame 512: 0.316 (1.21×); 640: 0.322 (1.86×).
- AV2, ours: v6-S 0.309, v6-M 0.311, v6-L 0.311.
- nuScenes, YOLOv8s heuristics + C1: tsp0@L 0.325, tcr3@S 0.313, tcr0@M 0.303.
- nuScenes, other controls with C1:
  - YOLO11m tsp2@S 0.414 (px 1.26; GFLOPs class L);
  - whole frame 448: 0.195 (0.99×); 512: 0.217 (1.28×).
- nuScenes, ours: v6-S 0.321, v6-M 0.394, v6-L 0.408.
- **Reading:** on AV2 dev, several heuristic controls with C1 are above ours at classes M / L. On nuScenes, ours leads at S / M, and at L it is within 0.006 of the YOLO11m heuristic. These rows are reported in phase 2 as baselines (R5); none is a candidate of ours.

**§2m follow-up results (2026-10-13 00:35 +07 / VUW 06:32; dev only, not selection inputs).** Tables: results/k28/tables/{parity,seeds,basewf,baseheur}_fu.csv.
- Parity (parity_fu): 100% identical on all 82 rows (nuScenes 6,117; AV2 23,665 frames):
  - v5.1-S = r1 in each part;
  - R3-2 with the seed-0 refiner (none / own, px) = r3;
  - YOLOv8m whole frame plain / + T4 (none / own, px) = r1 vwf-yolov8m.
- **(a) D for R3-2** (dev G with the own YOLOv8m refiner of seeds 0 / 1 / 2; refiner md5s d2a92d8f… (s1), c88c9742… (s2)):
  - nuScenes 0.394 / 0.399 / 0.395 (std 0.003);
  - AV2 0.311 / 0.311 / 0.317 (std 0.003).
- **(b) YOLOv8m same-detector baselines** (own refiner; GFLOPs class by the measured ratio):
  - **Whole frame:** every side is over the L cap (nuScenes 2.73× at 448 up to 35×; AV2 3.33× at 512 up to 50×).
    - Best with C1: nuScenes 1600 0.423 (35×), 1280 0.417 (22×); AV2 1280 0.337 (20×).
    - At the smallest side: nuScenes 448 + C1 0.197 (2.73×); AV2 512 + C1 0.326 (3.33×).
  - **Heuristics at the S crop budget:**
    - only two rows fall within a class: nuScenes tsp2@S + C1 0.381 (2.52×, L) and AV2 cor0@S + C1 0.245 (2.48×, L). Both are below v6-L (0.408 / 0.311);
    - the best rows are over the L cap: tsp8@S + C1 nuScenes 0.389 (3.02×), AV2 0.313 (3.08×).
  - No YOLOv8m baseline falls in class M, where v6-M uses YOLOv8m.

## 2n. Amendment 2026-10-13 (K28) — freeze of v6 and pre-registration of the ONE test pass (phase 2)

Written before any K28 test evaluation. No K28 code has run on a test unit, and no decision above used a test number (R1). This is the single further test pass allowed by R2. Nothing below changes after it starts.

**Frozen configurations (v6).** All three keep k_loop's v5.1-S policy (TPOL, arm "c", support context 2.0) and the YOLOv8s draft.

| Name | Crop detector | Crop long side | Temporal cue | Refiner |
|---|---|---|---|---|
| v6-S | YOLOv8s | 960 | C1 | v3a |
| v6-M | YOLOv8m | 768 | C1 | own YOLOv8m, seed 0 |
| v6-L | YOLO11m | 960 | C1 | own YOLO11m, seed 0 |

- C1 head: out/k28_temporal/temporal_head.pt.
- Weight md5s:
  - refiners: resnet18_v3a 6b29d05e…, resnet18_v3a_yolov8m ec73855c…, resnet18_v3a_yolo11m eec4782b…;
  - C1 head: 7bfdecec…;
  - detectors: yolov8s.pt 0a1d5d06…, yolov8m.pt 02adb754…, yolo11m.pt 2c2bcbb5…;
  - phase0/gflops_640.csv 4bf38b0b….
- Code md5s:
  - k28_test.py acb53791…, k28_analyze_test.py d0dd013a…, k28_test_chain.sh 32daaa9b…;
  - k_loop.py 480cf13a…, k28_analyze.py 18af0ccc…;
  - k24_test.py d2ae37cc… (units), k24_analyze.py 3ee3290b… (K24 rows, routers, Boot, verdict);
  - k28_ttrain.py 4f093e32…, k28_tdata.py 751b2e31…, k27_dev.py 33e0b3c6…, k10_masked_run.py 4abc786b….
- Dev smoke parity: identical on 100% of frames in two runs (§2l record; STATUS 2026-10-12 21:10) for v6-S / v6-M / v6-L, v5.1-S, every whole-frame row and the K24-form heuristics.

**Test data (K24 protocol, suffix _v6).**
- nuScenes new test: 228 scenes.
- AV2 val: 150 logs, with the ego mask through k10_masked_run.py.
- Argoverse-HD val: 24 sequences. Every frame runs for the histories; tables use the frames with 3D labels, as in K24.

**Rows of the pass.**
- **Ours:** v5.1-S (parity with K24) and v6-S / v6-M / v6-L.
- **Prior works:** the published methods (AdaScale, FOVEA, LZU, SAHI, UFP / UFPMP-Det, ESOD, ZoomDet, Budget-Adaptive Routing) and K24's routers.
  - They run with their official code and settings. Their K24 test results are reused unchanged (k24_analyze.merged). They are not re-run and do not receive C1.
- **Controls (R4).** Each control is scored plain, + T4 and + C1, on its own same-detector history.
  - YOLOv8s whole frame:
    - nuScenes 448 / 512 / 640 / 960 / 1280 / 1536 / native;
    - AV2 512 / 640 / 960 / 1280 / 1536 / native;
    - Argoverse-HD 448 / 512 / 640 / 960 / 1280 / 1536 / native.
    - These are the K24 forms plus the dev-best sides per class of basewf_r2. Argoverse-HD has no dev split and takes the union.
  - YOLO11m and YOLOv8m whole frame at the B sides, with own refiners:
    - nuScenes 448 / 640 / 768 / 960 / 1280 / 1600;
    - AV2 512 / 640 / 896 / 1280 / 1536 / 2048;
    - Argoverse-HD 512 / 640 / 896 / 1280 / 1536 / 1920.
  - YOLOv8s heuristics, the 26 configurations of §2k (E dev-best ∪ K24 forms): unc2 / unc4 / tcr0 / tcr3 / cor0 / cor8 / tsp2 / tsp8 / rnd @S; unc2 / unc8 / tcr0 / tcr3 / cor0 / cor8 / tsp4 / tsp8 / rnd @M; unc2 / unc6 / tcr3 / cor0 / cor8 / tsp0 / tsp8 / rnd @L.
  - YOLO11m and YOLOv8m heuristics at the S crop budget, with the 9 S configurations and own refiners.

**Analysis (k28_analyze_test.py, run once when every part has every unit).**
- Unit bootstrap 2,000, seed 0, by scene / log / sequence (K24 Boot).
- Verdict on the (v6 − row) interval with refiner:
  - outperforms: lo > 0;
  - matches: lo > −0.02;
  - else below.
- GFLOPs class by the measured test ratio vs v5.1-S. Over-budget rows are compared with v6-L and flagged.
- Pre-registered outputs:
  - headline_v6: v6-X vs v5.1-S, per dataset;
  - class_matched_v6: every baseline vs the v6 configuration of its class;
  - table_main_<ds>_v6: every row vs v6-S / M / L and v5.1-S;
  - not_outperformed_v6: every pair whose interval is not above 0;
  - parity_v6.
- Paper tables (the user decides): (1) system table, ours vs prior works; (2) controlled table, every selection method with C1. Both come from these outputs, and every row is reported (R5).

**Procedure.**
- k28_test_chain.sh launches every part, in groups of 3 shard processes, through run_when_gpu_free.sh (SHARE=1). Groups with ended shards are re-run twice at most.
- No test value is looked at before the pass is complete and the analysis has run. The only checks made earlier are crash checks (Traceback, unit counts).
- If a crash needs a code fix, the fix touches only the failing path and is disclosed with its diff and new md5. Finished units are kept.

**§2n launch record.** k28_test_chain.sh (md5 32daaa9b…) launched 2026-10-13 00:36 +07 (VUW 06:36), PID 3843551. Before the launch, no `*_test_v6*` result folder existed.

### 2n.1 Addendum 2026-10-13 ~02:30 +07 — phase-2 items of the K28 task missing from §2n (written while the pass runs, before any test value was looked at)

On re-reading the task's phase-2 list, §2n omitted two of its items:
- "routers and the oracle re-matched";
- "support only / field only / 2x2 ablation".

They are added to the same single pass. They are fixed by the task text, not chosen from any result. Up to this point only crash checks (Traceback, unit counts) had been made on the pass. Nothing else in §2n changes.

- **Ablation of v6-S** (k28_test_abl.py, md5 692a3716…: k28_test.py with one added part `abl`; k28_test.py itself is unchanged).
  - Loops on the same test units and draft, each with its own history and refiners none / v3a:
    - so51S and fo51S: v5.1-S, support crops only / field crop only, with T4. Their parity target is K24 supp so51S / fo51S.
    - so6S and fo6S: the same with C1.
  - The 2 × 2 {refiner v3a} × {C1} uses v6-S (main pass) and v5 (K24 main; the same loop with no temporal cue). T4 in place of C1 is v5.1-S.
- **Routers and the oracle re-matched** (k28_analyze_test2.py, md5 26468b2b…).
  - The K24 routers (K18 protocol: the best Budget-Adaptive Routing and the best decision-value router, each picked on val) and the true-V oracle are re-matched to the GFLOPs of v6-S, v6-M and v6-L.
  - Matching uses the YOLOv8s-equivalent tensor px, within 2%, because the routers escalate to u960 with YOLOv8s. Each router is compared with the v6 configuration it is matched to.
  - Argoverse-HD has no trained router, so only the oracle is computed there.
- **Outputs:** ablation_v6S_<ds>.csv, routers_v6_<ds>.csv, parity_abl_v6.csv (unit bootstrap 2,000, seed 0; K24 verdicts).
- **Chain:** k28_test_abl_chain.sh (md5 4a70322b…). A dev smoke run (one nuScenes dev scene) is checked for crashes and for v5.1-S parity only.

**§2n.1 launch record.**
- Dev smoke (nuScenes scene-0108, 39 frames): no crash. J_v51S|v3a and px_v51S are identical to K24 r1 on 39 / 39 frames.
- k28_test_abl_chain.sh (md5 4a70322b…) was launched 2026-10-13 02:27 +07 (VUW 08:27), PID 4179495. Before the launch, no `*_test_v6abl` folder existed.
- At launch, the main pass's own unit counts were checked for completeness only.

**§2n.2 Analysis record (phase 2, 2026-10-13).** Every test unit finished at VUW 16:32:52.
- k28_analyze_test.py (d0dd013a) and k28_analyze_test2.py (26468b2b) crashed before writing any table, with `KeyError: 'J_cheap_ours'`. Cause: the merged column list held J_cheap twice (J_cheap also matches the "J_" prefix), and the @k28 / @abl rename renamed both copies.
- Fix, analysis code only: J_cheap is excluded from the prefix list (ca1c34b4 / 725e481e). Both scripts were run once on the unchanged result files.
- Required parity checks are 100% identical: K28 v5.1-S and whole frame 960 vs K24, and the ablation's v5.1-S / support only / field only vs K24.
- Results: docs/paper/handoff/K28/phase2/REPORT.md.
