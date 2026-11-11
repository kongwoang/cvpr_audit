# CVPR 2027 direction: Decision-Sensitivity Fields (working draft)

*Draft of 2026-09-27. The evidence sections are filled from `docs/directions_log.md` as the exploratory tests land.*

## 1. The idea in one paragraph

Perception compute in a driving stack is allocated by perception-centric signals: uncertainty, difficulty,
saliency, expected mAP gain. DEEP showed that the downstream value of extra perception is signed and that these
signals, and allocators trained on its decision value, barely predict it before escalation. The missing ingredient is
the **planner**. The decision itself defines where perception matters: a missed pedestrian 40 m ahead in the ego
corridor at 15 m/s changes the plan; the same miss 3 m to the side does not; a phantom car is harmless unless the plan
would react to it. We make this explicit with a **decision-sensitivity field (DSF)**, a map the planner induces over
the scene (and, by projection, over the image), that says where a change of perception would change the decision. The
DSF is computed at run time, without ground truth, by probing the planner. It drives *where* perception spends
compute, at three granularities under one budget: which frames to escalate, which image regions to re-perceive at
high resolution, and which already-detected objects to verify and re-range.

## 2. Formalism

Scene `s`, image `x`, cheap perception `z = f_c(x)`, a set of objects with estimated geometry. Planner `π` with its
own cost `C_π` (never ground truth). Downstream loss `L(π(z), s)`, used only for evaluation.

**Decision-sensitivity field.** For a location `u` on the ground grid `U` and a perturbation `τ` from
{insert an object at `u`, delete the object at `u`, shift its range},

    S_τ(u; z) = 1[ π(z ⊕ τ_u) ≠ π(z) ]      (graded variant:  C_π(π(z); z ⊕ τ_u) − C_π(π(z ⊕ τ_u); z ⊕ τ_u))

**First-order value of information of a compute action.** Let an action `a` (escalate the frame, re-perceive a
region `R`, verify an object `o`) change perception at `u` with probabilities `q⁺_a(u)` (reveals a real missed
object), `q⁻_a(u)` (introduces a phantom) and, for verification, removes a phantom or corrects a range. To first
order in these probabilities (one change at a time),

    VOI(a; z) ≈ Σ_u [ q⁺_a(u) Δ⁺(u) − q⁻_a(u) Δ⁻(u) ] S_ins(u; z)  +  Σ_{o ∈ piv(z)} p_err(o) Δ_o

with `piv(z) = {o : S_del(o; z) = 1}` the decision's pivots. The first term is the frame/region value (benefit of
revealed objects minus harm of phantoms, both only where the decision is sensitive); the second is the value of
verifying what the decision already rests on. `q⁺` comes from the recall gain of the high-fidelity mode as a function
of apparent size/range (estimated on training data); `p_err` from a verifier.

**Budgeted allocation.** With measured costs `c_a` (Jetson ms or mJ) and budget `B`, choose actions by a
multiple-choice knapsack on `VOI(a; z) / c_a` — frames, regions and objects compete for the same budget.

**Sign.** Harm of escalation (DEEP: 34–54% of affected inputs) enters as the `q⁻ Δ⁻` term and as mis-ranged pivots.
Region re-perception confines new detections to where they are needed; pivot verification removes phantom pivots and
re-ranges real ones. Both attack harm without predicting it from pre-escalation features, which DEEP found impossible.

## 3. Method (to build)

1. **DSF by probing** for modular planners (braking controller, receding-horizon Planner B, PDM-Closed, IDM): a
   (range × lateral) grid of hypothetical insertions and one deletion per detected object.
2. **Amortised DSF** for learned planners, where probing is expensive: a light head predicting the field from the
   cheap output (trained on probed labels), or input gradients of the planner.
3. **Compute actions.** Frame escalation (DEEP's cascade); region re-perception with TensorRT crop engines; a
   **pivot verifier–refiner**: a crop network (high-resolution crop around the pivot) predicting existence and metric
   range, trained on KITTI / nuScenes 3D labels on GPU.
4. **One scheduler** over all actions, charged its own measured cost.

## 4. Claims and the tests that decide them

| claim | test | status |
|---|---|---|
| C1 DSF gating beats perception-signal and learned allocators | D4 probe VOI on DEEP's test split | KITTI: yes on 3/4 cells; nuScenes running |
| C2 DSF regions give more decision value per pixel than frames or static crops | D3 region proxy | running |
| C3 verifying pivots beats escalating frames per unit compute | D2 oracle + noisy headroom | nuScenes mono: yes with ≤5–10% verifier error |
| C4 the gains come with fewer harmful changes | harm share under each action | pending |
| (motivation) better detectors can drive worse | D5 model-upgrade regressions | running |

## 5. What is new relative to prior work

- Planner-aware *evaluation* (PKL, planning-aware metrics, TIP, task-aware risk) scores perception after the fact;
  the DSF is computed at run time from the planner alone and *allocates* compute before and after escalation.
- Foveation (FOVEA, LZU, policy-based foveated imaging) chooses pixels by objectness or task accuracy; the DSF
  chooses them by decision sensitivity, which is zero for most detectable objects.
- Detector routing (ORIC, budget-adaptive routing) decides per frame and replaces outputs; the DSF acts per frame,
  region and object, and verifies what the decision rests on.
- Risk-object identification (counterfactual removal) explains a driver's behaviour; here counterfactual probing
  prices perception actions under a compute budget.

## 6. What the exploratory tests of 2026-09-27 say (details in `directions_log.md`)

1. **The decision hinges on one object (D32).** 96–99% of decision changes caused by escalation are reproduced by a
   single element of the cheap→full difference (94–99.6% of |V|), while an escalation changes 3.6–7.4 elements. Frame
   escalation pays for ~5–7 perception changes to buy one.
2. **Two kinds of value, two regimes.** *Discovery* (a miss the cheap mode did not see, where a miss would change the
   decision) dominates where the cheap mode is blind (KITTI braking). *Verification* (the current decision's pivot is
   a phantom or mis-ranged) dominates where harm comes from the lift (KITTI mono traj: verifying one object per
   ~2.5 frames is worth 3× the best frame-escalation policy; nuScenes mono brake: 0.309 against 0.278 for the frame
   oracle). Escalation + verification exceeds the frame-arbitration ceiling ΣV⁺ in every monocular cell (D2).
3. **The planner is a cheap, strong allocator (D4, D16).** Probing the planner with hypothetical objects gives a
   training-free score that beats all of DEEP's learned allocators on 3/4 KITTI cells and, as a feature, yields the
   best nuScenes gate; one consumer's field transfers to the other (it measures decision reach).
4. **Time matters (D35, D22).** Escalating one frame in five and carrying the full mode's extra objects forward keeps
   65–100% of the value of escalating every frame; decision changes are sticky, and whether an escalation flipped the
   decision is a free online signal.
5. **What does not work.** A classifier on cheap post-escalation metadata (D1, pre-registered NO-GO); a Bayes consumer
   with calibrated existence (D15); a second full-frame opinion as a stopping rule (D6).
6. **Motivation beyond resolution (D5).** Swapping YOLOv8s-640 for RT-DETR-l-640 at an equal box count changes the
   braking decision on 19% of frames, 45% of them for the worse; the net gain is a fifth of what a decision-congruent
   swap would keep.

## 7. The method this points to: pivot-centric perception (working name)

    VOI(action) = discovery value  +  verification value
                = Σ_u q⁺(u)·Δ(u)·S_ins(u; z)      +  Σ_{o∈piv(z)} p_err(o)·Δ_o

- **Discovery**: the DSF says where an unseen object would change the decision; spend high-resolution pixels only
  there (region crop), or escalate the frame when the field is large; carry discovered objects forward in time.
- **Verification**: the decision's pivots (usually one object) get a crop-level second look — an *asymmetric*
  existence check (remove only near-certain phantoms) plus metric range refinement; the scale-consistency prior (D14)
  and temporal support are its free inputs.
- **Scheduler**: one knapsack over {region crops, pivot checks, frame escalations} with measured Jetson costs, charged
  its own cost, with flip feedback as an online correction.

The paper's formal contribution is the decomposition and the DSF; the vision contribution is the pivot
verifier–refiner and DSF-guided cropping; the systems contribution is the measured-budget scheduler on an edge device.

## 8. Closest prior work found on 2026-09-27 (to cite and separate from)

| work | what it does | how the DSF direction differs |
|---|---|---|
| Self-Supervised Relevance Modelling via Counterfactual Analysis (arXiv 2606.10688, 2026) | learns, from synthetic counterfactual data, how relevant each *detected* object is to the driving decision; filters objects to save compute and limit noise | the DSF also probes *unseen* locations (where a missed object would matter) to decide where to re-perceive; verifies pivots; judged by signed decision value under measured budgets. **Closest; must be cited and contrasted.** |
| Object Importance Estimation via Counterfactual Reasoning (arXiv 2312.02467); risk-object identification (Who Make Drivers Stop, RiskBench) | counterfactual removal to rank object importance | explanation/importance of existing objects, not compute allocation |
| Periphery-Fovea driving model (arXiv 1903.09950) | fovea chosen by predicted *human* attention for end-to-end control | decision sensitivity of the actual planner, GT-free at run time; harm accounting |
| LZU (CVPR 2023), FOVEA (ICCV 2021), Learning When and Where to Zoom (CVPR 2020), Dynamic Zoom-in (CVPR 2018), policy-based foveated imaging (2026) | zoom/foveate by saliency, object priors or RL on accuracy | saliency = decision sensitivity; objective = decision value; D41 plugs the DSF into LZU/FOVEA-style warps |
| VeteranAD / Perception-in-Plan (2025), DriveTeach-VLA (ECCV 2026) | trajectory-guided perception queries / prompts in E2E models | architectural coupling for accuracy, not compute allocation or signed value |
| EMPERROR (2024) | generative perception-error model to stress-test planners | offline testing; the DSF probes at run time to allocate compute |
| EneAD (2025), Think-as-Needed (2026), Chanakya (NeurIPS 2023) | adaptive perception by NDS/mAP/streaming AP | objective is perception accuracy |

## 9. Candidate papers (as of 00:30, 2026-09-27) and what each still needs

**A. Speculative Perception (main CVPR candidate).** *Draft cheap, verify what the decision depends on.* Cheap
perception drafts the scene and the decision; the planner's decision-sensitivity field and the draft decision's
pivots define a few crops where the high-fidelity model runs (verification), with an optional "decision-lossless"
property borrowed from speculative decoding: every decision change with a single sufficient element (96–99% of them,
D32) is caught. Components: DSF by probing (D4), discovery crop (D3), pivot verification/refinement (D2/D9/D40),
temporal carry (D35), a scheduler (D33), measured on Jetson.
*Needs:* D47 agreement/value numbers on KITTI and nuScenes; real-crop confirmation (D3-real, running); crop packing
for milliseconds (D54); a learned asymmetric verifier and a crop-level range refiner (GPU); PDM-Closed/nuPlan; ideally
a learned object-list planner (PLUTO / Diffusion Planner) with gradient DSF.

**B. Decision churn in perception upgrades (second paper / workshop).** A decision-level negative-flip analysis of
detector swaps, resolution changes and (to add) quantisation/pruning across consumers (D5), and decision-congruent
fine-tuning (D25). *Needs:* more model pairs (INT8, YOLOv8m/l, RT-DETR variants), a congruent-training method on GPU.

**C. Planner saliency (longer term).** The DSF as a planner-grounded saliency map: compare with human gaze (D23), use it
as the threat model for phantom attacks (D38), amortise it for camera-input end-to-end planners via counterfactual
image edits (D30).

**Seven-week plan for A (deadline 2026-11-16).**
1. Week 1: finish the exploratory runs (nuScenes D47, real crops); freeze the formal definitions (DSF, pivot, the
   decision-lossless proposition and its assumptions); write the pre-registration for the main experiments.
2. Weeks 2–3 (GPU): train the crop verifier (existence, asymmetric) and range refiner on KITTI/nuScenes 3D labels;
   build crop-packing TensorRT engines; add PDM-Closed on nuPlan with probing.
3. Weeks 4–5: main experiments on the frozen DEEP splits with measured latency/energy; baselines: DEEP allocators,
   uncertainty/criticality, FOVEA/LZU-style static saliency, relevance filtering (arXiv 2606.10688), periodic keyframes.
4. Week 6: ablations (discovery only / verification only / no carry / no feedback), sensitivity, failure analysis.
5. Week 7: writing, figures, supplementary; anonymised DEEP citation.

## 10. A formal core for paper A: when is speculative perception decision-exact?

**Setting.** Scenes are finite sets of objects with geometry. Cheap scene C, full scene F, difference elements E
(additions, removals, geometry moves). A crop set R ⊂ image; the verify operator replaces C by F inside R:
`C ⊕_R F`. Locality assumption (L): running the full model on R returns exactly the full-frame detections whose centre
lies in R (holds for the proxy; measured for real crops by D3-real).

**Max-composable decisions.** A decision rule is *max-composable* if π(S) = g(max_{o∈S} φ(o)) for a per-object
demand φ and a monotone quantiser g — the braking controller (φ = required deceleration), IDM (φ = demand of the most
constraining leader), and the IDM-proposal stage of PDM-Closed are of this form.

**Proposition (sketch).** Let π be max-composable, and let R contain (i) every location u where inserting an object
with demand φ ≥ the draft's next threshold is possible (the insertion-sensitive DSF region for φ), and (ii) the box of
every draft pivot (objects attaining the max). Then under (L), π(C ⊕_R F) = π(F).
*Idea of proof.* Outside R, no full-only object can exceed the draft's decisive demand (else its location would be
insertion-sensitive), and no cheap-only object outside R attains the max (else it would be a pivot). Inside R the
scenes coincide with F. So the maximum demand, hence the action, equals F's. For general (non-max-composable) rules,
exactness holds when decision changes are *1-sensitive* (a single element suffices, D32: 96–99% of observed changes),
up to the probe grid's resolution; the iterated second round covers chains of two.

**Consequences.** (1) An allocator that runs the full model on R is as good as always running it, with pixel cost
|R|/|image| (D3: 2.7% on KITTI); (2) with pivot *verification* (existence + range refinement) on top, it can be
strictly better than the full model, because F's own phantoms and mis-ranges inside R are then corrected (D2: escalation
+ verification exceeds ΣV⁺). This turns DEEP's negative finding (allocators cannot predict harm) into a design:
do not predict where escalation helps — make escalation cheap enough to do where the decision is sensitive, and check
what the decision rests on.

## 11. State of the evidence at 02:45, 2026-09-27 (read this first tomorrow)

**What holds (KITTI, test split and all units):** speculative perception — cheap draft, full-resolution detector only
on the DSF discovery crop and the pivot crops, learned range refiner on the pivots — beats uniform full-frame
escalation on all four monocular KITTI entries (brake 0.134 vs 0.111 test; traj +0.095 vs −0.024 test) at ~5% of the
full pass's pixels; with real crops (no proxy) braking reaches 0.166 vs 0.111 on test.

**What is partial (nuScenes):** real speculative crops recover 69% (all) / 43% (test) of uniform escalation's value
at ~7% of the pixels; discovery alone fails there; the box-feature refiner does not transfer (needs an image-level
range head); the ensemble-sensitivity allocator (D62) is training-free, significantly better than random on KITTI and
significantly better than DEEP's learned gate on KITTI mono; on nuScenes its point estimates are the best of any
allocator but its bootstrap intervals include zero (underpowered test split).

**What must be fixed before any efficiency claim:** at YOLOv8s scale on Xavier, a crop pass costs ~53% of a full pass
in milliseconds (2.02 vs 3.83 ms), so the gain today is decision value per *pixel*, not per millisecond. Options:
heavier/higher-resolution full modes (e.g. 1280 px or a larger backbone, where full passes dominate), crop packing
across frames or cameras, energy accounting, or an honest framing as "decision quality at equal compute" rather than
"compute savings". This is the main open risk of paper A.

**Next concrete steps (GPU):** (1) image-level pivot verifier/refiner (existence with hard negatives, metric range);
(2) regime-adaptive scheduler combining D62 ensemble scores, discovery crops and pivot refinement; (3) a heavier full
mode so that crops save milliseconds; (4) nuPlan/NAVSIM with PDM-Closed to report standard driving scores; (5) the
decision-exactness proposition written out with its assumptions (max-composable rules, locality, 1-sensitivity).

**Update 02:45.** The pre-escalation gate for paper A now has a candidate that holds on both datasets: a ridge mixture
of the training-free ensemble-sensitivity components (D62b) is significantly better than random on both nuScenes
braking cells (0.482 [0.236, 0.674]; 0.453 [0.089, 0.738]) and matches DEEP's best learned allocators in point
estimate. Unattended runs after this session (`scripts/overnight_queue2.sh`): speculative crops at the 960 scale with
288×960 vs 96×288 engine timings (the milliseconds question), and real speculative crops under reference geometry;
their tables land in `docs/overnight_results.md`.
