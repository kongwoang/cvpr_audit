# Directions log — method ideas built on decision value

One entry per direction: hypothesis, the quickest test that could kill it, result, verdict. Tests run on DEEP's cached
detections and consumers (no detector re-run) unless stated. "Pre-registered" entries follow a written protocol;
"exploratory" entries are post hoc and only rank directions for further work.

Shares are of the all-cheap loss `L_c` of the cell. `G_full` = uniform escalation; `G_max` = sum of positive V (every
helpful escalation, no harmful one).

---

## D1. Escalate-then-arbitrate with cheap post-escalation features — pre-registered, **NO-GO**

- **Hypothesis.** After escalation, the consumer is run on both outputs, and a small classifier on detection/lift
  features of the pivotal element separates helpful from harmful decision changes.
- **Protocol.** `docs/phase0h_prereg.md`; code `scripts/01_flip_tables.py`, `scripts/02_arbiter.py`;
  gate V1 (recomputed losses equal DEEP's to 1e-15) passed on all four tables.
- **Result (test split).**

| cell | G_full | G_arb | G_max | capture | arb − full, 95% CI |
|---|---|---|---|---|---|
| KITTI mono brake | 0.111 | **0.157** | 0.279 | 0.27 | [0.014, 0.105] |
| KITTI oracle brake | 0.269 | **0.327** | 0.386 | 0.50 | [0.017, 0.154] |
| KITTI mono traj | −0.024 | 0.032 | 0.053 | 0.73 | [−0.004, 0.126] |
| KITTI oracle traj | 0.519 | 0.519 | 0.521 | — | no headroom |
| nuScenes mono brake | 0.165 | 0.120 | 0.249 | −0.53 | [−0.176, 0.084] |
| nuScenes oracle brake | 0.142 | 0.173 | 0.192 | 0.63 | [−0.012, 0.113] (25 affected) |
| nuScenes planner ×4 | ≈0.00–0.03 | no gain | 0.03–0.08 | ≤ 0.12 | all include 0 |

  C1 4/10, C2 2/10, C3 not met (pooled arb − rule 0.015 [−0.001, 0.039]) → **NO-GO**.
- **What it taught.** (i) Harm *is* recognisable after escalation where data are plentiful and the consumer is
  object-driven (both KITTI braking cells, CI > 0; +41% and +22% over uniform escalation). (ii) It does not transfer
  to nuScenes (57 and 25 affected test inputs) or to the learned planner, whose value is not carried by one object.
  (iii) An *existence* oracle verifier of the pivot captures 0.87–0.98 of the headroom in KITTI mono traj and
  nuScenes oracle brake but ~0 in the monocular braking cells, where the pivot is a real object with a wrong range
  (non-test KITTI mono traj: 119 of 228 harmful flips have a correct, real pivot). Harm is two things — phantom
  pivots and mis-ranged pivots — and a verifier must handle both.

---

## D2. Verify the decision's pivot instead of escalating the frame — exploratory (oracle headroom)

- **Hypothesis.** The decision hinges on one or two objects. Spending compute on *those objects* (is it real? what
  is its range?) buys more decision value per unit of compute than re-running the whole frame at high fidelity.
- **Test.** `scripts/03_headroom.py`, `scripts/04_headroom_summary.py`. Pivot = the object whose removal alone
  changes the decision. An oracle verifier removes it if it matches no reference box, otherwise gives it the
  reference geometry; up to K pivots in turn. A noisy verifier (existence error `e`, range noise `s`) measures the
  accuracy a learned verifier would need. All units; shares of the all-cheap loss.
- **Result so far (nuScenes mono brake).** Uniform escalation 0.165; oracle frame escalation at 20% 0.278;
  **cheap + verify 1 pivot, no escalation: 0.209 at 0.27 verifications per frame**; escalation + 1 pivot 0.336.
  Noisy verifier: e = s = 0.05 keeps it (0.247 / 0.298), e = s = 0.10 halves it (0.160 / 0.217), 0.20 destroys the
  cheap-side gain (−0.032). (KITTI and nuScenes-oracle rows: see below when the runs finish.)
- **Reading.** Object-level compute on the pivot beats frame-level escalation at a fraction of the pixels, but only
  with a verifier whose existence error is ≲ 5–10% on pivotal objects. The verifier is a vision problem (a crop-level
  existence + metric-range specialist), trainable on GPU from KITTI/nuScenes 3D labels.

---

## D4. Planner probing as a pre-escalation value-of-information score — exploratory

- **Hypothesis.** Without running the full mode, the consumer itself says where an unseen object would change the
  decision: insert a hypothetical static obstacle at range r in the corridor of the cheap scene and see if the
  decision flips. Weighted by the full mode's recall gain at r (training units only), this is a first-order expected
  value of information.
- **Test.** `scripts/10_voi_probe.py`; nDG at a 20% selection budget on DEEP's frozen test split, DEEP's scorer.
- **Result (KITTI test).**

| cell | random | ego speed | **probe VOI (no training)** | DEEP gate GBM | DEEP router MLP | DEEP router GBM |
|---|---|---|---|---|---|---|
| KITTI mono brake | 0.079 | 0.415 | **0.436** | 0.158 | 0.172 | 0.215 |
| KITTI oracle brake | 0.139 | 0.584 | **0.611** | 0.226 | 0.313 | 0.360 |
| KITTI oracle traj | 0.199 | 0.795 | **0.835** (×speed: 0.880) | 0.538 | 0.422 | 0.484 |
| KITTI mono traj | −0.091 | −0.691 | −0.149 | 0.184 | 0.127 | 0.059 |

- **Reading.** A training-free, planner-derived score beats every learned allocator of DEEP on three of four KITTI
  cells and edges ego speed, the trivial baseline that dominates KITTI. It fails where escalation is net harmful
  (mono traj), as a benefit-only score must. nuScenes, where ego speed is weak, is the real test (running).

---

## D5. Decision regressions of perception *upgrades* (motivation-level phenomenon) — exploratory

- **Question.** Is signed decision value specific to cheap-to-full *resolution* escalation, or does any perception
  upgrade — a stronger model family, a larger input — regress the downstream decision on many inputs?
- **Test.** `scripts/11_multimode.py`, `scripts/14_multimode_analysis.py`; KITTI, all 8,008 frames, eight cached
  modes, DEEP's consumers and shared history, one operating threshold (0.25) for every mode.
- **Result (mono geometry).**

| upgrade | misses | false positives | affected | harmed | harm ratio | net loss change (braking) |
|---|---|---|---|---|---|---|
| YOLOv8s 320→640 | −46% | +124% | 2,075 | 34% | 0.17 | −51.7% |
| YOLOv8s 512→640 | −17% | +18% | 1,470 | 43% | 0.62 | −9.5% |
| YOLOv8s 640→960 | −21% | +62% | 1,412 | 46% | 0.89 | −2.5% |
| **YOLOv8s-640 → RT-DETR-l-640** | **−27%** | +279% | 2,057 | **61%** | **1.09** | **+3.0% (worse)** |
| RT-DETR 480→640 | −13% | +5% | 1,947 | 42% | 0.72 | −8.6% |
| YOLOv8s-320 → RT-DETR-l-640 | −61% | +751% | 2,874 | 45% | 0.28 | −50.2% |

  Planner B shows the same pattern (harmed 39–54% of affected inputs in every pair).
- **Reading.** In every upgrade 39–61% of the inputs whose decision changes are made worse, and a detector that
  misses 27% fewer objects makes the braking loss *worse* overall. Caveat to close: RT-DETR at a shared threshold emits
  far more boxes; rerun at a calibrated operating point (queued). If it holds, "better detector, worse driving" is a
  headline motivation, and a *decision-congruent upgrade* (keep the old model's correct decisions, as positive-congruent
  training does for labels) is a method direction.

---

## D6. Decision-convergence cascade (anytime perception stopped by decision agreement) — exploratory, **NO-GO as is**

- **Hypothesis.** If a slightly better mode (384) gives the same decision as the cheap one (320), the full mode (640)
  rarely changes it; escalate only on disagreement.
- **Result (KITTI mono).** The signal is real: P(640 changes the decision | 320 and 384 disagree) = 0.66 against
  0.15 when they agree (braking); 0.72 against 0.035 (Planner B). But running 384 costs 6.1 ms, two thirds of a
  640 pass; at equal GPU cost the cascade realises 0.442 against 0.447 for random frame escalation (braking) and
  0.092 against 0.093 (Planner B).
- **Reading.** A second opinion is informative but must be far cheaper than a frame pass — which points back to a
  crop-level second look at the pivot (D2), not a second full frame.

### D4, nuScenes (test split, nDG at 20%)

| cell | random | ego speed | probe VOI | GBM (speed, #dets, conf, a_req, action) | **same + probe** | DEEP best |
|---|---|---|---|---|---|---|
| nuScenes mono brake | 0.132 | 0.148 | 0.071 | 0.487 | **0.524** | 0.482 (gate GBM) |
| nuScenes oracle brake | 0.148 | 0.208 | 0.177 | 0.472 | **0.628** | 0.421 (router GBM) |
| nuScenes planner ×4 | −0.01–0.09 | ≈0 | ≈0 | ≈0 | ≈0 | 0.03–0.13 |

- **Reading.** Alone, the probe does not carry nuScenes (the full mode's recall gain there is small and flat in range),
  but as one feature of a five-feature gate that also sees the cheap decision state (required deceleration, action)
  it gives the best allocator of any tested on both braking cells. The *decision-state* features themselves (0.487,
  0.472) already match or beat DEEP's gates. Nothing predicts the learned planner's value (as in DEEP). Single split,
  no interval yet — to be bootstrapped before any claim.

### D2 caveat (nuScenes)

The oracle verifier calls a detection real if it overlaps a reference box at IoU ≥ 0.5. nuScenes reference boxes are
projections of 3D boxes and are loose; in the oracle-geometry cell, verification therefore deletes real objects and
the "cheap + verify 1 pivot" row goes to −0.113. The nuScenes D2 numbers are not trustworthy at IoU 0.5; a rerun with
IoU 0.3 is running. KITTI's 2D-annotated boxes do not have this problem.

---

## D12. Anatomy of harmful escalations: what kind of pivot, how far, how persistent — exploratory (analysis)

- **Test.** Flip tables of D1, all units (`results/final/d12_pivot_anatomy.csv`).
- **Result.** A pivot the full mode *adds* at **< 15 m** is harmful in 83% (KITTI mono brake), 77% (KITTI oracle
  brake) and 100% (both nuScenes brake cells) of cases: the cheap mode already sees near objects (recall ≈ 0.8 within
  10 m), so a near object only the full mode sees is usually a phantom or mis-lift. Added pivots at 15–45 m are harmed
  in 23–48%. A pivot also present in the previous cheap frame is harmed less often (KITTI oracle brake 0.17 against
  0.30 for a new one), a modest effect. Moves (range changes of matched objects) carry little net value in KITTI and
  most of it in nuScenes mono.
- **Reading.** Harm has a physical signature — *resolution-inconsistent* evidence — which motivates D14.

## D14. Scale-consistency verification — exploratory (test split, fitted on non-test)

- **Idea.** A real object of apparent height h is missed at low resolution with a probability that falls fast with h.
  A full-only detection that the cheap mode *should* have seen is suspect. Table E[V | added pivot, height bin] on
  non-test flips; reject an added-pivot flip whose bin has negative mean value.
- **Result.** Height only: nuScenes mono brake 0.165 → **0.189**, nuScenes oracle brake 0.142 → **0.174** (the two
  cells where D1's learned arbiter failed), KITTI unchanged; adding a confidence dimension overfits (KITTI mono brake
  0.111 → 0.059). `results/final/d14_scale_consistency.csv`.
- **Reading.** A physics prior, not a learned classifier, is what transfers to the small nuScenes cells. It belongs
  inside the verifier (as a feature or a prior) rather than as a method on its own.

## D16. Is the decision-sensitivity field consumer-specific? — exploratory

- **Test.** KITTI test: rank frames for Planner B by the *braking* probe, and vice versa.
- **Result.** The two consumers' probe scores correlate at 0.89. The braking probe ranks Planner B's valuable frames
  as well as or better than Planner B's own probe (oracle traj 0.894 against 0.835; mono traj +0.177 against
  −0.149). Recall-gain weighting adds nothing on KITTI (`reach` ≈ `voi`).
- **Reading.** Consistent with DEEP's finding that valuable-input rankings transfer across similar planners; the field
  measures a generic *decision reach* (how far ahead, at this speed and with these obstacles, an unseen object would
  change what the car does). Good for a method (one field serves several consumers); it also means the field is close
  to a function of speed and the nearest obstacles — a baseline the paper must beat explicitly.

## D19. View-level allocation for multi-camera stacks — design only

- **Idea.** In a 6-camera BEV stack, the DSF decides which *views* get high resolution or a heavier backbone (front
  when cruising, the side views when merging or turning).
- **Status.** Not testable locally: the nuScenes blob here holds CAM_FRONT only (3,376 frames). Needs the full
  camera set and a BEV detector on the GPU server. Strong CVPR fit if the frame/region results hold.

## D22. Flip feedback: after an escalation, whether the decision changed is free, online supervision — exploratory

- **Observation.** Decision changes are sticky in time. KITTI (10 Hz): P(flip at t | flip at t−1) = 0.59–0.66
  against 0.02–0.15 otherwise; nuScenes (2 Hz): 0.29–0.32 against 0.03–0.06. As an (optimistic, full-information)
  allocator, "previous frame flipped" reaches nDG@20 0.298 / 0.333 on nuScenes mono / oracle brake against 0.148 /
  0.208 for ego speed.
- **Causal test.** Only escalated frames reveal a flip. Policy: escalate while the last escalated frame (within 3
  KITTI frames / 1 nuScenes frame) flipped, else explore with probability ε; token bucket at 20%. Realised gain at
  the realised rate is 1.3–2.3× random on KITTI (except mono traj, where escalation is net harmful) and 1.0–1.3× on
  nuScenes (`results/final/d22_flip_feedback_causal.csv`).
- **Reading.** A self-supervised, ground-truth-free online signal with moderate value; a natural "exploit" term for
  a scheduler (bandit view of escalation), not a paper by itself.

## D15. A Blackwell-consistent consumer (Bayes action under calibrated existence) — exploratory, **NO-GO**

- **Idea.** Harm arises because a fixed rule consumes the full output as truth; a consumer that takes the expected
  loss under calibrated existence probabilities (conf × apparent-height bins, per mode, fitted on training units)
  should not be hurt by extra, uncertain detections.
- **Result (KITTI brake).** All-cheap loss −1.0% / −0.6%, all-full −1.1% / −1.0% relative to the threshold rule;
  harmed share of affected inputs 0.342 → 0.341 (mono, all units), 0.246 → 0.234 (oracle).
- **Reading.** The harm is not the planner ignoring confidence: the harmful detections are confident and wrong, or
  real and mis-ranged (P(real) is only 0.50 even at conf ≥ 0.7 for boxes under 15 px). Existence calibration cannot
  fix it; per-object *evidence* (a second look, geometry) can.

---

## Design-only directions (not testable on this machine today; each with the test that would decide it)

**D23. Planner saliency vs human gaze.** The DSF is a planner-grounded saliency map. Compare it with driver-attention
datasets (BDD-A, DR(eye)VE) and ask whether foveating by human gaze or by the DSF buys more decision value per pixel.
*Kill test:* on BDD-A frames with a modular stack, nDG of gaze-guided vs DSF-guided crops at equal pixels. *CVPR fit:*
high (attention/saliency is core CVPR); *needs:* BDD-A download, a detector, a planner.

**D24. Decision-weighted distillation into the cheap model.** Train the cheap detector to match the full one only
where the DSF is high (DSF-weighted feature/box distillation), so the value of escalation is bought at training time.
*Kill test:* D3's region value is the ceiling at zero run-time cost; a fine-tuned YOLOv8s-320 must recover ≥ 30% of it
on the test split without raising FP-driven harm. *Needs:* GPU, 1–2 days of training.

**D25. Regression-free perception upgrades.** If D5 survives calibration: a decision-level negative-flip rate (NFR_dec)
for any perception change and a *decision-congruent* fine-tuning (positive-congruent training with pivot-weighted focal
distillation from the old model). *Kill test:* NFR_dec of YOLOv8s→RT-DETR at a calibrated operating point ≥ 30% of
affected inputs; congruent fine-tuning halves it at equal mAP. *Needs:* GPU.

**D28. DSF-targeted test-time augmentation.** Spend compute on flip/scale TTA only on crops around pivots — a second
look that needs no second model. *Kill test:* as D9 with TTA instead of the 640 detector.

**D30. Counterfactual image edits as the DSF of camera-input end-to-end planners.** For UniAD/VAD/SparseDrive-style
models there is no object list to probe; insert or remove objects in the image (inpainting / 3DGS scene editing) and
measure the planner's trajectory change, offline, to train an amortised DSF head that then allocates visual tokens or
resolution at run time. *Kill test:* on NAVSIM, token budgets allocated by the amortised DSF keep PDMS within 1 point
of full compute at ≤ 50% tokens, beating uniform and attention-based pruning. *CVPR fit:* very high (generative +
end-to-end driving + efficiency); *risk:* heavy; editing quality; needs the GPU server and NAVSIM data.

**D31. Language- or route-conditioned DSF for VLA driving models.** The decision depends on the instruction/route;
the DSF becomes command-conditioned (turn left → the left view matters). *Kill test:* conditioned vs unconditioned
DSF token allocation on a VLA benchmark. *Risk:* crowded area (AutoVLA-style adaptive reasoning).

### D5, calibrated operating point (closes the caveat)

RT-DETR thresholds set so that each RT-DETR mode emits as many boxes on the training units as the YOLOv8s mode it
replaces (0.42 / 0.424 / 0.444). `results/final/d5_upgrade_regressions_calibrated.csv`, KITTI mono, all units.

| upgrade (equal box count) | misses | FPs | affected | harmed | harm ratio | net loss change | if congruent (ΣV⁺) |
|---|---|---|---|---|---|---|---|
| YOLOv8s-640 → RT-DETR-l-640, braking | −5% | +6% | 1,499 (19% of frames) | 45% | 0.81 | −5.2% | −26.4% |
| YOLOv8s-512 → RT-DETR-l-480, Planner B | −4% | +4% | 316 | 53% | 1.68 | **+2.9% (worse)** | −4.3% |
| YOLOv8s-640 → RT-DETR-l-640, Planner B | −5% | +6% | 239 | 41% | 0.72 | −1.4% | −4.9% |

- **Reading.** At a fair operating point the headline "better detector, worse driving" holds only for some
  consumer/pair combinations. What holds everywhere is **decision churn**: swapping one good detector for a slightly
  better one changes the decision on up to a fifth of all frames, and 41–53% of those changes are for the worse; the
  net gain is a fifth of what a decision-congruent upgrade would keep. This is a clean motivation for D25 (congruent
  upgrades) and for pivot verification at deployment (D2/D9), and it generalises DEEP's finding from compute
  escalation to *any* perception change.

---

## D32. One escalation changes many objects; the decision hinges on one — foundational observation

- **Test.** Flip tables, all units (`results/final/d32_single_pivot.csv`).
- **Result.** In 96.0–99.4% of decision changes a *single* element of the cheap→full difference is sufficient to
  reproduce the full decision, and it carries 94.2–99.6% of |V|; 84–97% of changes have an element that is both
  sufficient and necessary. Meanwhile an escalation changes 3.6–7.4 elements of the perceived scene on average.
- **Reading.** The natural unit of perception compute for a decision is the pivotal object, not the frame. This is
  the empirical backbone of D2/D9 and of the object-level part of the DSF paper: frame escalation pays for ~5–7
  changes to buy one.

**D36. Decision-aware data curation (what to label next).** Frames whose cheap/full or model-A/model-B decisions
disagree harmfully are the frames the detector must get right. Rank unlabeled frames by probed decision sensitivity ×
cross-model disagreement for annotation or pseudo-labelling. *Kill test:* fine-tune YOLOv8s on N frames chosen this
way vs uncertainty-based active learning at equal N; compare decision loss, not mAP. *CVPR fit:* moderate; *needs:* GPU.

**D37. Per-frame selection between two equal-cost detectors.** D5 shows a large *oracle* headroom between detectors of
similar quality: choosing per frame between YOLOv8s-640 and RT-DETR-l-640 (equal box counts) would cut braking loss by
26.4% against 5.2% for swapping wholesale. With both run, a disagreement between their decisions localises a pivot
(D32), which a crop-level second look (D9) can adjudicate. *Kill test:* D1's arbiter on the Y640/R640 pair, where the
two branches are symmetric and neither is "cheap". *CVPR fit:* moderate (ensembling is not new; the decision-level
arbitration is).

**D38. The DSF as a threat model for phantom attacks.** Physical/adversarial phantom objects (projected, patched,
spoofed) only matter where they change the decision; the DSF says where, and pivot verification is the defence.
*Kill test:* insert synthetic phantoms (copy-pasted objects) inside vs outside the DSF on KITTI; measure decision harm
with and without pivot verification. *CVPR fit:* good for the adversarial-robustness track; a possible second paper.

### D2, nuScenes mono with a verifier criterion of IoU ≥ 0.3 (addresses the caveat)

All units: uniform escalation 0.165; oracle frame escalation @20% 0.278; **cheap + verify 1 pivot 0.309**
(0.27 verifications per frame, no escalation); 2 pivots 0.327; escalation + 1 pivot 0.389; every cheap detection
verified 0.422. Noisy verifier (existence error e, range noise s): e = s = 0.05 → 0.241; 0.10 → 0.153; 0.20 → 0.060.
Test split only (24 scenes, small loss totals): 0.214 against 0.165 uniform and 0.249 oracle@20; the noisy rows are
unstable there (−0.03 at 0.05). **Reading:** with a good verifier, checking about one object in four frames beats
the best possible frame-level escalation policy on nuScenes; the verifier's accuracy on *pivotal* objects is the
single number the method lives or dies by (target ≤ 5% existence error, ≤ 5–10% range error).

### D2, KITTI (all units; KITTI's 2D boxes make the oracle verifier reliable)

| | KITTI mono brake | KITTI mono traj | KITTI oracle brake | KITTI oracle traj |
|---|---|---|---|---|
| uniform escalation | 0.517 | 0.128 | 0.679 | 0.683 |
| oracle frame escalation @20% (= ΣV⁺ here) | 0.621 | 0.178 | 0.738 | 0.686 |
| **cheap + verify 1 pivot** (0.39 verif./frame) | 0.110 | **0.542** | 0.004 | −0.032 |
| **full + verify 1 pivot** | **0.672** | **0.807** | 0.724 | 0.672 |
| full + verify 2 pivots | 0.684 | 0.844 | 0.721 | 0.652 |
| cheap + 1 pivot, noisy e = s = 0.05 | 0.061 | 0.397 | −0.063 | −0.395 |
| full + 1 pivot, noisy e = s = 0.05 | 0.597 | 0.609 | 0.623 | 0.183 |

- **Reading.** Verification and escalation buy different things. Where the cheap mode is *blind* (KITTI braking:
  empty cheap frames carry most of the value), verifying what it saw cannot help — escalation (or a region search) is
  needed to find the miss. Where the harm is *mis-ranging* (KITTI mono traj), verifying one object per ~2.5 frames
  is worth 3× the best possible frame-escalation policy. Escalation + pivot verification exceeds the arbitration
  ceiling ΣV⁺ in every monocular cell (0.672 > 0.621; 0.807 > 0.178; nuScenes 0.389 > 0.278): the full output plus a
  checked pivot is better than either output alone. **Symmetric verifier errors are dangerous**: calling a real object
  a phantom 5% of the time turns KITTI oracle traj from −0.03 to −0.40 (collisions). A deployable verifier must be
  asymmetric — remove only near-certain phantoms, refine range freely.

## D35. Amortise escalations over time: escalate periodically, carry the full mode's extra objects — exploratory

- **Test.** `scripts/17_temporal_reuse.py`: escalate every k-th frame; on the frames in between, add the last
  escalated frame's full-only objects that the cheap mode still does not see, their range reduced by the ego's travel.
- **Result (KITTI mono brake).** All units: every 2nd frame + carry 0.464 (all-full 0.517; random at 1/2 0.258);
  every 5th + carry 0.376 (random at 1/5 0.103; oracle @20% 0.621). Test split: every 2nd + carry 0.112, every 5th +
  carry 0.107 — both at the level of escalating **every** frame (0.111), against 0.022 for random at 1/5.
- **Reading.** At 10 Hz, most of the value of an escalation persists for several frames; a static per-frame allocator
  (DEEP's setting) leaves it on the table. Keyframe detection with propagation is well known in video detection, so
  this is not a contribution by itself; it is a **mandatory baseline and scheduler component** — decision-triggered
  keyframes (escalate when the DSF changes or a carried pivot is about to matter) is the novel version.
- **KITTI oracle brake.** All units: every 5th + carry 0.489 (72% of all-full 0.679; random at 1/5 0.136); test:
  0.176 (65% of 0.269; random 0.054). The carry keeps most of the value in both geometries.

## D39. Trigger range refinement by the disagreement of the two monocular range cues — exploratory, **NO-GO**

- **Idea.** The lift fuses a ground-plane cue and a class-height cue; if the decision changes between them, the
  pivot's range is decision-ambiguous — refine only then (a free trigger for the refiner).
- **Result (KITTI mono brake).** The cues flip the decision on 3.8% of frames; refining only those captures 0.018 of
  the all-cheap loss against 0.109 when every frame's pivot is refined (oracle range, existence untouched; test split
  0.016 against **0.166**, which itself exceeds uniform escalation, 0.111).
- **Reading.** The mono cues agree while both being wrong; cue disagreement is not where the decision-relevant
  range errors are. The refiner must be a learned, better range estimate (crop-level depth), not a consistency check.

---

## D3. DSF-guided region re-perception (decision-sensitivity foveation) — exploratory, **strongest lead so far**

- **Test.** `scripts/12_region_proxy.py`. Probe the consumer on a (range 10–80 m × lateral ±6 m) grid of hypothetical
  obstacles on the cheap scene (mono geometry, no reference state); project the sensitive cells into the image; the
  crop is their bounding box. Proxy for running the 640 detector on the crop: the cached full-frame 640 detections
  whose centre lies in the crop replace the cheap ones there. Cost = crop pixels / image pixels.
- **Result (KITTI mono brake).** Mean crop **2.7% of the image** (a crop in 48% of frames). All units: **DSF region
  0.502** against uniform frame escalation 0.517 (100% of pixels), the frame oracle at the same pixel budget 0.397,
  a static horizon crop of equal area 0.120, random frames 0.014. Test split: 0.098 against 0.111, 0.120, 0.044,
  0.003. Harmed share of affected inputs 0.245 against 0.342 for frame escalation. KITTI mono traj: 0.109 against
  0.128 (all units; the static crop is harmful, −0.015); on its test split escalation itself is net harmful.
- **Reading.** Where the decision is sensitive is a tiny part of the image, and re-perceiving only that part keeps
  ~90–97% of the value of re-perceiving everything, beats the best possible *frame* selection at the same pixel budget,
  and beats an equal-area static crop 4×. Caveats: (i) proxy — a real crop run changes detections (context, scale);
  (ii) pixels are not milliseconds on a GPU (a 2.7% crop does not cost 2.7% of a pass; batching crops across frames or
  a tiled engine is needed); (iii) KITTI only so far (nuScenes queued).

## D33. How complementary are the actions? Oracle mix under one pixel budget — exploratory

- **Test.** `scripts/21_action_mix.py`; per frame one of {none, pivot second look, DSF region, frame escalation}.
- **Result (KITTI mono brake).** At a 5% pixel budget the oracle mix reaches **0.632** (all units) — the full
  arbitration ceiling ΣV⁺ is 0.621 — against 0.477 for frames only, 0.536 regions only, 0.086 second looks only.
  Test: 0.300 against 0.168 frames only. Planner B: regions alone already reach the frame ceiling.
- **Reading.** A scheduler that can choose the *granularity* per frame has large headroom over frame-level allocation
  at tiny budgets; the achievable share depends on value estimators (D4 probe, D22 flip feedback).

## D9. A real second look at the pivot (640 detector on a crop around it) — exploratory

- **Result.** nuScenes mono brake: **0.127 (all units) at 2.2% of the pixels** (uniform escalation 0.165 at 100%;
  random frames at equal pixels 0.004); test 0.058. KITTI: 0.036 (brake) and ≈0 (traj). Existence-only (keep the cheap
  geometry, drop unconfirmed pivots): ≈0 everywhere — the value comes from the *better box at 640* (hence a better lift),
  not from deleting phantoms.
- **Verifier accuracy (KITTI, `results/final/d2_verifier_accuracy.csv`).** Only 9.6% of cheap-decision pivots are
  phantoms (26% of other detections). Re-detection at 640/960 or with RT-DETR almost never calls a real pivot a phantom
  (≤ 0.5%) but confirms 46–82% of phantoms: phantoms are *systematic* across resolutions and model families (some may
  be unlabeled real objects in DontCare regions). **Re-detection is not a verifier**; existence needs other evidence
  (temporal/geometric consistency, a classifier trained on the detectors' own hard negatives).

## D7. Stakes-proportional evidence thresholds — exploratory, **NO-GO**

- **Result (KITTI mono, test).** Raising the confidence a pivot needs lowers the value monotonically (0.111 at 0.25 →
  0.082 at the non-test-selected 0.45 → 0.024 at 0.7); Planner B likewise. Low-confidence pivots are more often real
  recovered misses than phantoms; confidence does not separate help from harm.
- **D7, nuScenes mono (test).** Here the rule helps: 0.165 → **0.189** with t_pivot = 0.40 chosen on non-test (the
  same gain as D14's size prior). Verdict revised to **mixed**: stakes-proportional evidence helps where harm is
  phantom-driven (nuScenes) and hurts where low-confidence pivots are mostly real recovered misses (KITTI). A
  per-dataset threshold is a weak, brittle method; the information belongs in a learned verifier.

**D42. Decision-aware coding for offload/V2X.** Allocate bitrate per block by the DSF when frames are compressed for a
remote perception server or shared over V2X. *Prior art:* task-aware encoder control (CVPR 2024), block-level RL QP
control for task-aware video compression (RL-RC-DoT, CVPR 2025) — they optimise detection/tracking accuracy. *New
here:* signed decision value as the target and a planner-derived, GT-free saliency. *Kill test:* at equal bitrate,
DSF-guided QP maps vs task-aware saliency maps on DEEP's decision value. *Fit:* compression track; incremental unless
the harm (sign) story transfers.

**D43. DSF-driven sensor ROI readout.** High-resolution sensors can read out a region of interest at full resolution
and the rest binned (e.g. 200 MP dual-stream foveated sensors, policy-based foveated imaging 2026; event-based ROI
selection, arXiv 2609.17134). The DSF crop of D3 is exactly such an ROI, chosen by the planner. *Kill test:* D3 with
the crop read from a higher-resolution source than the full mode (KITTI full_960 cache as the "sensor") at the pixel
budget of the 640 frame. *Fit:* computational imaging; hardware-flavoured, strong story, needs a sensor for a demo.

**Prior art found for the design-only directions.** D36 (decision-aware data curation): ActiveAD, planning-oriented
active learning for end-to-end driving (arXiv 2403.02877) — selection by planning criteria exists; the decision-value
and cross-model-churn criterion would be the difference. D24/D25 (decision-weighted training): a safety-adapted loss
for pedestrian detection (arXiv 2402.02986) weights detection errors by criticality; DEEP already showed criticality
weighting does not track decision value, so a decision-value-weighted loss is distinct but must be compared with it.
Framing for the main paper: the DSF is a myopic **value of computation** (Russell & Wefald; rational metareasoning,
recently applied to LLM reasoning budgets, arXiv 2410.05563) for *perception* actions, computed by probing the planner.

**D44. Pivot tracks: verify once, trust for a while.** Combine D35 and D2: a pivot verified at t keeps its verified
status and refined range (propagated with ego motion) while it stays the pivot, so verification cost is paid per
*track*, not per frame. *Kill test:* D2's noisy-verifier numbers at 1/k of the verifications.

**D45. Decision-aware conformal ranges.** Calibrate a conformal interval on each pivot's range (split conformal on
training units, per range band) and let the planner act on the interval's worst case only for the pivot; target
*decision coverage* (the action is right with probability ≥ 1−α) rather than range coverage. *Prior art:* conformal
decision theory (Lekeufack et al.) and conformal safe planning with trajectory predictors. *Kill test:* shortfall-free
rate and phantom-braking rate against the lift and the oracle on KITTI/nuScenes braking. *Risk:* conservative intervals
re-create phantom braking; needs D40's refiner to shrink them.

**D46. Physical-consistency phantom tests as a zero-shot verifier.** Ground contact (box bottom vs lifted range),
apparent size vs class prior at the lifted range (D14), image motion vs ego motion (static-world parallax), and
presence in the previous frames — each a closed-form likelihood ratio; combined as naive Bayes on the pivot only.
*Kill test:* phantom-called-real rate at ≤ 0.5% real-called-phantom, against re-detection (D9: 46–82%).
- **D46 tested (KITTI test pivots, 1,405 pivots, 157 phantoms; `results/final/d46_physical_verifier.csv`).** A GBM on
  physical-consistency features (range-cue agreement, ground contact, size/aspect, presence in the 3 previous frames,
  entropy, class) catches 68–83% of phantom pivots but at 26–36% real pivots called phantom; a learned combination of
  three re-detections (YOLOv8s 640/960, RT-DETR-l 640) catches 61% at 4.6%; both combined 81% at 22%. Thresholds set
  in-sample for 0.5–5% false removals do not hold on test. **NO-GO as a zero-shot verifier**: the asymmetric operating
  point the method needs (≤ 0.5% real pivots removed) requires an image-level verifier trained on the detectors' own
  hard negatives (GPU), with cross-validated thresholds.
- **D3, nuScenes mono (does not replicate).** Mean crop 3.5% of the image; DSF region 0.060 (all units) / 0.006 (test)
  against uniform escalation 0.165 and an equal-area static horizon crop 0.096 / 0.067; harmed share 0.47 vs 0.40.
  On nuScenes the value is mostly *range corrections of objects the cheap mode already sees* (D12: moves and pivots),
  which an insertion-only discovery region does not target (and near cells, < 10 m, are excluded by design). Discovery
  alone is dataset-dependent: it carries KITTI (where the cheap mode is blind) and not nuScenes (where it mis-ranges).
  The speculative variant (D47), which adds pivot crops, is the test of whether the two halves together generalise.

---

## Cross-field idea bank (2026-09-27) — methods borrowed from unrelated fields

| id | source field | borrowed idea | mapping to decision value | status |
|---|---|---|---|---|
| D47 | LLM inference | **speculative decoding** (draft small, verify large, lossless) | cheap perception drafts the decision; the full model verifies only the DSF region and the pivots; "decision-lossless" when every flip has a single sufficient element (D32) | **testing** (`24_speculative.py`) |
| D48 | marketing / epidemiology | **uplift modelling / CATE** (T-learner) | V is the individual treatment effect of escalation; model J_cheap and J_full separately | testing (`25_allocator_crossfield.py`) |
| D49 | information retrieval | **learning to rank for top-k** | train the allocator on top-k membership, not on V regression | testing (proxy; no LambdaMART on this machine) |
| D50 | finance | **risk-averse allocation** (lower quantile) | rank by the 25% quantile of V to avoid escalations that may harm | testing |
| D51 | compilers | **program slicing** | the pivot set is the backward slice of the action over the perceived scene; a formal name and algorithm for D32 | framing |
| D52 | computer graphics | **level of detail by screen-space error** | choose each object's perception fidelity by its *decision-space* error (DSF) instead of pixel error | design |
| D53 | ecology | **marginal value theorem** (when to leave a patch) | stop spending compute on a frame/region when the marginal decision gain falls below the budget's average rate | design (anytime scheduler) |
| D54 | systems | **mosaic / canvas packing** | pack all DSF and pivot crops of a frame (or several frames) into one fixed-size engine input, turning D3's pixel savings into milliseconds | design; needed to close D3's pixels ≠ ms caveat |
| D55 | neuroscience | **active inference** (expected free energy = pragmatic + epistemic) | DSF = pragmatic value, recall gain / miss probability = epistemic value; the VOI decomposition of §7 is its first-order form | framing |
| D56 | clinical statistics | **decision curve analysis** (net benefit vs threshold) | report allocators by net benefit = benefit − w·harm across w, not one nDG | evaluation framing |
| D57 | finance | **real options / optimal stopping** | escalating now vs waiting (value persists, D35; flips are sticky, D22) as an option exercise problem | design |
| D58 | coding theory | **redundancy only where errors matter** | a *second cheap model* (different family) only inside the DSF region instead of a higher resolution | testing (R320 in the DSF crop) |
| D60 | safe control | **Hamilton–Jacobi / backward reachability** | the decision-sensitive region as the backward reachable tube of a collision within the horizon; a certified superset of the probed DSF | design |
| D61 | economics | **auctions / VCG for a shared resource** | several consumers (braking, lane change, parking, prediction) bid DSF values for one perception budget | design (multi-consumer stacks) |
| D62 | weather forecasting | **targeted observations by ensemble sensitivity** | sample plausible scenes (lift error, phantoms, misses) from the cheap scene; observe where the decision is unstable | **testing** (`26_ensemble_sensitivity.py`) |
| D63 | real-time systems | **earliest-deadline-first with value** | each pivot has a decision deadline (time until the planner must commit, from range and speed); schedule verification by value/deadline | design |
| D64 | active learning theory | **query-by-committee** | disagreement among a committee of cheap perceptions (models/augmentations) *about the decision* chooses where to spend | partly covered by D6/D37; design |

### D3-real — the DSF crop actually detected (closes the proxy caveat) — **headline result so far**

`scripts/22_real_crop.py`: YOLOv8s (same weights) run on the DSF crop of the original KITTI image at the full mode's
scale (640/1242), boxes shifted back, DEEP's ego-frame monocular lift, merged with the cheap detections outside the
crop; DEEP's consumers and shared history. Network pixels are counted after /32 padding against the 192×640 full pass.

| KITTI mono | uniform escalation (100% pixels) | proxy | **real crop (4.0% / 5.1% of the full pass's network pixels)** |
|---|---|---|---|
| brake, all units | 0.517 | 0.502 | **0.509** |
| brake, test | 0.111 | 0.098 | **0.129** |
| traj, all units | 0.128 | 0.109 | **0.135** |
| traj, test | −0.024 | −0.024 | **−0.013** |

**Reading.** Running the high-resolution detector only where the planner says an unseen object would change the
decision recovers 98.5% (all units) to 117% (test) of the decision value of running it on the whole frame, at ~4% of
its network pixels, and is *less* harmful than full escalation on the trajectory controller. Crops do slightly better
than the proxy (crop-level context does not hurt; fewer out-of-region false positives enter). Remaining caveats: KITTI
only (nuScenes needs the pivot half, D47); pixels → milliseconds needs crop packing/dynamic-shape engines (D54);
eager PyTorch was used for the crops (quality test, not a latency test).

### D48–D50 results — allocator learners from other fields (test split, nDG@20; `results/final/d48_allocator_crossfield.csv`)

| cell | direct V regression | uplift T-learner | sign × |V| | top-k membership | lower q25 | probe only |
|---|---|---|---|---|---|---|
| KITTI mono brake | 0.440 | **0.491** | 0.444 | 0.465 | 0.079 | 0.436 |
| KITTI mono traj | −0.020 | −0.267 | **0.116** | −0.261 | −0.017 | −0.149 |
| KITTI oracle brake | 0.600 | 0.604 | 0.595 | **0.639** | 0.139 | 0.611 |
| KITTI oracle traj | 0.848 | 0.700 | 0.709 | **0.931** | 0.199 | 0.835 |
| nuScenes mono brake | **0.524** | 0.412 | 0.516 | 0.231 | 0.132 | 0.071 |
| nuScenes oracle brake | **0.628** | 0.541 | 0.174 | 0.483 | 0.148 | 0.177 |

- **Reading.** Each borrowed learner wins somewhere — the uplift T-learner where the cheap loss itself is predictable
  (KITTI mono brake), the top-k objective where value is concentrated (KITTI oracle), the |V|-weighted sign where
  escalation is net harmful (KITTI mono traj) — but none everywhere, repeating DEEP's "no allocator wins every cell".
  The risk-averse quantile fails (V is zero on most inputs, so its quantiles carry no ranking). Worth keeping as
  **ablations/baselines** for the scheduler's value estimator, not as a contribution; the bigger lever is changing
  the *action* (crops, pivots), not the frame ranker.
- **D2, nuScenes oracle, IoU 0.3.** All units: cheap + verify 1 pivot 0.050; escalation + 1 pivot 0.186 (uniform
  escalation 0.140; frame oracle @20% 0.241); a noisy verifier (e = s = 0.05) turns it negative (−0.059). With reliable
  geometry, verification can only remove phantoms, a small share of the value; its benefit is the range half, which the
  oracle geometry has already supplied. Consistent with D32/D12: verification pays where the lift is the problem.
- **D9, nuScenes oracle.** Second look 0.023 (all) / 0.029 (test) at 1.9% of the pixels, against uniform escalation
  0.140 and 0.003 for random frames at equal pixels; K = 2: 0.032. With reference geometry the second look can only add
  or drop objects, and it buys ~15–20% of the frame value at ~2% of the pixels — ten times random, far below nuScenes
  mono (0.127), where the better 640 box fixes the lift.

### D3 on standard perception metrics (answers "does it look bad on existing benchmarks?") — `results/final/d3_standard_metrics.csv`

KITTI mono, all frames, proxy crops, IoU ≥ 0.5, boxes ≥ 10 px:

| population | recall cheap | **recall DSF-fused** | recall full | precision cheap | precision DSF-fused | precision full |
|---|---|---|---|---|---|---|
| every object | 0.475 | **0.536** | 0.717 | 0.755 | 0.737 | 0.676 |
| objects in the braking corridor ≤ 80 m | 0.530 | 0.644 | 0.767 | — | — | — |
| objects inside the DSF crop | 0.362 | 0.753 | 0.758 | 0.829 | 0.708 | 0.708 |

- **Reading.** On a whole-image detection benchmark the method closes only a quarter of the cheap→full recall gap
  (0.536 vs 0.717), while it keeps ~98% of the *decision* value (D3-real). This is by design (DEEP: perception gain and
  decision value disagree) but it means the paper cannot be sold on mAP; it must be evaluated on driving outcomes
  (nuPlan/NAVSIM scores with a published planner), with mAP reported as the honest trade-off and a periodic full-frame
  refresh (D35) as the knob for consumers that need the whole scene.

### D62 results — ensemble decision sensitivity (weather-forecast targeted observations), KITTI test, nDG@20

| cell | ego speed | probe VOI (D4) | ensemble: range noise | ensemble: phantoms | ensemble: misses | ensemble: all |
|---|---|---|---|---|---|---|
| KITTI mono brake | 0.415 | 0.423 | 0.009 | 0.030 | **0.439** | 0.386 |
| KITTI oracle brake | 0.584 | 0.599 | 0.016 | 0.059 | **0.611** | 0.532 |

- **Reading.** Only the *missed-object* part of the ensemble ranks frames (≈ the deterministic probe, slightly better);
  instability under range noise or phantom removal is not where escalation pays on KITTI, and mixing them in dilutes
  the score. A principled Monte-Carlo generalisation of D4 with no gain over it here; nuScenes (queued) is where range
  instability should matter.
- **D3, KITTI oracle geometry (proxy).** Brake: DSF region 0.658 (all) / 0.209 (test) against uniform 0.679 / 0.269,
  frame oracle at equal pixels 0.505 / 0.188, static crop 0.192 / 0.154; harmed share 0.13 vs 0.25. Traj: 0.628 /
  0.309 against 0.683 / 0.519 (the test split's traj value sits partly outside the DSF crop). With reference geometry
  the region still keeps 92–97% of the value on all units at ~2.7% of the pixels, and halves the harmed share.
- **D43 tested (proxy): the DSF crop read at 960 px scale instead of 640.** KITTI mono all units: brake 0.505 (640-crop
  0.502; full-frame 640 0.517); traj **0.141** (640-crop 0.109; full-frame 640 0.128) — the higher-resolution ROI beats
  escalating the whole frame to 640 on the trajectory controller at 2.25 × 3.4% ≈ 7.7% of the 640 frame's pixels.
  Test split: 0.096 / −0.026 (unchanged). Modest but positive: spending the saved pixels on *more* resolution inside
  the ROI is a knob worth keeping (sensor ROI readout, D43).
- **D35, nuScenes mono (2 Hz keyframes).** Carrying objects forward over 0.5 s gaps does not work: every 2nd + carry
  0.114 (all) vs no carry 0.122; every 5th + carry 0.042 (all) / −0.092 (test) — carried objects are stale after 1–2 s
  of ego and object motion. Temporal amortisation is a high-frame-rate effect (KITTI 10 Hz), not a 2 Hz one; a real
  stack runs at 10–30 Hz, so this favours KITTI-like conditions, but the nuScenes benchmark cannot show it.

## D40. A learned range refiner for the decision's pivot — exploratory, **strong (realistic, not oracle)**

- **Test.** `scripts/20_learned_refiner.py`. Refiner v0: gradient boosting on box-level features of the 640 px
  detection that confirms the cheap decision's pivot (a second look on a crop; box geometry, class, confidence, both
  lift cues), predicting log(true range / lifted range); fitted on reference-matched full detections of the **training
  units only** (14,605 boxes). At test time the pivot's range and lateral extent are rescaled; nothing is deleted.
- **Pivot range error (|log ratio|, test units).** Cheap lift 0.143, 640 lift 0.138, **refined 0.070**.
- **Decision value (share of the all-cheap loss).**

| cell | uniform escalation | frame oracle @20% | pivot with 640 geometry | **pivot, learned refiner (no escalation)** |
|---|---|---|---|---|
| KITTI mono traj, test | −0.024 | 0.053 | −0.038 | **+0.258** |
| KITTI mono traj, all units* | 0.128 | 0.178 | 0.003 | **0.390** |
| KITTI mono brake, test | 0.111 | 0.132 | 0.007 | 0.024 |

  (*all units include the refiner's training units.)
- **Reading.** Where harm comes from the lift, a small learned correction applied to *one object per frame* is worth
  ~5× the best frame-escalation policy on the held-out split and turns a cell where escalation hurts into a large gain.
  Where value comes from misses (braking), it cannot help — that is the discovery crop's job (D3-real). The two halves
  of speculative perception cover the two regimes found in D2/D12. The GPU version is an image-crop depth/3D head
  trained on KITTI/nuScenes 3D labels (and an existence head with hard negatives), which should beat this box-feature GBM.
- **D39/D2, nuScenes mono: refining the pivot's range (oracle range, nothing deleted).** 0.283 (all) / 0.181 (test)
  against uniform escalation 0.165 / 0.165 — range refinement of one object per frame beats escalating every frame on
  nuScenes too. The cue-ambiguity trigger captures 0.065 / 0.105 of it (fires on 6–9% of frames). The learned refiner
  (D40) on nuScenes is queued (`20_learned_refiner.py --dataset nuScenes`).

## D47 results — speculative perception (proxy crops), KITTI mono, probes from 10 m

| cell | pixels | decision = full's | … on frames where full flips | G speculative | G uniform escalation | frame oracle, equal pixels |
|---|---|---|---|---|---|---|
| brake, all units | 4.7% | **94.2%** | 79.2% | **0.529** | 0.517 | 0.470 |
| brake, test | 5.1% | 90.2% | 67.3% | **0.117** | 0.111 | 0.175 |
| traj, all units | 5.4% | **99.3%** | 92.0% | 0.111 | 0.128 | 0.178 |
| traj, test | 5.8% | 99.2% | 83.2% | −0.030 | −0.024 | 0.053 |

- **Reading.** "Draft cheap, verify what the decision depends on" reproduces the full mode's decision on 94–99% of
  frames at ~5% of its pixels and matches or exceeds its decision value on braking (its verification crops also
  replace the draft's own phantoms). Not lossless yet — misses below 10 m (excluded from the probe), the coarse probe
  grid and a canonical probe object, and multi-object interactions; the 5 m variant is running. On the trajectory
  controller, where escalation itself is harmful, faithfully reproducing the full decision reproduces its harm too —
  which is exactly where the learned refiner (D40, +0.258 on test) must sit inside the loop.

### D58 result — a second *cheap* model (RT-DETR-l at 320) inside the DSF crop instead of higher resolution (proxy)

KITTI mono: brake 0.445 (all) / 0.112 (test) against the 640 crop 0.502 / 0.098 and uniform escalation 0.517 / 0.111;
traj 0.121 / **+0.017** against −0.024 for uniform escalation (test). Harmed share rises on braking (0.39 vs 0.25 for
the 640 crop) because RT-DETR at a shared 0.25 threshold emits many more boxes. **Reading:** model diversity inside the
decision-sensitive region is a viable alternative to resolution and less harmful on the trajectory controller; with a
calibrated RT-DETR threshold it is a cheap "redundancy where it matters" variant (and D37's disagreement signal comes
for free when both are run in the crop).

### More directions (design; each could be a separate paper or a section)

| id | idea | why it is new | kill test | fit |
|---|---|---|---|---|
| D65 | **Lazy, query-driven perception** (databases: approximate query processing). The planner issues typed queries — "is region R free up to d?", "range of object o within ±ε?" — and a query optimiser picks the cheapest perception operator (crop, model, resolution, temporal reuse) that answers each at the needed precision. | reframes perception as answering the decision's queries, with speculative perception as one plan | cost to answer the braking/Planner B queries at a fixed decision error vs frame escalation | systems + vision |
| D66 | **Decision-aware self-supervision.** Pretrain region embeddings to separate content that changes the decision (probed DSF labels, free and unlimited) from content that does not. | a supervision signal from the planner, not from labels or pixels | linear probe of V / pivots from frozen features vs ImageNet/driving pretraining | representation learning |
| D67 | **Minimax DSF.** An adversary places the most harmful plausible miss/phantom; allocate compute to minimise worst-case decision regret. | robust counterpart of the expected-value DSF; links to D38 | worst-case shortfall under D38's phantom injections | robust AD |
| D68 | **Decision-aware copy-paste augmentation.** Paste objects into DSF-sensitive locations during detector training so the *cheap* model stops missing what matters. | augmentation targeted by the planner, not by class frequency | fine-tuned YOLOv8s-320: decision value of the cheap mode alone vs standard copy-paste | GPU, CVPR-friendly |
| D71 | **Decision-aware cooperative perception (V2X).** Request or send features only for the ego's DSF region, not for "confident" regions. | Where2comm (NeurIPS 2022) and successors pick what to share by perception confidence; the decision is never the criterion | OPV2V/V2X-Sim with a planner: decision value at equal bandwidth vs Where2comm | **high novelty, CVPR fit** |
| D72 | **DSF-steered LiDAR / adaptive scanning.** Concentrate points of a steerable (solid-state) LiDAR in the DSF region. | adaptive LiDAR sampling exists for reconstruction/detection accuracy, not decisions | simulate beam budgets on nuScenes LiDAR by subsampling outside the DSF | hardware-flavoured |
| D75 | **Decision-focused test-time adaptation.** Under shift (night, rain), adapt only on decision-relevant regions/objects (entropy minimisation weighted by the DSF). | TTA objectives are image- or detection-wide | nuScenes night subset: decision value after TTA vs standard TTA | CVPR TTA track |

---

## Status at 00:55, 2026-09-27 — runs still queued (results land in `docs/overnight_results.md`)

Running/queued without supervision (`scripts/queue_nusc.sh` for nuScenes, one at a time; `scripts/overnight_queue.sh`
for KITTI): D47 speculative KITTI (probes from 5 m) and nuScenes; D47+D40 speculative + learned refiner (KITTI);
D47-real with real crops on KITTI (full, pivots only, discovery only) and nuScenes (full, pivots only); D62 ensemble
sensitivity nuScenes; D40 learned refiner nuScenes; D3-real KITTI oracle; crop-engine timings (64×192, 96×320 vs the
full 640 engine). `scripts/collect_results.sh` gathers every final table into `docs/overnight_results.md` when the
queues finish; interpret them against the entries above (D3, D40, D47 are the ones that decide paper A).

### D47, probes from 5 m instead of 10 m (KITTI mono, proxy)

Agreement with the full decision rises to 96.7% (brake, all; 87.8% on flip frames) and **100% on the traj test split**
(lossless there), with G = 0.528 / 0.128 (brake all / test) — but pixels jump from ~5% to 22–26%, because near cells
project to large image regions. **Trade-off:** rmin = 10 m ≈ 5% pixels at 94–99% agreement; rmin = 5 m ≈ 23% pixels
at 97–100%.

**D76 (new, follows directly): scale-matched DSF crops.** Near cells are large in the image and the cheap mode already
resolves them; far cells are small and need the full scale. Detect each sensitive cell at the resolution its apparent
object size needs (near crops downsampled, far crops at 640/960 scale — a level-of-detail rule, D52), packed into one
canvas (D54). Expected: the 5 m variant's near-lossless agreement at close to the 10 m variant's cost. *Kill test:*
`24_speculative.py` with per-cell scale; agreement ≥ 97% at ≤ 8% of the full pass's network pixels.

---

## Overnight results (queues finished 01:41, 2026-09-27; interpreted 02:45) — raw tables in `docs/overnight_results.md`

### Correction: milliseconds, not pixels (D54)
trtexec FP16 GPU compute, median, same conditions: YOLOv8s full 192×640 **3.83 ms**, cheap 96×320 2.20 ms, crop canvas
64×192 **2.02 ms**, crop canvas 96×320 2.21 ms. At these sizes the Xavier GPU is overhead-dominated: a crop escalation
costs ~53% of a full-frame escalation, not the ~4% its pixel share suggests (the earlier "4.6× cheaper" used DEEP's
9.27 ms profile and is withdrawn). The pixel saving becomes a latency saving only with larger models or inputs
(1280 px, multi-camera, heavier backbones), batched crops across frames/cameras, or energy accounting — to be shown
before any efficiency claim. Decision-value claims below are per pixel, not per millisecond.

### D47 + D40 — speculative perception with the learned pivot refiner (KITTI mono, proxy crops): **beats uniform escalation everywhere**

| | uniform escalation | speculative | **speculative + learned refiner** |
|---|---|---|---|
| brake, all units | 0.517 | 0.529 | **0.539** |
| brake, test | 0.111 | 0.121 | **0.134** |
| traj, all units | 0.128 | 0.111 | **0.391** |
| traj, test | −0.024 | −0.030 | **+0.095** (frame oracle @20%: 0.053) |

At ~5% of the full pass's pixels. The only configuration so far that beats full-frame escalation on all four KITTI
monocular entries, including the one where escalation is harmful.

### D47-real — real YOLOv8s crops (discovery + pivots), braking

| | uniform escalation | **speculative, real crops** | pivots only | discovery only | network pixels |
|---|---|---|---|---|---|
| KITTI mono, all | 0.517 | **0.519** | 0.003 | 0.509 | 6.1% (pivots only 2.1%) |
| KITTI mono, test | 0.111 | **0.166** | 0.022 | 0.129 | 6.5% |
| nuScenes mono, all | 0.165 | 0.113 | 0.072 | 0.060 (proxy) | 6.8% (pivots only 1.1%) |
| nuScenes mono, test | 0.165 | 0.071 | 0.038 | 0.006 (proxy) | 7.5% |

- KITTI: the real speculative system matches uniform escalation on all units and exceeds it by 50% on the test split;
  the discovery crop carries braking, pivots add +0.036 on test.
- nuScenes: the pivot half rescues what discovery alone could not (0.113 vs 0.060 all units; 0.071 vs 0.006 test) —
  69% / 43% of uniform escalation's value at ~7% of its pixels; pivots alone give 44% of it at 1.1% of the pixels.
  Still short of full-frame escalation on nuScenes: the refiner half is what is missing there (next item).
- Proxy speculative on nuScenes (all 98.7% decision agreement with the full mode; 0.153 all / 0.064 test) confirms
  the gap is mostly in the test split's few affected frames.

### D40 on nuScenes — the box-feature refiner is not enough there
Pivot range error (test): lift 0.181 → refined 0.113 (−38%, against −51% on KITTI). Decision value 0.045 (all) /
0.013 (test), far below the oracle range refinement (0.283 / 0.181). nuScenes needs an **image-level** range refiner
(crop depth / 3D head on GPU); the box-feature GBM does not carry across datasets.

### D62 on nuScenes — ensemble decision sensitivity is a training-free allocator that works where the probe failed

| cell (test, nDG@20) | ego speed | probe VOI (D4) | ens: range | ens: phantoms | ens: misses | **ens: all** | DEEP best learned |
|---|---|---|---|---|---|---|---|
| nuScenes mono brake | 0.099 | 0.128 | **0.493** | 0.341 | 0.091 | 0.426 | 0.482 (gate GBM) |
| nuScenes oracle brake | 0.240 | 0.223 | 0.307 | **0.487** | 0.304 | 0.428 | 0.421 (router GBM) |
| KITTI mono brake | 0.415 | 0.423 | 0.009 | 0.030 | **0.439** | 0.386 | 0.215 |
| KITTI oracle brake | 0.584 | 0.599 | 0.016 | 0.059 | **0.611** | 0.532 | 0.360 |

- **Reading.** The borrowed weather-forecasting idea is the first *training-free* allocator that is competitive with
  or better than DEEP's learned allocators on **both** datasets. Its components line up with the two regimes found
  in D2/D12: miss insertion carries KITTI (blind cheap mode), range instability carries nuScenes mono (mis-ranging),
  phantom removal carries nuScenes oracle. The combined score (ens: all) is 0.39–0.53 in all four cells; a
  regime-adaptive mixture (weights fitted on non-test) is the obvious next step. Bootstrap intervals still to add.

### D3-real, KITTI oracle geometry
Real crops at ~4–5% of the pixels: brake 0.654 / 0.244 (all / test) against uniform 0.679 / 0.269 (96% / 91%);
traj 0.577 / 0.307 against 0.683 / 0.519 (84% / 59%).

### Second unattended batch (queued 02:47; results appended to `docs/overnight_results.md` by `overnight_queue2.sh`)

1. `logs/ensemble_bootstrap.log` / `results/final/d62_ensemble_bootstrap.csv` — paired unit-bootstrap intervals
   (1,000 draws, DEEP's scorer) of the D62 ensemble allocators against random and DEEP's gate GBM, 4 braking cells.
   Claim D62 only where `minus_random_lo > 0`; "competitive with DEEP" only where the `minus_gate` interval covers 0
   or is positive.
2. `logs/realspec_s960_KITTI_mono.log` — speculative perception with crops detected at the **960** scale. Compare with
   uniform escalation to 960 computed from the multimode tables: **0.529 (all) / 0.125 (test)** (uniform to 640:
   0.517 / 0.111). With the 288×960 engine timing (`engines/time_288x960.log`) against the 96×288 crop canvas
   (`engines/time_96x288.log`) this answers whether crops save *milliseconds* once the full mode is heavy.
3. `logs/realspec_KITTI_oracle.log`, `logs/realspec_nuScenes_oracle.log` — real speculative crops under reference
   geometry (does the method hold when the lift is not the problem?).

### D62 intervals (paired unit bootstrap, 1,000 draws, nDG@20 differences; `results/final/d62_ensemble_bootstrap.csv`) — **claim revised**

| cell | ens: all − random | ens: all − DEEP gate GBM | best component − random |
|---|---|---|---|
| KITTI mono brake | **[0.087, 0.371]** | **[0.001, 0.344]** | misses [−0.069, 0.487] |
| KITTI oracle brake | **[0.063, 0.481]** | [−0.256, 0.401] | misses [−0.054, 0.591] |
| nuScenes mono brake | [−0.028, 0.482] | [−0.362, 0.174] | range [−0.143, 0.561] |
| nuScenes oracle brake | [−0.148, 0.691] | [−0.237, 0.625] | phantoms [−0.083, 0.722] |

**Revised reading.** The training-free ensemble score is significantly better than random on both KITTI cells and
significantly better than DEEP's learned gate on KITTI mono. On nuScenes its point estimates are the highest of any
allocator tested, but every interval includes zero (24 test scenes, 57 and 25 affected test inputs): **not resolved**.
The earlier sentence "competitive with DEEP's learned allocators on both datasets" holds only as point estimates.

### D62b — regime-adaptive mixture of the ensemble components (fitted on non-test units) — `results/final/d62b_ensemble_mixture.csv`

| cell | selected single component | ridge mixture | GBM mixture | DEEP's best learned (point) |
|---|---|---|---|---|
| KITTI mono brake | 0.439 (misses) [−0.069, 0.487] | 0.435 [−0.066, 0.493] | **0.439 [0.129, 0.488]** | 0.215 |
| KITTI oracle brake | 0.611 (misses) [−0.054, 0.591] | 0.614 [−0.052, 0.600] | 0.598 [−0.032, 0.591] | 0.360 |
| nuScenes mono brake | 0.341 (phantoms) [−0.056, 0.631] | **0.482 [0.236, 0.674]** | 0.277 [−0.096, 0.645] | 0.482 |
| nuScenes oracle brake | 0.428 (all) [−0.148, 0.691] | **0.453 [0.089, 0.738]** | 0.438 [−0.002, 0.827] | 0.421 |

(nDG@20 on test; brackets: paired bootstrap interval of the difference with random.) **Reading.** A five-feature ridge
over the training-free ensemble-sensitivity components (range instability, phantom instability, miss insertion, all,
speed) is the first allocator in this exploration that is significantly better than random on **both** nuScenes
cells, matching or exceeding DEEP's best learned allocators in point estimate with far fewer features. On KITTI the
GBM mixture is significant on mono. Together with D3-real/D47 this gives paper A a principled pre-escalation gate:
*simulate what the cheap scene could be wrong about, and escalate where the decision is unstable*.

### Second batch (finished 03:33) — interpreted 07:45

| real speculative crops, braking | uniform escalation | speculative | network pixels vs full 640 |
|---|---|---|---|
| KITTI mono, crops at **960** scale, all / test | 640: 0.517 / 0.111 · 960: 0.529 / 0.125 | **0.542 / 0.182** | 12.5–13.4% |
| KITTI oracle, all / test | 0.679 / 0.269 | 0.675 / **0.284** | 5.6–5.9% |
| nuScenes oracle, all / test | 0.140 / 0.142 | **0.194** / 0.087 | 6.4–7.3% |

**Milliseconds (trtexec FP16, median):** full 288×960 **7.44 ms**; full 192×640 3.83 ms; crop canvas 96×288 **2.12 ms**;
64×192 2.02 ms; cheap 96×320 2.20 ms. With the full mode at 960, a crop escalation (≈2.1 ms per canvas, 0.74 crops per
frame) is ~3.5× cheaper than a full 960 pass, cheaper than even a full 640 pass, and gives more decision value than
escalating every frame to either 640 or 960 (KITTI mono test 0.182 vs 0.111 / 0.125). This largely answers the
pixels-vs-milliseconds concern **for heavier full modes**; at 640 the time saving stays ~1.9×. Caveat: large crops can
exceed one 96×288 canvas; packing/tiling cost still to be measured end to end.

---

## B2 — compute accounting, done (2026-09-27 08:20)

**Closed-form field and pivots** (`src/vtf/dsf.py`, `scripts/32_dsf_exactness.py`): for the braking controller the
insertion-sensitive grid and the pivots have closed forms (max-composable rule). On all 8,008 KITTI frames they
**equal the probed ones exactly** (0 grid mismatches, 0 pivot mismatches) and cost **0.41 ms** instead of 21.6 ms per
frame (Python; probing Planner B costs 74 ms and still needs vectorising).

**Deployed pipeline, end to end on the Jetson** (`scripts/33_e2e_latency.py`; TensorRT FP16; preprocessing + inference +
NMS + lift + fusion; image decode excluded; KITTI mono test split, 3,536 frames; every pipeline's decisions computed on
the same TensorRT branch with the cheap branch's action history):

| pipeline | braking decision value (test) | mean ms | median ms | p95 ms |
|---|---|---|---|---|
| cheap only | 0 | 11.9 | 12.4 | 15.1 |
| uniform full 640 (alone) | 0.123 | 16.8 | 16.8 | 20.3 |
| uniform cascade cheap + 640 | 0.123 | 28.7 | 29.0 | 35.7 |
| uniform full 960 (alone) | 0.129 | 24.4 | 23.3 | 29.4 |
| **speculative v1** (closed-form field, pivots, 96×288 crop engine, 0.71 crops/frame) | **0.237** | **21.1** | 21.8 | 35.7 |

Paired unit bootstrap (2,000 draws): speculative − uniform 960 = **+0.108 [+0.037, +0.180]**; − uniform 640 =
**+0.115 [+0.044, +0.205]**; harmed share of affected frames 0.21 vs 0.43 for uniform 960.

**Reading.** On KITTI the deployed method gives about twice the decision value of escalating every frame, at 13% less
mean time than a full 960 pass and 26% less than DEEP's cascade — but not less than a lone 640 pass, and with a worse
p95 (frames with two or more crops). Per-call Python overhead (~9–10 ms of letterbox/NMS per detector call) dominates:
GPU-only, cheap 2.2 + 0.71 × 2.1 ms ≈ 3.7 ms vs 7.4 ms for full 960. Engineering to close the gap: batch all crops of a
frame into one engine call, GPU preprocessing, C++ post-processing; cap crops per frame for the p95.
(The value 0.237 exceeds the earlier real-crop run's 0.182 because the 96×288 letterbox upsamples small crops beyond
the 960 scale; the effect of crop scale is itself an ablation to report.)
