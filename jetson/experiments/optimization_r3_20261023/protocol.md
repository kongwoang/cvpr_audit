# Round 3: repairing observed failure mechanisms (2026-10-23)

User authorizes broad, compute-intensive sandbox experiments. No main source,
frozen weight, K34 record, threshold, or test split may be changed. All attempted
rows, failed implementations, negative results and selection criteria are retained.
Main sources are hashed before/after jobs. GPU jobs use the existing unchanged
run_when_gpu_free.sh with SHARE=1 MIN_FREE_MB=10000, two workers per group. All
remote writes stay under the CVPR workspace. Other users' jobs are untouched.

## Work packages, specified before their outcomes

1. **Identity-aware training evidence.** Regenerate training/REF-VAL records with
   full unit/frame/box provenance and GT identities for every temporal observation.
   Keep incomplete, unmatched and identity-switched chains as separate categories.
   Add cheap-history/current-verified bridge examples. GT is a training label only;
   inference associations use images, boxes, class and earlier frames only.
2. **Missing-evidence motion.** Learn a small masked temporal head over frozen C1
   image embeddings, using actual observed crops and an explicit availability mask.
   Compare a point estimate, a conditional uncertainty estimate, and direct frozen
   C1 on mixed-fidelity history. Include motion transfer from the cheap draft as a
   simple control. Missing history must not masquerade as observed zero closing.
3. **Joint geometry/motion.** Audit/refit range and rate together, including a
   decision-aware auxiliary objective. Compare geometric-only vs decision-aware
   fitting. Separately test preserving closing speed during a range correction;
   this is a sensitivity hypothesis, not a physical guarantee or free improvement.
4. **Acquisition for unobserved hazards.** If motion evidence improves, test queries
   driven by motion/uncertainty and previous object extent. Fixed vc5/vc10 and probe
   union were already evaluated in K30; repeating them alone is not a new result.

Training uses nuScenes new TRAIN (original hash holdout) and AV2 REF-TRAIN/REF-VAL.
CFG-DEV = 152 nuScenes val scenes + 300 AV2 logs, never test. Epoch/model selection
uses training holdouts only; dev comparisons remain exploratory after several
rounds of adaptation. No confirmatory claim. Paired unit bootstrap 2,000, seed 0,
pointwise 95% intervals. Original episode definitions and cheap-previous-action
jerk convention. Report severity, newly lost episodes, false alarms, loss, G,
requested detector pixels and executions; actual Jetson time only when measured.

All useful-looking rescues require attribution to the annotated decisive object;
wrong-object compensation remains in scores but cannot establish mechanism repair.
New inference heads/query policies require fresh in-loop histories and exact
baseline replay. Frozen-query postprocessing is explicitly labelled. A plausible
quality gain is required before any main-code integration (none is authorized by
this protocol alone). Jetson uses its exclusive-window protocol and existing engines.
