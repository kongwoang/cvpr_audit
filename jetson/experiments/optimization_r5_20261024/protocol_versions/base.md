# Round 5: conditional decision repair (2026-10-24)

Focus: remaining decision errors, not a speed-only improvement. Sandbox only;
frozen core, weights, K34 and all test data remain untouched. This is adaptive
development, not a confirmatory study or a claim of globally optimal components.

Before fitting or evaluating the new policies, fix this scope:

1. Audit all 152 nuScenes / 300 AV2 CFG-DEV units from R4. Separate absence of the
   decisive detection, missing temporal history, corridor exclusion, refiner
   downgrade, and residual underestimation. These are observable failure
   signatures, not mutually identifiable causal effects. Report loss-weighted
   counts, all remaining missed episodes and severity, not selected anecdotes.
2. Train a small ordinal braking-evidence classifier on the ORIGINAL R3
   TRAIN/REF-TRAIN data. Use raw monocular geometry, frozen C1 output where a
   complete history exists, confidence, class, image-normalized box, ego speed,
   full/cheap source and history availability. No new CNN, no future frames, no
   GT or identity labels at deployment. A three-class tree classifier estimates
   the annotated object's KEEP/DECELERATE/HARD_BRAKE probabilities. This changes
   the experimental perception readout; it is not a new theorem for the frozen
   geometric selector. Original thresholds and consumer costs are unchanged.
3. Include unmatched detections as annotation-negative examples; label these
   honestly as unmatched, not manually certified phantoms. Original R3 matched
   rate regression excluded them. Exclude bridge duplicates; retain crop, u960
   and cheap examples, with source explicitly observable. Use GT mono to label
   matched objects; exclude invalid matched geometry. Train separately per
   dataset, no dev/test fitting. Fixed HistGradientBoostingClassifier:
   150 iterations, 7 leaves, min leaf 50, learning rate .05, L2=1, seed 0,
   no early stopping. Also fit a matched-only negative control identically.
   Report original unit-disjoint holdout calibration/precision/recall and all
   training provenance. No hyperparameter or confidence selection from dev.
4. Fixed confidence ladder q in {.5, .8, .95}, all rows retained:
   - rollback: permit a raw-over-refined braking threshold only when predicted
     probability of the required action-or-higher reaches q;
   - missing: same confidence rule adds evidence only for verified objects with
     no complete temporal chain (including genuine retrospective recovery);
   - both: union of the above;
   - median_gate: retain the R4 object's median proposal only if the classifier
     corroborates its upward threshold crossing; leave baseline otherwise.
   All corrections are upward only. Therefore shortfall cannot increase by
   construction, but false braking can: risk reduction alone is insufficient.
   Matched-only control: both at q=.8. These fixed policies do not alter queries,
   histories or future state, so replay of stored round1 final scenes is exact
   for this declared readout scope. They do not repair absent detections.
5. Full-frame controls: cheap, frozen v6-S, round1 raw/v3a, R4 median and the
   strongest prior R3 point+extent result (as a separate acquired-scene control).
   Every row: missed episodes, onset delay, insufficient loss, severe loss,
   false episodes AND false/hard-brake frames, excess cost, G, requested pixels.
   Paired unit bootstrap 2,000, seed 0; pointwise exploratory intervals. Keep
   jerk referenced to previous cheap action, as in K30. Audit every rescued and
   newly lost episode, including decisive-object evidence and hard adequacy.

Success requires useful missed-episode/severe-risk gains without merely moving
to a much worse false-braking operating point. Report the joint tradeoff rather
than declaring success from G alone. No arbitrary post-hoc noninferiority margin.
If no candidate is credible, retain the negative result and diagnose why;
Jetson timing is justified only for a quality candidate, not as a distraction.

Training acquisition differs from B2K-A+recovery; quantify this domain shift in
holdout versus dev calibration. Stored embeddings are float16: reconstructing
frozen C1 is a training approximation, never replace the exact dev baseline.
Do not claim the repeatedly inspected dev population is untouched evaluation.

Execution: CPU tasks on VUW/VUW1, source absolute env.sh before every remote
action/write, one thread per CPU task, all files/caches inside B. GPU work, if
needed later, uses the unchanged run_when_gpu_free.sh with SHARE=1,
MIN_FREE_MB=10000 and two workers per group. Snapshot sources/checkpoints before
and after. No Jetson reservation is taken by this CPU study.
