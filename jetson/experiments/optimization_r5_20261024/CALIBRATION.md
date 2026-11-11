# Classifier and conditional-calibration audit

Eight small tree classifiers were trained, all from original R3 train/REF-TRAIN
units, with the original separate holdout. No development/test fit or threshold
selection. 383 nonempty nuScenes units and 400 AV2 units contribute examples.
No new CNN weights were learned. Frozen C1 embeddings were cached float16, so
training C1 reconstruction is approximate; exact stored C1/v3a defines dev.

The first model uses 20 observable features: raw z/lateral/rate, ego speed,
demand/ego-only demand, confidence, class, normalized box, full-source and
complete-history flags. HistGradientBoostingClassifier is fixed at 150
iterations, 7 leaves, min leaf 50, learning rate .05, L2=1, seed0, no early stop.
Separate dataset models; no bridge duplicates. Annotation-unmatched detections
are negative examples, not manually certified false detections. A matched-only
control uses the same learner and hyperparameters. Provenance, invalid-label
counts, confusion matrices, hashes and every holdout prediction are in training/.

Adding unmatched examples improves overall holdout log loss:

| Dataset | Matched-only | All detections | Train examples, all | Holdout examples |
|---|---:|---:|---:|---:|
| nuScenes | .08912 | .07980 | 43,762 | 7,033 |
| AV2 | .10232 | .09618 | 139,681 | 25,031 |

This does **not** mean q is a calibrated probability on the repair population.
On AV2 dev, q=.8 DECELERATE precision is 4,444/4,679 (94.98%) among all verified
objects but only 29/62 (46.77%) among raw≥1/refined<1 disagreements. Overall
accuracy hides the population where the gate is actually invoked. Precision
also differs from frame-level false braking: a wrong object can coexist with
another legitimate braking object.

The follow-up computes the existing frozen v3a on original training crops and
adds refined geometry, log scale, demand and corridor change. Original support
selection is reconstructed within unit/frame/acquisition kind. One model fits
all full-source examples; another fits only discrete-action downgrades. This
support/acquisition distribution is still different from B2K-A+recovery and
especially point+extent; there is no claim that the training acquisition matches
deployment. No classifier is selected by development outcomes.

Conditional training has only 458 examples (412 KEEP / 44 DECELERATE / 2 HARD)
for nuScenes and 5,715 (5,045 / 640 / 30) for AV2. Holdout downgrades number 54
and 968; the nuScenes apparent high precision is based on very few positives.
On AV2 holdout, conditional q=.8 predicts 28 positives, 20 correct (71.4%).
On actual dev raw≥1/refined<1 objects, conditional q=.8 predicts one positive,
zero correct; pooled+refiner q=.8 predicts eight, four correct. Thus this round
does not solve selected-population generalization by adding a conditional head.
The detailed full-downgrade tables also include HARD→DECELERATE, so use the
explicit raw≥1/refined<1 scope when comparing the quoted precision numbers.

Conclusions supported by the data:

- Do not interpret a confidence threshold as a false-braking guarantee.
- Broad confidence/geometry cues lack reliable evidence for rare hard errors;
  conditional fits have very few hard positive examples and a substantial
  acquisition-distribution shift.
- Including unmatched negatives reduces some false predictions but may suppress
  real fragments that fail the annotation matcher. The matched-only control is
  retained; neither labelling choice is a universal solution.
- Every q=.5/.8/.95 row is reported. High q mostly abstains; that is a negative
  recall result, not an unreported “safe” configuration.

Files: training/, conditional_training/, evaluation_*/dev_object_thresholds.csv,
conditional/*/disagreement_predictions.csv and dev_disagreement_thresholds.csv.
These are descriptive calibration diagnostics; primary paired scene/log CIs
are in all_intervals.csv and all_contrasts.csv.
