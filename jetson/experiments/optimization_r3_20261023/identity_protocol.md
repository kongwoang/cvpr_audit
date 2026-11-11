# Addendum: separate identity discrimination kill test

The low-weight shared validity output predicts mostly the majority class. Test a
separate small pair classifier on frozen C1 embeddings and relative 2D geometry,
using only regenerated TRAIN/REF-VAL identity labels. No GT input at inference.
Known historical pairs (including incorrect pairs) are kept; unknown matches are
excluded from supervised identity scoring, not relabelled as negative.

Pair features: L2-normalized current/past 512-D embeddings, absolute difference,
appearance cosine, box-center displacement normalized by current size, log size
ratios, lag*dt and source kind. MLP128->32->1; 15 epochs, AdamW1e-3, wd1e-4,
balanced BCE for the rare switched class, seed0. Select epoch by holdout ROC AUC.
The balancing changes probability calibration: scores are not identity probabilities.

Audit recall of switched pairs at a threshold retaining 99% of valid holdout
pairs; also AUC, average precision and unit-bootstrap intervals. A tentative
feasibility gate is AUC>=.8 and switch recall>=.2 at <=1% valid rejection, separately
per dataset. Passing this diagnostic would still require fresh pipeline evaluation;
failing it rules out deploying this classifier in this round. No hard identity
gate is introduced into any running policy by this experiment.
