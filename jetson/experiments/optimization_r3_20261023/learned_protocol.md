# Learned adapter experiment, fixed before fitting

Regenerate all original training units, including full unit IDs, frame indices,
three boxes/GT track IDs, masks and ego/GT geometry. Original T4+T2_1 training
acquisition; nuScenes keyframes and every second AV2 frame, original AV2 hood mask.
Retain crop/u960/cheap and current-crop + previous-cheap bridge examples, including
incomplete or identity-switched histories. No manual case becomes training data.

The original C1 ResNet18 encoder is frozen. Three 512-D crop embeddings, original
16 normalized C1 features, explicit availability, source kind, ego speed, lateral
extent, relative box centers/aspects and pairwise appearance cosine feed a 256-D
MLP. Initialize overlapping point/quantile weights from C1; new inputs start with
zero weight. Missing crops are zeroed after encoding, with explicit masks.
Outputs: point closing rate; log range factor (ray-coupled lateral correction);
rate quantiles .1/.5/.9; two historical identity-validity probabilities. Identity
is a supervised auxiliary target, not a GT gate at inference. Quantiles are
empirical conditional estimates, not certified bounds or a safety guarantee.

Fit independently per dataset, seed 0, 20 epochs, AdamW lr1e-4 weight decay1e-4,
cosine schedule, batch about512, gradient norm5. Base loss: rate SmoothL1 beta.02
+ .2 log-range SmoothL1 beta.05 + .2 pinball + .02 identity BCE. Decision variant
adds .2 SmoothL1 beta.1 on log(1+demand), using the frozen stopping/closing formula
and a smooth lateral gate of .15m during training. Point predictions clamp [0,2],
log range factors [-1.1,1.1]. GT factors use the same clipping range; unbounded
geometric errors must also be reported to expose clipping limitations.

Epoch selection on original training holdouts only: rate MAE+.2 range-log MAE;
decision variant additionally .2 log-demand MAE. Original C1 previously used those
holdouts for checkpoint selection: these are not untouched validation/test data.
Report all epochs, models, availability/identity strata, empirical interval
coverage and false-demand/required-demand errors. No decision threshold tuning.

Planned fresh in-loop rows: point adapter only for missing chains; point adapter
on all verified objects; decision-aware adapter on all verified objects; upper
quantile only for missing chains as a conservative stress test; and at most one
cheap-object query when lower/upper quantile hypotheses imply different actions.
Existing C1/round-1 controls replay exactly. Report rate-only with v3a separately
from the adapter's jointly predicted geometry. Every policy has its own history.
Retain negative results; further directions require a dated addendum before runs.
