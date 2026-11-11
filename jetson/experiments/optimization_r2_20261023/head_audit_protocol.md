# Frozen C1 calibration audit — before prediction extraction

Question from observed valid-chain cases: does the frozen C1 underestimate
strong closing rates? The previous same-class identity switch confounds that
case and cannot establish regressor bias. The source also constructs training
triplets from IoU-linked detections but labels only the current GT match; it
does not check historical GT track identity. The available NPZ has no frame/box
IDs for a complete retrospective label audit. Do not invent clean-track labels.

Run the **unchanged seed-0 checkpoint**, eval mode and original preprocessing,
on every finite-target `split == val` sample in the original tdata shards.
These are AV2 REF-VAL and the nuScenes training-scene internal holdout, not the
152/300 CFG-DEV experiment. It is the checkpoint-selection validation set,
so results are descriptive, not independent generalization evidence.

Report C1 and stored T4 on bins r=0, (0,.1), [.1,.25), [.25,.5), [.5,2] /s,
by dataset and crop/u960. Freeze bins; no fitted correction or new checkpoint.
Metrics: MAE, signed bias, under/overestimation fractions, fraction estimating
below half the true rate for r>0. NaN/invalid targets excluded exactly as original
training (counts reported separately); clip rates to [0,2] as original evaluation.
Reproduce original aggregate validation MAE to numerical tolerance before
interpreting strata. Paired unit bootstrap 2,000 seed0 for C1-minus-T4 MAE/bias.
Keep raw predictions with source filename/sample index/sequence for audit.

GPU only through unchanged wrapper, SHARE=1 MIN_FREE_MB=10000. Two workers
(one dataset each), CPU threads at most4, inference batch128, one NPZ loaded at
a time. Save exact checkpoint/data/source hashes and errors. No training or
main source change. Do not interpret conditional-bin bias as proof of a fix.
