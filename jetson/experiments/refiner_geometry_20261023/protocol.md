# Refiner geometry attribution — CPU-only feasibility screen, 2026-10-23

Motivation: recovery reduces some misses but the refiner/no-refiner gap remains.
Do not assume that removing refinement or conservatively vetoing every downgrade
is beneficial; previous K30 dev results and the current screen oppose that claim.

Use all completed original W1 dev baseline traces, 152 nuScenes scenes and 300
AV2 CFG-DEV logs, after the original run completes. No new images, inference,
training, thresholds, test units, or main-source changes.

Predeclare five stored-scene rows: raw/no-refiner; deployed v3a; range-only
(refined z, raw lateral/TTC); lateral-only (raw z/TTC, refined lateral); deployed
geometry with TTC scaled by refined/raw depth (holds the estimated z/TTC closing
velocity fixed where finite). Report every row. First prove reconstructed raw/
deployed actions and costs match stored records on every frame.

These are post-loop geometry counterfactuals. The frozen loop/history use raw
boxes and one-pass refiner support, so existing corrections can be reused for
this diagnostic. This does not apply automatically to post-refinement closure.
Range-only/lateral-only may violate the camera ray/depth coupling; favorable
decision metrics alone do not establish physically correct geometry. There are
no applicable generic whole-frame controls yet, so no selector-superiority or
new-method-selection claim is permitted.

Count required-frame action downgrades and which range-only/lateral-only change
individually suppresses the action. Mechanisms overlap and are not causal shares
of all misses. Report episode misses, false episodes/1,000, insufficient loss, G,
and 95% paired unit bootstrap 2,000 seed0 for all rows/contrasts. Original G jerk
convention: preceding cheap action. No best-parameter search or promotion.
