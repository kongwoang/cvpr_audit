# Addendum: identity gate plus explicit uncertain fallback, nuScenes only

Separate pair classifier gate passed on nuScenes training holdout: AUC .944,
switch recall .579 at valid-pair rejection .00990. AV2 AUC .877 but recall .166
fails the predeclared .2 feasibility gate; do not deploy the AV2 classifier here.
These are model/threshold selection diagnostics, not independent accuracy claims.

New nuScenes candidate starts from upper-quantile missing-history + B2K-A+recovery.
For every actual complete verified/retro chain, score both historical identities
with the fixed pair classifier. If either score exceeds its holdout threshold,
exclude that chain from C1 and use the masked head's CURRENT-ONLY upper quantile.
Do not reuse its rejected past boxes as a cheap bridge. Valid chains retain C1;
originally missing chains keep the existing bridge/masked policy. Keep the object
detection itself and all genuine observations for future frames. No GT or future
images enter inference. Threshold/model frozen before the dev pilot.

Run original upper-quantile policy and v6-S/round-1 controls alongside. This differs
from the negative R2 geometric gate by using learned identity evidence and an
explicit missing-evidence fallback instead of silently imposing TTC infinity.
It can still reject useful histories, hallucinate braking or increase encoder work.
Every action, false/missed episode and request count is retained. No core change.
