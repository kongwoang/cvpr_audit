# Addendum: remove fallback image-encoder cost

Jetson smoke shows that detector-pixel parity alone hides substantial extra work
from encoding historical cheap-box ROIs. Warm-up includes first model load, so
smoke means are not latency conclusions; nevertheless the added encoder work is
architecturally avoidable if motion can be inferred from observed box dynamics.

Fixed compact control: retain original C1 for every actual full chain. For missing
chains use a small MLP over original16 C1 features and16 mask/kind/ego/relative-box
features ONLY, no image embeddings, no new CNN crops. Use the same causal cheap
association and explicit masks as the image adapter. Train on the regenerated
bridge examples (current verified + actual earlier cheap observations), finite
targets only, TRAIN units only. Current-only examples remain explicit, not an
invented motion observation. No GT, future data or development labels as inputs.

Architecture32->128->64->1, ReLU, normalized on training data, SmoothL1 beta.02,
AdamW3e-4 wd1e-4,20epochs cosine,seed0,batch512. Select epoch by original holdout
bridge rate MAE. Preserve all epochs and holdout predictions. Predict rate in[0,2];
same original >.001/dt threshold for finite TTC. Range remains original geometric
lift and final v3a, no jointly trained range correction.

Fresh in-loop dev candidates: compact point fallback, and compact point with >=2
real observations only. Controls v6-S, round1 and full-image point fallback. Every
row retained and negative tradeoffs exposed. No deployment claim unless quality
and actual runtime support it. This is an additional exploratory development
iteration, not a confirmatory evaluation or an assumed improvement.
