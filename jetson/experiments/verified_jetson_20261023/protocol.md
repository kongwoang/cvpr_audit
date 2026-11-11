# Corrected AV2 Jetson timing — 2026-10-23, before new execution

The legacy K29 `av2_dev` cache is not wholly current CFG-DEV: the initial 512-frame
timed prefix contains 242 frames from four REF-TRAIN logs. Preserve all old rows
with a dataset correction. They are runtime engineering evidence, not held-out
method-quality evidence. nuScenes prefix membership is unchanged/verified.

Training audit: available C1/refiner training and selection unit IDs resolve
unambiguously; no overlap with the full 152-scene/300-log quality screen. Sample
counts and both feature-normalization arrays match the frozen checkpoints
exactly. Original training-time per-NPZ hashes are unavailable; state this limit.

Freeze verified_log_manifest.json before inference: original K29 cache ordering,
current CFG-DEV only, exclude every available training/selection unit for both
components. No outcome/latency-based unit selection. Check cache presence.

Use existing trt32 engines/reference crop runtime and the exact fast bookkeeping
helper after its three-repetition parity gate. Same four method rows, no weights,
threshold or policy changes. First 712 chronological filtered frames, 200 warm-up
=512 timed, three repetitions, fixed ego mask 1830, rotating pipeline order.

Before timing, original-vs-fast helper smoke on the first 80 filtered frames must
match every action, cost, pixel, execution and recovery counter. Then time fast
helper only, with all four policies on the same verified frames. Histories and
current detections are independent between policies; no timing cache sharing.

Record actual image filename and annotation timestamp, source/engine/manifest
hashes, retained-image memory, all rows, paired unit bootstrap 2000 seed0 and
all common-frame VUW comparisons. Full development quality stays in the VUW
screen; prefix G/loss are diagnostic only. This corrected timing is a distinct
dataset/protocol and must not be spliced with the older AV2 mixed-prefix numbers.
Exclusive Jetson window; existing engines only; restore every owned control.
