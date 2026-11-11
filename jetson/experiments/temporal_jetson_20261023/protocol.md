# Targeted C1 history recovery on Jetson — feasibility protocol, 2026-10-23

User authorized separate feasibility/tradeoff experiments. Initial fixed dev
results motivate measuring actual latency of recovery and its B2K-A combination.
No method promotion, test tuning, new training, thresholds, engine build, or edit
of frozen source. Existing trt32 path and reference crop runtime only.

Four rows: v6-S, v6-S+recovery, B2K-A, B2K-A+recovery. Original v3a and independent
pipeline histories. Same recovery helper/gate as the VUW dev experiment, genuine
two prior images, unique compatible matches, <=2 historical detector executions.
Keep the source crop's own letterbox dimensions and actual common-canvas padding
when replaying historical crops. Detector/backend differences from VUW remain
possible; do not claim numerical parity between servers and Jetson.

Each recovery pipeline retains its own two previously uploaded GPU frame tensors;
current upload is timed normally for that pipeline. No sharing GPU histories or
detector outputs between comparator pipelines. Report this memory tradeoff and
all extra historical execution/pixel counts; historical inference is timed.

Gate: VUW factorial smoke must pass. Wait for the runtime buffer experiment and
its reporting to release the Jetson. Then an 80-frame disabled-recovery port
check on each dev dataset must show exact action/cost/pixel/call parity between
each duplicate pair. Compare base rows against the prior runtime reference
smoke on identical frames. Enabled recovery smoke checks call/pixel budgets and
actual past-image provenance. Save attempts even if gate fails.

Timing: first 712 chronological nusc_val / av2_dev frames, discard first 200 =
512 timed frames, three repetitions, rotate four pipeline orders by repetition.
Fixed AV2 mask y=1830. All four rows retained, including unfavorable outcomes.
Record episode metrics on the prefix only as development/conditional evidence;
full VUW dev remains the quality screen. Report recovery-triggered frame latency
separately so an inactive prefix cannot establish its tail cost.

Outputs: mean/p95/p99, >50/66.7/100ms, detector executions and pixels, recovery
attempts/success, retained-GPU-image memory estimate, all actions/costs. Paired
unit bootstrap 2000 seed0, repetitions nested within units, all rows. The new
port need not agree with VUW end to end; diagnose and report differences rather
than using favorable backend results to choose a method.
