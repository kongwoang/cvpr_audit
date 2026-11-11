# R3 learned missing-history fallback: Jetson feasibility

Sandbox runtime experiment, no core/engine modification. Fixed candidates: frozen
v6-S, round-1 B2K-A+recovery, and B2K-A+recovery+the geometric adapter's point rate
only for objects without an actual complete C1 chain. Valid C1 chains unchanged;
v3a refinement and thresholds unchanged. This tests the lower-false-alarm R3 option,
not the upper-quantile policy. No result-driven frame selection.

Use the prior frozen nuScenes val prefix and corrected AV2 CFG-DEV manifest, 200
warm-up frames then 512 timed frames, three repetitions, rotate pipeline order.
Existing trt32 C1 encoders, existing detector/refiner engines; learned adapter is
a small torch FP32 MLP. Share exact encoder features with the pipeline's existing
C1 bank, keyed by frame/box; never share histories or timing work across pipelines.
Only missing-chain eligible objects enter the new MLP, preserving all-object
cheap association semantics. No new engine builds, no GPU use outside the window.

Before timing: baseline port parity against the prior verified harness on 80 frames;
learned Torch-vs-trt32 encoder replay on the same detector outputs (actions/rates),
and trained-adapter weights/normalization portability checks. Report differences,
not an assumed equivalence. VUW common-frame parity is checked separately because
Torch and TensorRT detectors and bounded padding may differ. No clock or engine
modification without restore. Own .jetson_pause, wait busy.sh only 'paused', no
.r3_profiling, MAXN/clocks/fan255, restore/release in finally.

Report mean/p95/p99, fractions >50/66.7/100ms, actual padded detector pixels,
executions, additional encoded objects, and paired unit bootstrap 2,000 seed0.
Training/adaptive development prefix timings are not confirmatory test evidence.
