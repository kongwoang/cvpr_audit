# Runtime buffer experiment — registered 2026-10-23 before GPU execution

Intent: preserve frozen v6-S/B2K-A decisions and crop geometry while reducing
intermediate crop tensors, stacking and the extra TensorRT input copy.
Sandbox code only; original modules/engines remain unchanged. No engine build.

Candidate staged: letterbox directly into persistent dynamic-engine input views,
keeping interpolation, normalization, exact shapes, crop order, padding,
thresholds and postprocessing. Enqueue that engine on the current CUDA stream.
Do not introduce uint8/fused arithmetic, new shape buckets or FP16 conversions.
The output buffer must be consumed before the next execution overwrites it.

Dev timing only: nusc_val and av2_dev (fixed ego mask 1830), first 712 chronological
frames, discard first 200 as warm-up = 512 timed frames per dataset. Record exact
unit/frame manifest and hashes. Every pipeline has independent history and runs
its own detection/upload; no timing cache shared across pipelines.
Reference and staged run in separate sequential processes; alternate order by
repetition, three repetitions. Ours, <=2, B2K-A all retained. Existing trt32 C1.

Gate: synthetic real-engine crop parity first, then 80-frame end-to-end dev smoke
for both implementations/datasets. Compare every action and crop/engine counter.
Any difference blocks full timing pending diagnosis; do not relax the gate to
claim an exact optimization. Save failures and all attempted rows.

Timing includes upload, draft, crop verification, C1, refiner and action; excludes
decode. Report mean/p95/p99 and >50/66.7/100ms, paired unit bootstrap 2000 seed0,
all repetitions nested in each unit. The first-712 dev prefix is a runtime
screen, not the full dev quality evaluation or confirmatory test.

Jetson exclusive pause protocol; record hardware/clocks/telemetry and source/
engine hashes, restore owned controls on all exits. Notify through the shared
pause state before starting; wait for existing heavy work/profiling to finish.
If hardware or resource conditions prevent timing, keep the job queued.
