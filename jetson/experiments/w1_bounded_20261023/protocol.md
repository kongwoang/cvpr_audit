# Bounded verification × selective temporal evidence — 2026-10-23

User authorized autonomous feasibility assessment and separate experiments on VUW,
VUW1 and Jetson. Main sources/weights and frozen K34 remain unchanged.

Motivation is post-hoc: current dev results favor targeted temporal evidence over
closure/recheck; K34 establishes a tail-latency benefit of B2K-A. Their interaction
is unmeasured. No assumption that independent improvements add.

Data: all 152 nuScenes new-val scenes and all 300 AV2 CFG-DEV logs; fixed AV2 ego
mask y=1830; verify training disjointness and identical frame sets as K30/W1.
No A-HD/test selection, training, threshold sweep, or automatic promotion.

Fixed 2x2 factorial: frozen v6-S, v6-S+selective C1 retrospective recovery,
B2K-A (two verification rounds, retain pending truncated detections at cap), and
B2K-A+the identical recovery gate/budget. Both with and without v3a. Recovery
allows at most one historical crop canvas at each of t-1/t-2 (two executions).
Always count this extra work separately; the two-round cap excludes recovery.
No widened gate, action floor, new refiner, or combined closure/recheck.

Reuse the immutable completed W1 records for the full-loop recovery row and
generic whole-frame controls. Recompute frozen v6-S in every new unit and verify
actions/cost/GT/frame parity. Each new variant owns its history. GPU detection
caching can share identical current inputs for dev throughput only; counters
record each pipeline's logical executions/pixels. GPU wall time is not latency.

Add hybrid tsp8 and unc2 controls at the combined candidate's total crop-pixel
allowance INCLUDING retrospective pixels. These are requested-budget controls:
guards can overrun the allowance, so report all realized costs and overruns.
Do not claim exact equal total cost or selector superiority from this screen.

Smoke: first four sorted reference units per dataset; check frozen parity, N<=2,
retro calls<=2, all extra pixels, real past images and ambiguous-match rejection.
Smoke units are a subset of dev. Then full dev, two shard workers per unchanged
run_when_gpu_free.sh wrapper (SHARE=1), OMP=4. One group on each server; avoid
adding work to the active original AV2 GPU. Do not modify a running artifact.

Report every row, paired unit bootstrap 2,000 seed 0: required-episode misses,
onset delay, false episodes/1,000, insufficient-braking loss and severity, G,
realized pixels/calls, bound hits, recovery successes, and interaction contrasts.
G retains original cost convention, jerk relative to preceding cheap action.

Next gate is a dev Pareto assessment and actual Jetson timing, with all candidate
rows retained regardless of outcome. Any quality/false-alarm/latency selection
margins must be declared before selection. This phase reports tradeoffs and
does not promote a winner. Jetson uses existing engines only, exclusive window,
three repetitions after smoke; no new engine build required.
