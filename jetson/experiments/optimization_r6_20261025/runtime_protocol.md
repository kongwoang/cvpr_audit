# Jetson component feasibility, declared before timing

Measure the two bounded observation readouts, after proposal deduplication, using
stored GT-free scene/proposal/history inputs for every 152/300 development unit.
Three repetitions, alternate lag-1/lag-2 execution order, warm up first 200 frames.
Reset two-frame history at each scene/log. Time only the function's matching,
current geometry demand, real-observation C1 and historical-action cap. Include
early returns in full-population means; also report conditional active frames.

Validate all final actions against the VUW reference on all frames; the optimized
function is first checked remotely against the full 20-rule reference. Runtime
inputs exclude GT/images and contain only original fixed parent observations.
Paired scene/log bootstrap 2,000 seed0; retain repetitions within a sampled unit.
Report mean/p95/p99 and per-repetition summaries, plus lag-2 minus lag-1 intervals.

This is a **component microbenchmark**, not end-to-end latency. It excludes the
unchanged detector, cross-crop duplicate suppression, tensor transfers, proposal
buffering, R5 classifier, image decode and refiner. It cannot establish total
overhead, FPS, real-time tails, or trt32 decision parity. No network improvement
is inferred from it. It is useful even for a quality-negative row: it separates
decision risk from the CPU feasibility of the small added rule.

Use the agreed exclusive Jetson protocol, MAXN, locked clocks, fan255, one CPU
thread; record hardware and restore owned controls in finally. Frozen source
hashes are checked before/after; no engine, checkpoint or core write.
