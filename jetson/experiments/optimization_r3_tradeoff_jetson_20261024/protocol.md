# R3 tradeoff Jetson window

Six sandbox rows: frozen v6-S, round1 B2K-A+recovery, image-point fallback, compact box-only point fallback, critical-extent persistence, corroborated upper fallback. All defined on training/CFG-DEV; no thresholds changed, no test data. Same fixed 712-frame prefixes, 200 warm-up, 512 timed, three repetitions; rotated pipeline order. Each pipeline has independent histories.

Compact 12,545-parameter MLP executes on CPU FP32 and consumes the same observed-box/mask features. Image point/corroborated use original trt32 C1 image encoder + FP32 readout. Exact old-Jetson baseline/image-point parity is checked on 80 frames/dataset. Torch/TRT image-encoder differences are reported for all candidate actions. Extent seed is inserted before support at first bounded round, matching the VUW order. No engine builds.

Report all quality/runtime rows regardless of full-development outcomes. Detector padding/backend differ from VUW; cross-backend decisions must be compared explicitly. Exclusive window, MAXN/fan255/locked clocks, restore in finally. Main source and engine hashes must match. This is an exploratory feasibility comparison, no core promotion.
