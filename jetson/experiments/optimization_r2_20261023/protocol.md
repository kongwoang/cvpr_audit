# Round 2: case-led method and execution improvements — 2026-10-23

User authorized a second isolated optimization round and direct inspection of
failure cases. Main sources, weights, earlier runs and K34 stay frozen. All code,
records, images and reports for this round live in this experiment or its handoff.
Remote writes remain under /vol/dl-nguyenb5-solar/users/hoangpc/CVPR.

Development only for new method experiments: 152 nuScenes val scenes and 300 AV2
CFG-DEV logs. No new A-HD/test-based selection. Earlier inspected tests motivated
hypotheses; exploratory development evidence is not confirmatory evidence.
Paired unit bootstrap: 2,000, seed 0. Preserve every attempted row and negative
result. G keeps the original cheap-previous-action jerk convention. Report miss,
onset, false episodes, insufficient loss, high-demand strata, logical/padded
pixels, executions, latency and resource costs separately.

1. Inspect representative development cases from stored final scenes, real
   images and GT annotations: remaining missed episodes with missing C1 chain,
   refiner action drop, and absent decisive object, plus cheap false alarms
   correctly removed by v6-S. Freeze the case manifest before image review.
   Selection is diagnostic and intentionally outcome-conditioned, not an
   unbiased quality sample. Record overlapping flags, don't infer causality.
2. Audit new proposals against every stored baseline development scene before
   expensive inference. Temporal evidence reuse must use only earlier frames;
   predicted/borrowed evidence must not be labelled a measured C1 observation.
   Physically meaningful refiner variants must retain camera-ray coupling.
   Stored-scene counterfactuals don't establish behavior after altered queries.
3. Specify exact candidate policies and falsifiers in an addendum before their
   evaluation. Advance promising mechanisms to in-loop dev experiments with
   independent histories, fresh inference for changed inputs, and baseline
   replay parity. Choose scheduling for server load, never for favorable results.
4. For Jetson consider the measured launch/padding tradeoff, not logical pixels
   alone. Any canvas change is an input change and needs decision evaluation;
   exact execution changes require parity. Existing engines only. Exclusive
   window, locked clocks/fan, restore owned controls. No timing overlap.

GPU jobs only via unchanged run_when_gpu_free.sh, SHARE=1, MIN_FREE_MB=10000;
two workers per wrapper, modest CPU threading. CPU analyses use one thread.
Record child status, source hashes, manifests and progress; resume atomic units
after failures. Local handoff: docs/paper/handoff/OPTIMIZATION_R2_20261023/.
