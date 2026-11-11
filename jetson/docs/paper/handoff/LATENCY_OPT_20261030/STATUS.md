# LATENCY_OPT_20261030 status — final 2026-11-03

**Closed (2026-11-03).**
- JP01–JP03 fail the pre-registered tail screens.
- The full adaptive-deadline benchmark (K41 J04 / J03) and the stage breakdown (J02) are measured: see K41/REPORT.md.
- No optimization candidate was adopted. J01 was not run, because no candidate passed the screen.
- Final answers are at the top of REPORT.md. The 2026-10-31 text below is history.

**Authorized Jetson feasibility pilots JP01–JP03 complete. Neither exact candidate meets the pre-registered DESP tail-improvement screen on either dev dataset.** See [FEASIBILITY_SUMMARY](FEASIBILITY_SUMMARY.md). The broader Appendix C/K41 workload remains partial; other RUNPLAN jobs are pending authorization.

JP01 columns-only: p95 baseline→candidate58.748696→59.018695ms nuScenes,111.691552→111.361024ms AV2. Δ95% CIs[−0.881167,+1.378397]/[−2.371471,+0.697829]ms. Prefix parity passed712+712 frames,47 endpoints/frame; clean scalar parity66 dataset/endpoint pairs, zero mismatches. Detailed report, all repetitions, stage/conditional profiles, quality/pixel/deadline/metadata tables retained.

JP02 fixed-context C1 encoder probe:120 exact synthetic comparisons; batch1 mean1.920978→1.797600ms,0.123378ms/6.42% saved. Larger buckets improve0.29–2.12%; this is not a pipeline gain.

JP03 C1-only full pipeline: p95 baseline→graph61.886440→61.149196ms nuScenes (1.19%),115.241185→114.971293ms AV2 (0.23%). Δ95% CIs[−1.514577,+0.847415]/[−1.870542,+2.777268]ms. Mean decreases0.076/0.122ms; nuScenes p99 increases0.596ms, AV2 decreases1.629ms, both tail intervals include0. All47 debug endpoints/frame pass on712+712 frames;3,167 actual encoder calls pass graph/direct exact equality. Clean parity112 dataset/endpoint pairs ×1,536 frame/repetitions=172,032 comparisons, zero mismatches.12 clean timings complete; all summaries/plots collected and validated.

Protocol:512 timed frames/dataset after200 warmup,3 alternating repetitions/arm,13 nuScenes scenes/8 AV2 logs. Frozen inputs/weights/configuration, equally optimized whole-frame and bounded rows. No test selection, holdout or energy claim. Different pilot baselines belong to different windows; no combined-candidate result measured.

Operational limitations: JP01 original wrapper exit/restoration unobserved, device subsequently rebooted; all20 DONE/artifact sets verified and controls restored during explicit recovery. JP03 wrapper/tool session disappeared during repetition2; fifth clean child finished, exit code unknown. Original state preserved; five complete trials retained, seven continued in original order with a detached controller. No device reboot then; idle acquisition gap disclosed, no latency-selected discard/rerun. Final clocks/fan restoration succeeded; pause removed;88 protected hashes and source/engine hashes unchanged. Whole JP03 window max GPU53°C/CPU52.5°C; these do not certify per-row energy or total TRT memory.

| ID | Status |
|---|---|
| LAT-1 | Audit plus baseline profile complete. J02 measured per-stage means (all frames and ≥ p95) for draft / crop rounds / C1 / refiner / rest: the tail is crop rounds + C1. Finer 13-stage kernel / byte attribution was not done (per-stage syncs already add about 4%) |
| LAT-2 | Existing K29 ROI extraction audited;285 CPU pixel fixtures passed; not a new gain |
| LAT-3 | Existing bank audited; C1 graph synthetic/real-prefix parity and3-repeat pipeline timing complete; no meaningful tail gain |
| LAT-4 | Columns-only implemented/tested; prefix parity passes, DESP tail screen fails |
| LAT-5 | Dependency / merge / geometry audit complete; blind dependent batching rejected. Guard and next-round cost is covered by the J02 crop-round stage; a separate guard-depth profile was not done |
| LAT-6 | Refiner audit complete; existing support/batch/buffer optimizations retained, no new backend adopted |
| LAT-7 | Existing bounded rows measured with decision/pixel/deadline trade-offs; new scheduling variants unrun |
| LAT-8 | Two fair exact-candidate pilots fail. Full test adaptive / in-loop deadline benchmark measured (J04: 18 rows, D = 33.3 / 50 / 66.7 / 100 / ∞, 3 repetitions). No row beats DESP after Holm; adverse n.s. gaps are reported. Energy at window level only. Optimized-vs-original lane is identical (no adopted candidate) |

CPU evidence retained:102 release tests pass/3 CUDA skipped; original pixel/cache/NMS/column fixtures pass;800 quantile-bootstrap cases agree with literal resampling (max difference2.22e−16). GPU-visible OpenCV/Torch/TensorRT/CUDA preflight passes. Historical K39 and original Tab.6 protocols remain separate. Analysis source snapshots/hashes, raw windows and operational recovery records make scope reviewable.

2026-11-01 J04/J02/J03 window started on the Jetson (see K41/STATUS.md and the K41 PREREG J04 addendum): same K39-1 backend for every row (staged fast, trt32, static CUDA graphs), adaptive K33 rows + uncertainty crops added, in-loop deadlines for iterated rows. No optimization candidate is included (JP01/JP03 failed the screen), so LAT-8's "optimized vs original" lane is identical to the original; J01 is not run.

2026-11-02 J04 window COMPLETE (12/12 runs, window released, hashes unchanged); analysis in K41/REPORT.md "J04 / J02 / J03" and K41/J04/*.csv. DESP v6-S / B2K-A parity with K38-1 and K34-3 is exact (0 mismatches). No row beats DESP after Holm at any D. AV2 shows adverse, non-significant gaps vs whole frame at D ≥ 66.7 ms and vs u640 at 50 ms; DESP collapses on nuScenes at 33.3 ms. J02: the tail is crop rounds + C1. The energy figure is window level only. J01 is not run (no adopted candidate).
