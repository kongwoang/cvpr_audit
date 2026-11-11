**Result (A)** (post-hoc analysis; AV2 with the ego mask). Matched on TENSOR px rather than crop area, ours final (with refiner) is at or above every crop heuristic + refiner. On AV2 it is the best point at its px.
- **Equal-px rows.** The heuristic budget is a per-frame cap on its crops' canvases; guard re-detections come on top. So the equal-px comparison is the 0.5× budget row, whose REALISED px is 0.96–1.18× ours' (heuristics get up to 18% more px).
- **nuScenes new test** (ours 0.170 at 0.100 px):
  - above temporal_crit + ref [+0.018, +0.118], random + ref [+0.044, +0.381] and ego-corridor + ref [+0.021, +0.656];
  - uncertainty + ref [−0.031, +0.490] and traj_support + ref [−0.005, +0.413] n.s.;
  - without refiner, above all five heuristics.
- **AV2 masked** (ours 0.122 at 0.053 px):
  - above traj_support + ref [+0.013, +0.073], ego-corridor + ref [+0.018, +0.082] and random + ref [+0.063, +0.183];
  - uncertainty + ref (0.099) [−0.031, +0.081] and temporal_crit + ref (0.076) [−0.007, +0.100] n.s.;
  - heuristics + ref reach ours' G only at ≥ 1.5× its tensor px (uncertainty 0.119 at 1.5×, 0.133 at 1.9×).
- **Frame routers.** At 0.5 / 1 / 2 / 4× ours' tensor px the best Budget-Adaptive Routing and the decision-value router stay at G ≈ 0 (−0.02 … +0.11). Ours is above both at 1× ([+0.076, +0.262] nuScenes, [+0.064, +0.182] AV2). The oracle router is above ours (0.342 / 0.315).
- **Missing AV2 cell** (ego-corridor at ours' crop AREA, K14 run): G 0.029 without refiner and 0.113 with refiner, at 0.066 tensor px. Ours final minus it: [+0.043, +0.150] without refiner, [−0.018, +0.040] with refiner.
- **(B) Jetson tail latency** (frames of K11 / K13(e); bounded-call ours are NEW configurations, reported as variants).
  - **Frozen ours.** nuScenes: p95 / p99 / max 53 / 69 / 125 ms vs native 34 / 35 / 36 ms. AV2 masked: 82 / 110 / 134 ms vs native 66 / 88 / 128 ms.
  - **N = 2 detector calls after the draft:**
    - nuScenes: 40 / 49 / 97 ms, G 0.159 vs 0.173 (ours − it [−0.011, +0.051]), same action as frozen ours on 99.2% of frames, bound hit on 12%;
    - AV2: 58 / 77 / 94 ms, G 0.167 vs 0.173 ([−0.003, +0.023]), 98.6% agreement, bound hit on 32%. Every AV2 tail statistic is now below native's.
  - **N = 3:** nuScenes 48 / 60 / 112 ms, G 0.170; AV2 71 / 94 / 114 ms, G 0.174; 99.4% / 99.1% agreement.

# K18: fairness of the adaptive baselines

**(A) Decision value against tensor px** (`pareto_{nusc,av2}.csv`, figure `pareto.pdf/png`, details in `heur_{ds}.csv` and `routers_{ds}.csv`)
- **Heuristics** (`remote/A_method/k18_heur.py`). Ours (frozen) runs first on each frame. Then each heuristic adds crops in its own order while their summed canvas px (32-px rounding, as the detector sees them) stays within b × ours' crop tensor px on that frame, b ∈ {0.5, 1, 2, 4}. The tube heuristics take the widest tube whose canvas fits.
  - **Realised px, nuScenes:** 0.96–1.15× ours at b = 0.5; 1.3–1.8× at b = 1.
  - **Realised px, AV2:** 1.11–1.18× ours at b = 0.5; 1.2–1.7× at b = 1.
  - **Saturation:** uncertainty, traj_support and the corridor saturate near 1.9–2.1× ours' px because they run out of candidates, so 2× and 4× are not reached for them. The realised ratio is in `px_ratio_to_ours`.
  - **Intervals:** paired unit bootstrap of ours final minus each row (2,000), with and without refiner.
- **Routers** (`k18_routers.py`). Same scores and calibration as K12. The routing share is searched until the router's mean tensor px is within 2% of the budget; all rows are within 2% except skip-type routers at 4× on nuScenes, which cannot reach it.
- **Ours' other operating points.**
  - nuScenes: the K8 operating table (v_close, rounds, crop scale; px area-based, as in K8).
  - AV2: v_close 5 / 10 with the car probe (K17; tensor px 0.068 / 0.075, G with refiner 0.081 / 0.082). The frozen point (0.053, 0.122) dominates them.
- **Whole frame.** u960 / native (with refiner) come from K14; u640 / u1280 / u1536 from the K1 / K10 tables.

**(B) Tail latency on the Jetson** (`latency_bounded_{nusc,av2_masked}.csv`, figure `latency_bounded.pdf/png`; script `scripts/59_bounded_latency.py`; exclusive Jetson windows, DEEP pause protocol).
- **Frames and pipelines.** Same frames, whole-frame engines, lift and fast machinery as K11 (nuScenes, 3,176 timed frames) and K13(e) (AV2 with the ego mask, 586 frames). Every pipeline is timed per frame; mean / median / p95 / p99 / max are reported.
- **Frozen reproduces K11.** Ours (frozen, batched refiner) reproduces K11's mean (25.4 ms nuScenes) and K13(e) (36.9 ms, G 0.173 AV2).
- **Bounded variant `ours_bN`.** At most N detector calls after the draft:
  - each call detects, in ONE batch on a common canvas, the new field + support crops of the current merged scene and the grown crops that the guard queued in the previous call (guard growth is deferred by one call);
  - the scene is merged and lifted after each call, as in speculative.run;
  - at the bound, the current scene's action is returned, then the refiner is applied in one batched call.
- **Engine executions per frame** (draft + detector calls + refiner): 2.5 / 2.7 mean, max 5 / 6. For comparison, frozen ours makes 3.1 / 4.0 engine calls per frame on average, max 16.
- **Tensor px.** The common canvas costs a little: nuScenes 0.103 (N = 2) and 0.112 (N = 3) vs 0.096; AV2 0.068 / 0.074 vs 0.057.

| | p95 / p99 / max ms (nuScenes) | G | p95 / p99 / max ms (AV2 masked) | G |
|---|---|---|---|---|
| native | 34 / 35 / 36 | 0.175 | 66 / 88 / 128 | 0.125 |
| u1280 | 25 / 25 / 27 | 0.143 | 33 / 43 / 69 | 0.131 |
| ours (frozen) | 53 / 69 / 125 | 0.173 | 82 / 110 / 134 | 0.173 |
| ours, N = 2 (variant) | 40 / 49 / 97 | 0.159 | 58 / 77 / 94 | 0.167 |
| ours, N = 3 (variant) | 48 / 60 / 112 | 0.170 | 71 / 94 / 114 | 0.174 |

- **Interpretation.** Bounding the calls removes most of ours' tail: on AV2, N = 2 is below native on p95, p99 and max at nearly the same G.
- **Where ours still loses.** On nuScenes, native's very flat tail (34–36 ms) stays below ours at any bound.
- **Comparing G with the paper's numbers.** G on these timed frames is the board's subset (K5 / K11 frames), not the full test split.
