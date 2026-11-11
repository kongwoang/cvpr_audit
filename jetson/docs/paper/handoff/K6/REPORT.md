# K6: Bench2Drive closed loop, 220 routes × 5 perception branches (single run), 2026-10-04

## What was run
- **Benchmark:** Bench2Drive, all 220 base routes, CARLA 0.9.15 headless. The leaderboard is the carla_garage copy, with two crash-cleanup patches listed in `third_party/carla_garage/PROVENANCE_CVPR.txt`.
- **Planner:** PDM-Lite, the privileged rule-based expert, with **perception in the loop**: `remote/A_closedloop/perception_pilot.py` (md5 ff4d5c29…).
- **How perception enters the loop:**
  - Every vehicle or walker that projects into the front camera (1600×900, FOV 70) is kept only if a detection of the branch matches it (Hungarian matching, IoU ≥ 0.3, compatible class).
  - Matched actors keep their **true** state. Unmatched in-camera actors are hidden from the planner.
  - Unmatched detections become static pseudo-obstacles.
  - Actors outside the camera, the map, the traffic lights and the scenario information stay privileged.
- **Branches** (frozen COCO YOLOv8s): privileged, cheap 192×320, u960, u1600 (native), and ours = nuScenes v3 (draft → braking field + support crops at 0.6, guard, ≤ 3 rounds).
- **The refiner is not used.** Matched actors keep their true range, so range correction cannot act under this protocol; this is the "ours" of this task.
- **Commands:** dispatcher `b2d_dispatch.sh` (MAXJ = 4, through run_when_gpu_free.sh, SHARE=1), then `python scripts/A_closedloop/b2d_summary.py` (md5 089525db…). It merges each branch with Bench2Drive's official `tools/merge_route_json.py`.
- **Definitions** (Bench2Drive):
  - DS = Σ score_composed / 220.
  - Success = status Completed and no infraction other than min-speed.
  - Paired route bootstrap of ours − branch, 2,000 resamples.
- **Sources:** `results/vuw_summaries/b2d/` (copied here), from VUW `results/b2d/summary/`.

## Results (b2d_summary.csv; per route in b2d_per_route.csv)

| Branch | Routes run | DS (official, /220) | SR | Route completion | Vehicle collisions | Pedestrian collisions | Layout collisions | ours − it: DS [95% CI] | ours − it: SR [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| privileged | 219 | **96.00** | 0.909 | 98.3 | 7 | 0 | 2 | −3.26 [−5.92, −0.84] | −0.073 [−0.123, −0.027] |
| cheap 320 | 220 | 92.74 | 0.850 | 97.3 | 20 | 5 | 1 | +0.29 [−1.19, +1.75] | −0.014 [−0.046, +0.014] |
| u960 | 219 | 92.63 | 0.850 | 96.9 | 19 | 1 | 2 | +0.12 [−1.88, +2.15] | −0.014 [−0.055, +0.027] |
| u1600 native | 219 | 91.86 | 0.841 | 96.9 | 26 | 0 | 2 | +0.90 [−1.64, +3.45] | −0.005 [−0.046, +0.037] |
| **ours** | 219 | **92.75** | 0.836 | 98.0 | 19 | 5 | 0 | — | — |

The paired comparisons use the 219 routes common to the branches.

**Perception statistics per frame** (front camera, from `perception_<branch>.jsonl`; only routes run after the 2026-10-03 01:15 logging fix):

| Branch | Actors in camera | Hidden | Unmatched detections ("phantoms") | Crops | Routes logged |
|---|---|---|---|---|---|
| cheap | 3.27 | 1.16 | 0.73 | 0 | 125 |
| u960 | 3.36 | 0.91 | 1.36 | 0 | 202 |
| u1600 | 3.36 | 0.91 | 1.34 | 0 | 155 |
| ours | 3.10 | **1.19** | 0.92 | 1.17 | 223* |

\* Includes re-runs from the crashed chunk0 attempts; weighted by frames.

## Reading and caveats (against the method)
1. **No closed-loop gain.** Ours ties cheap, u960 and u1600: all four perception branches sit at DS 91.9–92.8, and every CI includes 0. All of them lose about 3 DS points to privileged perception, significantly.
   - On these routes ours hides as many in-camera actors as the cheap draft (1.19 vs 1.16 per frame; u960 hides 0.91).
   - The braking field ignores actors that do not affect the ego's braking decision, which is by design. PDM-Lite, however, also uses those actors for lane changes and for yielding at junctions.
2. **The failures are shared and mostly not about detection quality.** `b2d_failures_by_scenario.csv`:
   - AccidentTwoWays and ConstructionObstacleTwoWays fail on 4–5 of 5 routes for every perception branch and 0 for privileged. Cones, warning triangles and construction props are not COCO classes and are not CARLA actors handled by the matching, so a likely cause is shared by all detectors (not verified per route).
   - Red-light and merging failures also occur for privileged.
3. **Not all failures are driving failures.** 6–13 routes per branch end with "Failed – TickRuntime" (a simulator / runtime timeout). Under the official scoring these count as failures; they are not driving errors.
4. **Route 11755 (Town12):** CARLA crashed repeatedly in four branches (privileged, u960, u1600, ours), so those branches have 219 routes. The official DS divides by 220, which counts that route as 0.
5. **Parked-car caveat.** CARLA's parked cars are map props, not actors, so a correct detection of one counts as an unmatched "phantom" and becomes a static pseudo-obstacle.
   - Unmatched detections per frame: cheap 0.73, ours 0.92, u960 1.36, u1600 1.34. Most of the extra "phantoms" of the high-resolution branches are probably such props, but we cannot separate them: prop geometry is not exposed by the evaluator.
6. **Three runs:** see the update below. Ours' perception statistics were logged on a subset of routes (table above).

## Update 2026-10-07: three runs (`b2d_repeats_summary.csv`, `b2d_repeats_per_run.csv`)

**How the runs were made**
- Runs 2 and 3: `remote/A_closedloop/b2d_repeats.sh`, with the same dispatcher, branches and code.
- Route 11755, which crashes, is removed from every run, leaving the 219 routes that ran in every branch.
- Run 2 was paused 2026-10-05 05:55 → 10-06 13:28, because VUW was reserved for other jobs, and was resumed from its checkpoints.
- Summary script: `remote/A_closedloop/b2d_repeats_summary.py`.
- **Definitions:**
  - DS here is the mean over the 219 routes (not the official /220 with route 11755 counted as 0).
  - The interval is a paired route bootstrap (2,000) on the run-averaged per-route DS / success.

| Branch | DS run 1 / 2 / 3 | **DS mean ± std** | SR mean ± std | RC mean ± std | ours − branch DS [95% CI] | ours − branch SR [95% CI] |
|---|---|---|---|---|---|---|
| privileged | 96.44 / 96.72 / 96.74 | **96.63 ± 0.17** | 0.915 ± 0.003 | 98.53 ± 0.25 | −3.83 [−6.24, −1.56] | −0.070 [−0.114, −0.027] |
| cheap 320 | 92.89 / 91.22 / 92.71 | 92.27 ± 0.92 | 0.834 ± 0.018 | 97.59 ± 0.40 | +0.53 [−0.66, +1.68] | +0.011 [−0.012, +0.033] |
| u960 | 93.06 / 92.46 / 93.93 | 93.15 ± 0.74 | 0.857 ± 0.010 | 96.92 ± 0.65 | −0.35 [−1.96, +1.21] | −0.012 [−0.046, +0.020] |
| u1600 native | 92.28 / 91.14 / 92.44 | 91.95 ± 0.71 | 0.837 ± 0.013 | 96.94 ± 0.07 | +0.85 [−1.13, +2.70] | +0.008 [−0.030, +0.041] |
| **ours** | 93.18 / 92.07 / 93.16 | **92.80 ± 0.63** | 0.845 ± 0.012 | 97.65 ± 0.44 | — | — |

**Reading:** the single-run conclusion holds over three runs.
- The four perception branches lie within one run-to-run standard deviation of each other (0.6–0.9 DS), and every ours − branch interval includes 0.
- All four perception branches are about 3.8 DS below privileged perception, significantly.
- **There is no closed-loop gain or loss from ours** on Bench2Drive with PDM-Lite under this protocol.
