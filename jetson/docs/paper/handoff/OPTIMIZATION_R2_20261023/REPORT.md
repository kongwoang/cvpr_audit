# Optimization round 2 — case-led analysis and isolated experiments

**Completed:** direct review of nine actual three-frame cases and a full 452-unit / 29,782-frame CPU screen (20 rows).
**Scope:** development exploration; main code, weights, thresholds and frozen K34 results unchanged. No method promoted.
**Intervals:** paired unit bootstrap, 2,000, seed 0; pointwise and exploratory.

## Findings from the images and code

- A gap in verified observations can remove the C1 chain even when cheap detects the object. Fresh instrumentation also confirms a same-class ID switch in retrospective recovery (GT track 21 linked to 25). Restoring actual history in another case changes TTC from infinity to 6.10 s, still above GT 4.21 s.
- Refined range can be closer to GT but cross the wrong action threshold: the consumer takes the maximum of an ego stopping term and a TTC-derived closing term. See the [measured case figure](range_decision_valley.pdf).
- Episode-level flags must not be assigned to every frame. The cyclist case is matched through its person box at frames 37–40; the missing match occurs at frame 41. The initial visualization used overly strict class equality and was corrected to the frozen person/cyclist compatibility rule; aggregate evaluations were unaffected. A severe crossing truck and clipped edge object additionally expose spatial/visibility limitations.
- Some cheap false alarms are correctly removed. Preserving cheap braking or weakening every correction has a measurable cost.

Evidence and all nine image links: [CASE_REVIEW.md](CASE_REVIEW.md).

## Full-development CPU screen

Refiner postprocessing rows are exact on unchanged acquisitions; motion rows are **frozen-query counterfactuals**, requiring fresh inference for actual method claims. No threshold fitting or GT enters a policy.

| Dataset | Row | Missed / episodes | False / 1,000 | Insufficient loss / frame | G |
|---|---|---:|---:|---:|---:|
| av2 | v6-S | 59/366 | 7.860 | 0.37983 | 0.30871 |
| av2 | cheap | 30/366 | 35.960 | 0.31396 | 0.00000 |
| av2 | no refiner | 44/366 | 12.508 | 0.31324 | 0.27601 |
| av2 | round1 recovery | 49/366 | 8.451 | 0.36838 | 0.31996 |
| av2 | camera anchored | 60/366 | 7.691 | 0.38151 | 0.31297 |
| av2 | median correction 3 | 59/366 | 8.409 | 0.38273 | 0.30847 |
| av2 | half log correction | 52/366 | 9.592 | 0.33783 | 0.32030 |
| av2 | looming height 3 | 57/366 | 8.113 | 0.37929 | 0.30749 |
| av2 | looming width-height 3 | 57/366 | 7.944 | 0.37966 | 0.30862 |
| av2 | looming width-height 2 missing | 57/366 | 8.663 | 0.37200 | 0.31183 |
| nusc | v6-S | 40/99 | 2.943 | 0.38735 | 0.32117 |
| nusc | cheap | 34/99 | 27.955 | 0.50385 | 0.00000 |
| nusc | no refiner | 39/99 | 6.539 | 0.39114 | 0.31297 |
| nusc | round1 recovery | 38/99 | 3.270 | 0.38482 | 0.32469 |
| nusc | camera anchored | 40/99 | 2.943 | 0.38735 | 0.32117 |
| nusc | median correction 3 | 40/99 | 2.943 | 0.38846 | 0.31911 |
| nusc | half log correction | 39/99 | 4.414 | 0.38985 | 0.31833 |
| nusc | looming height 3 | 40/99 | 3.106 | 0.38701 | 0.32156 |
| nusc | looming width-height 3 | 40/99 | 2.943 | 0.38735 | 0.32117 |
| nusc | looming width-height 2 missing | 39/99 | 4.414 | 0.38428 | 0.31411 |

**Tradeoff, not a solved weakness:** half-log correction rescues 7 AV2 episodes, with no newly missed baseline episode, but adds 1.733 [0.760, 2.834] false episodes/1,000. Loss/frame falls 0.0420 [0.0326, 0.0523]; G difference +0.0116 [−0.0003, 0.0237] is uncertain. On nuScenes it rescues one episode, increases false alarms, and does not establish a loss/G improvement.

The three-observation width/height floor rescues two AV2 episodes with a very small loss effect; it changes no nuScenes actions. Camera anchoring loses one AV2 episode. Median smoothing rescues two and loses two. These negative and limited results are retained.

[All summary rows](screen/summary.csv), [intervals](screen/intervals.csv), [paired contrasts](screen/contrasts.csv), [rescued/newly missed counts](screen/episode_transitions.csv), [unit concentration](screen/concentration.csv), [tradeoff figure](screen_tradeoffs.pdf).

## Fresh in-loop experiment

Five acquisition rows, each with no-refiner/v3a/half-log controls: original v6-S, round-1 combined, proactive completion, proactive + recovery, and verified looming + recovery. Each owns its temporal history. Pending completion keeps one target for two future frames, coalesces existing coverage and admits only actual detections into C1 history. All bounded rows retain the two-batch cap; crop executions/pixels can increase.

Protocol and code: `experiments/optimization_r2_20261023/inloop_protocol.md`, `runner.py`, `proactive.py`, `estimators.py`. Smoke uses the full sequences of the nine diagnostic cases and checks baseline/round-1 parity. The coordinator then runs all 152/300 development units through the unchanged GPU wrapper on VUW/VUW1.

**Full in-loop development finished:** all 452 units and 32 rows. [In-loop report](INLOOP_REPORT.md), [summary](dev/summary.csv), [paired contrasts](dev/contrasts.csv).

| Dataset | Acquisition (v3a) | Missed / episodes | False / 1,000 | Loss / frame | G | Requested px / native |
|---|---|---:|---:|---:|---:|---:|
| nusc | v6-S | 40/99 | 2.943 | 0.38735 | 0.32117 | 0.08037 |
| nusc | round1 combined | 38/99 | 3.923 | 0.36845 | 0.35054 | 0.07497 |
| nusc | proactive | 39/99 | 3.760 | 0.37103 | 0.34646 | 0.07525 |
| nusc | proactive + recovery | 37/99 | 3.923 | 0.36810 | 0.35069 | 0.07558 |
| nusc | looming + recovery | 38/99 | 3.923 | 0.36845 | 0.35054 | 0.07497 |
| av2 | v6-S | 59/366 | 7.860 | 0.37983 | 0.30871 | 0.05556 |
| av2 | round1 combined | 49/366 | 9.001 | 0.36851 | 0.31942 | 0.05141 |
| av2 | proactive | 54/366 | 9.212 | 0.37753 | 0.30898 | 0.05134 |
| av2 | proactive + recovery | 48/366 | 9.423 | 0.36642 | 0.32078 | 0.05221 |
| av2 | looming + recovery | 47/366 | 9.043 | 0.36834 | 0.31937 | 0.05141 |

## Mechanism checks and next decisions

**Small improvements do not yet justify a method change.** Proactive+recovery saves one more episode per dataset than round-1 combined, but its sole nuScenes rescue brakes because another object has an erroneous TTC. The actual decisive object remains unmatched. On AV2, it adds .423 [.169, .719] false episodes/1,000 and +1.55% requested pixels. The looming addition saves two AV2 episodes, with a tiny loss effect and no nuScenes change. No additional hard-demand episode is rescued by either addition.

A stronger association gate rejects a confirmed wrong identity but loses a useful early recovery: the fresh nine-unit pilot worsens nuScenes and leaves AV2 actions unchanged. See [identity audit and negative pilot](ASSOCIATION_REVIEW.md), [actual-image identity switch](recovery_identity_switch.pdf), and [coincidental rescue](coincidental_rescue.json).

**C1 audit completed:** all 14,696 original validation examples; original MAE reproduced within 3e-7. C1 beats T4 overall. High-GT-rate examples tend to be underestimated, but AV2 is close to calibrated when grouped by its runtime prediction; a blanket upward correction is not supported. The original training pipeline labels current detections without checking the historical GT identity. [Full head review](HEAD_REVIEW.md).

**Recommended next experiments:** identity-aware training-data audit, a temporal validity/uncertainty output, and joint range/motion decision-loss calibration; measured-motion queries for low ego speed need their own cost budget. [Feasibility, tradeoffs and falsifiers](NEXT_STEPS.md). These are proposals, not measured improvements.

No new Jetson timing claimed. New proposals will need actual padded-pixel/launch accounting and an exclusive timing window if their quality/cost tradeoff justifies it. GPU development elapsed time is not Jetson latency.

Verification: all stored baseline actions/costs replay exactly in the CPU screen; semantic tests cover ambiguous matches, stationary/shrinking motion, conservative growth, camera-ray coupling, pending expiry and genuine-chain completion. One initial CPU attempt failed on a pandas column-name access before producing aggregate rows; the failed status is retained as `screen/FAILED_attempt1.json`, corrected and rerun in full.
