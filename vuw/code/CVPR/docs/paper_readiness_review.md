# Can speculative perception become a complete CVPR 2027 paper? — readiness review

*2026-09-27. Evidence from `directions_log.md`; new numbers in this review: `results/final/headline_bootstrap.csv`
(paired unit bootstrap, 2,000 draws, test split only) and the probe timing below.*

## 1. Verdict

**Promising core, not yet a paper.** The idea is coherent and novel, the KITTI evidence is real, and two headline
results are statistically resolved. But four gaps each block acceptance at CVPR on their own: the method does not yet
generalise to the nuScenes test split, its compute accounting ignores the planner probes (which currently cost more
than the detector time they save), it has no result on a recognised driving benchmark, and every number so far is
exploratory on splits that were looked at many times. All four are fixable in seven weeks with the GPU server, if the
scope is held tight and a go/no-go checkpoint is respected (§6).

## 2. The claims a paper would make, and where each stands

| # | claim | status | evidence (test split unless stated) |
|---|---|---|---|
| C1 | Decision changes hinge on one object (1-sensitivity) | **solid** | 96–99% single-element sufficient, 94–99.6% of \|V\| (D32, all units, 6 cells) |
| C2 | Speculative crops match or beat uniform escalation at a few % of pixels | **KITTI: solid; nuScenes: fails on test** | real crops 640: +0.055 over uniform [+0.021, +0.103]; 960: +0.071 [+0.031, +0.150] (KITTI mono brake); nuScenes mono −0.094 [−0.280, +0.067], oracle −0.055 [−0.181, +0.059] |
| C3 | A pivot refiner turns harmful escalation into gain | **KITTI: solid; nuScenes: not with the current refiner** | KITTI mono traj refiner alone +0.282 over uniform [+0.150, +0.389]; nuScenes box-feature refiner 0.013 vs 0.165 |
| C4 | Speculative + refiner beats uniform in every cell | **not resolved** | KITTI proxy: brake +0.023 [−0.022, +0.074], traj +0.119 [−0.010, +0.215] (86% / 97% of draws positive); not run with real crops + refiner; not run on nuScenes |
| C5 | A training-free pre-escalation gate (ensemble sensitivity) | **partly solid** | significant vs random on KITTI (D62) and on both nuScenes cells with a ridge mixture (D62b) |
| C6 | Decision-exactness for max-composable planners | **sketch only** | proposition and proof idea (cvpr_direction §10); 94–99% empirical agreement with rmin 10 m, 97–100% at 5 m |
| C7 | Less compute than uniform escalation | **not shown end to end** | detector: crop canvas 2.1 ms vs full 960 7.4 ms / full 640 3.8 ms; **probes: 30 ms (braking) / 74 ms (Planner B) per frame in Python** — not yet charged |

KITTI has only **6 test sequences**, so every KITTI interval is wide and rests on few units; nuScenes has 24 test
scenes but only 57 / 25 test frames where escalation changes the braking decision.

## 3. Blocking gaps

**B1. Generalisation beyond KITTI.** On nuScenes the real speculative system gives 43–61% of uniform escalation's
value on test (point estimates below uniform; intervals include zero). The analysis explains why — nuScenes value is
range correction of objects already seen, and the box-feature refiner that works on KITTI (range error −51%) barely
works there (−38%, no decision gain). *Fix:* an image-level pivot refiner (crop → metric range / 3D box head, trained
on nuScenes and KITTI 3D labels) and an asymmetric existence head with hard negatives; more nuScenes scenes (the
local blob holds 85 of 850; the full trainval gives ~10× the affected frames and real statistical power).

**B2. Compute accounting.** The planner probes (72 insertions + one deletion per detection) take 30 ms per frame for
the braking controller and 74 ms for Planner B in Python on one Xavier core — an order of magnitude more than the
2–5 ms of detector time a crop saves. The proposition's region is analytic for max-composable rules: for the braking
controller the insertion-sensitive set is {(r, y): y in the corridor, a_req(r, v) ≥ next threshold}, i.e. r ≤ r*(v),
computable in microseconds; Planner B can batch all probes in one vectorised rollout; PDM-Closed/IDM have the same
leader-gap structure. Learned planners need an amortised field (a small head trained on probe labels) or gradients.
*Fix:* closed-form/vectorised DSF, crop packing, then **end-to-end latency and energy on the Jetson** against uniform
640 and uniform 960 (the 960 comparison is where the method has a real time margin: 7.4 ms full vs ~2.1 ms per crop).

**B3. A recognised benchmark.** Current consumers are DEEP's rule-based braking controller and receding-horizon
planner, open loop; CVPR driving reviewers expect nuPlan/NAVSIM scores with a published planner. *Fix:* camera
detections → lift → PDM-Closed on nuPlan (DEEP already has this pipeline for front-camera real perception) or NAVSIM,
reporting the benchmark's own scores vs compute, with DEEP's decision value as the analysis tool, not the headline.
The perception trade-off must be reported honestly (whole-image recall 0.536 vs 0.717 for full, D3 standard metrics).

**B4. Confirmatory evidence.** Around sixty exploratory analyses touched the KITTI and nuScenes test splits; grid,
crop context, rmin and the refiner were chosen while looking. *Fix:* freeze the method now (rmin 10 m, grid, CTX 2,
refiner recipe, gate), write a pre-registration, and evaluate once on data not used today — nuPlan/NAVSIM and the
unused nuScenes scenes — keeping KITTI/nuScenes-blob as development data.

## 4. Must-haves that are not blocking but will be asked

- **Baselines at equal compute:** uniform 640/960; static horizon crop (done: 4× worse); saliency/objectness crops in
  the FOVEA/LZU style; uncertainty-driven crops; DEEP's learned routers and ORIC-style reward; periodic keyframes with
  carry-over (D35); object-relevance filtering (arXiv 2606.10688).
- **Ablations:** discovery only / pivots only / refiner only (partly done), probe grid and rmin, crop context,
  existence head on/off, gate on/off.
- **Consumers:** at least one learned planner that consumes object lists (PLUTO or Diffusion Planner on nuPlan) with an
  amortised or gradient DSF — otherwise "only works for rule-based planners" is the obvious rejection.
- **Perception stack:** a sentence-level answer to "why 2D detection + monocular lift in 2026" — either run one BEV
  multi-camera detector as the full mode on nuScenes, or scope the paper explicitly to edge monocular stacks (Jetson).
- **Theory:** the decision-exactness proposition with explicit assumptions (max-composable rule, locality of crop
  detection, 1-sensitivity) and a proof; report how often each assumption fails.
- **Positioning:** DEEP is under review at ICLR — cite it anonymously and do not re-claim its findings.

## 5. Novelty check

The closest work (self-supervised object relevance via counterfactuals, arXiv 2606.10688) scores *detected* objects to
filter them; foveation work (FOVEA, LZU, periphery-fovea) zooms by saliency, objectness or human gaze; detector routing
(ORIC, budget-adaptive routing) escalates whole frames by accuracy gain. None probes the planner for *unseen* objects,
verifies the decision's pivots, or is evaluated by signed decision value. The speculative-decoding framing with an
exactness property is, as far as the searches went, new. The novelty is adequate; the risk is empirical (B1–B3), not
conceptual.

## 6. Seven-week plan with checkpoints (deadline 2026-11-16)

| week | work | checkpoint |
|---|---|---|
| 1 (Sep 28–Oct 4) | closed-form/vectorised DSF; crop packing; end-to-end Jetson latency vs uniform 640/960 on KITTI; freeze method; write pre-registration | per-frame overhead of probes + crops < full-960 pass |
| 2–3 (Oct 5–18) | GPU: image-level pivot refiner + asymmetric existence head (KITTI + nuScenes 3D labels); full nuScenes trainval detections | **go/no-go Oct 18:** nuScenes test value ≥ uniform escalation (point) and KITTI gains hold with the learned heads |
| 4–5 (Oct 19–Nov 1) | nuPlan/NAVSIM with PDM-Closed (+ one learned object-list planner); baselines; confirmatory run | benchmark score ≥ uniform at lower measured compute |
| 6 (Nov 2–8) | ablations, failure analysis, proposition proof | — |
| 7 (Nov 9–16) | writing, figures, supplementary | — |

**If the Oct 18 checkpoint fails:** keep the KITTI + analysis story for a workshop or ICCV 2027 (March), or reframe as
"decision-exact speculative perception for monocular edge stacks" with KITTI + nuPlan only.

## 7. Recommendation

Pursue it, starting with B2 (cheap, a few days, decides whether the efficiency story exists) and B1 (the GPU refiner,
decides whether it generalises). Do not start writing the paper before the Oct 18 checkpoint.

## 8. Progress log

**2026-09-27 08:30 — B2 largely closed on KITTI.** The braking field and pivots have exact closed forms (0 mismatches on
8,008 frames, 0.41 ms instead of 21.6 ms). The deployed pipeline (TensorRT, end to end) gives braking decision value
0.237 vs 0.129 for uniform 960 and 0.123 for uniform 640 on the KITTI test split — speculative − uniform 960 = +0.108
[+0.037, +0.180] — at a mean 21.1 ms vs 24.4 ms (full 960) and 28.7 ms (DEEP's cascade); not below a lone 640 pass
(16.8 ms), and p95 35.7 ms. Remaining engineering: batch the crops of a frame into one call, GPU pre/post-processing,
cap crops per frame; vectorise Planner B probing. Claim C7 is now "cheaper than uniform 960 escalation at higher decision
value", not "cheaper than a 640 pass".

**Next:** B1 — the image-level refiner (dataset exported, training on the Jetson GPU), then speculative v1 with real crops
+ refiner on the nuScenes test split (`scripts/36_real_spec_refine.py`). B3 — nuPlan: DEEP's real-perception pipeline
(`113`–`115`) can take a speculative detection branch; needs crops on CAM_F0 images and one PDM-Closed run per branch
(several hours; only 45 affected states in nuPlan mini, so more logs are needed for a confirmatory result).
