# R6: observation corroboration after negative verification

Declared 2026-10-25 before R6 outcomes. User requested a final creative,
paper-informed improvement round. Frozen core, weights, K34, thresholds and
test splits remain unchanged. This is adaptive development after five rounds,
not a confirmatory result or promotion.

## Evidence and scope

R5 found current cheap evidence lost after crop verification, especially a night
car and a near edge bus. Learned gates generalize poorly on selected refiner
disagreements. Other important losses have no current matched detection, or arise
from fragmented/edge objects; retention alone cannot solve them. The finite
controller also leaves continuous loss even for the reference discrete action.

ByteTrack (https://arxiv.org/abs/2110.06864), algorithm 1/section 3, motivates
matching low-score detections to existing evidence and forbidding low-score
births. We borrow this principle, not its tracker or MOT performance claim.
OC-SORT (https://arxiv.org/abs/2203.14360) motivates anchoring state recovery to
actual observations. We do not interpolate future observations or claim to
implement OC-SORT. Bernoulli filtering with thresholded measurements
(https://arxiv.org/abs/2606.09573) explains why absence is conditioned on sensing;
its simulated multi-sensor detection model is unavailable here, so no invented
existence probabilities or independence assumption is used. Selective one-sided
prediction (https://proceedings.mlr.press/v130/gangrade21a.html) motivates showing
coverage/error tradeoffs; its guarantees do not apply to these hand rules.

## Fixed acquisition and data

All 152 nuScenes new-val scenes and 300 AV2 CFG-DEV logs. Original AV2 geometry exclusion for boxes at least half inside y>=1830 (the original wrapper; no new image mask). Parent acquisition: R3 point fallback + critical extent retention,
`v6S_b2A_missing_extent_retro`; preserve its actual queries, geometry, history,
refiner outputs and future state. R6 changes final action readout only. No
training, new threshold fitting, or test access. Controls: cheap, frozen v6-S,
round1, R3 parent, R5 parent+residual q=.5/.8/.95.

The frozen detector already postprocesses at confidence .10 before the consumer
filters at IE.THR. Re-run exactly the recorded CURRENT crops to recover these
proposals. Do not add queries or change crop shape/scale/NMS. Merge duplicate
proposals with the existing same-class NMS IoU .65; record source crop provenance.
Audit every stored final full box against the re-run high-score proposals; a
parity failure blocks claims of exact cached-query reuse. Offline replay compute
is recorded separately; it is not additional deployed detector work. Postprocess,
matching and buffering overhead remains unmeasured until timed.

## Fixed hypotheses and rows

All matches are one-to-one compatible vehicle-vs-VRU IoU>=.3 using the existing
R4 matcher. No GT enters eligibility, matching, demand or action. A residual cheap
object is current, unmatched to the final parent scene, its centre is inside an
executed crop, and its original demand reaches DECELERATE.

Cheap residual actions, each applied as max with the R3 parent action:
1. ungated (ceiling/tradeoff control);
2. observed in one previous cheap frame;
3. observed in two consecutive previous cheap frames;
4. action also corroborated in one previous cheap frame (min current/prior action);
5. action corroborated in two previous cheap frames;
6. current low-score crop box corroborates its existence;
7. current low-score crop box also corroborates its ego-only demand (min actions);
8. current low-score OR one previous cheap observation;
9. current low-score AND one previous cheap observation;
10. current low-score OR two previous cheap observations;
11. current low-score AND two previous cheap observations.
The low-score subset is .10<=confidence<IE.THR; high-score proposals discarded by
the final merge are an explicit additional `proposal_any` diagnostic row.

Low-score current crop objects not already matched to the final scene:
12. anchored to a previous full observation; current geometry, TTC infinity
    unless a genuine two-lag same-fidelity chain exists;
13. require that genuine two-lag chain;
14. additionally clip action to the previous full object's refined action;
15. two-lag chain and clip to both previous refined actions.
Use frozen C1's median pairwise looming/gate on these explicit compatible chains.
Low boxes never update the parent history or create persistent unobserved tracks.

Compositions, declared before outcomes: low+one-observation cheap residual with
row12 and with row13; each also max-composed with R5 q=.5. Report all component
rows and combinations. An action increase alone is not a genuine rescue: annotate
matched decisive-object evidence and whether HARD demand is adequately answered.

## Evaluation and resources

Paired unit bootstrap, 2,000, seed0, pointwise 95% intervals. Required episodes,
severity, onset delay, insufficient-braking loss, G (original previous-cheap jerk),
false episodes/frames, false HARD frames, excess braking cost, network pixels
mean/p95, query calls and gain concentration. Report every row and negative result.
No claim that tracking accuracy implies better braking or that unchanged detector
pixels imply measured latency. Test labels and GT matcher remain unchanged.

VUW/VUW1: source absolute env.sh; unchanged GPU wrapper, SHARE=1,
MIN_FREE_MB=10000, two workers per group, OMP4; two shards per dataset. CPU analyses
one thread. Durable status, per-unit atomic files, source hashes before/after.
If a useful quality candidate warrants it, a separate exclusive Jetson window
must measure its actual incremental or combined runtime and decision parity.

Outputs: docs/paper/handoff/OPTIMIZATION_R6_20261025/REPORT.md, all CSVs and PDF/PNG
plots, mirrored results/vuw_summaries/OPTIMIZATION_R6_20261025/, paper indexes.
