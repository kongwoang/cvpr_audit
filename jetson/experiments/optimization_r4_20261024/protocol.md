# Round 4: simpler mechanics before more learned modules

User requests component-by-component review and simple alternatives. All work stays
in this sandbox. Frozen source/checkpoints, K34 and test data remain untouched.
This is another adaptive development round, not confirmation or evidence that any
component is globally optimal. No additional training in this round.

Fixed evaluation: original 152 nuScenes development scenes and 300 AV2 CFG-DEV
logs; original action thresholds, costs and K31 episodes. Report every row, all
changed episodes, hard-demand loss, false episodes, mean/p95 requested pixels.
Paired unit bootstrap 2,000, seed 0, pointwise 95% intervals. Actual latency only
from a later exclusive Jetson window if justified; no GPU-server timing claim.

## Fresh acquisition suite (specified before outcomes)

Controls: frozen v6-S, round1 B2K-A+recovery, round3 critical-extent persistence.
No-refiner and original v3a for every acquisition, independent histories.

1. **Containment merge:** a cheap box is replaced if at least 95% of its area is
   inside ONE executed primary/guard crop, or a compatible full box overlaps at
   IoU >= .5. Original merge uses cheap centre in a primary crop, plus overlap
   after guard. This tests whether partial observation is too readily treated as
   negative evidence. It can retain phantoms; no unconditional safety claim.
2. **Union coverage:** compute exact union area of previously executed rectangles
   for the existing .95 coverage test, instead of the current maximum single-crop
   intersection. This may avoid redundant queries, but fragmented visibility can
   still be worse for detection than one intact crop. Fresh inference required.
3. **Semantic consistency:** use vehicle vs vulnerable-road-user compatibility
   for temporal association, guard matching and cross-crop NMS; keep the existing
   IoU/intersection thresholds. Person/cyclist remain compatible to avoid obvious
   class flicker. No learned identity module; same-class switches remain possible.
4. **One-round extent:** same extent seed as R3, bounded to ONE acquisition round
   with B2K-A pending retention and retrospective recovery. Cost/quality control
   for whether a second adaptive round is worth its overhead, not a novel theorem.

All acquisition variants initially isolated; no combination selected from smoke
outcomes. Smoke uses the same nine diagnostic units as R3 only for implementation
and baseline parity, then all rows run on the full development population.

## Exact final-readout CPU suite (specified before outcomes)

Round1 histories/queries use raw scene estimates and boxes; its final refiner
does not affect subsequent acquisition/history. Therefore a policy explicitly
changing ONLY the final readout can be evaluated on stored raw/v3a scenes without
pretending to rerun a different acquisition. Use original consumer unchanged.

- Objectwise median demand of compatible matched cheap / raw verified / refined
  verified estimates; unmatched objects keep the deployed estimate.
- Hard-brake corroboration: when cheap and raw verified estimates both demand
  HARD_BRAKE, disallow a final refiner downgrade for that object; all other objects
  use deployed refinement.
- Refiner abstention when cheap/raw agree at each action threshold (equivalent
  to objectwise majority); quantify how many final refiner objects could be
  skipped WITHOUT altering this candidate's output. This is a cost opportunity,
  not measured speed. The estimators are correlated; no ensemble guarantee.
- Framewise median action as an explicit weaker matching-free control; unlike
  objectwise voting it can combine evidence from different objects.
- Whole-refiner removal and fixed half-log refinement controls from R3.

Matching is deterministic same compatibility groups, greedy IoU >= .3. The
objectwise median keeps max-composable scalar object demands but changes the
perception readout; do not claim the original exactness guarantee automatically
extends through this heuristic. Any implementation skipping v3a must verify exact
decision parity against the full median computation on common frames.

## Evidence audit

Count cheap braking detections removed without adequate whole-box coverage and
without a compatible full detection; cross-class overlaps/history links; current
single-crop-vs-union coverage disagreements; refinement improvements and harms.
Describe overlap/association labels as diagnostics, not causal fractions or
manually certified errors. Inspect all newly lost episodes and representative
severe rescues before recommending a candidate. Keep negative results.

VUW/VUW1: source env.sh, all writes inside B, unchanged run_when_gpu_free.sh,
SHARE=1 MIN_FREE_MB=10000, two workers per group. Hash main sources/weights before
and after every job. Jetson uses owned pause/exclusive/restore protocol only.

## Dated extension: exact bottleneck-score cache (2026-10-24, before its runs)

Code review found K31 NAVSIM field_support recomputes the entire candidate/object
matrix for each inserted probe. For the fixed selector s_j(S)=min(b_j,min_i h_ji),
s_j(S+o)=min(s_j(S),h_j(o)). Batch all probe-only columns once and reuse the base
scores/H matrix. This is algebraic implementation simplification, not a new
selector, grid, learned component or configuration.

Fixed audit population: K34/K20's exact existing 1,000-token, 125-unit NAVSIM
navtest runtime manifest; explicitly post-hoc runtime/parity only. No quality
selection or detector rerun. Compare original K31 field_support and cached code
on the same host/process, one CPU thread, five warm-up repetitions, three timed
repetitions, alternating method/grid order, grids 7x7 and 3x3. Scope includes
scene construction, selector initialization, projection, field/support extraction;
excludes disk loading. Preserve exact action, every field box and every support
box/order in all comparisons, plus K31 original first-round counts. Report mean,
p95, paired unit bootstrap 2,000 seed0, actual candidate/object evaluations and
both outcomes even if no speedup. It does not accelerate general PDM-Closed by
assumption; no planner identity change or full perception latency claim.
