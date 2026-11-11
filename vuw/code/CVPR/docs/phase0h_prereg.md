# Phase 0H pre-registration: can a harmful escalation be recognised after it has run?

Written 2026-09-26, before any Phase 0H table was built and before any arbiter was fitted or scored.
Amendments are appended at the end with a timestamp and never edit the text above them.

## 1. Question

DEEP found that pre-escalation allocators capture part of the benefit of escalation but select harmful
escalations about as often as random. The question here is different: once an input has been escalated,
the system holds both the cheap and the full perception output and can run the downstream consumer on
both. Can a lightweight **arbiter**, using only this post-escalation information, tell a helpful change
of decision from a harmful one?

The arbiter acts only on inputs where the decision changes (for the planner cells, where the loss would
change is not observable, so it acts on every input whose planner output differs; see 3.3). Accepting
keeps the full-mode decision (value `V`); rejecting keeps the cheap-mode decision (value 0). With a
perfect arbiter the value of escalation becomes `max(V, 0)`.

## 2. Data and cells

The ten core cells of DEEP (YOLOv8s 320->640, lift frame `ego`, shared action history), with decision
values read from `risk-aware-perception/results/final/benchmark_decision_values.csv.gz` and the frozen
split in `risk-aware-perception/configs/benchmark_splits.json`:

| cell | consumer | geometry |
|---|---|---|
| KITTI brake mono / oracle | braking controller (`rap.planner`) | deployed monocular / reference for matched detections |
| KITTI traj mono / oracle | Planner B, `static_obstacles` (`rap.planner_b`) | same |
| nuScenes brake mono / oracle | braking controller | same |
| nuScenes plan ADE / FDE, mono / oracle | PKL's learned planner (`JC_ade`, `JC_fde`) | same |

No detector is re-run. Every perception output comes from the cached detections in
`risk-aware-perception/data/cache` (confidences down to 0.10).

**Gate V1 (must pass before anything else is computed).** The per-frame `J_cheap` and `J_full` recomputed
for the braking and Planner B cells must equal the canonical values to within 1e-9 on every frame. If
they do not, Phase 0H stops and the discrepancy is reported.

## 3. What the arbiter may see (deployable, post-escalation)

Nothing derived from the reference scene state enters a feature. Geometry in features is always the
**monocular** lift, also in the oracle-geometry cells, so that GT-matched geometry cannot leak through a
feature. In the oracle cells, the consumer's actions and the identity of the pivotal element are
computed on the cell's own geometry (that is the system being arbitrated). This may leak that an
element matched a reference object; the oracle cells are therefore reported with this caveat and the
deployability criterion C4 uses the monocular cells only.

### 3.1 Frame features (all cells)
Causal ego speed; kept detections per branch (`n_cheap`, `n_full`); added / removed / matched counts
(greedy confidence-ordered matching of full to cheap detections at image IoU >= 0.5); max and mean
confidence of added detections; min confidence of removed ones; count of added detections whose mono
geometry overlaps the braking corridor within 80 m; mean relative mono-range change over matched pairs;
share of added detections with a sub-threshold cheap detection (0.10 <= conf < 0.25) at IoU >= 0.3;
the cheap cache's image scalars (`fs_img_*`, `fs_motion_*`).

### 3.2 Decision and pivot features (braking and Planner B cells)
Cheap and full actions (braking: KEEP/DECELERATE/HARD_BRAKE and the flip direction; Planner B: the
longitudinal acceleration and lateral offset of each branch's plan); required deceleration of each
branch recomputed with mono geometry.

**Pivot.** Elements are the added detections, the removed detections and the matched pairs whose
consumer geometry differs. An element is *sufficient* if applying it alone to the cheap set yields the
full decision, and *necessary* if reverting it alone in the full set changes the full decision. The
pivot is the first of: sufficient and necessary, sufficient, necessary, ties broken by the nearest
consumer-geometry range; if no element qualifies the pivot type is `multi` and the pivot is the element
with the largest required deceleration under mono geometry. Pivot features: type (add / remove / move /
multi), numbers of sufficient and necessary elements, confidence, class, entropy, box height, width and
bottom (normalised), border contact, mono `z`, `z_ground`, `z_height`, `log(z_ground / z_height)`,
lateral centre and half-width, clipped TTC, best IoU and confidence of any detection (conf >= 0.10) of
the *other* branch at that location, best IoU with the previous and the second previous cheap frame
(conf >= 0.10), and for a move the relative range change, box IoU and confidence change.

### 3.3 Planner cells
The learned planner is not re-run, so no pivot is computed. They use the frame features plus the
features of the **salient element**: the added or removed detection with the largest required
deceleration under the braking rule and mono geometry (ties: nearest). The arbiter acts on every input
with `V != 0`, which for this continuous-output planner is every input whose output trajectory moves;
this is observable because the planner is run on both outputs.

## 4. Arbiter

Label `y = 1[V > 0]` on inputs with `V != 0`, sample weight `|V|`, so that a probability of at least 0.5
estimates `E[V | x] >= 0`. **Decision rule: accept iff p >= 0.5**, not tuned.

Four configurations: {L2 logistic regression (standardised, C = 1), gradient-boosted classifier
(`HistGradientBoostingClassifier`, depth 3, 200 iterations, learning rate 0.08, 20 samples per leaf,
L2 1.0, 64 bins, seed 0)} x {fitted per cell, pooled over the cells of the same consumer family with
dataset and geometry indicators}. For each cell one configuration is selected by grouped 5-fold
cross-validation over the non-test units (unit = sequence / scene), maximising the out-of-fold realised
arbitration gain; ties go to logistic, then per-cell. The selected configuration is refitted on all
non-test units and scored **once** on the test split.

## 5. Baselines

- accept-all = uniform full fidelity; reject-all = all-cheap.
- **Best single-feature rule**: for each of {pivot or salient confidence, other-branch support IoU,
  previous-frame support IoU, `|log(z_ground / z_height)|`, flip direction, added-count}, the direction
  and threshold maximising realised gain on the non-test units; the best of these on the non-test units
  is the rule baseline of that cell.
- Diagnostic only (uses GT): **verifier oracle**, accept iff the pivot / salient element is correct
  (added -> matches a reference box at IoU >= 0.5; removed -> the cheap detection matched none; move ->
  full mono range closer to the reference range). This measures how much of the arbitration ceiling a
  perfect object verifier would reach, i.e. the headroom for an image-based verifier.
- Negative control: the selected configuration refitted on labels permuted within units.

## 6. Metrics (test split, per cell)

`L_c` = all-cheap loss; `B` = sum of positive V; `H` = sum of |negative V|.
benefit retained = accepted positive V / B; harm removed = rejected |negative V| / H;
`G_full = sum V / L_c`; `G_arb = sum_{accepted} V / L_c`; ceiling `G_max = B / L_c`;
capture = `(G_arb - G_full) / (G_max - G_full)`.
Paired bootstrap over test units, 1,000 draws, seed 0: `G_arb - G_full` and `G_arb - G_rule`.

Secondary, not part of the verdict: realised decision gain at a 20% selection budget of DEEP's cached
deployable allocators (ridge and GBM gates, the four R1 routers, ego speed, random) with `V` against
their gain with the arbitrated value `V_arb`.

## 7. Verdict criteria

- **C1** harm removed >= 0.50 and benefit retained >= 0.80, in >= 6 of the 10 cells.
- **C2** paired 95% interval of `G_arb - G_full` above zero in >= 6 of the 10 cells.
- **C3** equal-weight mean over the 10 cells of `G_arb - G_rule` with its pooled paired interval above zero.
- **C4** C1 and C2 each hold in >= 3 of the 5 monocular cells.

**GO** = C1, C2, C3 and C4. **WEAK GO** = C2 in >= 4 cells, C3, and C2 in >= 2 monocular cells.
**NO-GO** otherwise. A cell whose test split has fewer than 10 inputs with `V != 0` counts as not met.

If the verifier oracle captures most of the ceiling while the learned arbiter does not, the next step is
an image-based verifier of the pivotal element (Phase 1), not a verdict on the idea.
