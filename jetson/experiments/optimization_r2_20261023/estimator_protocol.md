# Fixed CPU estimator screen, specified before aggregate evaluation

Motivation from nine inspected development cases: missing C1 observations,
successful recovery but underestimated closing rate, a refiner range improvement
that still crosses the wrong action threshold, and successful removal of cheap
false alarms. Illustrative cases are not representative outcome estimates.

Rows on every stored v6-S development scene:
- frozen v6-S, cheap, no-refiner and round-1 recovery references;
- camera-anchored refinement: preserve predicted ego x, recompute lateral
  coordinates by scaling about camera translation, rather than ego origin;
- temporal median: median of three log correction factors, only with mutually
  unique same-class links and all three original observations actually refined;
- half log correction: fixed 0.5 shrinkage control, retaining coupled geometry;
- three-observation height looming floor, assuming +/-1 detector pixel error
  in each measured size; min of both consecutive and the two-step rates;
- three-observation width-and-height floor, min across both dimensions;
- two-observation width-and-height fallback only when original C1 chain missing.

Motion links require mutually unique same-class IoU >=0.3. Observations must
come from verified detections; don't mix cheap/full box sizes. Known requested
scales are 0.6 nuScenes and 960/2048 AV2. Floors are clamped to [0,2]/s and
can only increase closing rate. The one-pixel perturbation is an engineering
sensitivity assumption, not a calibrated uncertainty or safety guarantee.
No GT, future frame, dataset-specific fit or threshold change enters a policy.

Refinement rows change only final estimation after the unchanged acquisition;
they can be evaluated exactly from stored raw/corrected geometry. Motion rows
are frozen-acquisition counterfactuals: newly supporting objects may need
refinement and queries, so promising rows require in-loop rerun before a method
quality claim. Evaluate every row; no claim of selected winner or held-out gain.
Falsifiers: lost baseline episodes, new false episodes, loss/G regressions,
benefit concentrated in one unit, or no activity in remaining failures.

Same full 152-scene/300-log dev manifests, baseline action/cost replay parity;
K31 gap<=1 episodes; paired unit bootstrap 2,000 seed0; report every row/CI.
