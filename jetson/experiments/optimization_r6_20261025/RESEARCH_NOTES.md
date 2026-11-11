# What the papers suggest, and what they do not establish

Read against frozen source and the R5 case audit, 2026-10-25.

1. **[ByteTrack](https://arxiv.org/abs/2110.06864)**, ECCV 2022, section 3 and
   algorithm 1. Associate high-score detections first; use low-score detections
   to recover unmatched tracks, and do not initialize tracks from unmatched
   low-score boxes. This fits negative crop verification directly: current
   weak evidence may be useful even when it is below the ordinary acceptance
   threshold. Our adaptation uses existing high-quality observations or a
   current cheap box, no new tracking model. Its limitation is fundamental:
   existence/identity does not establish range, lateral extent or closing speed.
   MOT benchmark gains are not evidence of braking gains here.

2. **[OC-SORT](https://arxiv.org/abs/2203.14360)**, CVPR 2023. Accumulated motion
   prediction error through occlusion motivates anchoring recovery to actual
   observations. Our causal adaptation requires a current box, an observed
   compatible predecessor, and two real lags before C1 looming. It never brakes
   on an extrapolated absent box. We do not implement its Kalman re-update or
   virtual-trajectory machinery. A wide image-IoU gate can still link different
   vehicles, and a strict one can miss a rapidly moving or fragmented truck.

3. **[Bernoulli Filtering for Multi-Sensor Tracking with Thresholded
   Measurements](https://arxiv.org/abs/2606.09573)**, 2026 preprint. Its simulated
   multi-sensor study explicitly models state/threshold-dependent missed
   detection. The transferable lesson is that an empty crop is not equally
   strong negative evidence for every object. The calibrated sensor model and
   independence assumptions needed for a probabilistic existence filter are
   unavailable here. Treating correlated cheap/crop YOLO scores as independent
   probabilities would manufacture confidence; R6 uses explicit observations
   and reports errors instead.

4. **[Selective Classification via One-Sided
   Prediction](https://proceedings.mlr.press/v130/gangrade21a.html)**, AISTATS
   2021. It formalizes error versus coverage for abstaining classifiers. This
   supports measuring errors in the selected disagreement subset, rather than
   citing high aggregate classifier precision. R5 demonstrated this mismatch.
   Our bounded rules are not its learned optimization and inherit none of its
   theoretical guarantees.

## Components considered

| Weakness | Feasible simple change | Main risk | R6 treatment |
|---|---|---|---|
| Cheap real object lost by verification | Current low crop proposal or prior cheap observation corroborates retention | Repeated phantom detections; wrong cheap geometry | All observation and demand-capped variants, full dev |
| Current cheap also misses a weak object | Low crop box continues a prior full observation | Wrong identity or continued erroneous prior demand | Current box required; one/two-lag and historical-action caps |
| Refiner lowers valid demand | Condition trust on deployed disagreements | R5 selected-population miscalibration, tiny HARD training count | No new larger learned head; retain R5 and R3 controls |
| Truck cab/trailer fragments | Temporal part association / larger physical extent | Phantom corridor occupancy if boxes are naively united | Keep original matcher; document unresolved cases, no arbitrary union |
| No current visible/matched object | New sensor/view or targeted new acquisition | Extra compute may not make hidden evidence observable | Availability audit; no guarantee from threshold lowering |
| Residual loss with correct discrete action | Different controller action set | Changes the consumer/reference and hence the task | No controller/GT change; quantify residual separately |

R6 final readout does not change the parent's next-frame state, crop selection or
refiner support. It therefore demonstrates a bounded decision repair under the
recorded acquisition, not a geometrically corrected scene or a new guarantee for
the selector. Positive cases are mechanisms to investigate, not evidence that the
method's remaining HARD weakness is solved.
