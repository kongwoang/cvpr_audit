# R5 direct case review

Nine deterministic panels were inspected: highest-loss absent/removed evidence,
highest-demand supported/unsupported rescue, and highest-cost new false braking.
They are examples, not a random population sample. Every changed episode is in
the final dataset CSVs, including comparisons directly against R3 point+extent.
Panels label **R5 both q50**; the residual-policy outcomes below are separate
rows from final_nusc/frames.csv and final_av2/frames.csv.

| Case | What the image and stored record show | What the experiment establishes |
|---|---|---|
| nuScenes scene-0461, f8 | Long crossing truck; the annotation spans cab and trailer, while several visible detections cover smaller parts. GT demand 9; no match at the frozen class/IoU≥.3 rule. | This is not visually “nothing detected”. All tested residual readouts still KEEP; the geometry/extent of a fragmented object remains unresolved. [Panel](cases/nusc/00_severe_absent_both.png). |
| nuScenes scene-1050, f36 | Night/glare; cheap box covers the car ahead, demand 6.008, but the final scene loses it. GT demand 8.288. The crop was executed. | R3 point+extent KEEP→residual q50/q80 HARD. This frame explains the entire q80 nuScenes risk improvement: 63.460 summed loss reduction. It improves a severe frame, not a newly rescued hard-peak episode. [Panel](cases/nusc/01_severe_lost_cheap.png). |
| nuScenes scene-0594, f3 | Sideways pickup, GT demand 1.606; final evidence remains below DECELERATE. | Both q50 and point+extent+residual q50 brake. The latter adds one episode over R3, with decisive-object evidence. q80 does not. [Panel](cases/nusc/02_supported_rescue.png). |
| nuScenes scene-0451, f1 | Large truck split into cab/bed detections; ground-truth box covers the whole vehicle. | R5 both q50 brakes, but the matched decisive detection does not support the action. A second fragment may belong to the same physical object; this does not justify silently relabelling attribution. Point+extent+residual leaves KEEP. [Panel](cases/nusc/03_unsupported_rescue.png). |
| nuScenes scene-0258, f24 | Urban road, distant detections and cones; GT required demand 0. | R5 both q50 adds DECELERATE at a false-braking cost. Residual q50 on R3 does not change this frame. [Panel](cases/nusc/04_new_false_braking.png). |
| AV2 b56e3f47…, f4 | GT decisive projection is a large box along the right/bottom image edge, partly overlapping the fixed ego mask; the visible annotated rectangle is not a clean fully visible vehicle. | GT demand 9 and all shown readouts KEEP. The image alone cannot certify an annotation error or prove invisibility. Keep this loss in every primary score. [Panel](cases/av2/00_severe_absent_both.png). |
| AV2 e4279e3e…, f66 | Large near bus at the right image edge. Cheap has a clear box and demand 3.27; final scene loses the match. GT demand 3.926. | R3 KEEP→residual q50 DECELERATE, still below HARD; q80 remains KEEP. This log contributes 67.9% of AV2 residual q50's incremental risk reduction. [Panel](cases/av2/01_severe_lost_cheap.png). |
| AV2 b1527e96…, f17 | Crossing vehicle partially occluded by a pole; GT demand 1.576, final/refined demand is below threshold. | R5 both q50 supplies matched decisive evidence and rescues an episode versus round1. This is not an incremental result over the strongest R3 control. [Panel](cases/av2/02_supported_rescue.png). |
| AV2 0b97f5dd…, f33 | Distant vehicles ahead, GT demand 0 on this frame. | R5 both q50 adds braking. Residual q50 on R3 does not. A confidence gate can still be wrong in the selected disagreement population. [Panel](cases/av2/03_new_false_braking.png). |

The conditional-model AV2 point+extent q50 row rescues one additional episode
over R3 at log 14bf638b… f49–51 (peak 1.377), but no braking frame has matched
decisive-object demand ≥1. This is the same kind of attribution weakness already
seen in prior rounds; do not count the improved miss total as a solved mechanism.

The full-population secondary audit qualifies the simple “absent” label:

- nuScenes unbraked/unmatched: 53 frames, loss 560.937. Four have a compatible
  possible contained fragment (≥.5 intersection/detection area and IoU≥.05),
  accounting for 159.715 loss. This overlap test is not identity certification.
- AV2 unbraked/unmatched: 96 frames, loss 783.109. Fourteen GT boxes touch the
  image edge (loss 547.952), six overlap the masked strip (loss 385.723), and
  21 contain possible fragments (loss 133.917). Flags overlap. None has ≥50%
  of its projected box area masked; “mask overlap” does not imply full invisibility.

No annotation, matcher, required episode or evaluation frame was removed.
These findings argue for diagnosing object extent, temporal continuity and
observability before adding another global rate or confidence correction.
