# R6 direct case review

All eight three-frame panels were visually inspected. These are deterministic
diagnostic selections, not a random sample; every frame and changed episode is
in the population CSVs. Red is the decisive annotation; blue cheap; cyan final
parent detections; magenta current low-score proposals. For AV2 the orange line
marks the original geometry-exclusion boundary, and pixels are not blacked out.

| Case | Observation | What the rule does / cannot establish |
|---|---|---|
| [nuScenes scene-1049 frame31](cases/nusc/00_new_supported_action.png) | A car is visible ahead at night. A full box exists at frame30; frame31 has a corresponding weak box (confidence about .13), below the normal acceptance threshold. R5 is KEEP, required demand 6.57. | One-lag bounded continuation gives HARD on the matched car. Frame29 is another supported recovery. The episode was already hit at another frame, so these are HARD-frame repairs, not a newly detected episode. All nuScenes incremental loss gain is in this scene. |
| [nuScenes scene-1016 frame37](cases/nusc/01_new_false_action.png) | A real bus passes on the right. The previous final box incorrectly gives HARD demand, and the current weak box follows that bus. Reference demand is zero. | One-lag continuation extends a false HARD decision. This is not simply a nonexistent-object false detection: observing a real object does not validate its corridor/range demand. Two-lag action corroboration suppresses this particular failure. |
| [AV2 e4279e3e… frame62](cases/av2/00_new_supported_action.png) | A bus is clearly visible immediately ahead. Final full boxes at frames60–61 support HARD; the current crop's bus box has confidence about .23 and is dropped by ordinary filtering. Required demand is 3.94. | Both bounded variants recover HARD from R5 KEEP using a matched current bus box. This earlier frame in the same log is distinct from the R5 frame66 cheap-retention gain. It contributes most AV2 incremental improvement. |
| [AV2 e125bb91… frame4](cases/av2/01_new_false_action.png) | The added weak box covers road/ego-hood area below the intersection, rather than the annotated distant car. The previous frame contains a similar erroneous full box. | One-lag continuation produces false HARD. The unchanged half-box ego-region rule does not eliminate every artifact straddling that boundary. Two-lag action corroboration suppresses this case. No mask tuning or GT exclusion is performed. |
| [nuScenes scene-0461 frame8](cases/nusc/02_unresolved_hard_case.png) | A crossing truck's annotation spans cab and trailer. Current weak boxes cover small fragments, not the full extent under the frozen IoU matcher. GT demand reaches9. | Bounded continuation remains KEEP. Lowering acceptance or keeping history does not reconstruct the correct whole-object extent/closing motion. Naively uniting fragments could invent corridor occupancy elsewhere. |
| [AV2 b56e3f47… frame4](cases/av2/02_unresolved_hard_case.png) | The large decisive annotation touches the right/bottom image edge; much of its image rectangle contains road/ego-region. The complete annotated object is not clearly visible in these panels. GT demand reaches9. | The bounded variants remain KEEP. This is a visibility/geometry ambiguity to audit, not proof of bad annotation. All original GT and frames stay scored. An aggressive variant's episode-level rescue here lacks decisive-object evidence. |
| [nuScenes scene-1017 frame15](cases/nusc/03_low_score_episode_rescue.png) | Oncoming headlights at night; cheap demand about5.3 has a matching weak current proposal. Frame15 GT demand is3.49; the episode later peaks above the HARD threshold. | The residual-low rule detects an episode missed by R5, but no adequately supported HARD action is established at the episode's HARD-required frames. This panel is explicitly the residual-low row, not the bounded-history composition. |
| [AV2 22dcf96c… frame48](cases/av2/03_low_score_episode_rescue.png) | The decisive annotation covers a person beside the parked pickup on the right. The added cheap/low-score evidence is a different, distant road vehicle. | The residual-low row happens to brake during the required episode; it does not recover the decisive person. All three extra AV2 episodes of this row lack decisive evidence. This distinction prevents claiming that the pedestrian weakness was fixed. |

The first render used the bounded-history title for every panel, including the
two residual-low examples. Their captions were corrected to identify the actual
row and action; the initial renders remain under case_caption_v1/. No inference,
selection, score or CI changed. Compilation caught a quote error in the caption
script before it was launched; the corrected render jobs completed normally.

The cross-case lesson is specific: retaining a current weak observation can fix
dropouts, but persistent evidence can also preserve wrong geometry or recurring
ego-region artifacts. The simple two-lag cap controls some such errors at the
cost of missing fast/new events. R6 is a small complementary repair, not a
solution to unobserved objects, fragmented extents, identity errors or all HARD
episodes.
