# K35-6 — reproducible selection rule for the qualitative panel figure (POST-HOC)

**figs/q6_* is not on the kit.** It lives in the writer's paper repository; the kit has its source panel sets.
- The paper's six-panel qualitative frames come from the K28 phase 3 §17 panel set (results/k28/p3/capture_frames.csv,
  30 frames, panels a–f per frame) and the K30 V3 / K31 sets that use the same renderer (k28_render.py).
- The writer should confirm which of those frames figs/q6_* shows; each set's examples.csv gives the frame IDs.

## How the existing frames were chosen

- capture_frames.csv records a category and a reason text per frame:
  - the K8 categories (spurious brake removed, range corrected, missed object recovered by the field, failure);
  - K15 missed recovered;
  - the K28 §13 failure categories.
- K28 phase 3 §17 states the counts per category. §13 lists the five costliest frames per category
  (failure_frames_v6.csv), of which "two per category" were rendered.
- **No rule for choosing a frame within a category is recorded.** No script on the kit or on VUW writes
  capture_frames.csv, and nothing states which two of the five costliest frames were taken. So the existing selection is
  not reproducible.
- Checked against the formal category definitions below, using frozen v6-S's stored test actions (existing_check.csv),
  5 of the 30 existing frames no longer meet their own category. They are mostly the K8 / K15 frames on AV2 and one
  on nuScenes, picked under earlier configurations (K8 / K15 predate v6-S and the AV2 ego mask):

| dataset | unit | frame | existing category | formal categories of that frame now |
|---|---|---|---|---|
| nuScenes | scene-0443 | 30 | K8 range_corrected | failure: under-reaction |
| AV2 | dc9077b9-2fe0-3d18-9b97-8067ff090874 | 6 | K8 spurious_removed | none |
| AV2 | 201fe83b-7dd7-38f4-9d26-7b4a668638a9 | 36 | K8 range_corrected | none |
| AV2 | 76916359-96f4-3274-81fe-bb145d497c11 | 31 | K8 missed_recovered_field | none |
| AV2 | 87ca3d9f-f317-3efb-b1cb-aaaf525227e5 | 17 | K15 missed recovered | none |

## Proposed rule (applied here; code remote/A_method/k35_k6.py, md5 265b8c82)

**Categories**, formalised from the stated reasons and computed from the stored K28 p3 test rows of frozen v6-S. Object
conditions use the per-pipeline "found" arrays over the frame's relevant annotated objects (demand ≥ τ1), which share one
order across pipelines.
- **K8 spurious_removed:** act_gt = KEEP; cheap ≥ DECELERATE; v6-S = KEEP; u960 + C1 + v3a = KEEP.
- **K8 range_corrected:** cheap finds a relevant object; cheap and v6-S without refiner differ from act_gt; v6-S equals
  act_gt.
- **K8 missed_recovered_field:** a relevant object is missed by cheap, found by u960 + C1, found by v6-S, and not found
  by v6-S's support-only ablation; v6-S equals act_gt.
- **K8 failure:** v6-S ≠ act_gt while u960 + C1 + v3a = act_gt.
- **K15 missed recovered:** a relevant object is missed by cheap and by unc2@S + C1 and found by v6-S; v6-S equals act_gt.
- **failure: …:** K28 §13 unchanged (J(v6-S) > J(u960 + C1 + v3a); then unnecessary action / missed / under-reaction /
  other, in that order).
- K8 and K15 apply to nuScenes and AV2, as in the original set; the failure categories apply to all three splits.

**Selection.** For each dataset, go through the categories in the order above. Take the first qualifying frame in time
order (unit name, then frame index) that was not already taken by an earlier category. No cost ranking and no visual
choice are involved.

**Rendering** (k35_q6_chain.sh, md5 e06e0f03):
- The selected units were re-run with the frozen pipeline in capture mode (k28_p3.py --part instr --capture, tag
  k35q6; AV2 with the ego mask; VUW1 2026-10-25), then rendered with the unchanged k28_render.py into six panels per
  frame (PDF + PNG).
- Panels: a draft, b BEV + field, c crops, d v6-S final scene, e native whole frame, f uncertainty crops.

## Selected frames (22)

| dataset | category | unit | frame | qualifying frames (units) | actions |
|---|---|---|---|---|---|
| nuScenes | K8 spurious_removed | scene-0122 | 7 | 318 (114) | act_gt 0, cheap 1, v6-S 0, v6-S no ref 0, u960+C1 0 |
| nuScenes | K8 range_corrected | scene-0289 | 19 | 47 (10) | act_gt 2, cheap 1, v6-S 2, v6-S no ref 1, u960+C1 2 |
| nuScenes | K8 missed_recovered_field | scene-0273 | 6 | 30 (13) | act_gt 1, cheap 0, v6-S 1, v6-S no ref 1, u960+C1 1 |
| nuScenes | K8 failure | scene-0123 | 0 | 71 (37) | act_gt 1, cheap 0, v6-S 0, v6-S no ref 0, u960+C1 1 |
| nuScenes | K15 missed recovered | scene-0273 | 7 | 39 (15) | act_gt 1, cheap 0, v6-S 1, v6-S no ref 1, u960+C1 1 |
| nuScenes | failure: unnecessary action | scene-0299 | 21 | 9 (8) | act_gt 0, cheap 0, v6-S 1, v6-S no ref 1, u960+C1 0 |
| nuScenes | failure: missed | scene-0123 | 1 | 22 (9) | act_gt 1, cheap 0, v6-S 0, v6-S no ref 0, u960+C1 1 |
| nuScenes | failure: under-reaction | scene-0131 | 4 | 40 (25) | act_gt 1, cheap 0, v6-S 0, v6-S no ref 0, u960+C1 1 |
| nuScenes | failure: other | scene-0274 | 2 | 9 (7) | act_gt 0, cheap 0, v6-S 0, v6-S no ref 0, u960+C1 1 |
| AV2 | K8 spurious_removed | 02678d04-cc9f-3148-9f95-1ba66347dff9 | 55 | 1211 (132) | act_gt 0, cheap 1, v6-S 0, v6-S no ref 1, u960+C1 0 |
| AV2 | K8 range_corrected | 02678d04-cc9f-3148-9f95-1ba66347dff9 | 34 | 178 (46) | act_gt 1, cheap 2, v6-S 1, v6-S no ref 2, u960+C1 1 |
| AV2 | K8 missed_recovered_field | 02678d04-cc9f-3148-9f95-1ba66347dff9 | 25 | 97 (26) | act_gt 1, cheap 0, v6-S 1, v6-S no ref 1, u960+C1 1 |
| AV2 | K8 failure | 02678d04-cc9f-3148-9f95-1ba66347dff9 | 27 | 136 (74) | act_gt 1, cheap 0, v6-S 0, v6-S no ref 1, u960+C1 1 |
| AV2 | K15 missed recovered | 02678d04-cc9f-3148-9f95-1ba66347dff9 | 26 | 111 (32) | act_gt 1, cheap 0, v6-S 1, v6-S no ref 1, u960+C1 1 |
| AV2 | failure: unnecessary action | 02a00399-3857-444e-8db3-a8f58489c394 | 47 | 49 (36) | act_gt 0, cheap 2, v6-S 2, v6-S no ref 2, u960+C1 0 |
| AV2 | failure: missed | 4e3fedbb-847c-3d5b-8a62-c9ff84550985 | 56 | 4 (4) | act_gt 1, cheap 0, v6-S 0, v6-S no ref 0, u960+C1 1 |
| AV2 | failure: under-reaction | 0aa4e8f5-2f9a-39a1-8f80-c2fdde4405a2 | 76 | 77 (38) | act_gt 1, cheap 1, v6-S 0, v6-S no ref 1, u960+C1 1 |
| AV2 | failure: other | 05fa5048-f355-3274-b565-c0ddc547b315 | 23 | 25 (14) | act_gt 0, cheap 1, v6-S 0, v6-S no ref 0, u960+C1 1 |
| Argoverse-HD | failure: unnecessary action | seq00 | 8 | 42 (15) | act_gt 1, cheap 2, v6-S 2, v6-S no ref 2, u960+C1 1 |
| Argoverse-HD | failure: missed | seq00 | 683 | 22 (5) | act_gt 1, cheap 2, v6-S 0, v6-S no ref 2, u960+C1 1 |
| Argoverse-HD | failure: under-reaction | seq00 | 748 | 55 (10) | act_gt 1, cheap 2, v6-S 0, v6-S no ref 1, u960+C1 1 |
| Argoverse-HD | failure: other | seq00 | 653 | 52 (8) | act_gt 0, cheap 0, v6-S 0, v6-S no ref 1, u960+C1 1 |

"Time order" across AV2 logs is lexicographic, because the log IDs are hashes. Within a unit, frame order is
chronological.

## Files
- capture_frames.csv: the selection.
- qualifying_counts.csv: frames and units per category.
- existing_check.csv: the existing panel frames against the formal definitions.
- panels/: 22 frames × 6 panels × PDF / PNG, plus examples.csv (actions, crops, px and the panel stem per frame).
