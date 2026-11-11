**Result:** on all four frames the cheap draft misses a decision-relevant object, and uncertainty crops at ours' crop area do not recover it. Both therefore KEEP when the annotated scene requires DECELERATE. Ours recovers the object and decelerates, using 5.5–12% of the frame's pixels. Whole-frame native also decelerates on three of the four frames, but at 101–103% of the frame's pixels. On nuScenes scene-0961 even whole frame (u960 and native) KEEPs.

# K15: Figure 1 on real frames (post-hoc analysis; AV2 with the K10 ego mask)

## Selection
- **Source.** Candidates come from the instrumented base runs (the K4 / K13 natural-miss sets): nuScenes new test (36 frames) and masked AV2 val (107 frames). All candidates are in `candidates.csv`.
- **Criteria.** On each candidate frame:
  - a decision-relevant annotated object is not matched in the draft;
  - it is not matched after uncertainty crops either;
  - it is matched in ours' final scene;
  - ours (with refiner) takes the required action, and that action is not KEEP.
- **Choice.** Among the candidates, the chosen frames are the ones where cheap and uncertainty crops both take the WRONG action. Daylight frames with several salient but irrelevant objects were preferred.

| dataset | scene / log | frame | why |
|---|---|---|---|
| nuScenes | scene-0273 | 6 | vehicle at 22 m missed; ours decelerates as required; 5 pedestrians on the pavement, daylight |
| nuScenes | scene-0961 | 8 | forklift (vehicle) at 15 m missed; ours decelerates; whole-frame u960 and native KEEP; daylight |
| AV2 | 87ca3d9f-f317-3efb-b1cb-aaaf525227e5 | 17 | cyclist at 38 m missed; ours decelerates; 26 other vehicles, bright daylight |
| AV2 | c85a88a8-c916-30a7-923c-0c66bd3ebbd3 | 17 | vehicle at 52 m missed; ours decelerates; 30 irrelevant objects (20 vehicles, 10 pedestrians) |

## Actions and network px (`panels.csv`; required action is DECELERATE on all four)

| frame | cheap | uncertainty crops (px) | ours (px) | u960 (px) | native (px) |
|---|---|---|---|---|---|
| nuScenes 0273/6 | KEEP | KEEP (0.080) | **DECEL (0.122)** | DECEL (0.384) | DECEL (1.031) |
| nuScenes 0961/8 | KEEP | KEEP (0.049) | **DECEL (0.115)** | KEEP (0.384) | KEEP (1.031) |
| AV2 87ca/17 | KEEP | KEEP (0.114) | **DECEL (0.056)** | DECEL (0.232) | DECEL (1.012) |
| AV2 c85a/17 | KEEP | KEEP (0.113) | **DECEL (0.056)** | DECEL (0.232) | DECEL (1.012) |

The refiner does not change ours' action on these frames (with and without refiner both DECEL).

## Panels
Format: K8, no text, PDF (vector overlays) + PNG at native resolution, ground truth green thin. The five panels per frame are in `panels/` and listed in `index.md`:
1. **Draft:** draft detections (blue); the missed object (dashed red).
2. **Whole frame native:** all detections (red); image tinted light orange.
3. **Uncertainty crops** at ours' crop area: crops (orange); their detections (red).
4. **Ours:**
   - field footprint (orange);
   - support crops (blue);
   - guard-grown crops (grey);
   - verified detections (red).
5. **BEV** (K8 style):
   - ego and corridor;
   - sensitive cells (orange);
   - draft objects (blue);
   - missed object (dashed red).

Script: `remote/A_method/k15_render.py`. It re-runs each unit from its start with the frozen configurations (nuScenes v3 + refiner v1, AV2 v4-lite + refiner v2; `k8_render.run_capture`, identical to `speculative.run`). AV2 goes through `k10_masked_run.py`. Selection script: `k15_select.py`.
