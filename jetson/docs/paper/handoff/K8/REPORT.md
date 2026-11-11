# K8: figure material (qualitative panels, pipeline figure, plot data), 2026-10-04

## What was run
- **Plot data:** `remote/A_method/k8_plotdata.py` (md5 48775d69…) → `plot_*.csv`. Done 2026-10-03; see `index.md` §3.
- **Panels:** `remote/A_method/k8_render.py` (md5 86c3de7d…), run on VUW (GPU, torch FP16, frozen COCO YOLOv8s):
  ```
  SHARE=1 MIN_FREE_MB=4000 run_when_gpu_free.sh logs/A_method/k8_render3.log \
    python scripts/A_method/k8_render.py results/handoff_K8/frames.csv results/handoff_K8/panels
  ```
- **Configurations:** frozen final ones. nuScenes v3 (v_close 10, crops at 0.6) + refiner v1; AV2 v4-lite (v_close 0, crops at 960/2048) + refiner v2.
- **How each frame is rendered:** the whole sequence is re-run from its start, so the lift and scene histories are exact. The loop is `instr_eval.run_variant` with capture hooks; it was asserted to make the same decisions as `speculative.run`.
- **Frame selection:** from the instrumented post-hoc runs (K2/K4): nuScenes new test, AV2 val. One frame per category per dataset. The category rules are in the `reason` column of `examples.csv`.
- **Label:** this is post-hoc analysis; nothing was tuned.

## Files
- `examples.csv`: one row per frame. Columns:
  - dataset, split, unit (scene or log), frame index, image path, category, reason, ego speed (m/s);
  - actions (0 keep, 1 decelerate, 2 hard brake) for: required (GT), cheap, ours (no refiner), u960, native;
  - crops (field / support / guard-grown), rounds, network px;
  - `panel_stem`.
- `panels/<stem>_{a_draft,b_bev,c_crops,d_final}.{pdf,png}`: four panels per frame. Native resolution; the PDF is the image with vector overlays; no text is baked in.
- **Colour code:**
  - (a) draft detections in blue, dashed if verification later removed them;
  - (b) bird's-eye view: ground-grid cells of the sensitivity field in orange, corridor as grey lines, ego in black with a 1-s speed arrow, draft objects as blue footprints (lateral axis stretched about 4× for legibility);
  - (c) field-footprint crops orange, support crops blue solid, guard-grown crops grey;
  - (d) final detections red when verified on a crop, blue when kept from the draft; ground truth green thin.

| # | dataset / split | unit, frame | category | required / cheap / ours / u960 / native action | crops (field / support / guard) | network px |
|---|---|---|---|---|---|---|
| 1 | nuScenes test | scene-0152, 27 | spurious draft object removed | 0 / 2 / 0 / 0 / 0 | 1 / 1 / 1 | 0.267 |
| 2 | nuScenes test | scene-0443, 30 | range corrected | 2 / 0 / 2 / 2 / 2 | 1 / 0 / 6 | 0.410 |
| 3 | nuScenes test | **scene-1081, 6** | missed object recovered by the field | 2 / 0 / 2 / 2 / 2 | 1 / 0 / 3 | 0.159 |
| 4 | nuScenes test | scene-1051, 32 | failure | 2 / 2 / 0 / 2 / 0 | 1 / 1 / 0 | 0.092 |
| 5 | AV2 val | dc9077b9, 6 | spurious draft object removed | 0 / 2 / 0 / 0 / 2 | 1 / 1 / 1 | 0.098 |
| 6 | AV2 val | 201fe83b, 36 | range corrected (by the refiner) | 2 / 1 / 1 / 1 / 1 | 0 / 1 / 1 | 0.167 |
| 7 | AV2 val | 76916359, 31 | missed object recovered by the field | 2 / 0 / 2 / 2 / 2 | 1 / 0 / 1 | 0.047 |
| 8 | AV2 val | 201fe83b, 32 | failure | 2 / 1 / 0 / 2 / 0 | 0 / 1 / 0 | 0.070 |

## Pipeline figure
- **Frame:** nuScenes scene-1081 frame 6, at night, ego 13.8 m/s. **It is the same frame as example 3.**
- **What the panels show:**
  - (a) the draft (320 px) detects nothing;
  - (b) with no draft objects, the whole corridor out to 70 m is sensitive;
  - (c) one field crop (orange) on the corridor, plus 3 guard-grown crops (grey) around objects cut by its edge;
  - (d) the lead car is found on the crop (red) and matches the GT (green), so the action changes from keep to hard brake, as required.
- **Network px:** 0.159.

## Caveats
1. **Selected frames are illustrative, not representative.** The "failure" rows show cases where ours is wrong and u960 is right:
   - nuScenes frame 4: ours removes the brake that cheap and u960 keep;
   - AV2 frame 8: ours under-brakes.
2. **Frame 5 (AV2):** the "spurious" draft object is the ego hood at the bottom of the frame (blue dashed in (a); the blue support crop in (c) verifies it). Native u2048 also brakes for it, which shows this is a COCO-detector artefact.
3. **Frame 6:** ours without the refiner keeps the draft's wrong action (1). Only the refiner v2 changes the range enough to give 2. Panel (d) shows boxes, not ranges.
4. **u960 / native action history:** in `examples.csv` it uses only the 2 frames before the target, whereas ours uses the full history. Their lift ttc may differ slightly from the K2 runs.
5. **Pre-fix render:** the raw render output with the AV2 split mislabelled "test" is kept as `panels/examples_raw_render_output.csv`. `examples.csv` corrects it to val; the frames are from AV2 val logs.
