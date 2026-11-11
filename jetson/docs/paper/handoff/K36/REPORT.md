# K36 — sensitivity-field renders (F1) and the Fig. 4b frame's stored rows (F2)

Post-hoc, CPU only, frozen v6-S. No detector was run, and frames were not selected on outcomes. Every rendered panel is
listed below.
- Code: remote/A_method/k36_f1.py, run on VUW 2026-10-25 (log logs/k36/f1.log). It writes results/k36/, copied here.
- The curved-tube candidate variant (§F1.3) was rendered by a short follow-up call to the same module's functions.

## Frame-selection rules (fixed before rendering; "time order" = unit name, then frame index)
- **nuScenes test** (K28 test records results/k28/p3/nusc_test_instr). Rule: the first frame with
  - stored v6-S action a_v6S|v3a = KEEP;
  - ego speed in [8, 12] m/s;
  - no draft object inside the corridor (lateral extent overlapping ±corridor_half_w = ±1.2 m) at range ≤ 70 m.

  → **scene-0110, frame 0** (v = 10.12 m/s; the first candidate examined: k36_selection.csv).
- **Argoverse-HD val, curved.** Curvature κ as in K31 item 3 (results/k31/r3/ahd_curv). Rule: the first frame with
  - |κ| > 0.01 1/m (K31's curved-path threshold);
  - stored a_v6S|v3a = KEEP;
  - no draft object inside the straight corridor (range ≤ 70 m) nor inside the curved corridor (path coordinates,
    arc length ≤ 70 m).

  → **seq01, frame 14** (v = 4.83 m/s, κ = −0.0104 1/m, i.e. a right-hand curve of radius ≈ 96 m). seq00 has no frame
  with |κ| > 0.01 and KEEP.

## Definitions (all frozen code paths, CPU recomputation)
- **Draft scene.**
  - nuScenes: the cached draft with its mono geometry, as every K28 / K30 run.
  - Argoverse-HD: the stored K30 apdump cheap scene (conf ≥ 0.25), lifted with predicted_geometry on the previous
    frame's draft, as k30_eval2.
  - Check: a_cheap recomputed from this draft equals the stored K28 record on every frame of both units
    (0 mismatches; checks.json).
- **Fields.**
  - Braking: the draft-round field dsf.insertion_grid(draft mono, s·v, R_GRID, Y_GRID, SP.OBJ_W, pp, v_close 0), with
    the ego speed scaled by s ∈ {0.5, 1, 1.5} exactly as K30 V8d.
  - TTC warning: instr_eval.TTCWarn(pp, 0) at s = 1 (K9 / K31 form).
  - Curved: k31_eval.curved_grid(κ) with probe KL.CAR, as K31.
  - Grid: ranges 12.5, 17.5, 22.5, 27.5, 35, 45, 55, 70 m × lateral −6…6 m in 1.5 m steps.
- **Footprint** (the field crop that would be requested): k_loop.field_crop_g (braking; asserted equal),
  spec_general.footprint (TTC warning; asserted equal), and the same formula on the curved field (the crop K31's module
  substitution requests).
- **Tube** = the hybrid tsp8@S's ego-corridor tube, TUBE[8]: near 10 m, far 70 m, 'fill'.
  - Exact per-frame rectangle from the k28_p3 / k31_eval code. Budget = the frame's v5.1-S crop px (stored cpx_v51S) minus
    the support crops filled first. The half-width grows in 0.25 m steps while the canvas fits.
  - Path-following tube: k31_eval tube_c, with the curved consumer's support.
  - Check of the reconstruction: on all 351 frames of seq01 that have a decisive object, K31's stored flags
    (decisive-object centre inside the straight / curved tube) are reproduced with 0 mismatches.
- **"In the tube" (BEV cell).** The cell's ground-contact point, under the method's own projection
  (u = cx − fx·y / r, v = cy + fy·h / r), lies inside the tube rectangle. The BEV outline is that preimage, clipped at
  80 m.
- **BEV style** = k8_render.bev_panel. Orange cells = sensitive; grey lines = corridor (path-following on the curved
  panel); blue bars = draft objects; ego and a 1 s speed arrow.
  - Added: black dashed = straight tube outline; black dotted = path-following tube band (arc length 10–70 m,
    ±half-width).
  - The lateral axis follows k8_render: lateral y plotted as is, left-positive to the right, so it is mirrored relative to
    the camera image. Lateral is stretched about 4×. No text is baked in.
- **Image panels** (k8_render.img_panel, as panel c). Orange = field footprint; black dashed = straight tube rectangle;
  black dotted = path-following tube rectangle.

## F1 numbers (k36_panels.csv; cells in k36_cells.csv)

| panel | consumer | ego speed | sensitive cells | farthest sensitive range | footprint area share | tube rect area share | sensitive cells in tube |
|---|---|---|---|---|---|---|---|
| nusc scene-0110 f0 | braking ×0.5 | 5.06 m/s | 3 | 12.5 m | 0.0536 | 0.174 | 100% |
| nusc scene-0110 f0 | braking ×1 | 10.12 m/s | 21 | 55 m | 0.0536 | 0.174 | 100% |
| nusc scene-0110 f0 | braking ×1.5 | 15.18 m/s | 24 | 70 m | 0.0536 | 0.174 | 100% |
| nusc scene-0110 f0 | TTC warning ×1 | 10.12 m/s | 15 | 35 m | 0.0536 | 0.174 | 100% |
| AHD seq01 f14 | curved braking ×1 | 4.83 m/s | 2 | 12.5 m (arc 12.6 m) | 0.0289 | 0.030 (straight) | 100% straight / 0% path-following (no tube) |

### Rectangles (pixels)
- nuScenes footprint: [586, 461, 1067, 621] (1600×900 image).
- nuScenes tube: [169, 468, 1484, 659], half-width 5.25 m at 10 m. The same tube is used for all four nuScenes panels,
  because the tube is fixed by the frame's S budget.
- AHD footprint: [868, 602, 1238, 782].
- AHD straight tube: [793, 622, 1144, 820], half-width 1.25 m (1920×1200 image).

### Reading
- **The field's reach scales with speed and consumer** (12.5 → 55 → 70 m for ×0.5 / ×1 / ×1.5; 35 m for the TTC
  warning). **The footprint crop does not change on this frame.**
  - Every panel's nearest sensitive cells are at 12.5 m, lateral ±1.5 m.
  - The bounding box is set by those near cells: their ground contact sets the bottom, a 1.6 m probe above the horizon
    sets the top, and their lateral extent sets the sides. The projections of the farther cells fall inside it.
  - The difference between consumers is therefore in the field (which cells are sensitive), not in the requested crop
    on this frame.
- **The hybrid's tube contains every sensitive cell here.**
  - The tube starts at 10 m, and its 'fill' wedge widens with range (±5.25 m at 10 m, so wider than the plotted ±6 m
    beyond about 11 m). The grid starts at 12.5 m.
  - The tube rectangle covers 3.3× the footprint's area (0.174 vs 0.054).
- **Curved frame.** At the frame's S budget (27,648 canvas px), the path-following tube is **not formed**: its
  narrowest rectangle (half-width 0.5 m: [972, 596, 1515, 822]) needs 36,864 px.
  - So the curved hybrid requests no tube on this frame (0 tube cells), only its support crops (none: KEEP).
  - The straight tube (1.25 m) covers both sensitive cells.

## Panels (24 files, each PDF + PNG, in panels/)
| file stem | content |
|---|---|
| nusc_scene-0110_0000_brake_x0.5_{bev,img} | braking field, speed ×0.5 |
| nusc_scene-0110_0000_brake_x1.0_{bev,img} | braking field, ×1 |
| nusc_scene-0110_0000_brake_x1.5_{bev,img} | braking field, ×1.5 |
| nusc_scene-0110_0000_ttcwarn_x1_{bev,img} | TTC-warning field, ×1 |
| ahd_seq01_0014_curved_{bev,img} | curved-consumer field; straight tube; no path-following tube (not formed) |
| ahd_seq01_0014_curved_candidate05_{bev,img} | the same, plus the path-following tube's narrowest candidate (0.5 m, dotted), which exceeds the budget and is NOT requested: geometry illustration only |

## F2 — nuScenes test scene-0122 frame 7 (stored K28 instr record; k36_f2.csv)

| quantity | value |
|---|---|
| required action | KEEP |
| v6-S (+ v3a) | KEEP |
| unc2@S + C1 + v3a | KEEP |
| native + C1 + v3a | KEEP |
| px v6-S | 0.0839 |
| px unc2@S | 0.0683 |

## Files
- k36_panels.csv: one row per panel (all numbers above).
- k36_cells.csv: sensitive and tube cells per panel.
- k36_selection.csv: the frames examined by the rules.
- k36_f2.csv; checks.json; panels/; TASK.md.
