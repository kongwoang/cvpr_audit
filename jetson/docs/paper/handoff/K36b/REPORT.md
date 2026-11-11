# K36b — how the draft-round field depends on the action and the speed, compared with the hybrid's fixed tube

Post-hoc, CPU only, frozen v6-S. No detector was run; frames were chosen by fixed rules. Every panel is listed.
- Code: remote/A_method/k36b.py. It reuses k36_f1's draft reconstruction and tube / footprint geometry, both checked in K36.
- Run on VUW 2026-10-26 (logs/k36/k36b.log); outputs results/k36b/ copied here.

**Check.** The draft rebuilt from stored data gives the stored a_cheap on every frame of the three test splits: 9,181
nuScenes / 11,818 AV2 / 15,062 Argoverse-HD frames, **0 mismatches** (checks.json).
- nuScenes uses the cached draft.
- AV2 and Argoverse-HD use the K30 apdump cheap scene lifted on the previous frame's draft. AV2 also uses the K10 ego mask,
  y ≥ 1830.

## Definitions
- **Field.** The draft-round braking field: dsf.insertion_grid on the draft scene at the frame's ego speed, v_close 0.
  These are the cells where one inserted probe raises the draft action to the next level. It is empty at HARD-BRAKE by
  construction.
- **Footprint.** k_loop.field_crop_g (the field crop v6-S requests in round 1).
- **Support crops.** v6-S's round-1 support crops of the draft action: k_loop.box_crop_ctx of SP.support, context
  SP.CTX. Area share = the area of their union / the image area.
- **Tube.** Hybrid tsp8@S's tube rectangle (TUBE[8]: 10–70 m, 'fill'), with the budget = the frame's v5.1-S crop px minus
  the hybrid's support crops filled first; as K36.
  - "Not formed" means the remaining budget fits no tube (e.g. budget 0, or all of it spent on support crops).
  - A BEV cell is in the tube iff its ground-contact point projects inside the rectangle.
- **px.** The stored network px of v6-S and hybrid tsp8@S (K28 test records).

## F1 — nuScenes test panels (panels/, PDF + PNG, no text; f1_panels.csv, f1_cells.csv)
Rules (time order = scene name, then frame index):
- (a) K36's frame (v6-S KEEP, 8–12 m/s, no draft object in the corridor ≤ 70 m);
- (b) the first a_cheap = DECELERATE at 8–12 m/s;
- (c) the first a_cheap = HARD-BRAKE at 8–12 m/s;
- (d) the first ego speed < 2.5 m/s with a_cheap = KEEP.

| panel | frame | draft action | ego speed | sensitive cells | farthest | footprint share | support-crop share | tube share (half-width) | sensitive in tube | px v6-S | px hybrid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) | scene-0110 f0 | KEEP | 10.12 | 21 | 55 m | 0.054 | 0 | 0.174 (5.25 m) | 100% | 0.114 | 0.238 |
| (b) | scene-0121 f4 | DECELERATE | 9.93 | 6 | 17.5 m | 0.055 | 0.004 (1 crop) | 0.076 (2.25 m) | 100% | 0.078 | 0.093 |
| (c) | scene-0123 f5 | HARD-BRAKE | 9.34 | 0 | – | 0 | 0.164 (2 crops) | not formed | – | 0.113 | 0.113 |
| (d) | scene-0122 f21 | KEEP | 1.99 | 0 | – | 0 | 0 | not formed | – | 0.043 (= draft only) | 0.043 |

Notes on the panels:
- (a) is K36's panel, re-rendered with the (empty) support crops.
- (c): the support crops use the whole S budget, so no tube remains; v6-S and the hybrid run the same crops here.
- (d): the field is empty, so v5.1-S requests nothing, the S budget is 0 and no tube forms; both rows run the draft only.
- File stems: `{a,b,c,d}_nusc_<scene>_<frame>_brake_x1_{bev,img}`. Image panels: orange = footprint, blue = support
  crops, black dashed = tube.

## F2 — all three test splits, by draft action × ego-speed bin (f2_groups.csv; per frame f2_frames.csv.gz)
Plain means over the evaluated frames. Footprint and tube shares count 0 when the field is empty or no tube is formed.
The last column is the paired unit bootstrap (2,000, seed 0) of the mean px difference v6-S − hybrid.

| dataset | action | speed (m/s) | frames | field non-empty | footprint share | tube formed | tube share | px v6-S | px hybrid | v6-S − hybrid px [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| nuScenes | KEEP | <2.5 | 2,818 | 0% | 0 | 0% | 0 | 0.043 | 0.043 | 0 |
| nuScenes | KEEP | 2.5–8 | 3,954 | 74% | 0.040 | 74% | 0.082 | 0.083 | 0.124 | −0.041 [−0.050, −0.032] |
| nuScenes | KEEP | 8–12 | 1,324 | 100% | 0.054 | 100% | 0.096 | 0.087 | 0.116 | −0.029 [−0.041, −0.020] |
| nuScenes | KEEP | >12 | 58 | 100% | 0.055 | 100% | 0.110 | 0.092 | 0.113 | −0.021 [−0.037, −0.006] |
| nuScenes | DECELERATE | <2.5 | 36 | 0% | 0 | 39% | 0.036 | 0.104 | 0.172 | −0.068 [−0.126, −0.027] |
| nuScenes | DECELERATE | 2.5–8 | 377 | 11% | 0.006 | 71% | 0.092 | 0.152 | 0.246 | −0.095 [−0.118, −0.074] |
| nuScenes | DECELERATE | 8–12 | 191 | 100% | 0.054 | 100% | 0.131 | 0.120 | 0.188 | −0.068 [−0.101, −0.042] |
| nuScenes | DECELERATE | >12 | 22 | 100% | 0.055 | 100% | 0.130 | 0.113 | 0.138 | −0.025 [−0.050, −0.007] |
| nuScenes | HARD-BRAKE | <2.5 | 16 | 0% | 0 | 63% | 0.092 | 0.122 | 0.292 | −0.170 [−0.249, −0.091] |
| nuScenes | HARD-BRAKE | 2.5–8 | 197 | 0% | 0 | 60% | 0.083 | 0.182 | 0.268 | −0.086 [−0.117, −0.058] |
| nuScenes | HARD-BRAKE | 8–12 | 151 | 0% | 0 | 40% | 0.050 | 0.119 | 0.152 | −0.034 [−0.056, −0.017] |
| nuScenes | HARD-BRAKE | >12 | 37 | 0% | 0 | 78% | 0.074 | 0.096 | 0.124 | −0.029 [−0.032, −0.028] |
| AV2 | KEEP | <2.5 | 2,843 | 0% | 0 | 0% | 0 | 0.026 | 0.026 | 0 |
| AV2 | KEEP | 2.5–8 | 2,882 | 60% | 0.030 | 60% | 0.061 | 0.044 | 0.053 | −0.009 [−0.011, −0.007] |
| AV2 | KEEP | 8–12 | 1,562 | 100% | 0.049 | 100% | 0.095 | 0.054 | 0.064 | −0.011 [−0.014, −0.008] |
| AV2 | KEEP | >12 | 353 | 100% | 0.049 | 100% | 0.079 | 0.047 | 0.052 | −0.005 [−0.008, −0.003] |
| AV2 | DECELERATE | <2.5 | 108 | 0% | 0 | 56% | 0.058 | 0.061 | 0.104 | −0.043 [−0.059, −0.029] |
| AV2 | DECELERATE | 2.5–8 | 1,432 | 10% | 0.005 | 79% | 0.080 | 0.083 | 0.134 | −0.051 [−0.066, −0.039] |
| AV2 | DECELERATE | 8–12 | 899 | 100% | 0.049 | 100% | 0.111 | 0.066 | 0.088 | −0.022 [−0.027, −0.017] |
| AV2 | DECELERATE | >12 | 45 | 100% | 0.049 | 100% | 0.072 | 0.047 | 0.054 | −0.006 [−0.012, −0.003] |
| AV2 | HARD-BRAKE | <2.5 | 125 | 0% | 0 | 66% | 0.062 | 0.067 | 0.114 | −0.048 [−0.069, −0.029] |
| AV2 | HARD-BRAKE | 2.5–8 | 805 | 0% | 0 | 87% | 0.099 | 0.097 | 0.152 | −0.054 [−0.065, −0.044] |
| AV2 | HARD-BRAKE | 8–12 | 725 | 0% | 0 | 76% | 0.083 | 0.079 | 0.101 | −0.022 [−0.029, −0.017] |
| AV2 | HARD-BRAKE | >12 | 39 | 0% | 0 | 54% | 0.036 | 0.053 | 0.051 | +0.001 [−0.005, +0.012] |
| Argoverse-HD | KEEP | <2.5 | 3,424 | 0% | 0 | 0% | 0 | 0.031 | 0.031 | 0 |
| Argoverse-HD | KEEP | 2.5–8 | 3,249 | 79% | 0.033 | 79% | 0.062 | 0.058 | 0.089 | −0.030 [−0.042, −0.018] |
| Argoverse-HD | KEEP | 8–12 | 2,392 | 100% | 0.042 | 100% | 0.073 | 0.059 | 0.090 | −0.031 [−0.050, −0.018] |
| Argoverse-HD | KEEP | >12 | 80 | 100% | 0.042 | 100% | 0.034 | 0.044 | 0.046 | −0.002 (1 unit) |
| Argoverse-HD | DECELERATE | <2.5 | 134 | 0% | 0 | 76% | 0.104 | 0.178 | 0.442 | −0.264 [−0.391, −0.048] |
| Argoverse-HD | DECELERATE | 2.5–8 | 1,233 | 6% | 0.003 | 65% | 0.048 | 0.086 | 0.141 | −0.056 [−0.077, −0.039] |
| Argoverse-HD | DECELERATE | 8–12 | 723 | 100% | 0.042 | 100% | 0.089 | 0.071 | 0.109 | −0.038 [−0.053, −0.026] |
| Argoverse-HD | HARD-BRAKE | <2.5 | 211 | 0% | 0 | 66% | 0.087 | 0.122 | 0.311 | −0.189 [−0.316, −0.021] |
| Argoverse-HD | HARD-BRAKE | 2.5–8 | 2,199 | 0% | 0 | 65% | 0.061 | 0.127 | 0.203 | −0.076 [−0.102, −0.049] |
| Argoverse-HD | HARD-BRAKE | 8–12 | 1,361 | 0% | 0 | 64% | 0.057 | 0.096 | 0.134 | −0.038 [−0.062, −0.025] |

Group sizes and units per group are in f2_groups.csv. Groups with very few units have degenerate intervals, e.g.
Argoverse-HD KEEP > 12 m/s, which has 1 unit. Empty groups (e.g. Argoverse-HD DECELERATE / HARD-BRAKE > 12 m/s) are
absent.

## Reading
1. **The field depends on the action and the speed. The tube does not.**
   - Below 2.5 m/s the field is empty at every action on all three splits. At KEEP it is non-empty on 60–79% of frames at
     2.5–8 m/s and on 100% above 8 m/s. At DECELERATE it is non-empty on only 6–11% of frames at 2.5–8 m/s, but on 100%
     at ≥ 8 m/s, where it shrinks to the near range (panel b: 17.5 m vs 55 m at KEEP). At HARD-BRAKE it is always empty.
   - The hybrid still forms its tube where the field is empty: on 39–87% of the DECELERATE / HARD-BRAKE frames in the groups where
     the field is (almost) always empty. Over all evaluated frames, a tube is formed with an empty field on 5.0% (nuScenes), 20.4% (AV2) and
     21.9% (Argoverse-HD) of frames.
   - The converse almost never happens: a field without a tube occurs on 0 / 0 / 0.007% of frames.
   - The hybrid's budget is the frame's v5.1-S crop px, so its spend inherits part of this dependence. But the
     budget-filling tube converts budget left over after its support crops into a corridor crop whether or not any
     corridor cell could change the action.
2. **Where both exist, the tube contains the field.** The share of sensitive cells inside the tube is 97–100% in every
   group; below 100% only on 0.4% (Argoverse-HD) and 0.07% (AV2) of frames. The tube's area is 1.5–2.4× the footprint's
   (KEEP / DECELERATE at ≥ 8 m/s; the exception is Argoverse-HD KEEP > 12 m/s, 0.8×, 1 unit).
3. **The computation difference.** v6-S uses fewer network px than the hybrid in every group except the two where both
   run the draft only (KEEP < 2.5 m/s: equal) and AV2 HARD-BRAKE > 12 m/s (n.s.; 39 frames).
   - The gap is largest where the field is empty but the tube is still formed: HARD-BRAKE and DECELERATE at low speed,
     e.g. −0.17 to −0.26 px at < 2.5 m/s on nuScenes / Argoverse-HD.
   - It is smallest at KEEP at high speed.
   - So the field differs from the fixed tube mainly by *not* requesting computation when no corridor insertion can
     change the decision, rather than by covering cells the tube misses.

## Files
f1_panels.csv, f1_cells.csv, panels/ (16 files), f2_groups.csv, f2_frames.csv.gz, checks.json, TASK.md.
