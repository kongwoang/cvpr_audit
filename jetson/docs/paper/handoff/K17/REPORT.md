**What the field changes, per dataset** (post-hoc analysis; FULL = frozen loop vs SUPPORT ONLY; with refiner unless stated)
- **nuScenes new test.** The field changes the action on **340 of 9,181 frames (3.7%)**, raising it on 275. Decision loss on those frames: **full 571 vs support only 1,937**, so the field removes 70% of support-only's loss there. Pixel cost: +0.050 px (0.100 vs 0.050).
- **AV2 val (ego mask).** It changes **911 of 11,818 frames (7.7%)**, raising on 746. Loss: **full 1,370 vs support only 1,374**, a wash. The gains from true discoveries (108 vs 345) and corrected draft objects (492 vs 734) are cancelled by extra braking from field-crop detections in the "other" group (687 vs 191). Spurious detections are not a net cost here (82 vs 104). Pixel cost: +0.012 px.
- **Argoverse-HD val.** It changes **1,849 of 15,006 frames (12.3%)**, raising on 1,669. Loss: **full 3,498 vs support only 2,907**, so the field costs more than it saves: true discoveries 588 vs 1,898, but "other" 1,722 vs 331 and spurious 418 vs 62. Pixel cost: +0.014 px.
- **In the regime the field is built for, it is decisive on all three datasets.** These are frames where braking is required but the draft holds no object in the support of the required action (3.4–5.7% of frames). G with refiner, full vs support only: 0.349 vs 0.013, 0.329 vs 0.124, 0.554 vs 0.006. Temporal_crit is second (0.343 / 0.240 / 0.455).
- **Reading.** The field finds what the draft missed. On AV2 and Argoverse-HD, the extra detections in its crops also produce false alarms of the same order, so its net effect over support only is zero there. At equal pixels it is not distinguishable from spending them on larger support crops or random crops.

# K17: mechanism of the unseen-object field

**Setup**
- **Datasets:** nuScenes new test (228 scenes), AV2 val with the ego mask (150 logs), Argoverse-HD val (24 sequences, frames with 3D labels).
- **Run:** `remote/A_method/k17_variants.py`, which uses the frozen loop with switches (`k_loop.run_g`). Its FULL and SO arms reproduce instr_eval bit for bit.
- **Statistics:** unit bootstrap, 2,000 resamples.
- **Tables:** `field_contribution_{nusc,av2,ahd}.csv`, sections 1–5. Figure: `field_frames.pdf/png`.

**(1) Frames where FULL and SO act differently, by what the field crop did** (with refiner; loss full / support only):

| group | nuScenes frames: loss | AV2 frames: loss | A-HD frames: loss |
|---|---|---|---|
| a. true discovery (relevant object missed by the draft) | 36: 88 / 998 | 117: 108 / 345 | 643: 588 / 1,898 |
| b. spurious (no annotated object, raises the action) | 33: 73 / 9 | 53: 82 / 104 | 143: 418 / 62 |
| c. draft object's box / range changed | 198: 264 / 891 | 358: 492 / 734 | 442: 771 / 617 |
| d. other (a field-crop detection of an annotated but non-relevant object, or a removed draft twin) | 73: 146 / 38 | 383: 687 / 191 | 621: 1,722 / 331 |

Without refiner the pattern is the same (section 1, `refiner = False`).

**(2) Events per 1,000 frames** (with refiner; full / support only, full − SO interval):

| | nuScenes | AV2 masked | Argoverse-HD |
|---|---|---|---|
| correct HARD_BRAKE | 24.5 / 22.3 [+0.2, +5.2] | 24.2 / 23.8 n.s. | 5.5 / 5.5 n.s. |
| correct DECELERATE | 36.2 / 30.2 [+3.2, +9.1] | 143.0 / 132.7 [+6.1, +14.9] | 119.2 / 90.5 [+5.5, +60.2] |
| unnecessary DECELERATE | 17.0 / 7.3 [+6.6, +12.8] | 69.6 / 44.6 [+18.6, +31.8] | 49.0 / 36.1 [+7.2, +19.5] |
| unnecessary HARD_BRAKE | 19.7 / 14.5 [+2.8, +7.8] | 112.2 / 98.9 [+7.7, +18.8] | 250.2 / 197.5 [+35.1, +71.3] |

**(3) The field's regime:** see the first lines; numbers are in section 3, which also gives uncertainty crops (0.119 / 0.169 / 0.047).

**(4) Equal compute** (G with refiner; full − variant interval).
- **Larger support crops.** On AV2, support only with context 3.5 instead of 2 reaches full's tensor px (ratio 1.01): G 0.115 vs 0.122, [−0.046, +0.062]. On Argoverse-HD, context 3 (ratio 1.01): 0.105 vs 0.091, [−0.138, +0.115]. On nuScenes even context 12 reaches only 0.79× full's px, because 80% of frames have no support crop; full is above every context [+0.008 … +0.43], except context 6 [−0.001, +0.371].
- **Random extra crops up to full's per-frame px** (realised 1.05–1.16×):
  - nuScenes −0.059, [−0.040, +0.468];
  - AV2 0.124, [−0.043, +0.040];
  - Argoverse-HD 0.127, [−0.143, +0.082].
- **Conclusion:** at equal px, the field is not significantly better than spending the same px elsewhere on any dataset (nuScenes intervals are wide).

**(5) Closing speed** (frozen v_close 0; car probe; test splits; section 5):

| | AV2 vc 0 / 5 / 10 / support only | Argoverse-HD vc 0 / 5 / 10 / support only |
|---|---|---|
| G with refiner | 0.122 / 0.081 / 0.082 / 0.122 | 0.091 / 0.018 / 0.010 / 0.127 |
| G without refiner | 0.031 / −0.030 / −0.036 / 0.083 | 0.039 / −0.048 / −0.058 / 0.116 |
| tensor px | 0.053 / 0.068 / 0.075 / 0.041 | 0.070 / 0.078 / 0.085 / 0.056 |
| agreement with π(F) | 0.860 / 0.863 / 0.863 / 0.834 | 0.814 / 0.814 / 0.814 / 0.772 |
| recovery, controlled deletion (2,923 / 2,723 objects) | 0.870 / 0.930 / 0.939 / 0.192 | 0.844 / 0.875 / 0.892 / 0.241 |

- **Recovery rises, G falls.** A faster probe recovers more deleted objects (vs 0.19–0.24 without the field), but its larger footprint brings more false-alarm detections, so G falls.
- **Contrast with K16.** K16 found that a 10 m/s probe fixes 21% of the disagreement with π(F) under EXACT locality. With real crops, agreement does not move (0.860 → 0.863).
