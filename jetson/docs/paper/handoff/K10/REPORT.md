**Ranking of Tab. 1 (AV2):** ours stays the best realizable method and is still significantly above every baseline row. The ranking below it changes:
- every whole-frame resolution becomes negative (u960 −0.021, native −0.058);
- the oracle frame router (0.315) moves ABOVE ours (0.122);
- ours' AV2 G shrinks from 0.561 to 0.122, because 60% of the cheap reference loss on AV2 was the hood artefact.

**Tab. 2 (Argoverse-HD) does not change**: no ego part is visible there.

# K10: the AV2 ego-hood artefact (post-hoc robustness check, 2026-10-05)

## Mask (defined once per dataset)
- **AV2 ring_front_center (2048 rows):** the hood is the full-width band y ≥ 1830. Inspected on 10 val logs: the hood's top edge is at rows ~1840–1870 at the image centre. The AV2 fleet shares the camera mount.
- **Argoverse-HD (1200 rows):** no ego part is visible in 8 inspected sequences; the analogous bottom 5% band (y ≥ 1140) is used.
- **nuScenes CAM_FRONT (900 rows):** no ego part is visible; the band y ≥ 855 is used.
- **"In the mask":** box containment ≥ 0.5. IoU with the band is never ≥ 0.5 for an object box, so containment is the operative rule.

## 1. How often does an ego-region detection drive the action? (`hood_share.csv`)
- **Script:** `remote/A_method/k10_hood.py` (md5 1aaff757…), unmasked, frozen configurations.
- **Driver:** the object with the largest required deceleration when the action is not KEEP.
- **Parity:** G reproduces the main tables (AV2 ours 0.561, u960 0.446, native 0.558; Argoverse-HD ours 0.091).

| AV2 val (11,818 frames) | hood-driven frames | share of the pipeline's decision loss from them |
|---|---|---|
| cheap | **50.9%** | **81.1%** |
| u640 | 36.9% | 67.7% |
| u960 | 11.7% | 31.1% |
| u1280 | 9.0% | 25.3% |
| u1536 | 2.8% | 8.7% |
| native u2048 | 0.3% | 0.9% |
| **ours final** | 5.6% | 21.2% |

- Ours puts a crop on the hood region in 54% of frames: the hood draft box is a support object, so the loop verifies it.
- **Argoverse-HD:** at most 1 of 15,006 frames for every pipeline.
- **nuScenes test (part 3):** 1 of 9,181 frames (cheap). Detections in the band: < 0.001 per frame for every pipeline. **There is no ego-part artefact on nuScenes.**

## 2. Masked re-evaluation (`table_main_av2_masked.csv`, `ahd_masked.csv`)
**How it was run**
- The existing table scripts were run unchanged in a shadow tree ($B/k10_masked, with its own results/), via `k10_masked_run.py` (md5 4abc786b…).
- That wrapper sets in-mask detections to range = +∞ before the lift, for every pipeline. They then carry no demand, are never support, and seed no crop.
- Scripts run this way: av2_eval_v4 (cheap, u640, u960, v3 / v4-lite ± refiner), native_uniform (u1280, u1536, native), av2_baselines (FOVEA, SAHI, uncertainty / temporal_crit / random crops), av2_sota labels + AdaScale + score, and argoverse_hd_eval.
- Chain: `k10_chain.sh` (md5 a5570590…). Tables: `k10_tables.py` (md5 c1de892f…).
- **Router rows:** the original scores and calibration, with ρ matched to ours' masked pixels.
- **Crop baselines:** keep the original per-frame budget, i.e. frozen v3's unmasked crop area.
- **All-cheap loss on AV2:** 23,174 unmasked → **9,294 masked**.

**Selected rows** (full table in the CSV; interval = ours final − row, log bootstrap):

| AV2 val | G unmasked | **G masked** | ours − it, masked |
|---|---|---|---|
| **ours final (v4-lite + refiner v2)** | 0.561 | **0.122** (px 0.049) | — |
| ours without refiner | 0.537 | 0.032 | [+0.046, +0.140] |
| v3 + refiner v1 | 0.545 | 0.077 | [+0.016, +0.080] |
| u960 / u1280 / u1536 / native | 0.446 / 0.484 / 0.540 / 0.558 | −0.021 / −0.008 / −0.031 / −0.058 | all > 0 (lowest bound +0.069) |
| uncertainty crops | 0.494 | 0.057 | [+0.007, +0.124] |
| AdaScale / FOVEA / SAHI (s10) | 0.317 / 0.328 / 0.088 | −0.057 / −0.023 / −0.248 | all > 0 |
| Budget-Adaptive Routing (official / decision-value) | 0.072–0.230 | −0.013 – −0.006 | all > 0 |
| **oracle frame router** | 0.371 | **0.315** | **[−0.245, −0.137]** |

**Argoverse-HD:** 7 of about 926k boxes fall in the band, and every G is unchanged to 4 decimals (ours + refiner 0.091; every whole-frame resolution −0.03 … −0.09).

## Reading (against the method)
1. **The AV2 G values in Tab. 1 are dominated by an artefact.** COCO-YOLOv8s detects the ego hood at low resolution, and the cheap draft brakes for it on half of the frames.
   - Whole-frame passes at higher resolution stop detecting the hood. That explains most of their large positive AV2 G (0.45–0.56), as it does ours.
   - Without the hood, AV2 behaves like Argoverse-HD: escalating the whole frame is harmful at every resolution, and ours is the only positive realizable method.
2. **The paper must report the masked AV2 numbers** (main text or a prominent robustness row). The 0.561 must not be presented as a clean decision-value gain.
3. The verification loop's crops on the hood (54% of frames) are wasted pixels for a fixed-mount camera. A static ego mask is the obvious deployment fix; it applies to every pipeline alike.
