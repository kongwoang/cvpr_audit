# K33 STATUS

**Goal:** Table 1 v7. Every row uses the same detector (COCO YOLOv8s: cheap pass at 320, target at the verification
scale) and the same components as v6-S: refiner v3a + C1 + fidelity-consistent history, plus the truncation guard and
merge rule for crop methods.
- Datasets: nuScenes new test, AV2 val with the ego mask, Argoverse-HD val.
- v6-S is frozen.
- Baselines are tuned on DEV only (nuScenes new val + AV2 CFG-DEV) with the K28 rule (same number of candidates).
  Argoverse-HD gets the settings transferred from AV2. Each baseline is evaluated once on test, post-hoc.
- The official-form rows go to the supplement.
- Runs on VUW / VUW1 (G on the Jetson), in parallel with K32: K32 has CPU priority (MAXJ 4 per host is unchanged), and
  K33 GPU jobs use few CPU threads.

## Inventory (2026-10-19, before any K33 run)

**Not stored: boxes of the published methods.**
- SAHI, FOVEA, AdaScale, UFP, LZU, ZoomDet and ESOD stored only per-frame costs / actions, lifted with DEEP's mono
  lift without C1 or the refiner (results/{sahi,fovea,adascale,ufp,lzu,zoomdet,esod}/eval_newtest.pkl;
  results/k30/prior has SAHI s1.0 on nuScenes and FOVEA KDE_I on AV2 with geometry, but no boxes).
- So "re-lift stored detections" applies only where boxes exist: the routers (results/router, results/av2_sota:
  boxes__{train,val,test}.pkl). Every other method in A is re-run.
- There is no Argoverse-HD run of any published method.

**Code that exists:** sahi_baseline.py, fovea_baseline.py, adascale.py (scale nets for nuScenes and AV2),
router_baseline.py / k12_routers.py, ufp_baseline.py (official UFP module), lzu_baseline.py, fovea_ft.py, ufp_ft.py,
ESOD / ZoomDet chains, av2_ft_prepare.py / av2_ft_eval.py.

**AV2 fine-tuned regime.**
- Weights exist, trained on the unmasked AV2 train images of data/av2_ft:
  - u960-ft: out/av2_ft/plain960;
  - ESOD: out/esod/av2_yolov5s;
  - ZoomDet: out/zoomdet/av2.
- Their stored scores are on the unmasked protocol with v4-lite.
- data/av2_ft, data/fovea_ft and data/ufp_ft were deleted in the 2026-10-19 cleanup. av2_ft_prepare.py is
  deterministic (fixed log split, every 5th / 10th frame), so data/av2_ft can be rebuilt identically.
- FOVEA-ft, LZU-ft and UFPMP-Det-ft exist for nuScenes only.

**Not available:** review/table1_design.md (named in the task) does not exist in the repo; it is written as part of
K33 (handoff/K33/table1_design.md).

## Pre-registration (2026-10-19, before any K33 run)

table1_design.md §1–§2 is the registered design, written before any K33 computation:
- the rows A–G;
- the 12 dev candidates per family, with the official / K28 form marked;
- the selection rule (highest mean dev G with components; never weaker than the official / K28 form);
- the pixel-cap rows and the extra metrics;
- the Argoverse-HD transfer;
- the F rows.

Changes only by a dated amendment here, before the affected run.

### Amendment 1 (2026-10-20 VUW 00:05, before any A / C run; D / E are unaffected)
- **px convention.** The cheap draft's px counts for every row that uses the draft (D, E, B, AdaScale, FOVEA KDE_I,
  routers, Chanakya, and v6-S as before). It does not count for SAHI, FOVEA KDE_S or whole-frame rows, which do not use
  the draft. The cost's jerk term uses the cheap action history for every row, as in K28.
- **SAHI** keeps its own cross-slice merge (GREEDYNMM over the slices plus the full-image pass) as its truncation
  handling. No guard is added: the slices tile the whole frame, so K28's merge rule is degenerate. The components are
  C1 on its own previous-frame detections, plus refiner v3a.
- **Routers and Chanakya** use K28's router convention: a routed frame's strong detections are lifted with the strong
  arm's continuous history. This favours the baseline (in deployment the previous frame is often cheap) and replaces
  the "history from the last strong frame" sentence of table1_design.md §2 A.
  - Their per-frame costs come from whole-frame + C1 + v3a rows at 640 / 768 / 960 / 1280:
    - dev: K28's stored YOLOv8s dev rows results/k28/{nusc,av2}_r2basewf_r2;
    - test and Argoverse-HD: whole-frame rows computed in K33's part pub;
    - training splits for Chanakya: a K33 run on nuScenes new train and AV2 train.
  - Routing and the policy are applied to those per-frame costs. Thresholds and shares are fixed on dev, before test.
  - Estimators: BAR = the stored official skip_offbin models (results/router, results/av2_sota; the AV2 model for
    Argoverse-HD); the decision-value router = cond_V, retrained deterministically from the stored training features
    (router_baseline.py stage_train, XGBoost random_state 42); scores computed in-run on every dev / test frame.

## Log
- 2026-10-19 (VUW 23:00): data/av2_ft rebuild started on VUW1 (CPU, nice 10). This is operational (re-creating
  deleted data; no selection): av2_ft_prepare.py md5 a08e8ca0, unchanged, so it produces the same images, labels and
  split as the existing u960-ft / ESOD / ZoomDet weights. Log: logs/k33/av2_ft_prepare.log.
- 2026-10-19 (VUW 23:55): table1_design.md written and registered (above).
  - Order, per the user's request ("chạy các job trên kit trước cho tối ưu"): the Jetson part G starts first.
  - Next: the shared baseline module, then the Jetson timing of the candidates and, in parallel, k33_eval.py on VUW.
- 2026-10-19 (VUW 23:46): D / E implemented. Code: src/vtf/k33sel.py (shared with the Jetson), scripts/A_method/k33_eval.py
  part de (md5 2a50d976; a copy of k31_eval.py plus part de and --setting del_all), and k33_de_chain.sh.
  - Smoke queued: 1 nuScenes val scene + 1 AV2 CFG-DEV log, every candidate.
- 2026-10-19 (local 17:50, VUW 23:50): G started first on the Jetson (the user: "chạy các job trên kit trước cho tối ưu").
  - scripts/69_k33_jetson.py = 67_k30_jetson.py + the D / E rows (same vtf.k33sel selection; crop detections of earlier
    rounds reused within the frame); scripts/k33_j_window.sh.
  - Smoke OK on 12 nuScenes test20 frames (48 rows; 12.5–45 ms mean; one first-shape outlier of 2.3 s).
  - Candidate timing running: nusc_test20 and av2 with the ego mask, 1 repetition. These runs store latency and actions
    only, no J, and the dev selection never reads them.
  - The Jetson rebooted at 17:20 local with nothing of ours running (cause unknown); /tmp was wiped. The hourly check
    moved to remote/ops/hourly_check.sh.
- 2026-10-19 (VUW 23:53): D / E smoke OK (nuScenes val scene-0108 39 frames; AV2 CFG-DEV 1 log 78 frames). All 48
  candidates and v6-S run, at ~1.5 s per frame.
- 2026-10-19 (VUW 23:54): D / E dev runs launched (all 48 candidates; OMP 2 to leave CPU to K32): nuScenes new val on VUW
  (3 shards), AV2 CFG-DEV on VUW1 (6 shards in 2 groups). Logs logs/k33/de_dev_*.log; output results/k33/<ds>_de_dev/.
- 2026-10-20 (VUW 00:33): A / B / C infrastructure in k33_eval.py part pub (md5 at the dev launch), with families:
  - ada, sahi, fov (A);
  - ufp (B);
  - rt (router scores and Chanakya state) and wf (whole frame + components);
  - --split train for Chanakya's training data.
  Supporting runs:
  - k33_fovsal.py (KDE_S saliencies, 4 per dataset): nuScenes bw64 / amp1 equals the stored results/fovea/dataset_saliency.pt
    exactly.
  - k33_condv.py (cond_V retrained): predictions on the stored test frames equal the stored scores (max |diff| 0) for both
    datasets.
  - Smoke of part pub (all families, 1 nuScenes val scene + 1 AV2 CFG-DEV log) queued.
- 2026-10-20 (VUW 00:42): D / E dev nuScenes DONE (152 / 152, exit 0); AV2 CFG-DEV at 209 / 300.
  - Jetson G candidate timing (D / E, both frame sets) DONE at 18:31 local.
  - Part pub smoke: nuScenes DONE, every family's rows present (~4 s per frame for all families together); AV2 running.
  - Part pub dev nuScenes launched on VUW (families ada, sahi, fov, ufp, rt; 3 shards; k33_eval.py md5 d1d2a6ee).
    Routers and Chanakya on dev use K28's stored whole-frame rows (amendment 1).
- 2026-10-20 (local 18:55): Jetson G for A / B.
  - Code: src/vtf/k33pub.py (sahi 0.12.7 slicing and GREEDYNMM re-implemented; FOVEA's official grid generator; UFP's
    official packing) and the pub block of 69_k33_jetson.py.
  - Rows within the dynamic engine's 1024 px: AdaScale top 960, SAHI ×12, FOVEA canvases 640 / 960, UFP mosaics
    512 / 768 / 1024.
  - Smoke OK on 10 frames: px equal to VUW's; 18–112 ms mean.
  - Candidate timing started (k33_j_window.sh candidates_pub: nusc_test20 and av2, 1 repetition, latency + action only).
  - Candidates above 1024 px (AdaScale top 1280 / native, FOVEA 1280, UFP 1333) are timed only if dev selects them,
    with any engine build at the end of the window.
- 2026-10-20 (VUW 00:54): F started on VUW1.
  - Builds: k33_av2ft.py (FOVEA warped 960 × 736 set, UFP mosaic set, LZU saliency 48 × 36), via k33_f_chain.sh build.
  - Then the trainings fovea / ufp / lzu: YOLOv8s, 15 epochs, 960, batch 32, 4 dataloader workers (d68_train.py now
    reads WORKERS; default 8 unchanged).
  - The existing u960-ft / ESOD / ZoomDet weights and stored AV2 val predictions are re-used. The ego mask acts in the
    lift (masked_pg), not on the images, so stored predictions stay valid.
- 2026-10-20 (VUW 01:10): D / E dev DONE (nuScenes val 152 scenes / 6,117 frames; AV2 CFG-DEV 300 logs / 23,665 frames).
  Selection by the registered rule (k33_select.py; dev_candidates.csv, dev_selection.csv):

  | family | selected | dev G (nuScenes / AV2) | official form | its dev G |
  |---|---|---|---|---|
  | AnA one pass | ana05hinfo (t 0.05, T_h ∞) | 0.283 / 0.339 | ana25hinfo | −0.008 / 0.312 |
  | AnA iterated | ana05hinfi | 0.282 / 0.339 | ana25hinfi | −0.008 / 0.314 |
  | CF one round | cf40q20o (p 40%, q 2 s) | −0.050 / 0.290 | cf20q10o | −0.062 / 0.289 |
  | CF iterated | cf40q20i | −0.052 / 0.289 | cf20q10i | −0.066 / 0.286 |

  Test rows (once): ana05hinfo, ana05hinfi, cf40q20o, cf40q20i, plus ana25hinfo (the registered "reported in any case"
  row). Each runs uncapped and @cap, then with --setting del_all. Launched on VUW1.
- 2026-10-20 (VUW 01:25): Chanakya's training-split run launched on VUW (part pub --split train --fam rt,wf):
  nuScenes new TRAIN scenes (3 shards) and AV2 REF-TRAIN every 3rd log (~113 logs, 3 shards).
- 2026-10-20 (VUW 01:25): correction before any selection.
  - Part pub's first dev / train launches used a rounded canvas: nuScenes FOVEA 960 → 544 × 960, AV2 960 → 960 × 736.
    The design says "K28 resolutions", i.e. k28_test.wf_mode: nuScenes 576 × 960, AV2 960 × 768.
  - Those runs (nuScenes dev ~40 min in, the train run just started) were stopped by exact PID, their outputs deleted
    and logs kept as *_stopped_canvas.log. Nothing was selected from them.
  - Fixed:
    - FOVEA canvases and the whole-frame rows use wf_mode;
    - AdaScale keeps its official modes (adascale.mode_for / av2_sota_labels.mode_for up to 960, wf_mode above).
  - Relaunched: pub dev nuScenes and the train run, k33_eval.py md5 03672a46.
  - The Jetson's pub candidate timing in progress used the rounded FOVEA canvases (a few % in px). The selected rows are
    re-timed with the corrected canvases (69_k33_jetson.py fixed).
- 2026-10-20 (VUW 01:31): part pub can now be restricted with --rows, and --cap adds "<row>@cap".
  - The cap is v6-S's px of the frame from the stored K28 phase-3 run.
  - Under the cap: AdaScale takes the largest ladder scale; SAHI shrinks its slice input; FOVEA its canvas; UFP its
    mosaic. Otherwise the row is cheap only.
  - md5 2c6312ca, pushed while the dev / train runs, which loaded 03672a46, were running. Part pub's code paths are
    identical without --rows / --cap.
  - F builds DONE (FOVEA warped 9,183 / 1,030; UFP mosaics 9,161 / 1,024; LZU saliency). FOVEA-ft training launched on
    VUW1, UFP-ft on VUW.

### Next steps (the hourly check follows this list)
1. **D / E test.** When logs/k33/de_test_{nusc,av2,ahd}.log all exit 0: on VUW1 run
   `k33_de_chain.sh testdel ana05hinfo,ana05hinfi,cf40q20o,cf40q20i,ana25hinfo`.
2. **pub dev AV2.** When de_test_* are done: on VUW1 run `k33_pub_chain.sh dev av2` (families ada, sahi, fov, ufp, rt;
   6 shards).
3. **Selection.** When pub dev (nusc 152, av2 300 units) and pub train (nusc, av2) are done, run
   `k33_select.py --parts de,pub` and `k33_rc.py --stage dev`. Log the selected rows here.
   - Pub test then needs `--fam <families of the selected rows>,rt,wf --rows <selected> --cap`, plus the del_all
     setting. Edit k33_pub_chain.sh test to pass --fam.
4. **F.** When logs/k33/f_train_fovea.log or f_train_ufp.log exits 0, launch `k33_f_chain.sh train_lzu` on that host.
   The F evaluation (part ft: u960-ft, v6-S-ft, FOVEA-ft, LZU-ft, UFP-ft + components; ESOD / ZoomDet re-lifted from
   the stored preds) is still to be written.
5. **Jetson.** When logs/k33j/window.log shows "K33 G PUB CANDIDATES DONE" and the selection exists: time the selected
   rows ×3 with @cap. k33_j_window.sh selected must pass `--k33 de,pub`.
- 2026-10-20 (VUW 02:31):
  - D / E test DONE (228 / 150 / 24 units, exit 0); del_all launched on VUW1 (next step 1).
  - Pub train DONE (nuScenes, AV2).
  - Pub dev AV2 launched on VUW1 (next step 2); pub dev nuScenes at 59 / 152.
  - UFP-ft trained (15 epochs in 48 min, out/k33_av2ft/ufp; final val mAP50 0.43). The log's tracebacks are NFS tmp
    cleanup at exit (Errno 16), not the training.
  - FOVEA-ft had waited 1 h for a 14 GB GPU on VUW1: its waiting wrapper (PID 10749, nothing started) was stopped and
    the training relaunched on VUW. LZU-ft is queued on VUW1.
  - Jetson: A / B candidate timing DONE (local 20:03); the board is free until the selection.
- 2026-10-20 (VUW 02:45): part ft (F rows) written in k33_eval.py. It runs on AV2 test once FOVEA-ft and LZU-ft are
  trained.
  - Rows: u960-ft, v6-S-ft, FOVEA-ft, LZU-ft, UFP-ft, ESOD, ZoomDet. Each runs + C1 + v3a, and in its official form
    ("_off": DEEP mono lift, no refiner) where the method has one.
  - ESOD / ZoomDet boxes come from their stored AV2 val prediction files (out/{esod,zoomdet}/av2_val/labels, 11,781
    files for ESOD).
- 2026-10-20 (VUW 03:30):
  - D / E del_all DONE (3 / 3, exit 0).
  - FOVEA-ft trained on VUW (exit 0; the tracebacks are the NFS tmp cleanup).
  - LZU-ft had waited 30 min for a 14 GB GPU on VUW1: its waiting wrapper (nothing started) was stopped and the
    training relaunched on VUW.
  - Pub dev: nuScenes 121 / 152, AV2 21 / 300 (about 8 h left at 6 shards).
- 2026-10-20 (local 21:31): Jetson: timing of the selected D / E rows started (ana05hinfo, ana05hinfi, cf40q20o, cf40q20i,
  ana25hinfo, plus each @cap; 3 repetitions; nusc_test20 and av2; tag k33selde_r<rep>).
- 2026-10-20 (VUW 05:26): the first part ft smoke failed: the trained LZU-ft checkpoint already contains the zoom /
  unzoom modules, so the second lzu_yolo.patch asserted. Fixed (no second patch; GRID_SHAPE set to 48 × 36 before
  loading); the smoke is re-queued.
  - Selected D / E rows on the Jetson: 3 repetitions DONE. Summarise crashed on the "@cap@trt32" names; it was fixed
    with rsplit and latency.csv rebuilt from the saved frames.pkl (no re-timing).
- 2026-10-20 (VUW 06:12): part ft smoke OK (AV2 test, 1 log, 79 frames; every F row and its official form present;
  ESOD / ZoomDet predictions found for every frame).
  - Part ft test launched on VUW (3 shards; once).
  - On this one log the official forms (DEEP mono lift, no refiner) are far below cheap (G −1.3 to −1.8). To be checked
    on the full run against K28's AV2 ft numbers before reporting.
- 2026-10-20 (VUW 07:15): part ft test DONE (AV2 val, 150 logs / 11,818 frames). Sanity check against earlier runs:
  - The fine-tuned u960 with the official lift costs the same as in the K28-era run: ΣJ 13,691 vs 13,856.
  - The G change from +0.40 (K28 era) to −0.47 is entirely the denominator. The ego-masked protocol lowers the cheap
    draft's ΣJ from 23,174 to 9,294 (the mask removes its hood phantoms), and the fine-tuned detector's cost does not
    change.
  - The F rows over-brake (HARD on KEEP frames):
    - plain: 16.9% fine-tuned vs 8.1% COCO u960;
    - + C1 + v3a: 10.3% vs 2.5%.
  - Likely mechanism, to be quantified in the report: the fine-tuned detectors learned DEEP's projected-cuboid boxes,
    which are taller and wider than COCO's tight boxes. The mono lift and refiner were set on tight boxes, so ranges come
    out short.
  - No code issue found (per-frame parity with the K28-era run).
- 2026-10-20 (VUW 13:13): pub dev DONE (nuScenes val 152, AV2 CFG-DEV 300).
  - Selection by the registered rule (k33_select.py --parts de,pub; dev_selection.csv):

    | family | selected | mean dev G | official / K28 form | its mean dev G |
    |---|---|---|---|---|
    | AdaScale | ada+1t1280 | 0.296 | ada+0t960 | 0.262 |
    | SAHI | sahi640o1s10 | 0.318 | sahi640o2s10 | 0.311 |
    | FOVEA KDE_I | fovi1280b64a2 | 0.297 | fovi960b64a1 | 0.267 |
    | FOVEA KDE_S | fovs1280b64a1 | 0.305 | fovs960b64a1 | 0.233 |
    | UFP | ufp20s512 | 0.030 | ufp15s1333 | −0.793 |

  - Routers / Chanakya selection running on VUW (k33_rc.py --stage dev; logs/k33/rc_dev.log).
  - Pub test launched once with the selected rows, --cap and the rt / wf families: nuScenes + Argoverse-HD on VUW,
    AV2 on VUW1. The del_all variant follows.
  - Jetson: the selected rows with canvases above 1024 px (ada top 1280, fov 1280) need the static engines;
    736 × 1280 and 1280 × 992 exist; others (e.g. 640 × 1120) are to be built at the end of the window.
- 2026-10-20 (VUW 13:20): k33_rc.py's first launch failed at once (the dict of tables was modified while iterating). It
  was fixed and relaunched before any output; no selection was made.
- 2026-10-20 (VUW 14:10): routers / Chanakya dev selection DONE (k33_rc.py; dev_candidates_rc.csv, dev_selection_rc.csv).
  - BAR (skip_offbin): bar1280_0.5 (strong 1280, 50% routed): dev G 0.175 / 0.173 at px 0.37 / 0.23.
  - Decision-value router (cond_V): dvr1280_0.5: dev G 0.307 / 0.184 at px 0.37 / 0.23.
  - Chanakya: the highest mean dev G is ch128B_reinforce (0.287 / 0.313), but its λ bisection ended at dev px 0.094 on
    nuScenes, +17% over v6-S's 0.080.
    - The registration requires the policy's dev px to equal v6-S's. Interpretation fixed now, before any Chanakya test
      number: a candidate counts as matched within ±5% on both splits.
    - Selected under it: **ch128B_csc** (hidden 128, features B, cost-sensitive learner): dev G 0.222 / 0.311 at px
      0.079 / 0.056. Actions: cheap 79% / u640 21% on nuScenes, 71% / 29% on AV2.
    - ch128B_reinforce and ch512B_reinforce (+9%) are reported as unmatched references.
- 2026-10-20 (VUW 15:13): pub test nuScenes (228) and AV2 (150) DONE; Argoverse-HD at 19 / 24. Pub del_all for nuScenes
  and AV2 launched on VUW1 (same rows, families rt / wf included); Argoverse-HD del_all follows its test.
- 2026-10-20 (VUW 16:13): pub test DONE on all three splits (228 / 150 / 24). del_all: nuScenes DONE, AV2 116 / 150,
  Argoverse-HD launched on VUW1.
- 2026-10-20 (VUW 16:46): routers / Chanakya test stage written (k33_rc.py --stage test [--setting del_all]) and run on the
  VUW CPU (exit 0; results/k33/rc/test_{nusc,av2,ahd}.pkl, per frame; no test number read).
  - The first run (16:44) used dev_selection_rc.csv as written by stage dev. That file still held the idxmax Chanakya
    pick (ch128B_reinforce), not the ±5% px-matched rule fixed at 14:10. The selection is now code (select() in
    k33_rc.py, used by both stages); dev_selection_rc.csv was rewritten (Chanakya = ch128B_csc) and the stage re-run. The
    wrong run's log is logs/k33/rc_test_wrongsel.log; its outputs were overwritten.
  - Rows: bar1280_0.5, dvr1280_0.5, ch128B_csc, + ch128B_reinforce (unmatched reference), each also @cap. Argoverse-HD
    takes AV2's thresholds / policy.
  - @cap rule for C: a frame whose action exceeds v6-S's px of the frame takes the largest whole frame within it
    (routers: any of 640 / 768 / 960 / 1280 up to the router's s; Chanakya: its own actions), else cheap only.
  - del_all: queued on VUW (logs/k33/rc_testdel.log), after Argoverse-HD's pub del_all (5 / 24 at 16:55).
- 2026-10-20 (local 10:38–): Jetson window (~/.jetson_pause; DEEP informed, extended to about 13:30 +07):
  - k33_j_window2.sh: the selected A / B rows ×3 (sahi640o1s10, fovi1280b64a2, fovs1280b64a1, ufp20s512; tag
    k33selpub_r<rep>);
  - then engines/build_k33.sh (yolov8s 640×1120 and 1120×864, static, fp16, workspace 1024, one at a time);
  - then ada+1t1280 ×3 (tag k33selada_r<rep>).
  - k33_j_window3.sh follows: the C rows on the device (69_k33_jetson.py: BAR = bgt-ada MobileNetV2-lite at 128 px,
    DVR = cond_V evaluated from its JSON by vtf.k33pub.xgb_json, Chanakya = ch128B_csc MLP), 30-frame smoke, then ×3
    (tag k33selrt_r<rep>).
    - xgb_json parity vs xgboost 3.4.1: max |Δ| 1.4e-6 (nuScenes) / 1.9e-6 (AV2) on 1,193 / 2,355 dev frames, and the
      same against the stored s_dvr.
    - The Chanakya checkpoints were re-saved without numpy-2 pickles (*_t.pt) for the Jetson's torch.
  - Pub @cap rows are not timed on the device (not implemented there); their px is at most the uncapped row's.
- 2026-10-20 (VUW 17:02): **spurious braking was not stored by the test runs.** The registered metric (table1_design.md:
  "spurious braking detections per 1,000 frames (K30 V7)") needs every output box's demand and its GT match, and the de /
  pub records only kept costs, found arrays and the decisive object.
  - k33_eval.py now stores spur_<row>|<ref> per frame (and spur_cheap): boxes with estimated demand ≥ thr_decel that
    match no GT box of any class at IoU ≥ 0.3, as K30 V7. On AV2 the ego-mask band gives those boxes no demand, which is
    the same as K30's removal. No other change.
  - The registered test rows are re-run once with --tag spur (k33_spur_chain.sh de | pub; 3 datasets in sequence, 3
    shards each). k33_report.py requires every cost, action and px of the re-run to equal the first run's.
  - Smoke (nuScenes scene-0110, 40 frames): all 59 J / a / px columns of part de are identical to the first run.
  - de launched on VUW (logs/k33/spur_de.log); pub follows after its smoke.
  - k33_report.py (tables: table1_v7.csv, missed_natural.csv, missed_deletion.csv, ft_rows.csv) and
    scripts/k33_latency.py (Jetson: results/k33j_latency.csv) are written; they run when the inputs are complete.
- 2026-10-20 (VUW 17:06): pub smoke: all 76 J / a / px / router-score columns identical to the first run; pub spur
  launched on VUW1 (logs/k33/spur_pub.log).
- 2026-10-20 (local 11:07–12:10) Jetson:
  - A / B selected rows ×3 DONE (11:07).
  - Engines yolov8s 640×1120 and 1120×864 built (11:23 / 11:42, rc 0).
  - ada+1t1280 ×3 DONE (11:53).
  - The C-row smoke failed at import: bgt-ada's features.py uses PEP 585 annotations, which Python 3.8 rejects. The
    window released the pause at 11:54.
    - Fixed by compiling the module with postponed annotations.
    - Pause re-taken at 12:06; DEEP informed.
  - The relaunch's smoke showed BAR scores far from VUW's (up to 28 apart). Cause: the cv2 INTER_AREA resize; the
    estimator's own transform is a PIL resize. Rep 1 was stopped by PID; its partial run dir is renamed *_stopped.
    - With the PIL transform: max |score − VUW s_bar| 0.33 on the 25 smoke frames.
    - DVR scores equal VUW's on 68% of the frames. The rest differ through the draft: VUW's nuScenes draft is the
      cached torch one, the Jetson's is TensorRT online.
  - C rows ×3 running (k33_j_window3.sh, tag k33selrt_r<rep>).
- 2026-10-20 (local 12:26): **Jetson G DONE.** The C rows ran ×3 (rc 0), the pause was removed and DEEP was messaged.
  - scripts/k33_latency.py → results/k33j_latency.csv (also in VUW results/k33/tables/).
  - Every selected row has 3 repetitions on nusc_test20 and av2. Pub @cap rows are not timed (see 16:46).
- 2026-10-20 (VUW 18:30–18:41): Argoverse-HD pub del_all DONE; rc del_all DONE (results/k33/rc/test_*_del_all.pkl);
  spur re-run part de DONE (3 datasets, exit 0). The spur re-run part pub is on AV2.
- Next: when logs/k33/spur_pub.log exits 0, run k33_report.py --latency results/k33/tables/k33j_latency.csv (reporting
  step), then write REPORT.md + table1_v7.csv + figure.
- 2026-10-20 (VUW ~20:50): k33_report.py run (reporting step; on the user's request to see the results). The spur re-run
  has nuScenes and AV2; its costs equal the first run's (the parity assertion passed). Argoverse-HD spur is still running,
  so those columns are empty.
  - Tables: results/k33/tables/{table1_v7,missed_natural,missed_deletion,ft_rows,ft_mechanism}.csv.
  - It is re-run, and the REPORT written, once the Argoverse-HD spur part is done.
- 2026-10-20 (VUW 23:40): **K33 DONE (tables and report).**
  - spur pub finished at 22:33. k33_report.py re-run on all three splits; the spur re-run's costs equal the first run's
    on every split.
  - REPORT.md, table1_v7.csv, missed_natural.csv, missed_deletion.csv, ft_rows.csv, ft_mechanism.csv, k33j_latency.csv
    and fig_table1_v7.pdf / .png are in this folder. The paper kit (03_results.md, 06_pending.md) is updated.
  - Mechanism checks for the not-better rows (AV2 AnA / u1280; nuScenes SAHI / u1280) were computed from the test
    records in the reporting step: REPORT §3.
