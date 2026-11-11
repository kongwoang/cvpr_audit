# K35-0 — untouched labelled holdout inventory (GT only; no pipeline run)

**Verdict: PASS for candidate (b), Argoverse 1.1 3D-tracking TRAIN (65 logs).** Candidates (a) and (c) are empty
(fully touched). (d) has no usable untouched dataset. K35-4 may proceed, after the K35-4b gate and the PREREG.

- Code:
  - provenance scan: remote/A_method/k35_provenance.sh (md5 611e0b88);
  - GT-only episodes: remote/A_method/k35_k0_episodes.py (md5 3b199a44);
  - download: remote/data/download_av1_train.sh.
- Run on VUW 2026-10-25.

## Candidates

| candidate | units | status | evidence |
|---|---|---|---|
| (a) AV2 sensor train logs outside REF-TRAIN / REF-VAL / CFG-DEV / C1 / v3a / R rounds / K29 av2_dev / Jetson prefixes | 0 | **empty** | results/k23/av2_split.csv assigns all 700 downloaded AV2 train logs: CFG-DEV 300, REF-TRAIN 340, REF-VAL 60 (= 700, the full AV2 train set) |
| (b) Argoverse 1.1 tracking TRAIN | 65 | **untouched; passes** | see below |
| (c) nuScenes trainval outside every used split | 0 | **empty** | results/nusc_full/split.csv covers all 850 scenes (new train 385 / val 152 / test 228; old train 45 / val 16 / test 24). The provenance scan finds every one of the 850 scene names in workspace results / caches / logs. |
| (d) other datasets on VUW with 3D boxes + a front camera | – | **touched or unsuitable** | nuPlan cam_f0 (64 logs): used by the A_nuplan PDMS chains. NAVSIM mini: the same 64 nuPlan logs. NAVSIM navtest / navhard: used (K30, K31). KITTI tracking: 21 labelled sequences (< 30 units), used (DEEP cells, K30). CODA: 2D boxes only, used in K30 V8. |

## (b) Argoverse 1.1 TRAIN — provenance proof

- **Never on VUW before K35.** The workspace held only tracking VAL (24 logs = Argoverse-HD val, the test split). The 65
  TRAIN logs (= Argoverse-HD train.json seq_dirs, ids_av1_train.txt) were stream-extracted for K35 on 2026-10-25:
  calibration, poses, amodal 3D labels, ring_front_center and city_info only.
- **Systematic scan.** Paths and binary-safe content of 705,886 files under results, data, logs, out, archive,
  experiments (incl. every Codex optimisation round), k10_masked/results, deep_ext, repro, tmp, the DEEP repo's
  data/results, code/CVPR and scripts (dirs.txt).
  - Path hits for the 65 IDs: 0.
  - Content hits: 2 files, both in a third-party library (deep_ext/envs/pcdet/.../av2/utils/synchronization_database.py
    and its .pyc). They hard-code one log ID (c6911883-…) as a known sensor-sync exception. That is not a use.
- **No script** in scripts/ or code/ reads Argoverse-HD train.json; the zoomdet references are its own AV2 train.json.
- C1 and v3a were trained on nuScenes / AV2 only (K28 manifests). Argoverse-HD (K7 / K28) used only AV1.1 VAL.
- Kit-side grep (results, logs, experiments, docs, scripts, engines, ~/datasets): the only hit is the Argoverse-HD
  train.json annotation file itself.

## (b) Size and power, from GT only

Protocol: the frozen Argoverse-HD evaluation (k24_test 'ahd' path) applied to train.json.
- Every ring_front_center frame; a frame is evaluated iff it has 3D labels.
- Objects with projected height ≥ 10 px.
- Deployed planner thresholds (τ1 1.0, τ2 3.5 m/s²).
- K31 required episodes over the evaluated frames.

**Protocol check:** the same code on AV1.1 VAL reproduces K31 / K34 exactly: 24 units, 15,006 labelled frames,
71 required episodes (61 below hard / 10 at or above). See av1_val_summary.json.

| | units | labelled frames | required episodes | below hard / at or above | units with episodes |
|---|---|---|---|---|---|
| AV1.1 TRAIN (holdout) | 65 | 39,265 | **154** | 137 / 17 | 50 |
| AV1.1 VAL (= Argoverse-HD test, reference) | 24 | 15,006 | 71 | 61 / 10 | 19 |

Pass criteria: proof of no use ✓, ≥ 30 units (65) ✓, ≥ 100 required episodes (154) ✓.

**Caveats.**
- The holdout has the same sensor, cities and annotation process as Argoverse-HD. It is an independent set of logs, not
  an independent domain.
- AV2's selections transfer, as they did for Argoverse-HD.
- Only 17 episodes reach hard demand, so severity-stratified holdout results will be imprecise.

## Files
av1_train_units.csv, av1_train_episodes.csv, av1_train_summary.json, av1_val_summary.json, ids_av1_train.txt, dirs.txt.
The scan outputs (paths.txt, path_hits.tsv, content_hits.tsv) are on VUW in results/k35/k0/.
