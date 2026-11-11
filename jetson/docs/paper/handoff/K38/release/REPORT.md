# K38 (code release): anonymous release of the frozen DESP v6-S

- **Repository:** https://github.com/anonymous221206/DESP (public).
- **Commit:** `1c0bb32e471ab80bf37d70272ca0c90627d662ca`. This is a single commit, "Initial release", on `main`.
- **Release:** https://github.com/anonymous221206/DESP/releases/tag/v1.0, with three assets:

  | asset | bytes | md5 |
  |---|---|---|
  | looming_c1.pt | 46,375,042 | 7bfdecec7d291a5dfe05de6426c3084e |
  | refiner_v3a.pt | 45,325,698 | 6b29d05e8f06d121ec9e11aeb9dc2730 |
  | MD5SUMS | 97 | |

  - yolov8s.pt is not re-hosted. `scripts/download_weights.sh` fetches the official Ultralytics asset (v8.2.0, md5
    0a1d5d06…) and checks all three md5 sums.
- **Built from:** a fresh export directory (kit /tmp/desp_release, and the copy CVPR/tmp/desp_release on VUW) followed by
  a clean git directory (kit /tmp/desp_release_git, created with `git init`).
- **Not modified:** the private repositories, the DEEP repository and the historical handoffs.

## Results of the gates

| gate | check | result |
|---|---|---|
| C | unit tests, fresh env from requirements.txt (CVPR/tmp/desp_env) | 102 / 102 passed (test_core.py) |
| C | `--help` of every script | rc 0 for every entry point; training/common.py and kitti.py are modules, not scripts |
| C | `reproduce_main.sh --subset`, fresh env | rc 0, about 9 min |
| C | per-frame identity on the subset, fresh env | **1,060 / 1,060 frames identical, 0 differing** (see the subset table below) |
| C | `pytest tests` (unit + regression), fresh env | 105 passed |
| C | training-pipeline smoke (1–2 units, 1 epoch) | rc 0; both retrained checkpoints load in `desp` |
| D | anonymity grep, file names and contents | every hit reviewed by hand; all benign (see below) |
| D | media, notebooks | none included; nothing to strip |
| D | git identity | repo-local only; author and committer `Anonymous <anonymous221206@users.noreply.github.com>`; global config untouched |
| E | remote empty before the push | `git ls-remote`: 0 refs, so a normal (non-force) push |
| E | push credential | gh account anonymous221206 (`gh api user` → anonymous221206); HTTPS with a repo-local credential helper; SSH disabled for the push |
| E | public clone → fresh env → weights from the Release → `pytest tests` | **105 passed; 1,060 / 1,060 frames identical** |

**Subset compared.** The subset is:
- nuScenes new test, the first 2 scenes in sorted order (scene-0110 and scene-0121, 80 frames);
- AV2 val, the first log in sorted order, with the ego mask (02678d04…, 79 frames);
- Argoverse-HD val, seq00 (901 frames).

**Fields compared per frame:**
- DESP: action, no-refiner action, draft action, rounds, number of crops, number of guard-grown crops, every crop and
  grown-crop rectangle (exact), and px (exact).
- u960 + C1 + v3a: action, no-refiner action and px.

**Reference outputs.** `tests/regression/expected.pkl` (md5 e85c1a35dcdd03c3c9e7e5f3a9182430) is the private frozen
pipeline's capture on the subset.
- It was cross-checked against the full-split reference records, with 0 differences.
- The nuScenes reference is the live-draft run (user decision: the online PyTorch YOLOv8s draft). AV2 and Argoverse-HD
  are checked against the K28 records.
- An earlier smoke run in the workspace venv against the same references also gave 0 differences.

**Clone environment.** The fresh environment for the clone was CVPR/tmp/desp_clone_env, built with pip from the clone's
requirements.txt (torch 2.6.0+cu124, ultralytics 8.4.163, numpy 2.5.3). All weights were downloaded by
`scripts/download_weights.sh`, and every md5 printed OK.

**Full-split G** (taken from the reference records, which the release reproduces exactly on the subset):

| split | v6-S (paper) | v6-S (this release) | u960 + C1 + v3a (release) |
|---|---|---|---|
| nuScenes new test, 228 scenes | 0.211 | 0.178 | 0.077 (paper: 0.112) |
| AV2 val with the ego mask, 150 logs | 0.294 | 0.294 | 0.338 |
| Argoverse-HD val, 24 sequences | 0.418 | 0.418 | 0.297 |

The nuScenes difference comes only from the draft:
- The paper used DEEP's cached draft detections, which came from TensorRT FP16. They cannot be reproduced in PyTorch:
  boxes differ by up to 1.7 px and confidences by up to 0.13.
- The release runs the draft online. Both rows move together, and DESP stays ahead of the whole-frame reference
  (+0.10 vs +0.10 in the paper).

The README states this next to the paper value.

## Anonymity gate (D)

**Search.** `grep -rniE` over the export, contents and file names, for these terms:
- names and contact details: kongwoang, hoang, pham, cong, luong, conglc, gmail, `@`, longnd, nguyen;
- institution and machines: vuw, wellington, students, `.nz`, ecs., jetson, xavier;
- paths and access: /home/, /Users/, /vol/, /tmp/, ssh, an IPv4 regex;
- internal project terms: `\bK[0-9]+\b`, `\bR[0-9]\b`, DESP_internal, `rap\b`, deep_ext, DEEP;
- other: github.com, anthropic, claude.

**Hits, each reviewed by hand:**
- `@`: matrix multiplication and Python decorators (`@functools.lru_cache`, `@pytest...`).
- `rap\b`: the word "bootstrap" (README:54, evaluate.py:3).
- `ecs\.`: `recs.append` (run_desp.py:60).
- IPv4 regex: `opencv-python==5.0.0.93`.
- DEEP: cited only as "arXiv:2609.35910", in README:10 and desp/consumers/__init__.py. There is no repository URL.
- jetson: only as the device name in time_jetson.py, which the task requires, and in export_tensorrt.py and README.
- github.com: only the release's own download URL, the official Ultralytics asset URL and s5cmd's homepage.
- No names, institutions, hosts, home or machine paths, usernames, internal task IDs or `rap` package remain.

**Other checks:**
- expected.pkl has no path strings.
- The checkpoints contain only state dicts plus arch, head and feature statistics. Their internal archive folders are
  "temporal_head" and "image_refiner".
- No notebooks, images or PDFs are included.
- The `.git` directory is new (`git init`); no history from a private repo is included.
- LICENSE is MIT, "Anonymous Authors"; the README notes that ultralytics is AGPL-3.0. CITATION.cff says "Anonymous,
  under review".
- **Commit message:** "Initial release", with no Co-Authored-By trailer. An AI co-author trailer would identify the
  tooling, and the task fixes both the message and the identity.

## Fixes made during the gates (training scripts only; no inference code changed)
- `training/extract_refiner_data.py`:
  - it raises if a KITTI image directory is missing, where it used to give 0 samples silently;
  - it keeps the array rank when no samples are found;
  - it has a new `--units` option for smoke runs.
  - Cause: VUW holds KITTI images only for sequences 0007–0009 and 0017–0019, so the smoke used 0007.
- `training/train_looming.py`: with an empty hold-out split (smoke runs only), it saves the last epoch. It used to save
  an empty state.
- README: the nuScenes live-draft G was filled in, and the compute notes now give the measured subset time (about 9 min
  on one RTX A5000) in place of estimates.

## Files released (64; `git ls-files`)

| file | bytes | md5 (first 8) |
|---|---|---|
| .gitignore | 95 | a6e339f8 |
| CITATION.cff | 296 | 6d2ac4ee |
| LICENSE | 1074 | cf2ff85b |
| README.md | 6173 | 33a14096 |
| configs/v6s.yaml | 1448 | 73fddac4 |
| configs/v6s_bounded.yaml | 1517 | e50fa8e5 |
| configs/wholeframe.yaml | 680 | 0b34075f |
| data/README.md | 1650 | 8f64be4e |
| data/splits/argoverse_hd_val.txt | 1032 | 999e456d |
| data/splits/av2_cfg_dev.txt | 11100 | 20a10481 |
| data/splits/av2_ego_mask.yaml | 110 | f193008b |
| data/splits/av2_ref_train.txt | 12580 | fe04ed26 |
| data/splits/av2_ref_val.txt | 2220 | be632875 |
| data/splits/av2_val.txt | 5550 | 1d24be35 |
| data/splits/nuscenes_test.txt | 2508 | 1235976b |
| data/splits/nuscenes_train.txt | 4235 | 65fcd894 |
| data/splits/nuscenes_val.txt | 1672 | 81f07411 |
| data/splits/refiner_kitti.txt | 217 | fd9f5754 |
| data/splits/refiner_nuscenes.txt | 9781 | 8e1cc204 |
| desp/__init__.py | 285 | 14c63800 |
| desp/config.py | 2313 | 9af1ce7f |
| desp/consumers/__init__.py | 710 | 8aa058b7 |
| desp/consumers/braking.py | 5617 | cf0571d7 |
| desp/consumers/probing.py | 1758 | 37da8391 |
| desp/consumers/ttc_warning.py | 2258 | 9670d82d |
| desp/crops.py | 769 | 9e454b7a |
| desp/datasets/__init__.py | 219 | fbb77ddd |
| desp/datasets/argoverse_hd.py | 7600 | deff37f8 |
| desp/datasets/av2.py | 6483 | 2e9f7980 |
| desp/datasets/common.py | 3050 | 9016b208 |
| desp/datasets/nuscenes.py | 6904 | 077e34bd |
| desp/datasets/registry.py | 1831 | 80e9482d |
| desp/detection.py | 6597 | 2aad9edd |
| desp/evaluation/__init__.py | 120 | fb067b25 |
| desp/evaluation/metrics.py | 5699 | 57ebd1ec |
| desp/fast/__init__.py | 113 | 81bfcdc7 |
| desp/fast/looming_trt.py | 3657 | 16863405 |
| desp/field.py | 3676 | 7b2d49f4 |
| desp/geometry.py | 6487 | ba05b842 |
| desp/looming.py | 4684 | 5f0d0a83 |
| desp/loop.py | 8565 | 5dec1264 |
| desp/pipeline.py | 5683 | be59b5ae |
| desp/refiner.py | 3197 | a4181401 |
| requirements.txt | 449 | 220fd095 |
| scripts/download_data.sh | 3492 | 0ce6909c |
| scripts/download_weights.sh | 1070 | ef2a3d9d |
| scripts/evaluate.py | 2497 | 6731068e |
| scripts/export_tensorrt.py | 1900 | 7908a9ee |
| scripts/prepare_data.py | 2391 | ab6aff48 |
| scripts/reproduce_main.sh | 1116 | ff3dad4c |
| scripts/run_desp.py | 3615 | e49155bc |
| scripts/run_wholeframe.py | 2856 | e8f5fac0 |
| scripts/time_jetson.py | 3015 | 9315cb41 |
| tests/conftest.py | 98 | 15f6010d |
| tests/regression/expected.pkl | 276640 | e85c1a35 |
| tests/test_core.py | 6005 | 1a3d0c1d |
| tests/test_regression.py | 3174 | 055ec3ef |
| training/__init__.py | 97 | 725d206b |
| training/common.py | 2398 | 274f442f |
| training/extract_looming_data.py | 8755 | 75735614 |
| training/extract_refiner_data.py | 9667 | dded2f49 |
| training/kitti.py | 3683 | 552931d9 |
| training/train_looming.py | 4767 | b2a0b223 |
| training/train_refiner.py | 5066 | e5be6c6d |

**Excluded:**
- paper sources, handoffs, results, logs, caches;
- K/R experiment scripts, launch scripts, machine paths;
- engines, data, weights (these go in the Release only);
- published baselines, heuristics, CARLA / K32, NAVSIM / nuPlan harnesses. Only the generic probing consumer is kept,
  as `desp/consumers/probing.py`.

**Private helpers** (not released): remote/A_method/release_ref_chain.sh, remote/ops/release_{smoke,gate,train_smoke,clone_check}.sh.
Logs are on VUW under CVPR/logs/release/: gate_*.log, train_smoke*.log, clone_*.log.
