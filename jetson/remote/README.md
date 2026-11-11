# VUW workspace layout (`/vol/dl-nguyenb5-solar/users/hoangpc/CVPR`, mirrored locally in `remote/`)

Rule: everything stays inside this directory; `source scripts/common/env.sh` (or the `env.sh` symlink) first — it
activates the venv and points pip / torch / CUDA / ultralytics / matplotlib caches, HOME-like dirs and TMPDIR inside
the workspace. GPU jobs only through `run_when_gpu_free.sh <log> <cmd…>`.

| path | content |
|---|---|
| `scripts/common/` | env.sh, env_nuplan.sh, GPU wrapper, refiner model/trainer, crop helper, layout scripts |
| `scripts/data/` | downloads (nuScenes, nuPlan, AV2, LiDAR extraction), detection caches, DEEP tables/splits |
| `scripts/A_method/` | paper A: evaluation of the loop (confirmatory, ablations, diagnostics, baselines, Planner B) |
| `scripts/A_nuplan/` | paper A on nuPlan: branch building (proxy / real crops), DEEP 115 planner runs, cells |
| `scripts/A_ext/` | paper A extensions and killed directions (LiDAR budget C, attacks D, gate E, D68, D75) |
| `scripts/B_churn/` | paper B: detector-upgrade churn (D5), congruent upgrades (D25, D25b, training) |
| `scripts/old/` | superseded scripts kept for provenance |
| `code/CVPR`, `code/risk-aware-perception` | code copies (vtf package; DEEP read-only copy with caches under data/cache) |
| `datasets/` | nuscenes (trainval, CAM_FRONT + LIDAR_TOP keyframes), nuplan (mini, maps, CAM_F0), av2 (sensor val) |
| `models/` | detector weights (COCO) |
| `data/` | derived training sets (refiner data, d68, d75, b_cong) |
| `out/` | trained weights (`resnet18/` = FROZEN refiner of the confirmatory run; `*_v2` etc. = later) |
| `results/` | result tables; `results/nusc_full/` = split, DEEP tables, evals (confirmatory: `*_v3_confirm*`) |
| `logs/` | job logs (flat for now; phase 2 moves them into topic folders) |
| `archive/` | upload bundles |
| `venv/`, `envs/nuplan`, `miniforge3/`, `.cache/`, `.home/`, `tmp/` | environments and redirected caches |

Transitional: the old root-level script names are symlinks into `scripts/` while earlier chains still run (phase 1,
2026-09-28). Phase 2 (after those jobs end): remove the symlinks, move logs/results into topic folders. New scripts are
run by their `scripts/<topic>/` path; shared modules import from `scripts/common` (on PYTHONPATH via env.sh).

Local workflow: edit `remote/<topic>/<file>`, then `remote/push.sh <topic>/<file>` (add `--src` after changing
`src/vtf`). Confirmatory code is frozen: `results/frozen_v3/` locally and md5s in `logs/confirm_chain.log`.
