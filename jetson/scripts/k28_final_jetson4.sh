#!/usr/bin/env bash
# K28 final Jetson window, part 4 (after the second hard reset at 04:35:45 +07 during the nuScenes v6-S run with six
# whole-frame engines, memory 92%): fan at 100% (user, 2026-10-14), MAXN as before, and the whole-frame engines split over two
# processes per dataset (fewer engines in memory). ours (+ bounded) is timed in both halves; the first half is the reported
# ours row. One process at a time; ~/.jetson_pause held.
set -u
cd "$(dirname "$0")/.."
REF=results/k28_ref
log() { echo "$(date '+%F %T') $*"; }
R() { log "start $*"; scripts/py scripts/61_k28_v6_latency.py "$@" 2>&1 | tail -n 40; local rc=${PIPESTATUS[0]}; log "end rc=$rc"; return $rc; }
R --dataset nusc --config v6S --max_units 25 --wf u448,u512,u640 --bounds 2,3 --tag final_a
R --dataset nusc --config v6S --max_units 25 --wf u960,u1280,native --bounds "" --tag final_b
R --dataset nusc --config v6M --max_units 25 --wf_det yolov8m --wf u448,u640 --bounds 2 --tag final
R --dataset nusc --config v6L --max_units 25 --wf_det yolo11m --wf u448,u640 --bounds 2 --tag final
R --dataset nusc --config v51S --max_units 25 --wf none --bounds 2 --tag final
R --dataset av2 --config v6S --max_units 25 --mask_y0 1830 --wf u512,u640,u960 --bounds 2,3 --ref_dir $REF/av2_test_v6ours --tag final_a
R --dataset av2 --config v6S --max_units 25 --mask_y0 1830 --wf u1280,u1536,native --bounds "" --ref_dir $REF/av2_test_v6ours --tag final_b
R --dataset av2 --config v6M --max_units 25 --mask_y0 1830 --wf none --bounds 2 --ref_dir $REF/av2_test_v6ours --tag final
R --dataset av2 --config v6L --max_units 25 --mask_y0 1830 --wf none --bounds 2 --ref_dir $REF/av2_test_v6ours --tag final
R --dataset av2 --config v51S --max_units 25 --mask_y0 1830 --wf none --bounds 2 --ref_dir $REF/av2_test_v6ours --tag final
R --dataset kitti --config v6S --bounds 2 --tag final
R --dataset ahd --config v6S --wf none --bounds "" --tag final
log "K28 FINAL JETSON DONE"
