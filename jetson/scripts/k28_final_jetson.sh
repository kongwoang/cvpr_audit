#!/usr/bin/env bash
# K28 final Jetson window (post-hoc; v6 frozen in §2n). One process at a time, ~/.jetson_pause held by the caller, DEEP
# informed at start and release. Part 1: engines (engines/build_k28_final.sh). Part 2: scripts/61_k28_v6_latency.py
#   smoke (1 nuScenes scene) -> nuScenes / AV2 (ego mask 1830) v6-S / M / L (+ bounded) with parity against the phase-2
#   per-frame results (results/k28_ref, copied from VUW) on the units of the phase-1 B timing (25) -> KITTI test with the
#   shared configuration -> Argoverse-HD per-frame timing of v6-S for the streaming-AP simulation (VUW).
set -u
cd "$(dirname "$0")/.."
REF=results/k28_ref
log() { echo "$(date '+%F %T') $*"; }
R() { log "start $*"; scripts/py scripts/61_k28_v6_latency.py "$@" 2>&1 | tail -n 40; local rc=${PIPESTATUS[0]}; log "end rc=$rc"; return $rc; }
log "builds"; bash engines/build_k28_final.sh || { log "BUILD FAILED"; exit 1; }
R --dataset nusc --config v6S --max_units 1 --max_frames 20 --wf u448 --bounds 2 --ref_dir $REF/nusc_test_v6ours --tag smoke || { log "SMOKE FAILED"; exit 1; }
R --dataset nusc --config v6S --max_units 25 --wf u448,u512,u640,u960,u1280,u1536,native --bounds 2,3 --ref_dir $REF/nusc_test_v6ours --tag final
R --dataset nusc --config v6M --max_units 25 --wf_det yolov8m --wf u448,u640 --bounds 2 --ref_dir $REF/nusc_test_v6ours --tag final
R --dataset nusc --config v6L --max_units 25 --wf_det yolo11m --wf u448,u640 --bounds 2 --ref_dir $REF/nusc_test_v6ours --tag final
R --dataset nusc --config v51S --max_units 25 --wf none --bounds 2 --ref_dir $REF/nusc_test_v6ours --tag final
R --dataset av2 --config v6S --max_units 25 --mask_y0 1830 --wf u512,u640,u960,u1280,u1536,native --bounds 2,3 --ref_dir $REF/av2_test_v6ours --tag final
R --dataset av2 --config v6M --max_units 25 --mask_y0 1830 --wf none --bounds 2 --ref_dir $REF/av2_test_v6ours --tag final
R --dataset av2 --config v6L --max_units 25 --mask_y0 1830 --wf none --bounds 2 --ref_dir $REF/av2_test_v6ours --tag final
R --dataset av2 --config v51S --max_units 25 --mask_y0 1830 --wf none --bounds 2 --ref_dir $REF/av2_test_v6ours --tag final
R --dataset kitti --config v6S --bounds 2 --tag final
R --dataset ahd --config v6S --wf none --bounds "" --tag final
log "K28 FINAL JETSON DONE"
