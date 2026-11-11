#!/usr/bin/env bash
# K28 final Jetson window, part 3 (after the board rebooted at ~01:40 +07 during the YOLOv8s 864x1536 TensorRT build, the
# third reboot from a large static engine build; that engine is NOT retried, so nuScenes whole frame u1536 is not timed).
# Steps of parts 1-2 not yet done: own-refiner engines (checkpoints re-saved without numpy-2 objects; originals kept as
# .orig.pt), smoke, then every latency run. One process at a time; ~/.jetson_pause held; DEEP informed.
set -u
cd "$(dirname "$0")/.."
REF=results/k28_ref
log() { echo "$(date '+%F %T') $*"; }
cd engines
for t in own_yolov8m_vuw own_yolo11m_vuw; do
  [ -f refiner_${t}.orig.pt ] || mv refiner_${t}.pt refiner_${t}.orig.pt
  cp refiner_${t}_np1.pt refiner_${t}.pt
  [ -f refiner_${t}_dyn.engine ] || { rm -f refiner_${t}_dyn.onnx; bash build_refiner_v3.sh $t; }
  [ -f refiner_${t}_dyn.engine ] && [ -f refiner_${t}_norm.npz ] || { log "REFINER BUILD FAILED $t"; exit 1; }
done
cd ..
R() { log "start $*"; scripts/py scripts/61_k28_v6_latency.py "$@" 2>&1 | tail -n 40; local rc=${PIPESTATUS[0]}; log "end rc=$rc"; return $rc; }
R --dataset nusc --config v6S --max_units 1 --max_frames 25 --warmup 5 --wf u448 --bounds 2 --tag smoke || { log "SMOKE FAILED"; exit 1; }
R --dataset nusc --config v6M --max_units 1 --max_frames 25 --warmup 5 --wf_det yolov8m --wf u448 --bounds 2 --tag smoke || { log "SMOKE M FAILED"; exit 1; }
R --dataset nusc --config v6S --max_units 25 --wf u448,u512,u640,u960,u1280,native --bounds 2,3 --ref_dir $REF/nusc_test_v6ours --tag final
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
