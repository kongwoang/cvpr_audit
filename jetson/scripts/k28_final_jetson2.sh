#!/usr/bin/env bash
# K28 final Jetson window, part 2 (same window, after k28_final_jetson.sh): the own refiners of v6-M / v6-L could not be
# exported in part 1 (their checkpoints hold numpy-2 arrays; the edge env has numpy 1). They were re-saved on VUW with
# fmu / fsd as float lists (state tensors and norms identical; tmp/k28_refconv) -> refiner_<tag>_np1.pt. Here: swap them
# in (originals kept as .orig.pt), export + build the dynamic refiner engines, then the v6-M / v6-L latency runs that
# failed in part 1 (nuScenes, AV2 with mask 1830). One process at a time; starts only when part 1 has finished.
set -u
cd "$(dirname "$0")/.."
REF=results/k28_ref
log() { echo "$(date '+%F %T') $*"; }
until grep -q "K28 FINAL JETSON DONE" logs/k28_final_jetson.log; do sleep 60; done
log "part 2 start"
cd engines
for t in own_yolov8m_vuw own_yolo11m_vuw; do
  [ -f refiner_${t}.orig.pt ] || mv refiner_${t}.pt refiner_${t}.orig.pt
  cp refiner_${t}_np1.pt refiner_${t}.pt
  rm -f refiner_${t}_dyn.onnx
  bash build_refiner_v3.sh $t
  [ -f refiner_${t}_dyn.engine ] && [ -f refiner_${t}_norm.npz ] || { log "REFINER BUILD FAILED $t"; exit 1; }
done
cd ..
R() { log "start $*"; scripts/py scripts/61_k28_v6_latency.py "$@" 2>&1 | tail -n 40; local rc=${PIPESTATUS[0]}; log "end rc=$rc"; return $rc; }
R --dataset nusc --config v6M --max_units 25 --wf_det yolov8m --wf u448,u640 --bounds 2 --ref_dir $REF/nusc_test_v6ours --tag final
R --dataset nusc --config v6L --max_units 25 --wf_det yolo11m --wf u448,u640 --bounds 2 --ref_dir $REF/nusc_test_v6ours --tag final
R --dataset av2 --config v6M --max_units 25 --mask_y0 1830 --wf none --bounds 2 --ref_dir $REF/av2_test_v6ours --tag final
R --dataset av2 --config v6L --max_units 25 --mask_y0 1830 --wf none --bounds 2 --ref_dir $REF/av2_test_v6ours --tag final
log "K28 FINAL JETSON PART 2 DONE"
