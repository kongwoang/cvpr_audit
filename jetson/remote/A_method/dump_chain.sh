#!/usr/bin/env bash
# Scene dumps of frozen v3 (+ refiner) for DEEP's PKL cells: old 85 scenes (DEEP's units) and the new test scenes.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
G=$B/run_when_gpu_free.sh
for SET in "old train+val+test" "new test"; do
  set -- $SET
  $G logs/A_method/dump_$1.log python scripts/A_method/nusc_ablate.py --set $1 --split $2 --device cuda:0 --threads 4 \
    --scale 0.6 --dump --tag _dump --refiners $B/out/resnet18/image_refiner.pt
done
echo "$(date +%T) DUMPS DONE" >> logs/A_method/dump_new.log
