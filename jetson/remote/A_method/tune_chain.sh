#!/usr/bin/env bash
# Tuning grid on the NEW VAL scenes (152) with both refiners; GPU runs through the wrapper.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
mkdir -p logs/A_method/tune
REF="$B/out/resnet18/image_refiner.pt $B/out/resnet18_v2/image_refiner.pt"
for cfg in "0 3 0.6" "5 3 0.6" "10 1 0.6" "0 1 0.6" "10 3 0.8" "0 3 0.8"; do
  set -- $cfg
  $B/run_when_gpu_free.sh logs/A_method/tune/vc$1_r$2_s$3.log python scripts/A_method/nusc_ablate.py --set new --split val \
    --device cuda:0 --threads 4 --v_close $1 --max_rounds $2 --scale $3 --tag _tune_vc$1_r$2_s$3 --refiners $REF
done
echo "$(date +%T) TUNE DONE" >> logs/A_method/tune/done.log
