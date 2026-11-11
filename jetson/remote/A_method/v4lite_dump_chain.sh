#!/usr/bin/env bash
# Merged-scene dump of v4-lite (v_close 0, 3 rounds, crop scale 0.6, refiner v2) on new test, then detection metrics of
# every dumped pipeline (v3 and v4-lite) with det_metrics_dump.py.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/v4lite_dump.log
$B/run_when_gpu_free.sh $L python scripts/A_method/nusc_ablate.py --set new --split test --device cuda:0 --threads 4 \
  --v_close 0 --max_rounds 3 --scale 0.6 --tag _v4lite_dump --dump --refiners $B/out/resnet18_v2/image_refiner.pt
[ -f results/nusc_full/scenes__new_test_v4lite_dump.pkl ] || { echo "V4LITE DUMP ABORT" >> $L; exit 1; }
python scripts/A_method/det_metrics_dump.py results/nusc_full/scenes__new_test_dump.pkl \
  results/nusc_full/scenes__new_test_v4lite_dump.pkl >> $L 2>&1
echo "V4LITE DUMP CHAIN DONE" >> $L
