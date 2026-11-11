#!/usr/bin/env bash
# Detection metrics v2: + relevant precision (single-frame lift) for every method, + ESOD and UFP-ft.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/det_metrics2.log
$B/run_when_gpu_free.sh $L python scripts/A_method/det_metrics.py --step 4 --tag _v2
echo "DET METRICS2 DONE" >> $L
