#!/usr/bin/env bash
# Baseline (4): learned frame routers (bgt-ada, official) -- labels (GPU) -> ORIC -> train (GPU) -> score.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/router.log
S=scripts/A_method/router_baseline.py
ok() { tail -3 $L | grep -q "ROUTER $1 DONE"; }
$B/run_when_gpu_free.sh $L python $S --stage labels; ok LABELS || { echo "ROUTER ABORT labels" >> $L; exit 1; }
python $S --stage oric >> $L 2>&1; ok ORIC || { echo "ROUTER ABORT oric" >> $L; exit 1; }
$B/run_when_gpu_free.sh $L python $S --stage train; ok TRAIN || { echo "ROUTER ABORT train" >> $L; exit 1; }
python $S --stage score >> $L 2>&1; ok SCORE || { echo "ROUTER ABORT score" >> $L; exit 1; }
echo "ROUTER CHAIN DONE" >> $L
