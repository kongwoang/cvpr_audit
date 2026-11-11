#!/usr/bin/env bash
# K30 V6 (ii): measured probing runtime per grid (k30_probe_time.py), 12 single-thread CPU processes (as K20), after the
# centerline stage
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/k30
until grep -qs "K30 NAV CENTER DONE" $L/nav_center_done.log; do sleep 120; done
. $B/scripts/common/env_navsim.sh
cd $B/code/risk-aware-perception
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for i in $(seq 0 11); do CUDA_VISIBLE_DEVICES="" python $B/scripts/A_navsim/k30_probe_time.py --shard $i/12 > $L/probe_time_$i.log 2>&1 & done
wait
echo "$(date +%T) K30 PROBE TIME DONE" >> $L/probe_time_done.log
