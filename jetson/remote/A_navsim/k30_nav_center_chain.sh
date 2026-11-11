#!/usr/bin/env bash
# K30 V6 (i): PDM-Closed route centerlines of every navtest token (CPU, 12 shards)
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/scripts/common/env_navsim.sh
cd $B/code/risk-aware-perception
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NAVSIM_EXP_ROOT=$B/results/navsim_exp
for i in $(seq 0 11); do CUDA_VISIBLE_DEVICES="" python $B/scripts/A_navsim/k30_nav_hybrid.py --stage center --shard $i/12 > $B/logs/k30/nav_center_$i.log 2>&1 & done
wait
echo "$(date +%T) K30 NAV CENTER DONE" >> $B/logs/k30/nav_center_done.log
