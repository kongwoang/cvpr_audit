#!/usr/bin/env bash
# K20 (CPU): planner-call cost on the 1,000-token subset, interacting pairs, draft plans for the corridor cells.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_navsim/k20; mkdir -p $L
. $B/scripts/common/env_navsim.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
cd $B/code/risk-aware-perception
for mode in cost pairs base; do
  for i in $(seq 0 11); do nice -n 5 python $B/scripts/A_navsim/k20_pdm.py --mode $mode --shard $i/12 > $L/${mode}_$i.log 2>&1 & done
  wait
  echo "$(date +%T) $mode done" >> $L/chain.log
done
echo "$(date +%T) K20 CPU CHAIN DONE" >> $L/chain.log
