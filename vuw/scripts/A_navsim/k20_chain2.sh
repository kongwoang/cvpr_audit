#!/usr/bin/env bash
# K20(2): exact fast implementation on the 1,000-token subset (parity on the first 100 tokens), then the calls of a
# token over N worker processes (200 tokens, N = 4, 8, 16). Starts after k20_chain.sh.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_navsim/k20
. $B/scripts/common/env_navsim.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
cd $B/code/risk-aware-perception
until grep -qs "K20 CPU CHAIN DONE" $L/chain.log; do sleep 300; done
for i in $(seq 0 11); do nice -n 5 python $B/scripts/A_navsim/k20_fast.py --mode fast --shard $i/12 > $L/fast_$i.log 2>&1 & done
wait
for w in 4 8 16; do python $B/scripts/A_navsim/k20_fast.py --mode parallel --workers $w --tokens 200 > $L/parallel_w$w.log 2>&1; done
echo "$(date +%T) K20 CHAIN2 DONE" >> $L/chain.log
