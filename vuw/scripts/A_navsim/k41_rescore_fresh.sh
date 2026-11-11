#!/usr/bin/env bash
# rerun D07 score (4 CPU shards) and P02 joint-effect score (2) with a fresh PDM-Closed instance per token, then summary
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
cd $B; source env.sh >/dev/null 2>&1; source $B/scripts/common/env_navsim.sh >/dev/null 2>&1
export B=$B CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; cd $B
for i in 0 1 2 3; do python scripts/A_navsim/k41_d07_k2.py --stage score --shard $i/4 > logs/k41/D07/score_fresh_$i.log 2>&1 & done; wait
for i in 0 1; do python scripts/A_navsim/k41_p02_score.py --stage score --shard $i/2 > logs/k41/P02/score_fresh_$i.log 2>&1 & done; wait
python scripts/A_navsim/k41_p02_score.py --stage summary --out_dir results/k41/p02_summary
echo "$(date +%T) rescore done: D07 $(ls results/k41/d07/score/*.pkl | wc -l) logs"
