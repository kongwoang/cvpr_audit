#!/usr/bin/env bash
# P03 (user-authorized 2026-11-02 beyond the cost check): P02 interaction diagnostics on the remaining 11,146 navtest
# tokens, 24 CPU shards (VUW 0-11, VUW1 12-23), fresh PDM-Closed per token. usage: bash k41_p03_launch.sh <first> <last>
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
cd $B && . $B/env.sh >/dev/null 2>&1; . $B/scripts/common/env_navsim.sh >/dev/null 2>&1
export B=$B CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; cd $B; mkdir -p logs/k41/P03
for i in $(seq $1 $2); do
  python scripts/A_navsim/k41_p02_interactions.py --subset remaining --fresh_per_token --shard $i/24 --out results/k41/p03 > logs/k41/P03/shard_$i.log 2>&1 &
done
wait; echo "$(date +%T) shards $1-$2 finished"
