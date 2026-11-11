#!/usr/bin/env bash
# P02 -> PDMS of every joint effect (2 CPU shards) -> summary CSVs, on VUW after the 4 P02 shards finish.
# (no set -u: env.sh references unset variables)
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
cd $B; source env.sh >/dev/null 2>&1; source $B/scripts/common/env_navsim.sh >/dev/null 2>&1
export B=$B CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2; cd $B
until [ "$(ls $B/results/k41/p02/*.pkl 2>/dev/null | wc -l)" = 125 ] && ! pgrep -f k41_p02_interactions >/dev/null; do sleep 120; done
echo "$(date +%T) P02 complete; scoring joint effects"
python scripts/A_navsim/k41_p02_score.py --stage score --shard 0/2 > logs/k41/P02/score_0.log 2>&1 &
python scripts/A_navsim/k41_p02_score.py --stage score --shard 1/2 > logs/k41/P02/score_1.log 2>&1 &
wait
python scripts/A_navsim/k41_p02_score.py --stage summary --out_dir results/k41/p02_summary
echo "$(date +%T) P02 score chain done"
