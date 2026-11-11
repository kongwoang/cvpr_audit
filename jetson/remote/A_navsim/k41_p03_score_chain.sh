#!/usr/bin/env bash
# P03: wait until both host launchers report their shards finished, then PDMS of every joint effect (8 CPU shards) and the summary
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
cd $B && . $B/env.sh >/dev/null 2>&1; . $B/scripts/common/env_navsim.sh >/dev/null 2>&1
export B=$B CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 K41_P02_IN=p03 K41_P02_SCORE=p03_score; cd $B
until grep -q finished logs/k41/P03_launch_vuw.log 2>/dev/null && grep -q finished logs/k41/P03_launch_vuw1.log 2>/dev/null; do sleep 300; done
echo "$(date +%T) P03 complete ($(ls results/k41/p03/*.pkl | wc -l) logs); scoring joint effects"
for i in $(seq 0 7); do python scripts/A_navsim/k41_p02_score.py --stage score --shard $i/8 > logs/k41/P03/score_$i.log 2>&1 & done; wait
python scripts/A_navsim/k41_p02_score.py --stage summary --out_dir results/k41/p03_summary
echo "$(date +%T) P03 score chain done"
