#!/usr/bin/env bash
# K39-5 (CPU): k39_5_navsim.py --stage run on 8 shards (nice), then --stage summary.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/k39; mkdir -p $L
. $B/scripts/common/env_navsim.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 B=$B
cd $B/code/risk-aware-perception
for i in $(seq 0 7); do nice -n 10 python $B/scripts/A_navsim/k39_5_navsim.py --stage run --shard $i/8 > $L/k5_run_$i.log 2>&1 & done
wait
python $B/scripts/A_navsim/k39_5_navsim.py --stage summary --out $B/results/k39/k5/tables > $L/k5_summary.log 2>&1
echo "$(date +%T) K39-5 CHAIN DONE rc=$?" >> $L/k5_summary.log
