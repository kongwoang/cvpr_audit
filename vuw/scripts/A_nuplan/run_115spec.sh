#!/usr/bin/env bash
# usage: run_115spec.sh <planner> <phase> <n_shards>; waits for the branch spec, runs shards in parallel, then summarises
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env_nuplan.sh
export HOME=$B/.home OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $B/code/risk-aware-perception
P=$1; PH=$2; N=$3
until [ -f data/cache/nuplan_real_spec/branches.pkl ]; do sleep 30; done
for i in $(seq 0 $((N-1))); do
  python $B/scripts/A_nuplan/nuplan_spec_counterfactual.py --planner $P --phase $PH --shard $i/$N > $B/logs/nuplan_115_${P}_${PH}_$i.log 2>&1 &
done
wait
python $B/scripts/A_nuplan/nuplan_spec_counterfactual.py --planner $P --phase $PH --shard 0/1 > $B/logs/nuplan_115_${P}_${PH}_summary.log 2>&1
