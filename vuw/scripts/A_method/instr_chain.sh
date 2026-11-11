#!/usr/bin/env bash
# Instrumented re-runs for K2/K3/K4/K9 (instr_eval.py), at most MAXP shard processes at a time (GPU shared with CARLA).
# Order: nuScenes new val (dev: K4 debug, K9 development) -> nuScenes new test -> AV2 val (post-hoc on test splits).
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/instr; mkdir -p $L
MAXP=${MAXP:-6}; NS=6
running() { local n=0; for p in /proc/[0-9]*; do grep -qs "instr_eval.py" $p/cmdline 2>/dev/null && n=$((n+1)); done; echo $n; }
JOBS="nusc:val:base:0 nusc:val:del_all:0 nusc:test:base:50 nusc:test:del_all:0 nusc:test:del_half:0 av2:test:base:0 av2:test:del_all:0 av2:test:del_half:0"
for j in $JOBS; do
  IFS=: read ds sp st dg <<< "$j"
  for i in $(seq 0 $((NS-1))); do
    while [ "$(running)" -ge $((MAXP * 2)) ]; do sleep 120; done        # wrapper + python per shard
    SHARE=1 MIN_FREE_MB=4500 $B/run_when_gpu_free.sh $L/${ds}_${sp}_${st}_$i.log python scripts/A_method/instr_eval.py \
      --dataset $ds --split $sp --setting $st --shard $i/$NS --diag_units $dg &
    sleep 150
  done
done
wait
echo "$(date +%T) INSTR CHAIN DONE" >> $L/chain.log
