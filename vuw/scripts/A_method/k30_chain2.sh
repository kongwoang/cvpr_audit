#!/usr/bin/env bash
# K30 VUW GPU runs of k30_eval2.py (post-hoc): apdump (V7 / V8e) on the three test splits, speed (V8d) on nuScenes; groups of 3
# shard processes per run_when_gpu_free.sh call (SHARE=1, MIN_FREE_MB 3500); AV2 through k10_masked_run.py
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/k30
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
cmd() { local ds=$1; shift
  if [ "$ds" = av2 ]; then echo "$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k30_eval2.py --dataset av2 $*"
  else echo "$T B=$B python $B/scripts/A_method/k30_eval2.py --dataset $ds $*"; fi; }
grp() { log=$1; shift; SHARE=1 MIN_FREE_MB=3500 setsid nohup $B/run_when_gpu_free.sh $log bash -c "$*" > /dev/null 2>&1 < /dev/null & sleep 10; }
for ds in nusc av2 ahd; do grp $L/apdump_${ds}_g0.log "$(cmd $ds --part apdump --shard 0/3) & $(cmd $ds --part apdump --shard 1/3) & $(cmd $ds --part apdump --shard 2/3) & wait"; done
grp $L/speed_nusc_g0.log "$(cmd nusc --part speed --shard 0/3) & $(cmd nusc --part speed --shard 1/3) & $(cmd nusc --part speed --shard 2/3) & wait"
echo "$(date +%T) K30 CHAIN2 LAUNCHED"
