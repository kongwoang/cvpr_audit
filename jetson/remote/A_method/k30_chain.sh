#!/usr/bin/env bash
# K30 VUW GPU runs of k30_eval.py (relaunch 2026-10-15 23:40 with MIN_FREE_MB 3500: one group uses ~2.5 GB; the
# groups already running, match_nusc_g0 and idm_nusc_g0, are skipped via SKIP) on the K28 test units (post-hoc), grouped 3 shard processes per run_when_gpu_free.sh call
# (SHARE=1, MIN_FREE_MB 10000), in priority order: match (V1a / V1c / V2 / V3), cons (V5b / V5c), idm (V2b), axis (V1b).
# AV2 runs through k10_masked_run.py in the shadow tree (ego mask, K10 / K28 protocol). Finished units are skipped.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/k30; mkdir -p $L
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
cmd() {   # $1 dataset, rest: k30_eval.py arguments
  local ds=$1; shift
  if [ "$ds" = av2 ]; then
    echo "$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k30_eval.py --dataset av2 $*"
  else
    echo "$T B=$B python $B/scripts/A_method/k30_eval.py --dataset $ds $*"
  fi
}
grp() { log=$1; shift; case " ${SKIP:-} " in *" $(basename $log .log) "*) return;; esac
        SHARE=1 MIN_FREE_MB=${MIN:-3500} setsid nohup $B/run_when_gpu_free.sh $log bash -c "$*" > /dev/null 2>&1 < /dev/null & sleep 10; }
three() {  # $1 part, $2 dataset, $3 shards (multiple of 3)
  local p=$1 ds=$2 n=$3 a=() i
  for ((i = 0; i < n; i++)); do a+=("$(cmd $ds --part $p --shard $i/$n)"); done
  for ((g = 0; g < n / 3; g++)); do
    grp $L/${p}_${ds}_g$g.log "${a[3*g]} & ${a[3*g+1]} & ${a[3*g+2]} & wait"
  done
}
for ds in nusc av2 ahd; do three match $ds 6; done
for ds in nusc av2 ahd; do three cons $ds 6; done
for ds in nusc av2 ahd; do three idm $ds 3; done
for ds in nusc av2 ahd; do three axis $ds 3; done
echo "$(date +%T) K30 CHAIN LAUNCHED"
