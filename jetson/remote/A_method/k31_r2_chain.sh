#!/usr/bin/env bash
# K31 R2 (post-hoc re-runs of k31_eval.py, pre-registered in handoff/K31/STATUS.md): parts ttcpx (R2a) and idmS (R2b) on the
# K28 test units, 6 shards per dataset and part, grouped 3 shard processes per run_when_gpu_free.sh call (SHARE=1,
# MIN_FREE_MB 3500). AV2 through k10_masked_run.py (ego mask). Finished units are skipped.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/k31; mkdir -p $L
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
cmd() {
  local ds=$1; shift
  if [ "$ds" = av2 ]; then
    echo "$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k31_eval.py --dataset av2 $*"
  else
    echo "$T B=$B python $B/scripts/A_method/k31_eval.py --dataset $ds $*"
  fi
}
grp() { log=$1; shift; case " ${SKIP:-} " in *" $(basename $log .log) "*) return;; esac
        SHARE=1 MIN_FREE_MB=${MIN:-3500} setsid nohup $B/run_when_gpu_free.sh $log bash -c "$*" > /dev/null 2>&1 < /dev/null & sleep 10; }
three() {
  local p=$1 ds=$2 n=$3 a=() i
  for ((i = 0; i < n; i++)); do a+=("$(cmd $ds --part $p --shard $i/$n)"); done
  for ((g = 0; g < n / 3; g++)); do
    grp $L/r2_${p}_${ds}_g$g.log "${a[3*g]} & ${a[3*g+1]} & ${a[3*g+2]} & wait"
  done
}
for ds in nusc av2 ahd; do three ttcpx $ds 6; done
for ds in nusc av2 ahd; do three idmS $ds 6; done
echo "$(date +%T) K31 R2 CHAIN LAUNCHED"
