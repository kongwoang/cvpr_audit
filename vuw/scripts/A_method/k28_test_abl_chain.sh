#!/usr/bin/env bash
# K28 PHASE 2, addendum §2n.1: the support only / field only ablation of v6-S (k28_test_abl.py --part abl) on the same test
# units as the main pass, then k28_analyze_test2.py (ablation 2 x 2 of v6-S, routers / oracle re-matched to v6) once the
# main pass's ours part and this part have every unit.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k28/test; mkdir -p $L
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
cmd() {
  local ds=$1; shift
  if [ "$ds" = av2 ]; then
    echo "$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k28_test_abl.py --dataset av2 --split test $*"
  else
    echo "$T B=$B python $B/scripts/A_method/k28_test_abl.py --dataset $ds --split test $*"
  fi
}
grp() { log=$1; shift; SHARE=1 MIN_FREE_MB=${MIN:-9000} $B/run_when_gpu_free.sh $log bash -c "$*" & sleep 10; }
declare -A NUNITS=([nusc]=228 [av2]=150 [ahd]=24)
launch() {
  for ds in av2 nusc ahd; do
    grp $L/abl_${ds}_g0$1.log "$(cmd $ds --part abl --shard 0/3) & $(cmd $ds --part abl --shard 1/3) & $(cmd $ds --part abl --shard 2/3) & wait"
  done
  wait
}
n_of() { ls $B/results/k28/$1 2>/dev/null | wc -l; }
complete() {
  for ds in nusc av2 ahd; do [ "$(n_of ${ds}_test_v6abl)" -ge ${NUNITS[$ds]} ] || return 1; done
}
launch ""
for i in 1 2; do
  complete && break
  echo "$(date +%T) abl incomplete; re-run $i" >> $L/abl_chain.log
  launch "_rerun$i"
done
complete && echo "$(date +%T) K28 V6 ABL DONE" > $L/abl_done.log || { echo "$(date +%T) K28 V6 ABL INCOMPLETE" > $L/abl_done.log; exit 1; }
until grep -q -e "K28 V6 TEST DONE" -e "INCOMPLETE" $L/done.log 2>/dev/null; do sleep 900; done
cd $B && $T B=$B python $B/scripts/A_method/k28_analyze_test2.py > $L/analyze2.log 2>&1
echo "$(date +%T) K28 V6 ANALYZE2 DONE" >> $L/abl_done.log
