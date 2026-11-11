#!/usr/bin/env bash
# K33 D / E (k33_eval.py part de; registered in handoff/K33). $1 = smoke | dev | test [ROWS], one run_when_gpu_free.sh group
# (SHARE=1, MIN_FREE_MB 3500) per dataset; AV2 through k10_masked_run.py (ego mask). Finished units are skipped.
#   smoke: 1 nuScenes val scene + 1 AV2 CFG-DEV log, every candidate (tag smoke)
#   dev nusc | dev av2: nuScenes new val 3 shards / AV2 CFG-DEV 6 shards (2 groups), every candidate (OMP 2)
#   test:  the three test splits, 3 shards each, rows $2 with --cap, then --setting del_all (rows $2)
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/k33; mkdir -p $L
T="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 B_REAL=$B"
cmd() {
  local ds=$1; shift
  if [ "$ds" = av2 ]; then
    echo "$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k33_eval.py --dataset av2 --part de $*"
  else
    echo "$T B=$B python $B/scripts/A_method/k33_eval.py --dataset $ds --part de $*"
  fi
}
grp() { log=$1; shift; SHARE=1 MIN_FREE_MB=${MIN:-3500} setsid nohup $B/run_when_gpu_free.sh $log bash -c "$*" > /dev/null 2>&1 < /dev/null & sleep 10; }
case $1 in
  smoke)
    grp $L/de_smoke.log "$(cmd nusc --split dev --max_units 1 --tag smoke) && $(cmd av2 --split dev --max_units 1 --tag smoke)" ;;
  dev)   # $2 = nusc | av2 (one dataset per host)
    if [ "${2:-nusc}" = nusc ]; then
    a=(); for i in 0 1 2; do a+=("$(cmd nusc --split dev --shard $i/3)"); done
    grp $L/de_dev_nusc.log "${a[0]} & ${a[1]} & ${a[2]} & wait"
    else
    for g in 0 1; do
      a=(); for i in 0 1 2; do a+=("$(cmd av2 --split dev --shard $((3 * g + i))/6)"); done
      grp $L/de_dev_av2_g$g.log "${a[0]} & ${a[1]} & ${a[2]} & wait"
    done
    fi ;;
  test)      # $2 rows; one dataset at a time (sequential groups in one wrapper chain)
    for ds in nusc av2 ahd; do
      a=(); for i in 0 1 2; do a+=("$(cmd $ds --split test --rows $2 --cap --shard $i/3)"); done
      grp $L/de_test_$ds.log "${a[0]} & ${a[1]} & ${a[2]} & wait"
    done ;;
  testdel)
    for ds in nusc av2 ahd; do
      a=(); for i in 0 1 2; do a+=("$(cmd $ds --split test --rows $2 --setting del_all --shard $i/3)"); done
      grp $L/de_testdel_$ds.log "${a[0]} & ${a[1]} & ${a[2]} & wait"
    done ;;
esac
echo "$(date +%T) K33 DE $1 launched"
