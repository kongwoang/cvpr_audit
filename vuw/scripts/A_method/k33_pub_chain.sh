#!/usr/bin/env bash
# K33 A (k33_eval.py part pub --fam ada,sahi,fov; registered in handoff/K33). $1 = smoke | dev | test [ROWS], one run_when_gpu_free.sh group
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
    echo "$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k33_eval.py --dataset av2 --part pub $*"
  else
    echo "$T B=$B python $B/scripts/A_method/k33_eval.py --dataset $ds --part pub $*"
  fi
}
grp() { log=$1; shift; SHARE=1 MIN_FREE_MB=${MIN:-3500} setsid nohup $B/run_when_gpu_free.sh $log bash -c "$*" > /dev/null 2>&1 < /dev/null & sleep 10; }
case $1 in
  fovsal)   # KDE_S saliencies (GPU, small), before any fov run
    grp $L/pub_fovsal.log "$T B=$B python $B/scripts/A_method/k33_fovsal.py --dataset nusc && $T B=$B python $B/scripts/A_method/k33_fovsal.py --dataset av2" ;;
  smoke)
    grp $L/pub_smoke.log "$(cmd nusc --split dev --max_units 1 --tag smoke --fam ada,sahi,fov,ufp,rt,wf) && $(cmd av2 --split dev --max_units 1 --tag smoke --fam ada,sahi,fov,ufp,rt,wf)" ;;
  dev)   # $2 = nusc | av2 (one dataset per host)
    if [ "${2:-nusc}" = nusc ]; then
    a=(); for i in 0 1 2; do a+=("$(cmd nusc --split dev --fam ada,sahi,fov,ufp,rt --shard $i/3)"); done
    grp $L/pub_dev_nusc.log "${a[0]} & ${a[1]} & ${a[2]} & wait"
    else
    for g in 0 1; do
      a=(); for i in 0 1 2; do a+=("$(cmd av2 --split dev --fam ada,sahi,fov,ufp,rt --shard $((3 * g + i))/6)"); done
      grp $L/pub_dev_av2_g$g.log "${a[0]} & ${a[1]} & ${a[2]} & wait"
    done
    fi ;;
  train)    # Chanakya's training data (families rt, wf): nuScenes new TRAIN 3 shards, AV2 REF-TRAIN every 3rd log 3 shards
    a=(); for i in 0 1 2; do a+=("$(cmd nusc --split train --fam rt,wf --shard $i/3)"); done
    grp $L/pub_train_nusc.log "${a[0]} & ${a[1]} & ${a[2]} & wait"
    a=(); for i in 0 1 2; do a+=("$(cmd av2 --split train --fam rt,wf --unit_stride 3 --shard $i/3)"); done
    grp $L/pub_train_av2.log "${a[0]} & ${a[1]} & ${a[2]} & wait" ;;
  test)      # $2 rows, $3 datasets (default "nusc av2 ahd"): families of the selected rows + rt, wf (routers / Chanakya), --cap
    for ds in ${3:-nusc av2 ahd}; do
      a=(); for i in 0 1 2; do a+=("$(cmd $ds --split test --fam ada,sahi,fov,ufp,rt,wf --rows $2 --cap --shard $i/3)"); done
      grp $L/pub_test_$ds.log "${a[0]} & ${a[1]} & ${a[2]} & wait"
    done ;;
  testdel)
    for ds in ${3:-nusc av2 ahd}; do
      a=(); for i in 0 1 2; do a+=("$(cmd $ds --split test --fam ada,sahi,fov,ufp,rt,wf --rows $2 --setting del_all --shard $i/3)"); done
      grp $L/pub_testdel_$ds.log "${a[0]} & ${a[1]} & ${a[2]} & wait"
    done ;;
esac
echo "$(date +%T) K33 PUB $1 launched"
