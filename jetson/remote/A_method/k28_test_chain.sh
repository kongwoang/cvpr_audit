#!/usr/bin/env bash
# K28 PHASE 2 — the ONE confirmatory test pass of v6 (method_freeze_draft §2n; K24 protocol, suffix _v6): nuScenes new test
# (228 scenes), AV2 val (150 logs, ego mask through k10_masked_run.py), Argoverse-HD val (24 sequences). Parts per dataset:
#   ours                                   v5.1-S + v6-S / v6-M / v6-L
#   basewf  --det yolo11m --wf <WF_ds>     YOLOv8s whole frame at the §2n sides + YOLO11m at the B sides (plain / + T4 / + C1)
#   basewf  --det yolov8m --wf ""          YOLOv8m whole frame at the B sides (plain / + T4 / + C1)
#   baseheur --configs <CFG>               YOLOv8s heuristics, E dev-best and K24 forms (plain / + T4 / + C1)
#   baseheur --hdet yolo11m|yolov8m --configs <CFG11>   heuristics at the S budget with the v6-L / v6-M crop detectors
# then k28_analyze_test.py (refuses to write tables until every part has every unit).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k28/test; mkdir -p $L
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
declare -A WF=([nusc]=448,512,640,960,1280,1536,native [av2]=512,640,960,1280,1536,native [ahd]=448,512,640,960,1280,1536,native)
CFG=unc2@S,unc4@S,tcr0@S,tcr3@S,cor0@S,cor8@S,tsp2@S,tsp8@S,rnd@S,unc2@M,unc8@M,tcr0@M,tcr3@M,cor0@M,cor8@M,tsp4@M,tsp8@M,rnd@M,unc2@L,unc6@L,tcr3@L,cor0@L,cor8@L,tsp0@L,tsp8@L,rnd@L
CFG11=unc2@S,unc4@S,tcr0@S,tcr3@S,cor0@S,cor8@S,tsp2@S,tsp8@S,rnd@S
NS=(nusc av2 ahd)
declare -A NUNITS=([nusc]=228 [av2]=150 [ahd]=24)
cmd() {   # $1 dataset, rest: k28_test.py arguments
  local ds=$1; shift
  if [ "$ds" = av2 ]; then
    echo "$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k28_test.py --dataset av2 --split test $*"
  else
    echo "$T B=$B python $B/scripts/A_method/k28_test.py --dataset $ds --split test $*"
  fi
}
grp() { log=$1; shift; SHARE=1 MIN_FREE_MB=${MIN:-9000} $B/run_when_gpu_free.sh $log bash -c "$*" & sleep 10; }
three() {  # $1 log name, $2 dataset, $3 shards per process group (3 or 6), rest: part arguments (quoted)
  local nm=$1 ds=$2 n=$3; shift 3
  local a=() i
  for ((i = 0; i < n; i++)); do a+=("$(cmd $ds "$@" --shard $i/$n)"); done
  for ((g = 0; g < n / 3; g++)); do
    grp $L/${nm}_${ds}_g$g$SFX.log "${a[3*g]} & ${a[3*g+1]} & ${a[3*g+2]} & wait"
  done
}
launch() {   # the longest parts first
  for ds in av2 nusc ahd; do three heur $ds 6 --part baseheur --configs $CFG; done
  for ds in av2 nusc ahd; do three ours $ds 3 --part ours; done
  for ds in av2 nusc ahd; do
    three heur11 $ds 3 --part baseheur --hdet yolo11m --configs $CFG11
    three heur8m $ds 3 --part baseheur --hdet yolov8m --configs $CFG11
  done
  for ds in av2 nusc ahd; do
    three wf11 $ds 3 --part basewf --det yolo11m --wf "${WF[$ds]}"
    three wf8m $ds 3 --part basewf --det yolov8m --wf '""'
  done
  wait
}
n_of() { ls $B/results/k28/$1 2>/dev/null | wc -l; }
complete() {
  for ds in "${NS[@]}"; do
    for p in ours basewf-yolo11m basewf-yolov8m baseheur baseheur-yolo11m baseheur-yolov8m; do
      [ "$(n_of ${ds}_test_v6$p)" -ge ${NUNITS[$ds]} ] || return 1
    done
  done
}
SFX="" launch
for i in 1 2; do          # a shard process that ended without an error is re-run (finished units are skipped)
  complete && break
  echo "$(date +%T) incomplete; re-run $i" >> $L/chain.log
  SFX="_rerun$i" launch
done
complete && echo "$(date +%T) K28 V6 TEST DONE" > $L/done.log || { echo "$(date +%T) K28 V6 TEST INCOMPLETE" > $L/done.log; exit 1; }
cd $B && $T B=$B python $B/scripts/A_method/k28_analyze_test.py > $L/analyze.log 2>&1
echo "$(date +%T) K28 V6 ANALYZE DONE" >> $L/done.log
