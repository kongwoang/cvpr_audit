#!/usr/bin/env bash
# K28 PHASE 3 (post-hoc, reporting only; v6 frozen in §2n) — VUW runs of k28_p3.py on the test units, started only after
# the phase-2 analysis (test/done.log "K28 V6 ANALYZE DONE"):
#   instr          base and del_all, nuScenes / AV2 (ego mask) / Argoverse-HD: K2, K4, K9, K16, K17 of v6-S
#   nomask         AV2 val WITHOUT the ego mask: ours (v5.1-S, v6-S / M / L), YOLOv8s whole frame, the 26 heuristics
#   y11s           YOLO11s as draft and verifier, no re-tuning (v3a, C1 as frozen): ours S, whole frame, 9 S heuristics
#   ft0 / ft1      nuScenes, fine-tuned YOLOv8s verifier seeds 0 / 1 (out/fovea_ft/plain[_s1]), COCO draft: same rows
# then k28_p3_analyze.py and k28_p3_p0.py (phase 0's items for v6).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k28/p3; mkdir -p $L
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
declare -A WF=([nusc]=448,512,640,960,1280,1536,native [av2]=512,640,960,1280,1536,native [ahd]=448,512,640,960,1280,1536,native)
CFG=unc2@S,unc4@S,tcr0@S,tcr3@S,cor0@S,cor8@S,tsp2@S,tsp8@S,rnd@S,unc2@M,unc8@M,tcr0@M,tcr3@M,cor0@M,cor8@M,tsp4@M,tsp8@M,rnd@M,unc2@L,unc6@L,tcr3@L,cor0@L,cor8@L,tsp0@L,tsp8@L,rnd@L
CFGS=unc2@S,unc4@S,tcr0@S,tcr3@S,cor0@S,cor8@S,tsp2@S,tsp8@S,rnd@S
declare -A NUNITS=([nusc]=228 [av2]=150 [ahd]=24)
FT0=out/fovea_ft/plain/weights/best.pt
FT1=out/fovea_ft/plain_s1/weights/best.pt
cmd() {   # $1 dataset, $2 mask (1 = AV2 ego mask), rest: k28_p3.py arguments
  local ds=$1 mk=$2; shift 2
  if [ "$ds" = av2 ] && [ "$mk" = 1 ]; then
    echo "$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k28_p3.py --dataset av2 --split test $*"
  else
    echo "$T B=$B python $B/scripts/A_method/k28_p3.py --dataset $ds --split test $*"
  fi
}
grp() { log=$1; shift; SHARE=1 MIN_FREE_MB=${MIN:-9000} $B/run_when_gpu_free.sh $log bash -c "$*" & sleep 10; }
three() {  # $1 log name, $2 dataset, $3 mask, $4 shards (3 or 6), rest: part arguments
  local nm=$1 ds=$2 mk=$3 n=$4; shift 4
  local a=() i
  for ((i = 0; i < n; i++)); do a+=("$(cmd $ds $mk "$@" --shard $i/$n)"); done
  for ((g = 0; g < n / 3; g++)); do
    grp $L/${nm}_${ds}_g$g$SFX.log "${a[3*g]} & ${a[3*g+1]} & ${a[3*g+2]} & wait"
  done
}
launch() {
  three instr nusc 1 6 --part instr; three instr av2 1 6 --part instr; three instr ahd 1 3 --part instr
  for ds in nusc av2 ahd; do three del $ds 1 3 --part instr --setting del_all; done
  three nomask_ours av2 0 3 --part ours --tag nomask
  three nomask_wf av2 0 3 --part basewf --det '""' --wf "${WF[av2]}" --tag nomask
  three nomask_heur av2 0 6 --part baseheur --configs $CFG --tag nomask
  for ds in nusc av2 ahd; do
    three y11s_ours $ds 1 3 --part ours --only_s --ver yolo11s --draft yolo11s --tag y11s
    three y11s_wf $ds 1 3 --part basewf --det '""' --wf "${WF[$ds]}" --ver yolo11s --draft yolo11s --tag y11s
    three y11s_heur $ds 1 3 --part baseheur --configs $CFGS --ver yolo11s --draft yolo11s --tag y11s
  done
  for k in 0 1; do
    w=FT$k
    three ft${k}_ours nusc 1 3 --part ours --only_s --ver ${!w} --tag ft$k
    three ft${k}_wf nusc 1 3 --part basewf --det '""' --wf "${WF[nusc]}" --ver ${!w} --tag ft$k
    three ft${k}_heur nusc 1 3 --part baseheur --configs $CFGS --ver ${!w} --tag ft$k
  done
  wait
}
n_of() { ls $B/results/k28/p3/$1 2>/dev/null | wc -l; }
complete() {
  for ds in nusc av2 ahd; do
    for p in instr instr-del_all ours_y11s basewf_y11s baseheur_y11s; do [ "$(n_of ${ds}_test_$p)" -ge ${NUNITS[$ds]} ] || return 1; done
  done
  for p in ours_nomask basewf_nomask baseheur_nomask; do [ "$(n_of av2_test_$p)" -ge 150 ] || return 1; done
  for p in ours basewf baseheur; do for k in 0 1; do [ "$(n_of nusc_test_${p}_ft$k)" -ge 228 ] || return 1; done; done
}
until grep -q "K28 V6 ANALYZE DONE" $B/logs/A_method/k28/test/done.log 2>/dev/null; do sleep 900; done
echo "$(date +%T) phase-3 runs start" >> $L/chain.log
SFX="" launch
for i in 1 2; do          # a shard process that ended without an error is re-run (finished units are skipped)
  complete && break
  echo "$(date +%T) incomplete; re-run $i" >> $L/chain.log
  SFX="_rerun$i" launch
done
complete && echo "$(date +%T) K28 P3 RUNS DONE" > $L/done.log || { echo "$(date +%T) K28 P3 RUNS INCOMPLETE" > $L/done.log; exit 1; }
cd $B && $T B=$B python $B/scripts/A_method/k28_p3_analyze.py > $L/analyze.log 2>&1
cd $B && $T B=$B python $B/scripts/A_method/k28_p3_p0.py --part all > $L/p0v6.log 2>&1
echo "$(date +%T) K28 P3 ANALYZE DONE" >> $L/done.log
