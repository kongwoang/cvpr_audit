#!/usr/bin/env bash
# Code release (handoff/K38/release): reference records of the frozen pipeline with the ONLINE PyTorch draft on nuScenes
# (k28_p3 --draft models/yolov8s.pt, i.e. not the TensorRT draft cache):
#   full   nuScenes test, --part instr --tag livedraft, 4 shards (two run_when_gpu_free.sh groups of two)
#   cap    the regression subset with --capture (per-frame crops): nuScenes test first 2 scenes (live draft), AV2 val first
#          log (ego mask), Argoverse-HD val first sequence; tag relcap
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/scripts/A_method/k28_p3.py
K=$B/k10_masked
L=$B/logs/release; mkdir -p $L
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
case $1 in
  full)
    r() { echo "$T B=$B python $S --dataset nusc --split test --part instr --draft models/yolov8s.pt --tag livedraft --shard $1 > $L/livedraft_${1/\//of}.log 2>&1"; }
    for g in "0/4 1/4" "2/4 3/4"; do
      set -- $g
      c="{ $(r $1) & p0=\$!; $(r $2) & p1=\$!; wait \$p0 && wait \$p1; }"
      SHARE=1 MIN_FREE_MB=${MIN:-3500} setsid nohup $B/run_when_gpu_free.sh $L/livedraft_group_${1/\//of}.log bash -c "$c" > /dev/null 2>&1 < /dev/null &
    done ;;
  cap)
    C=$B/tmp/release_capture.csv
    c="{ $T B=$B python $S --dataset nusc --split test --part instr --draft models/yolov8s.pt --capture $C --tag relcap > $L/cap_nusc.log 2>&1 & p0=\$!; \
$T B=$K K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $K/scripts/A_method/k28_p3.py --dataset av2 --split test --part instr --capture $C --tag relcap > $L/cap_av2.log 2>&1 & p1=\$!; \
$T B=$B python $S --dataset ahd --split test --part instr --capture $C --tag relcap > $L/cap_ahd.log 2>&1 & p2=\$!; wait \$p0 && wait \$p1 && wait \$p2; }"
    SHARE=1 MIN_FREE_MB=${MIN:-3500} setsid nohup $B/run_when_gpu_free.sh $L/cap_group.log bash -c "$c" > /dev/null 2>&1 < /dev/null & ;;
  *) echo "usage: $0 full|cap"; exit 1 ;;
esac
echo "$(date +%T) release reference $1 launched"
