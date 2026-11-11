#!/usr/bin/env bash
# K31 item 1 panels: capture run of k28_p3.py (--part instr --capture, tag k31cap) on the units of the 30 top onset delays,
# then k28_render.py into results/k31/panels/ (AV2 with the ego mask).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
C=$B/results/k31/meta/capture_frames.csv
$T B=$B python $B/scripts/A_method/k28_p3.py --dataset nusc --split test --part instr --capture $C --tag k31cap &
$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k28_p3.py --dataset av2 --split test --part instr --capture $C --tag k31cap &
$T B=$B python $B/scripts/A_method/k28_p3.py --dataset ahd --split test --part instr --capture $C --tag k31cap &
wait
mkdir -p $B/results/k31/panels
for ds in nusc av2 ahd; do
  awk -F, -v d=$ds 'NR == 1 || $1 == d' $C > $B/results/k31/meta/capture_$ds.csv
  if [ $ds = av2 ]; then
    $T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k28_render.py $B/results/k31/meta/capture_$ds.csv $B/results/k31/panels $B/results/k28/p3/${ds}_test_instr_k31cap
  else
    $T B=$B python $B/scripts/A_method/k28_render.py $B/results/k31/meta/capture_$ds.csv $B/results/k31/panels $B/results/k28/p3/${ds}_test_instr_k31cap
  fi
done
echo "K31 CAPTURE AND PANELS DONE"
