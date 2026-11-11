#!/usr/bin/env bash
# K35-6: capture run of k28_p3.py (--part instr --capture, tag k35q6) on the units of the frames selected by k35_k6.py,
# then k28_render.py into results/k35/q6_panels/ (AV2 with the ego mask). Same pattern as k31_capture.sh.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
C=$B/results/k35/k6/capture_frames.csv
$T B=$B python $B/scripts/A_method/k28_p3.py --dataset nusc --split test --part instr --capture $C --tag k35q6 &
$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k28_p3.py --dataset av2 --split test --part instr --capture $C --tag k35q6 &
$T B=$B python $B/scripts/A_method/k28_p3.py --dataset ahd --split test --part instr --capture $C --tag k35q6 &
wait
mkdir -p $B/results/k35/q6_panels
for ds in nusc av2 ahd; do
  awk -F, -v d=$ds 'NR == 1 || $1 == d' $C > $B/results/k35/k6/capture_$ds.csv
  if [ $ds = av2 ]; then
    $T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k28_render.py $B/results/k35/k6/capture_$ds.csv $B/results/k35/q6_panels $B/results/k28/p3/${ds}_test_instr_k35q6
  else
    $T B=$B python $B/scripts/A_method/k28_render.py $B/results/k35/k6/capture_$ds.csv $B/results/k35/q6_panels $B/results/k28/p3/${ds}_test_instr_k35q6
  fi
done
echo "K35 Q6 CAPTURE AND PANELS DONE"
