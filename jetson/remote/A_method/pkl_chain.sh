#!/usr/bin/env bash
# DEEP's nuScenes planning cells (PKL planner vs real trajectory) for all dumped pipelines; 4 processes at a time.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
mkdir -p logs/A_method/pkl
jobs=()
for V in oracle mono; do
  for c in 0 1; do jobs+=("results/nusc_full/scenes__old_train+val+test_dump.pkl $V 2 $c"); done
  for c in 0 1 2 3 4 5; do jobs+=("results/nusc_full/scenes__new_test_dump.pkl $V 6 $c"); done
done
printf '%s\n' "${jobs[@]}" | xargs -P 4 -L 1 sh -c \
  'OMP_NUM_THREADS=2 python scripts/A_method/pkl_cells.py --dump "$0" --variant "$1" --nchunks "$2" --chunk "$3" > logs/A_method/pkl/$(basename $0 .pkl)_$1_$3.log 2>&1'
python scripts/A_method/pkl_summary.py > logs/A_method/pkl/summary.log 2>&1
echo "$(date +%T) PKL DONE" >> logs/A_method/pkl/summary.log
