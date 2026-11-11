#!/usr/bin/env bash
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
OMP_NUM_THREADS=2 python scripts/A_method/pkl_cells.py --dump results/nusc_full/scenes__new_test_dump.pkl --variant oracle \
  --nchunks 6 --chunk 2 > logs/A_method/pkl/scenes__new_test_dump_oracle_2.log 2>&1
python scripts/A_method/pkl_summary.py > logs/A_method/pkl/summary.log 2>&1
echo "$(date +%T) PKL REDO DONE" >> logs/A_method/pkl/summary.log
