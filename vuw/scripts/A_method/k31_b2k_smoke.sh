#!/usr/bin/env bash
source /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR python /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/A_method/k31_eval.py --dataset nusc --part b2dev --split dev --max_units 3 --tag smoke || echo "FAIL nusc"
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR/k10_masked K10_MASK_Y0=1830 python /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/A_method/k10_masked_run.py /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/k10_masked/scripts/A_method/k31_eval.py --dataset av2 --part b2dev --split dev --max_units 2 --tag smoke || echo "FAIL av2"
echo "K31 B2K SMOKE DONE"
