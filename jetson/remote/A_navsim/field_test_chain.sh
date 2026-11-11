#!/usr/bin/env bash
# PDM-Closed probing field for navtest, computed ahead of the dev decision (CPU otherwise idle); used only if the
# spec_pdm variant wins on the DEV split.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_navsim/field_test
mkdir -p $L
. $B/scripts/common/env_navsim.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NAVSIM_DETS=$B/results/navsim_exp/dets_s05 NAVSIM_FIELD=$B/results/navsim_exp/field_test
cd $B/code/risk-aware-perception
for i in $(seq 0 11); do python $B/scripts/A_navsim/navsim_field.py --shard $i/12 > $L/field_$i.log 2>&1 & done
wait
echo "$(date +%T) FIELD TEST DONE $(ls $NAVSIM_FIELD | wc -l) logs" >> $L/done.log
