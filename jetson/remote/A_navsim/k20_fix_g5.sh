#!/usr/bin/env bash
# K20(3) fix-up: the g5 detection shard 1 was requeued (GPU-pick starvation fix) after k20_chain3 had already started
# the g5 PDMS on shard 0's logs. When the requeued shard has written its logs and chain3 is done, run the g5 PDMS for the
# missing logs (navsim_pdms skips logs already scored) and redo the g5 summary.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_navsim/k20; X=$B/results/navsim_exp; S=$B/scripts/A_navsim
until grep -qs "exit" $L/detect_g5_1.log && grep -qs "K20 CHAIN3 DONE" $L/chain.log; do sleep 300; done
. $B/scripts/common/env_navsim.sh
cd $B/code/risk-aware-perception
export SPEC_CROP_SCALE=0.5 NAVSIM_FIELD=$X/field_k20_g5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  NAVSIM_DETS=$X/dets_k20_g5 NAVSIM_PDMS_OUT=$X/pdms_k20_g5 NAVSIM_TAG=_k20_g5 NAVSIM_MAIN=spec_pdm NAVSIM_BRANCHES=spec_pdm
for i in $(seq 0 11); do nice -n 5 python $S/navsim_pdms.py --stage run --shard $i/12 > $L/pdms_g5_fix_$i.log 2>&1 & done
wait
python $S/navsim_pdms.py --stage summary > $L/pdms_g5_summary.log 2>&1
echo "$(date +%T) g5 fixed ($(ls $X/dets_k20_g5/*.pkl | wc -l) det logs, $(ls $X/pdms_k20_g5/*.pkl | wc -l) pdms logs)" >> $L/chain.log
