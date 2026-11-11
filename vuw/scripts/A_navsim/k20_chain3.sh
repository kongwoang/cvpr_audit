#!/usr/bin/env bash
# K20(3): field approximations (g5, g3, corridor of the plan) -> spec_pdm detections (GPU) -> PDMS on navtest (CPU).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_navsim/k20; X=$B/results/navsim_exp; S=$B/scripts/A_navsim
until grep -qs "base done" $L/chain.log; do sleep 300; done
. $B/scripts/common/env_navsim.sh
cd $B/code/risk-aware-perception
CUDA_VISIBLE_DEVICES= python $S/k20_variants.py > $L/variants.log 2>&1
for v in g5 g3 corridor; do
  C="export SPEC_CROP_SCALE=0.5 NAVSIM_FIELD=$X/field_k20_$v"
  E=". $B/scripts/common/env_navsim.sh; cd $B/code/risk-aware-perception; $C;"
  for i in 0 1; do
    SHARE=1 MIN_FREE_MB=6000 $B/run_when_gpu_free.sh $L/detect_${v}_$i.log bash -c "$E NAVSIM_REDO=spec_pdm NAVSIM_SRC_DETS=$X/dets_final NAVSIM_DETS=$X/dets_k20_$v python $S/navsim_detect.py --shard $i/2" &
    sleep 150
  done
  wait
  ( eval "$C"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= NAVSIM_DETS=$X/dets_k20_$v NAVSIM_PDMS_OUT=$X/pdms_k20_$v \
      NAVSIM_TAG=_k20_$v NAVSIM_MAIN=spec_pdm NAVSIM_BRANCHES=spec_pdm
    for i in $(seq 0 11); do nice -n 5 python $S/navsim_pdms.py --stage run --shard $i/12 > $L/pdms_${v}_$i.log 2>&1 & done
    wait
    python $S/navsim_pdms.py --stage summary > $L/pdms_${v}_summary.log 2>&1 )
  echo "$(date +%T) variant $v done" >> $L/chain.log
done
echo "$(date +%T) K20 CHAIN3 DONE" >> $L/chain.log
