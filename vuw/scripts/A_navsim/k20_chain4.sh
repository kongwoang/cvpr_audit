#!/usr/bin/env bash
# K20(4): interacting objects added to the support (subset of 1,000 navtest tokens) -> spec_pdm detections -> PDMS.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_navsim/k20; X=$B/results/navsim_exp; S=$B/scripts/A_navsim
until grep -qs "pairs done" $L/chain.log; do sleep 300; done
. $B/scripts/common/env_navsim.sh
cd $B/code/risk-aware-perception
CUDA_VISIBLE_DEVICES= python $S/k20_pairfield.py > $L/pairfield.log 2>&1
for v in pairs pairs_probe; do
  E=". $B/scripts/common/env_navsim.sh; cd $B/code/risk-aware-perception; export SPEC_CROP_SCALE=0.5 NAVSIM_FIELD=$X/field_k20_$v NAVSIM_DETS=$X/dets_k20_$v;"
  SHARE=1 MIN_FREE_MB=6000 $B/run_when_gpu_free.sh $L/pairdet_$v.log bash -c "$E python $S/k20_pairdet.py"
  ( export SPEC_CROP_SCALE=0.5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= NAVSIM_DETS=$X/dets_k20_$v NAVSIM_PDMS_OUT=$X/pdms_k20_$v \
      NAVSIM_TAG=_k20_$v NAVSIM_MAIN=spec_pdm NAVSIM_BRANCHES=spec_pdm
    for i in $(seq 0 7); do nice -n 5 python $S/navsim_pdms.py --stage run --shard $i/8 > $L/pdms_${v}_$i.log 2>&1 & done
    wait
    python $S/navsim_pdms.py --stage summary > $L/pdms_${v}_summary.log 2>&1 )
  echo "$(date +%T) $v done" >> $L/chain.log
done
echo "$(date +%T) K20 CHAIN4 DONE" >> $L/chain.log
