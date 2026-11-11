#!/usr/bin/env bash
# NAVSIM navtest, ONE final run of the configuration frozen on dev (method_freeze_draft §3): spec_pdm from the
# PDM-Closed probing field, next to every other branch (same detections as the s05 run).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_navsim/final
mkdir -p $L
S=$B/scripts/A_navsim
X=$B/results/navsim_exp
until grep -qs "FIELD TEST DONE" $B/logs/A_navsim/field_test/done.log; do sleep 300; done
C="export SPEC_CROP_SCALE=0.5 NAVSIM_FIELD=$X/field_test"
E=". $B/scripts/common/env_navsim.sh; cd $B/code/risk-aware-perception; $C;"
for i in 0 1; do
  MIN_FREE_MB=6000 $B/run_when_gpu_free.sh $L/detect_$i.log bash -c "$E NAVSIM_REDO=spec_pdm NAVSIM_SRC_DETS=$X/dets_s05 NAVSIM_DETS=$X/dets_final python $S/navsim_detect.py --shard $i/2" &
  sleep 200
done
wait
. $B/scripts/common/env_navsim.sh; eval "$C"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NAVSIM_DETS=$X/dets_final NAVSIM_PDMS_OUT=$X/pdms_final \
  NAVSIM_TAG=_final NAVSIM_MAIN=spec_pdm NAVSIM_BRANCHES=reference,cheap,full,full960,spec,spec_pdm,uncertainty,sahi,sahi_s033,fovea
cd $B/code/risk-aware-perception
for i in $(seq 0 11); do python $S/navsim_pdms.py --stage run --shard $i/12 > $L/pdms_$i.log 2>&1 & done
wait
python $S/navsim_pdms.py --stage summary > $L/pdms_summary.log 2>&1
echo "$(date +%T) NAVSIM FINAL CHAIN DONE" >> $L/pdms_summary.log
