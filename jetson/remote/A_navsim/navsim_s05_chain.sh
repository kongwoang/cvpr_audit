#!/usr/bin/env bash
# NAVSIM rerun with v3 as FROZEN (crop scale = 960 / long side = 0.5 on the 1920-px CAM_F0; the first run used the
# nuPlan-exploration value 1/3 = no magnification) + the u960 branch (whole frame at 576x960). Other branches reused.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_navsim/s05
mkdir -p $L
S=$B/scripts/A_navsim
X=$B/results/navsim_exp
E=". $B/scripts/common/env_navsim.sh; cd $B/code/risk-aware-perception; export SPEC_CROP_SCALE=0.5 NAVSIM_REDO=spec,uncertainty,full960 NAVSIM_SRC_DETS=$X/dets NAVSIM_DETS=$X/dets_s05;"
for i in 0 1; do
  MIN_FREE_MB=6000 $B/run_when_gpu_free.sh $L/detect_$i.log bash -c "$E python $S/navsim_detect.py --shard $i/2" &
  sleep 200
done
wait
n=$(ls $X/dets_s05/*.pkl 2>/dev/null | wc -l)
[ "$n" -ge 136 ] || { echo "NAVSIM S05 ABORT detect: $n logs" >> $L/detect_0.log; exit 1; }
. $B/scripts/common/env_navsim.sh; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NAVSIM_DETS=$X/dets_s05 NAVSIM_PDMS_OUT=$X/pdms_s05 NAVSIM_TAG=_s05 \
  NAVSIM_BRANCHES=reference,cheap,full,full960,spec,uncertainty,sahi,sahi_s033,fovea
cd $B/code/risk-aware-perception
for i in $(seq 0 11); do python $S/navsim_pdms.py --stage run --shard $i/12 > $L/pdms_$i.log 2>&1 & done
wait
python $S/navsim_pdms.py --stage summary > $L/pdms_summary.log 2>&1
echo "$(date +%T) NAVSIM S05 CHAIN DONE" >> $L/pdms_summary.log
