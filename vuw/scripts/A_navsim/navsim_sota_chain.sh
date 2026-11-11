#!/usr/bin/env bash
# NAVSIM ports of AdaScale and Budget-Adaptive Routing (navsim_sota.py): detections at the extra AdaScale scales on dev
# (OpenScene mini, + full960) and navtest, PDMS of s480 / s800 on navtest, then labels -> ORIC -> train -> score.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_navsim/sota
mkdir -p $L
S=$B/scripts/A_navsim
X=$B/results/navsim_exp
E=". $B/scripts/common/env_navsim.sh; cd $B/code/risk-aware-perception; export SPEC_CROP_SCALE=0.5;"
for i in 0 1; do
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/detect_dev_$i.log bash -c "$E NAVSIM_SPLIT=mini NAVSIM_CACHE=$X/metric_cache_dev NAVSIM_REDO=full960,s480,s800 NAVSIM_SRC_DETS=$X/dets_dev NAVSIM_DETS=$X/dets_dev_sota python $S/navsim_detect.py --shard $i/2" &
  sleep 200
  SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/detect_test_$i.log bash -c "$E NAVSIM_REDO=s480,s800 NAVSIM_SRC_DETS=$X/dets_final NAVSIM_DETS=$X/dets_sota python $S/navsim_detect.py --shard $i/2" &
  sleep 200
done
wait
for f in $L/detect_*.log; do grep -q "exit 0" $f || { echo "$(date +%T) NAVSIM SOTA ABORT detect ($f)" >> $L/chain.log; exit 1; }; done
. $B/scripts/common/env_navsim.sh; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NAVSIM_DETS=$X/dets_sota NAVSIM_PDMS_OUT=$X/pdms_sota \
  NAVSIM_TAG=_sota NAVSIM_BRANCHES=s480,s800 NAVSIM_MAIN=s800
cd $B/code/risk-aware-perception
for i in $(seq 0 11); do python $S/navsim_pdms.py --stage run --shard $i/12 > $L/pdms_$i.log 2>&1 & done
wait
python $S/navsim_pdms.py --stage summary > $L/pdms_summary.log 2>&1
NAVSIM_DETS_DEV=$X/dets_dev_sota NAVSIM_DETS_TEST=$X/dets_sota python $S/navsim_sota.py --stage labels > $L/labels.log 2>&1
grep -q "LABELS DONE" $L/labels.log || { echo "$(date +%T) NAVSIM SOTA ABORT labels" >> $L/chain.log; exit 1; }
# bgt-ada / AdaScale stages in the main venv (xgboost, torchvision); NAVSIM paths passed explicitly
V=". $B/scripts/common/env.sh; cd $B; export OPENSCENE_DATA_ROOT=$B/datasets/navsim NAVSIM_EXP_ROOT=$X NAVSIM_DETS_TEST=$X/dets_sota;"
bash -c "$V python $S/navsim_sota.py --stage oric" > $L/oric.log 2>&1
grep -q "ORIC DONE" $L/oric.log || { echo "$(date +%T) NAVSIM SOTA ABORT oric" >> $L/chain.log; exit 1; }
SHARE=1 MIN_FREE_MB=8000 $B/run_when_gpu_free.sh $L/train.log bash -c "$V python $S/navsim_sota.py --stage train"
grep -q "TRAIN DONE" $L/train.log || { echo "$(date +%T) NAVSIM SOTA ABORT train" >> $L/chain.log; exit 1; }
bash -c "$V python $S/navsim_sota.py --stage score" > $L/score.log 2>&1
echo "$(date +%T) NAVSIM SOTA CHAIN DONE" >> $L/chain.log
