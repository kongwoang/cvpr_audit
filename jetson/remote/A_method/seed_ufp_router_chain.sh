#!/usr/bin/env bash
# Second seed of the UFP fine detector (mosaics), then two more seeds of the trained routers (skipping estimators; the
# conditioned XGBoost estimators are deterministic, random_state 42 in bgt-ada).
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
cd $B
L=logs/A_method/seeds
mkdir -p $L
SEED=1 MIN_FREE_MB=14000 $B/run_when_gpu_free.sh $L/ufp_train.log python scripts/A_ext/d68_train.py mosaic ufp_ft 15 yolov8s 960 32 mosaic_s1
[ -f out/ufp_ft/mosaic_s1/weights/last.pt ] || { echo "SEED UFP ABORT" >> $L/ufp_train.log; exit 1; }
FT_SUFFIX=_s1 $B/run_when_gpu_free.sh $L/ufp_eval.log python scripts/A_method/ufp_ft.py --stage eval
for s in 1 2; do
  SEED=$s $B/run_when_gpu_free.sh $L/router_s$s.log python scripts/A_method/router_baseline.py --stage train --est skip_offbin skip_V
  SEED=$s python scripts/A_method/router_baseline.py --stage score >> $L/router_s$s.log 2>&1
done
echo "SEED UFP ROUTER CHAIN DONE" >> $L/router_s2.log
