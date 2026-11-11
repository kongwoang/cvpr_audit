#!/usr/bin/env bash
# Confirmatory nuScenes run of the frozen v3 method (docs/method_freeze_draft.md §2b), once every scene is detected.
# Order: wait for the detect loop -> DEEP's tables on all scenes (fixes the new-scene split) -> ONE eval run on the
# new test split -> D5 at scale (GPU when free).  Code hashes are logged first so the frozen state is on record.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh
cd $B
L=$B/logs/confirm_chain.log
echo "$(date '+%F %T') waiting for detect loop" >> $L
until grep -q "DETECT LOOP DONE" $B/logs/detect_loop.log; do sleep 300; done
md5sum code/CVPR/src/vtf/speculative.py code/CVPR/src/vtf/dsf.py nusc_full_eval.py nusc_full_tables.py out/resnet18/image_refiner.pt >> $L
for g in mono oracle; do python nusc_full_tables.py --geometry $g >> $L 2>&1; done
python - >> $L 2>&1 <<PY
import pandas as pd
sp = pd.read_csv("$B/results/nusc_full/split.csv"); print(sp.groupby(["set", "split"]).size().to_string())
PY
echo "$(date '+%F %T') confirmatory eval start" >> $L
python nusc_full_eval.py --set new --split test --device cpu --threads 16 --scale 0.6 --tag _v3_confirm \
  --refiners $B/out/resnet18/image_refiner.pt > $B/logs/eval_confirm_v3.log 2>&1
echo "$(date '+%F %T') confirmatory eval exit $?" >> $L
$B/run_when_gpu_free.sh $B/logs/d5_models.log python nusc_models_vuw.py --device cuda:0
echo "$(date '+%F %T') CHAIN DONE" >> $L
