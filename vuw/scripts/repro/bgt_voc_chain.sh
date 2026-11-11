#!/usr/bin/env bash
# Fidelity (A): reproduce Budget-Adaptive Routing's published VOC table (bgt-ada README Tab. 2) with the official code
# and the released checkpoints (seed 42): detect + prepare (regenerated), train skipped (released checkpoints),
# evaluate + analyse; then compare with the shipped results/voc/metrics/main_benchmark_table.csv.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/common/env.sh
R=$B/repro/bgt-ada
L=$B/logs/repro/bgt_voc.log
mkdir -p $B/logs/repro
cd $R
while pgrep -u $USER -x curl > /dev/null; do sleep 30; done      # the VOC2010 download
echo "$(date +%T) tar $(stat -c %s data/VOCtrainval_03-May-2010.tar) bytes" >> $L
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# detection batch 8 (their 32 needs > 23.5 GB): scripts/repro/bgt_run.py
MIN_FREE_MB=12000 $B/run_when_gpu_free.sh $L python $B/scripts/repro/bgt_run.py --phase detect prepare --dataset voc --seed 42
tail -40 $L | grep -q "PREPARE complete" || { echo "BGT VOC ABORT detect/prepare" >> $L; exit 1; }
mkdir -p results/full_eval/voc
[ -e results/full_eval/voc/checkpoints ] || ln -s ../../../checkpoints results/full_eval/voc/checkpoints
MIN_FREE_MB=12000 $B/run_when_gpu_free.sh $L python $B/scripts/repro/bgt_run.py --phase evaluate analyse --dataset voc --seed 42
python $B/scripts/repro/bgt_voc_compare.py >> $L 2>&1
echo "BGT VOC REPRO DONE" >> $L
