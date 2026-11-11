#!/usr/bin/env bash
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh
cd $B
python nusc_ablate.py --set new --split test --device cpu --threads 10 --scale 0.6 --cheap_dir nusc_det_ft \
  --tag _ftdraft --refiners $B/out/resnet18/image_refiner.pt > logs/eval_ftdraft.log 2>&1 &
$B/run_when_gpu_free.sh $B/logs/d5_rtdetr.log python nusc_models_vuw.py --device cuda:0 --models rtdetr-l
python d5_analysis.py > logs/d5_analysis.log 2>&1
echo "$(date +%T) D5 ANALYSIS DONE" >> logs/d5_analysis.log
wait
echo "$(date +%T) FTDRAFT DONE" >> logs/eval_ftdraft.log
