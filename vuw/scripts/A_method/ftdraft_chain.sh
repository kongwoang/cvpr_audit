#!/usr/bin/env bash
# Follow-up to D68: the frozen v3 loop with the domain-fine-tuned draft (plain arm) on the new test scenes.
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/env.sh
cd $B
$B/run_when_gpu_free.sh $B/logs/ftdraft_detect.log python nusc_detect_ft.py --weights out/d68/plain/weights/best.pt \
  --out code/risk-aware-perception/data/cache/nusc_det_ft --device cuda:0 --only new:test
python nusc_ablate.py --set new --split test --device cpu --threads 10 --scale 0.6 --cheap_dir nusc_det_ft \
  --tag _ftdraft --refiners $B/out/resnet18/image_refiner.pt > logs/eval_ftdraft.log 2>&1
echo "$(date +%T) FTDRAFT DONE" >> logs/eval_ftdraft.log
