#!/usr/bin/env bash
# ESOD training was killed by a CUDA OOM (shared GPU) after epoch 27/29; the first chain tested last.pt (kept as *_ep27).
# Resume their train.py to the full 30 epochs, then test / measure / eval again. GPU via run_when_gpu_free.sh (lock),
# each step re-entering envs/esod.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_method/esod_resume.log
E=". $B/scripts/A_method/esod_env.sh; cd $B/third_party/esod;"
cp -n $B/results/esod/summary_newtest.csv $B/results/esod/summary_newtest_ep27.csv
cp -n $B/logs/A_method/esod_measure.log $B/logs/A_method/esod_measure_ep27.log
for try in $(seq 1 12); do                        # shared GPUs: another workload grabs whole GPUs at random times
  grep -q "^ *29/29" $B/out/esod/nusc_yolov5s/results.txt && break
  [ $try -gt 1 ] && sleep 1200
  # YOLOv5's select_device() sets CUDA_VISIBLE_DEVICES to --device (resume: opt.yaml), so pass the picked physical GPU
  MIN_FREE_MB=14000 $B/run_when_gpu_free.sh $L bash -c "$E sed -i \"s/^device: .*/device: '\$CUDA_VISIBLE_DEVICES'/\" $B/out/esod/nusc_yolov5s/opt.yaml; python train.py --resume $B/out/esod/nusc_yolov5s/weights/last.pt"
done
grep -q "^ *29/29" $B/out/esod/nusc_yolov5s/results.txt || { echo "$(date +%T) ABORT: epoch 29 not reached" >> $L; exit 1; }
W=$B/out/esod/nusc_yolov5s/weights/best.pt
rm -rf $B/out/esod/test_new
MIN_FREE_MB=12000 $B/run_when_gpu_free.sh $L bash -c "$E python test.py --data data/nuscenes.yaml --weights $W --batch-size 8 \
  --img-size 1536 --device \$CUDA_VISIBLE_DEVICES --task test --save-txt --save-conf --project $B/out/esod --name test_new --exist-ok"
MIN_FREE_MB=8000 $B/run_when_gpu_free.sh $B/logs/A_method/esod_measure.log bash -c "$E python test.py --data data/nuscenes.yaml \
  --weights $W --batch-size 1 --img-size 1536 --device \$CUDA_VISIBLE_DEVICES --task measure --project $B/out/esod --name measure --exist-ok"
n=$(ls $B/out/esod/test_new/labels 2>/dev/null | wc -l)
[ "$n" -gt 1000 ] || { echo "$(date +%T) ABORT: only $n prediction files" >> $L; exit 1; }
. $B/scripts/common/env.sh
python $B/scripts/A_method/esod_eval.py $B/out/esod/test_new/labels $B/logs/A_method/esod_measure.log >> $L 2>&1
echo "$(date +%T) ESOD RESUME DONE" >> $L
