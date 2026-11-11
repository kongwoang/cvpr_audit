#!/usr/bin/env bash
# Fidelity (A): ESOD's published VisDrone-val numbers with their released weights (Table III of the TIP paper).
# Everything runs inside envs/esod (SAM, skimage); the GPU wait is inlined (run_when_gpu_free.sh activates the main venv).
. /vol/dl-nguyenb5-solar/users/hoangpc/CVPR/scripts/A_method/esod_env.sh
L=$B/logs/repro/esod_visdrone.log
mkdir -p $B/logs/repro $B/repro/visdrone
gpu() { while true; do g=$(pick_gpu); if [ -n "$g" ]; then sleep 60; g2=$(pick_gpu); [ "$g" = "$g2" ] && break; fi; sleep 120; done; echo $g; }
cd $B/repro/visdrone
if [ ! -d VisDrone2019-DET-val ]; then
  gdown 1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59 -O val.zip >> $L 2>&1
  unzip -q val.zip && rm val.zip
fi
[ -d VisDrone2019-DET-val/images ] || { echo "ESOD VISDRONE ABORT download ($(ls))" >> $L; exit 1; }
# drop outputs of the earlier prepare run made without SAM (main venv by mistake)
rm -rf VisDrone2019-DET-val/labels VisDrone2019-DET-val/masks split
find VisDrone2019-DET-val/images -name "*_masked.jpg" -delete
cd $B/third_party/esod
[ -e VisDrone ] || ln -s $B/repro/visdrone VisDrone
python -c "import skimage" 2>/dev/null || pip install -q scikit-image >> $L 2>&1
g=$(gpu); echo "$(date +%T) prepare on GPU $g" >> $L
CUDA_VISIBLE_DEVICES=$g python $B/scripts/repro/esod_visdrone_prepare.py >> $L 2>&1
tail -3 $L | grep -q "PREPARE DONE" || { echo "ESOD VISDRONE ABORT prepare" >> $L; exit 1; }
grep -q "segment-anything" $L && { echo "ESOD VISDRONE ABORT no SAM" >> $L; exit 1; }
for m in yolov5m yolov8m; do
  g=$(gpu); echo "=== esod_$m val 1536 on GPU $g" >> $L
  CUDA_VISIBLE_DEVICES=$g python test.py --data data/visdrone.yaml --weights $B/repro/esod_weights/esod/esod_$m.pt \
    --batch-size 8 --img-size 1536 --device 0 >> $L 2>&1
done
echo "ESOD VISDRONE REPRO DONE" >> $L
