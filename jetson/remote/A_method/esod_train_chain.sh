#!/usr/bin/env bash
# ESOD baseline, after esod_chain.sh (prepare): train ESOD-YOLOv5s on nuScenes (their recipe: VisDrone hyps, SAM masks,
# pretrained YOLOv5s), then predict the new test keyframes with their test.py and measure GFLOPs, then score decisions.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
L=$B/logs/A_method/esod_train.log
# start only on the prepare script's own success line (written once, at its very end)
until grep -qs "ESOD PREPARE DONE" $B/logs/A_method/esod_prepare.log; do
  if grep -qsE "ESOD PREPARE exit [1-9]|Traceback" $B/logs/A_method/esod_prepare.log; then echo "prepare failed" >> $L; exit 1; fi
  sleep 300
done
gpu_wait () {
  while true; do
    . $B/scripts/A_method/esod_env.sh
    g=$(pick_gpu); if [ -n "$g" ]; then sleep 60; g2=$(pick_gpu); [ "$g" = "$g2" ] && break; fi
    echo "$(date +%T) waiting for a free GPU" >> $L; sleep 120
  done
}
# test split in ESOD layout: all keyframes of the new test scenes (images + GT labels; no masks needed at test time)
. $B/scripts/common/env.sh
python - >> $L 2>&1 <<PY
import os, pandas as pd
from pathlib import Path
from rap.nusc import NuScenesDB, make_adapter
B = Path(os.environ["B"]); R = B / "data" / "esod_nusc" / "test_new"
(R / "images").mkdir(parents=True, exist_ok=True); (R / "labels").mkdir(parents=True, exist_ok=True)
sp = pd.read_csv(B / "results" / "nusc_full" / "split.csv")
db = NuScenesDB(str(B / "datasets" / "nuscenes"), "v1.0-trainval"); ad = make_adapter(db)
by = {db.scene_name(t): t for t in db.scenes}; C = {"vehicle": 0, "person": 1, "cyclist": 2}; paths = []
for s in sorted(sp[(sp["set"] == "new") & (sp.split == "test")].seq):
    g = ad.geometry(s)
    for fr, tk in enumerate(db.samples(by[s])):
        ip = R / "images" / f"{s}_{fr:03d}.jpg"
        if not ip.exists(): ip.symlink_to(db.image_path(tk))
        q = g[(g["frame"] == fr) & ((g["y2"] - g["y1"]) >= 10)]
        (R / "labels" / f"{s}_{fr:03d}.txt").write_text("".join("%d %.6f %.6f %.6f %.6f\n" % (C[str(o["coarse"])],
            (o["x1"] + o["x2"]) / 3200, (o["y1"] + o["y2"]) / 1800, (o["x2"] - o["x1"]) / 1600, (o["y2"] - o["y1"]) / 900) for o in q))
        paths.append(str(ip))
(B / "data" / "esod_nusc" / "split" / "test_new.txt").write_text("\n".join(paths) + "\n")
print("test list", len(paths))
PY
sed -i "s#^test: .*#test: $B/data/esod_nusc/split/test_new.txt#" $B/third_party/esod/data/nuscenes.yaml
cd $B/third_party/esod
gpu_wait; echo "$(date +%T) train on GPU $g" >> $L
CUDA_VISIBLE_DEVICES=$g python train.py --data data/nuscenes.yaml --cfg models/cfg/esod/nuscenes_yolov5s.yaml \
  --weights weights/pretrained/yolov5s.pt --hyp data/hyps/hyp.nuscenes.yaml --batch-size 8 --img-size 1536 --epochs 30 \
  --device 0 --workers 8 --project $B/out/esod --name nusc_yolov5s --exist-ok >> $L 2>&1
echo "$(date +%T) train exit $?" >> $L
W=$B/out/esod/nusc_yolov5s/weights/best.pt
[ -s "$W" ] || { echo "$(date +%T) ABORT: no trained weights at $W" >> $L; exit 1; }
rm -rf $B/out/esod/test_new
gpu_wait
CUDA_VISIBLE_DEVICES=$g python test.py --data data/nuscenes.yaml --weights $W --batch-size 8 --img-size 1536 --device 0 \
  --task test --save-txt --save-conf --project $B/out/esod --name test_new --exist-ok >> $L 2>&1
CUDA_VISIBLE_DEVICES=$g python test.py --data data/nuscenes.yaml --weights $W --batch-size 1 --img-size 1536 --device 0 \
  --task measure --project $B/out/esod --name measure --exist-ok > $B/logs/A_method/esod_measure.log 2>&1
n=$(ls $B/out/esod/test_new/labels 2>/dev/null | wc -l)
[ "$n" -gt 1000 ] || { echo "$(date +%T) ABORT: only $n prediction files" >> $L; exit 1; }
. $B/scripts/common/env.sh
python $B/scripts/A_method/esod_eval.py $B/out/esod/test_new/labels $B/logs/A_method/esod_measure.log >> $L 2>&1
echo "$(date +%T) ESOD DONE" >> $L
