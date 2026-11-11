#!/usr/bin/env bash
# K11 exclusive Jetson window (DEEP session's pause protocol): wait until DEEP's busy.sh prints only "paused", then
# (1) engine micro-timings: dynamic vs static YOLOv8s at the crop canvases, (2) phase-1 runs of scripts/58 on nuScenes
# and AV2 (base = no breakdown syncs; + batched refiner) with parity against the K5 follow-up runs; release the pause.
cd "$(dirname "$0")/.."
BUSY=$HOME/research/risk-aware-perception/scripts/busy.sh
touch $HOME/.jetson_pause
# DEEP's TensorRT engine builds are trtexec runs that busy.sh does not see: wait for their end marker first
until grep -q BUILD_DONE $HOME/research/risk-aware-perception/logs/xl_build_engines.log; do sleep 60; done
# busy.sh must show only "paused" twice, 3 min apart (a DEEP step that passed its check just before the pause file
# appeared would show up here), and no python of the DEEP repo may be running
ok() { [ "$(bash $BUSY | tr -d '\n')" = "paused" ] && ! ps -eo args= | grep -q "^[^ ]*python[0-9.]* .*risk-aware-perception\|^[^ ]*python[0-9.]* scripts/1[0-9][0-9]_"; }
until ok && sleep 180 && ok; do sleep 60; done
echo "$(date +%T) window start" >> logs/k11_window.log
TE=/usr/src/tensorrt/bin/trtexec
for s in 96x288 320x320 640x640; do
  h=${s%x*}; w=${s#*x}
  $TE --loadEngine=engines/yolov8s_crop_$s.engine --iterations=300 --warmUp=500 --useSpinWait 2>/dev/null | grep "GPU Compute Time: min" | sed "s/^/static $s /" >> logs/k11_window.log
  $TE --loadEngine=engines/yolov8s_dyn.engine --shapes=images:1x3x${h}x${w} --iterations=300 --warmUp=500 --useSpinWait 2>/dev/null | grep "GPU Compute Time: min" | sed "s/^/dynamic $s /" >> logs/k11_window.log
done
RN=results/raw/20261004_143529_large_latency_opt_nusc; RA=results/raw/20261004_144842_large_latency_opt_av2
# phase 1 (decision-identical): no breakdown syncs (+ batched refiner on AV2, where the refiner matters)
scripts/py scripts/58_large_latency_fast.py --dataset nusc --ref_run $RN > logs/k11_nusc_base.log 2>&1
scripts/py scripts/58_large_latency_fast.py --dataset av2 --ref_run $RA > logs/k11_av2_base.log 2>&1
scripts/py scripts/58_large_latency_fast.py --dataset av2 --refbatch --ref_run $RA > logs/k11_av2_refbatch.log 2>&1
# phase 2 (NEW configurations, selected on the nuScenes frames of this board = the 85 old DEEP scenes, nuScenes' dev data):
# wider support context / field-crop margin to reduce the sequential guard re-detections
for cfg in "--ctx 2.5" "--fmargin 0.15" "--ctx 2.5 --fmargin 0.15"; do
  tag=$(echo $cfg | tr -d ' -' | tr '.' 'p')
  scripts/py scripts/58_large_latency_fast.py --dataset nusc $cfg --ref_run $RN > logs/k11_nusc_$tag.log 2>&1
done
rm -f $HOME/.jetson_pause
echo "$(date +%T) window end" >> logs/k11_window.log
