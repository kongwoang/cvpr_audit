#!/usr/bin/env bash
# K11 exclusive Jetson window 2 (phase-2 configurations only, after the dynamic-wrapper memory fix) (DEEP session's pause protocol): wait until DEEP's busy.sh prints only "paused", then
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
echo "$(date +%T) window2 start" >> logs/k11_window.log
RN=results/raw/20261004_143529_large_latency_opt_nusc
# phase 2 (NEW configurations, selected on the nuScenes frames of this board = the 85 old DEEP scenes, nuScenes' dev data):
# wider support context / field-crop margin to reduce the sequential guard re-detections
for cfg in "--ctx 2.5" "--fmargin 0.15" "--ctx 2.5 --fmargin 0.15"; do
  tag=$(echo $cfg | tr -d ' -' | tr '.' 'p')
  scripts/py scripts/58_large_latency_fast.py --dataset nusc $cfg --ref_run $RN > logs/k11_nusc_$tag.log 2>&1
done
rm -f $HOME/.jetson_pause
echo "$(date +%T) window2 end" >> logs/k11_window.log
