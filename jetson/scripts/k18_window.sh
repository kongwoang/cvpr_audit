#!/usr/bin/env bash
# K18(B) exclusive Jetson window (DEEP session's pause protocol): bounded-engine-call variants of ours + tail latency of
# every pipeline, on the K11 nuScenes frames and the K13(e) masked AV2 frames. Smoke first, then the full runs.
cd "$(dirname "$0")/.."
BUSY=$HOME/research/risk-aware-perception/scripts/busy.sh
touch $HOME/.jetson_pause
ok() { [ "$(bash $BUSY | tr -d '\n')" = "paused" ] && ! ps -eo args= | grep -q "^[^ ]*python[0-9.]* .*risk-aware-perception\|^[^ ]*python[0-9.]* scripts/1[0-9][0-9]_"; }
until ok && sleep 180 && ok; do sleep 60; done
echo "$(date +%T) window_k18 start" >> logs/k11_window.log
scripts/py scripts/59_bounded_latency.py --dataset nusc --max_frames 30 --warmup 5 > logs/k18_smoke.log 2>&1 || { rm -f $HOME/.jetson_pause; echo "$(date +%T) window_k18 smoke FAILED" >> logs/k11_window.log; exit 1; }
scripts/py scripts/59_bounded_latency.py --dataset nusc > logs/k18_nusc.log 2>&1
scripts/py scripts/59_bounded_latency.py --dataset av2 --mask_y0 1830 > logs/k18_av2_masked.log 2>&1
rm -f $HOME/.jetson_pause
echo "$(date +%T) window_k18 end" >> logs/k11_window.log
