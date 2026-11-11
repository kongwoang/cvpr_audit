#!/usr/bin/env bash
# Sequential KITTI jobs (small memory) alongside queue_nusc.sh; collects all results at the end.
cd "$(dirname "$0")/.."
run() { echo "$(date +%T) start $*"; ./scripts/py "$@"; echo "$(date +%T) done $*"; }
# wait for the KITTI jobs already running, to keep the CPU sane
while pgrep -f "24_speculative.py --dataset KITTI" >/dev/null || pgrep -f "29_spec_refine.py" >/dev/null; do sleep 30; done
run scripts/27_real_speculative.py --dataset KITTI --geometry mono > logs/realspec_KITTI_mono.log 2>&1
run scripts/27_real_speculative.py --dataset KITTI --geometry mono --no_discovery > logs/realspec_nodisc_KITTI_mono.log 2>&1
run scripts/27_real_speculative.py --dataset KITTI --geometry mono --no_pivots > logs/realspec_nopiv_KITTI_mono.log 2>&1
run scripts/22_real_crop.py --geometry oracle > logs/realcrop_KITTI_oracle.log 2>&1
./scripts/collect_results.sh
# final collection once the nuScenes queue is also done
while pgrep -f queue_nusc.sh >/dev/null; do sleep 60; done
./scripts/collect_results.sh
echo "$(date +%T) ALL QUEUES DONE"
