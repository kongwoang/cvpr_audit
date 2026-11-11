#!/usr/bin/env bash
# Run the memory-heavy nuScenes jobs one at a time (each loads the nuScenes DB, ~6 GB).
cd /home/kongwoang/research/CVPR
run() { echo "$(date +%T) start $*"; ./scripts/py "$@"; echo "$(date +%T) done $*"; }
run scripts/03_headroom.py --dataset nuScenes --geometry mono --verify_iou 0.3 > logs/headroom_iou03_nuScenes_mono.log 2>&1
run scripts/15_pivot_second_look.py --dataset nuScenes --geometry mono > logs/secondlook_nuScenes_mono.log 2>&1
run scripts/13_stakes_threshold.py --dataset nuScenes --geometry mono > logs/stakes_nuScenes_mono.log 2>&1
run scripts/12_region_proxy.py --dataset nuScenes --geometry mono > logs/region_nuScenes_mono.log 2>&1
run scripts/16_bayes_consumer.py --dataset nuScenes --geometry mono > logs/bayes_nuScenes_mono.log 2>&1
run scripts/03_headroom.py --dataset nuScenes --geometry oracle --verify_iou 0.3 > logs/headroom_iou03_nuScenes_oracle.log 2>&1
run scripts/15_pivot_second_look.py --dataset nuScenes --geometry oracle > logs/secondlook_nuScenes_oracle.log 2>&1
run scripts/17_temporal_reuse.py --dataset nuScenes --geometry mono > logs/reuse_nuScenes_mono.log 2>&1
run scripts/19_range_ambiguity.py --dataset nuScenes > logs/range_amb_nuScenes.log 2>&1
run scripts/24_speculative.py --dataset nuScenes --geometry mono --rmin 10 > logs/speculative_nuScenes_mono_r10.log 2>&1
run scripts/26_ensemble_sensitivity.py --dataset nuScenes > logs/ensemble_nuScenes.log 2>&1
run scripts/27_real_speculative.py --dataset nuScenes --geometry mono > logs/realspec_nuScenes_mono.log 2>&1
run scripts/27_real_speculative.py --dataset nuScenes --geometry mono --no_discovery > logs/realspec_nodisc_nuScenes_mono.log 2>&1
run scripts/20_learned_refiner.py --dataset nuScenes > logs/learned_refiner_nuScenes.log 2>&1
