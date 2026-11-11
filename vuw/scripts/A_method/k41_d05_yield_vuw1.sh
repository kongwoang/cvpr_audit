#!/usr/bin/env bash
# 2026-11-02: D03 base full_ahd (non-SHARE wrapper) starved by the 5 SHARE D05 workers on VUW1. Pause the worker chains
# (SIGSTOP the bash chains only; running grid jobs finish), wait until the D03 ahd detector has started, then SIGCONT.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR; cd $B
W="3915604 3915646 3944335 3971069 3971121"
kill -STOP $W; echo "$(date +%T) paused D05 workers $W"
until pgrep -u $(whoami) -f 'k41_d03_detector.py --dataset ahd --part pip --setting base' | grep -qv "^$$\$" && \
      nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -qf <(pgrep -u $(whoami) -f 'k41_d03_detector.py --dataset ahd'); do sleep 60; done
echo "$(date +%T) D03 ahd detector on GPU; resuming"; sleep 120; kill -CONT $W; echo "$(date +%T) resumed"
