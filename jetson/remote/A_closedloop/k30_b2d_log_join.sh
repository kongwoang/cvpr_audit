#!/usr/bin/env bash
# K30 V4: wait until both hosts' dispatchers (k30_b2d_log_dispatch3.sh on cuda12 / cuda13) are complete, then write the
# marker the watchdog and k30_finalize.sh wait for.
L=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR/logs/k30/b2dlog
until grep -qs "HOST COMPLETE" $L/dispatch_cuda12.log && grep -qs "HOST COMPLETE" $L/dispatch_cuda13.log; do sleep 600; done
echo "$(date +%T) K30 B2D LOG ALL CHUNKS COMPLETE" >> $L/dispatch.log
