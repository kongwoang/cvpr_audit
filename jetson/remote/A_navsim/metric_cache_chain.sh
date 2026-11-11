#!/usr/bin/env bash
# NAVSIM v1.1 metric cache for navtest (official run_metric_caching.py; CPU, 12 processes).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
. $B/scripts/common/env_navsim.sh
L=$B/logs/A_navsim
mkdir -p $L $NAVSIM_EXP_ROOT
cd $NAVSIM_DEVKIT_ROOT
python navsim/planning/script/run_metric_caching.py train_test_split=navtest \
  cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache worker=single_machine_thread_pool worker.use_process_pool=True \
  worker.max_workers=12 > $L/metric_cache.log 2>&1
echo "$(date +%T) exit $? ; $(find $NAVSIM_EXP_ROOT/metric_cache -name metric_cache.pkl | wc -l) caches" >> $L/metric_cache.log
echo "NAVSIM METRIC CACHE DONE" >> $L/metric_cache.log
