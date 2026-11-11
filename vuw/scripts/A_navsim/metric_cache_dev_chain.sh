#!/usr/bin/env bash
# NAVSIM metric cache for the DEV set (OpenScene mini split, navtest-like filter, every 4th frame; config navdev_mini).
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
. $B/scripts/common/env_navsim.sh
L=$B/logs/A_navsim
cd $NAVSIM_DEVKIT_ROOT
python navsim/planning/script/run_metric_caching.py train_test_split=navdev_mini \
  cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache_dev worker=single_machine_thread_pool worker.use_process_pool=True \
  worker.max_workers=8 > $L/metric_cache_dev.log 2>&1
echo "$(date +%T) exit $? ; $(find $NAVSIM_EXP_ROOT/metric_cache_dev -name metric_cache.pkl | wc -l) caches" >> $L/metric_cache_dev.log
echo "NAVSIM DEV METRIC CACHE DONE" >> $L/metric_cache_dev.log
