#!/usr/bin/env bash
# K29: after window C ("K29 TEST WINDOW DONE"), build the final MAXN tables (65_k29_report.py --sel trt32 -> handoff/K29)
# and write logs/k29/after_window.log. The Jetson stays paused (~/.jetson_pause): the session releases it and tells DEEP.
cd "$(dirname "$0")/.."
until grep -qs "K29 TEST WINDOW DONE" logs/k29/test_window.log; do sleep 300; done
scripts/py scripts/65_k29_report.py --sel trt32 > logs/k29/report_final.log 2>&1
echo "$(date '+%F %T') K29 REPORT exit $?" >> logs/k29/after_window.log
