#!/usr/bin/env bash
# K30: copy the final tables / figures from VUW into results/vuw_summaries/k30/ and docs/paper/handoff/K30/ (no smoke files)
R=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR/results
L=$HOME/research/CVPR
S=$L/results/vuw_summaries/k30; H=$L/docs/paper/handoff/K30
mkdir -p $S/b2d $S/navsim $H
scp -q "VUW:$R/k30/tables/*.csv" $S/ && rm -f $S/*_smoke.csv
scp -q "VUW:$R/k30/b2d/*.csv" $S/b2d/ 2>/dev/null
scp -q "VUW:$R/navsim_exp/k30_probe_time/v6ii_probe_time.csv" $S/navsim/ 2>/dev/null
scp -q "VUW:$R/navsim_exp/pdms_summary_k30*.csv" $S/navsim/ 2>/dev/null
scp -q "VUW:$R/navsim_exp/k30_v2/navhard_summary.csv" "VUW:$R/navsim_exp/k30_v2/navtest_summary.csv" $S/navsim/ 2>/dev/null
scp -q "VUW:$R/k30/b2d/inloop_*.csv" $S/b2d/ 2>/dev/null
scp -q "VUW:$R/k30/tables/dev_devref_selection.txt" $H/ 2>/dev/null
scp -q "VUW:$R/k30/figs/*" $H/ 2>/dev/null
for f in $S/*.csv $S/b2d/*.csv $S/navsim/*.csv; do       # the b2d tables are already named b2d_*
  b=$(basename $f); case $b in v7_detection_nusc_av2.csv) continue;; esac
  [ -f "$f" ] && cp $f $H/$b
done
ls $H | wc -l
