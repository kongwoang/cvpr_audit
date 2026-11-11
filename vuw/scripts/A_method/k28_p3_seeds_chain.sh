#!/usr/bin/env bash
# K28 PHASE 3 (post-hoc) — S6 on the test units: the ours part of k28_p3.py with --seeds (v6-S / M / L scored with the
# refiner seeds 0 / 1 / 2 of §2k-§2m; the loop is refiner-independent, so seed 0 = the phase-2 pass), nuScenes / AV2
# (ego mask) / Argoverse-HD, 3 shards each; completeness re-run built in; then k28_p3_seeds.py.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k28/p3; mkdir -p $L
T="env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 B_REAL=$B"
declare -A NUNITS=([nusc]=228 [av2]=150 [ahd]=24)
cmd() {
  if [ "$1" = av2 ]; then echo "$T B=$S K10_MASK_Y0=1830 python $B/scripts/A_method/k10_masked_run.py $S/scripts/A_method/k28_p3.py --dataset av2 --split test --part ours --seeds --tag seeds --shard $2"
  else echo "$T B=$B python $B/scripts/A_method/k28_p3.py --dataset $1 --split test --part ours --seeds --tag seeds --shard $2"; fi
}
launch() {
  for ds in nusc av2 ahd; do
    SHARE=1 MIN_FREE_MB=9000 $B/run_when_gpu_free.sh $L/seeds_${ds}_g0$1.log bash -c "$(cmd $ds 0/3) & $(cmd $ds 1/3) & $(cmd $ds 2/3) & wait" &
    sleep 10
  done
  wait
}
complete() { for ds in nusc av2 ahd; do [ "$(ls $B/results/k28/p3/${ds}_test_ours_seeds 2>/dev/null | wc -l)" -ge ${NUNITS[$ds]} ] || return 1; done; }
launch ""
for i in 1 2; do complete && break; echo "$(date +%T) incomplete; re-run $i" >> $L/seeds_chain.log; launch "_rerun$i"; done
complete || { echo "$(date +%T) K28 SEEDS RUNS INCOMPLETE" >> $L/seeds_chain.log; exit 1; }
echo "$(date +%T) K28 SEEDS RUNS DONE" >> $L/seeds_chain.log
cd $B && $T B=$B python $B/scripts/A_method/k28_p3_seeds.py > $L/seeds_analyze.log 2>&1
echo "$(date +%T) K28 SEEDS ANALYZE DONE (exit $?)" >> $L/seeds_chain.log
