#!/usr/bin/env bash
# K10 post-hoc robustness check: re-run the AV2 val table scripts and the Argoverse-HD evaluation unchanged, with the
# fixed ego-region mask applied to every pipeline (k10_masked_run.py), in the shadow tree $B/k10_masked (own results/).
# AV2 ring_front_center (2048 rows): hood band y >= 1830 (inspected on 10 val logs; the hood's top edge is at rows
# ~1840-1870 at the image centre). Argoverse-HD (1200 rows): no ego part visible (8 sequences inspected); the band of the
# bottom 5 % of rows (y >= 1140) is masked as the analogous check.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k10
mkdir -p $S/results/av2 $S/results/av2_sota $S/results/argoverse_hd $S/results/nusc_full $L
for d in datasets models out scripts third_party code; do [ -e $S/$d ] || ln -s $B/$d $S/$d; done
cp -n $B/results/av2/eval__val_v3_confirm2.pkl $S/results/av2/
cp -n $B/results/nusc_full/split.csv $S/results/nusc_full/
for f in frames__train.pkl frames__val.pkl boxes__train.pkl boxes__val.pkl adascale_net.pt scores__skip_offbin.pkl \
         scores__cond_moric.pkl scores__skip_V.pkl scores__cond_V.pkl; do cp -n $B/results/av2_sota/$f $S/results/av2_sota/; done
R="python $B/scripts/A_method/k10_masked_run.py"
SC=$S/scripts/A_method
run() { log=$1; y0=$2; shift 2; SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/$log.log env B=$S K10_MASK_Y0=$y0 $R "$@"; }
run av2_eval_v4 1830 $SC/av2_eval_v4.py --split val --tag _v4_final &
sleep 150
run av2_baselines 1830 $SC/av2_baselines.py &
sleep 150
run av2_labels_test 1830 $SC/av2_sota_labels.py --splits test &
sleep 150
for i in 0 1 2 3; do run ahd_$i 1140 $SC/argoverse_hd_eval.py --stage run --shard $i/4 & sleep 150; done
wait
run native_av2 1830 $SC/native_uniform.py --dataset av2 &
sleep 150
run av2_ada_eval 1830 $SC/av2_sota.py --stage ada_eval &
wait
env B=$S K10_MASK_Y0=1830 $R $SC/av2_sota.py --stage score > $L/av2_score.log 2>&1
env B=$S K10_MASK_Y0=1140 $R $SC/argoverse_hd_eval.py --stage score > $L/ahd_score.log 2>&1
echo "$(date +%T) K10 MASKED CHAIN DONE" > $L/chain_done.log
