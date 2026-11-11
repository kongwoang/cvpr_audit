#!/usr/bin/env bash
# K13: every pre-K10 AV2 analysis redone WITH the ego mask (rows >= 1830, containment >= 0.5, every pipeline; frozen
# configurations; k10_masked_run.py in the shadow tree $B/k10_masked), plus the Argoverse-HD ablation (no mask needed)
# and the fine-tuned hood check. Post-hoc analysis.
B=/vol/dl-nguyenb5-solar/users/hoangpc/CVPR
source $B/env.sh
S=$B/k10_masked
L=$B/logs/A_method/k13; mkdir -p $L
for d in datasets models out scripts third_party code data; do [ -e $S/$d ] || ln -s $B/$d $S/$d; done
mkdir -p $S/results/nusc_full; cp -n $B/results/nusc_full/split.csv $S/results/nusc_full/
R="python $B/scripts/A_method/k10_masked_run.py"
SC=$S/scripts/A_method
m() { log=$1; shift; SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/$log.log env B=$S K10_MASK_Y0=1830 $R "$@"; }
u() { log=$1; shift; SHARE=1 MIN_FREE_MB=5000 $B/run_when_gpu_free.sh $L/$log.log python "$@"; }
for i in 0 1 2; do m instr_base_$i $SC/instr_eval.py --dataset av2 --split test --setting base --shard $i/3 & sleep 120; done
for st in del_all del_half; do for i in 0 1; do m instr_${st}_$i $SC/instr_eval.py --dataset av2 --split test --setting $st --shard $i/2 & sleep 120; done; done
for i in 0 1; do m loc_$i $SC/k12_locality.py --shard $i/2 & sleep 120; done
for i in 0 1; do m k9curve_$i $SC/k9_tune.py --dataset av2 --split test --configs 0:0 --unc_mults 0.75,1,1.25,1.5,2 --shard $i/2 & sleep 120; done
m ft_dets $SC/av2_ft_eval.py --stage dets --det plain960 & sleep 120
m train_v4 $SC/av2_eval_v4.py --split train --tag _confirm3 & sleep 120
for i in 0 1 2 3; do u ahd_abl_$i $B/scripts/A_method/k13_ahd_ablation.py --shard $i/4 & sleep 120; done
u ft_hood $B/scripts/A_method/k13_ft_hood.py & sleep 60
wait
# CPU stages (masked where the scores involve AV2 lifts)
env B=$S K10_MASK_Y0=1830 $R $SC/av2_ft_eval.py --stage preds --method esod --pred $B/out/esod/av2_val/labels > $L/ft_preds_esod.log 2>&1
env B=$S K10_MASK_Y0=1830 $R $SC/av2_ft_eval.py --stage preds --method zoomdet --pred $B/out/zoomdet/av2_val/labels > $L/ft_preds_zoomdet.log 2>&1
env B=$S K10_MASK_Y0=1830 $R $SC/av2_ft_eval.py --stage score > $L/ft_score.log 2>&1
env B=$S python $SC/instr_analyze.py av2 test > $L/instr_analyze.log 2>&1
env B=$S python $SC/k12_locality.py --analyze > $L/loc_summary.log 2>&1
env B=$S python $SC/k9_tune.py --analyze --dataset av2 --split test --unc_mults 0.75,1,1.25,1.5,2 > $L/k9curve_summary.log 2>&1
python $B/scripts/A_method/k13_ahd_ablation.py --analyze > $L/ahd_abl_summary.log 2>&1
echo "$(date +%T) K13 CHAIN DONE" > $L/chain_done.log
