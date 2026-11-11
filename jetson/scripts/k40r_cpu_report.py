#!/usr/bin/env python3
"""Render K40R CPU tables already computed on VUW; no new operating point selection."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KIT = Path(__file__).resolve().parents[1]
SRC = KIT/'results/vuw_summaries/k40r/cpu'
OUT = KIT/'docs/paper/handoff/K40R'
for f in SRC.glob('*'):
    if f.suffix in ('.csv', '.json'):
        shutil.copyfile(f, OUT/f.name)
W = pd.read_csv(SRC/'k40_2_weights.csv')
S = pd.read_csv(SRC/'k40_2_safety.csv')
T = pd.read_csv(SRC/'k40_1_target.csv')
D = pd.read_csv(SRC/'k40_1_decomposition.csv')
P = pd.read_csv(SRC/'loss_parity.csv')
RUN = json.loads((SRC/'run.json').read_text())
ROWS = {'J_v6S|v3a':'DESP', 'J_v6S|none':'DESP no refiner',
        'J_wf512+C1|v3a':'u512 + comp.', 'J_wf640+C1|v3a':'u640 + comp.',
        'tab1.J_wf960|v3a':'u960 + comp.'}
DS = ['nuscenes', 'av2', 'argoverse_hd']
for kind in ('safety', 'G_pixels'):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, ds in zip(axs, DS):
        if kind == 'safety':
            d = S[(S.dataset==ds)&(S.family=='uncapped')&(S.level==1)]
            x,y,xlo,xhi,ylo,yhi = ('false_episodes_per_1000_frames', 'missed_episode_rate',
                'false_episodes_per_1000_frames_lo','false_episodes_per_1000_frames_hi',
                'missed_episode_rate_lo','missed_episode_rate_hi')
        else:
            d = W[(W.dataset==ds)&(W.family=='uncapped')&(W.weight_insufficient==1)]
            x,y,xlo,xhi,ylo,yhi = 'px_mean','G','px_mean_lo','px_mean_hi','G_lo','G_hi'
        ax.scatter(d[x],d[y],c='0.8',s=18,label='other stored rows')
        for row, label in ROWS.items():
            q=d[d.row==row].iloc[0]
            ax.errorbar(q[x],q[y],xerr=[[max(0,q[x]-q[xlo])],[max(0,q[xhi]-q[x])]],
                yerr=[[max(0,q[y]-q[ylo])],[max(0,q[yhi]-q[y])]],fmt='o',capsize=2,label=label)
        ax.set_title(ds); ax.set_xlabel('False episodes / 1,000 frames' if kind=='safety' else 'Mean network pixels / native pixels')
        ax.set_ylabel('Missed required episode share' if kind=='safety' else 'G')
        ax.grid(alpha=.2)
    axs[-1].legend(fontsize=7,loc='best')
    fig.suptitle('Post-hoc; deployed thresholds; 95% unit-bootstrap intervals')
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'k40_2_{kind}.{ext}',dpi=180)
    plt.close(fig)

def ci(r, name, lo, hi):
    return f'{r[name]:.4f} [{r[lo]:.4f}, {r[hi]:.4f}]'

lines = ['# K40-1(a,b), K40-2 — measured CPU results', '',
    'Date: 2026-10-30. Pre-specified post-hoc analysis, original allowed test splits. '
    '2,000 paired unit bootstrap draws, seed 0, percentile 95% intervals. '
    'No holdout; no detector or GPU inference. See PREREG.md and inventory.json.', '',
    'Command on VUW (after `source env.sh`): '
    '`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '
    'python scripts/A_method/k40r_cpu.py`. '
    f'Script md5: `{RUN["md5"]}`. Output source: `$B/results/k40r/cpu/`; '
    'KIT mirror: `results/vuw_summaries/k40r/cpu/`. '
    'Figures rendered by `python3 scripts/k40r_cpu_report.py`.', '',
    f'Loss reconstruction passed all {len(P)} dataset/row checks; maximum absolute error '
    f'{P.max_abs_J_error.max():.3g}. The evaluator uses the same previous cheap action for every row. '
    'Thus agree-frame current-action and temporal-carryover differences are exactly zero. '
    'This is an audited implementation fact, not a general claim about temporal losses.', '',
    '| Dataset | Row | G [95% CI] | Δ vs target [95% CI] | Action agreement |',
    '|---|---|---|---|---|']
for _,r in T.iterrows():
    lines.append(f'| {r.dataset} | {r.row} | {ci(r,"G","G_lo","G_hi")} | '
                 f'{ci(r,"delta_G","delta_lo","delta_hi")} | {r.agreement:.4%} |')
lines += ['', 'AV2 is a counterexample to the premise that no-refiner DESP always has higher G than its target. '
    'Its insufficient-braking contribution is negative (−0.07635), partly offset by less excessive braking '
    '(+0.02918). Argoverse-HD’s positive no-refiner difference is mainly less excessive braking '
    '(+0.13085); only that dataset’s no-refiner total difference excludes zero. '
    'These terms do not identify spurious detections: K40-1(c) requires object/state matching.', '',
    '| Dataset | Insufficient weight | DESP G [95% CI] | Rank / uncapped rows | Δ vs u960 + comp. [95% CI] |',
    '|---|---|---|---|---|']
for ds in DS:
    for w in (1,2,5,10):
        fam=W[(W.dataset==ds)&(W.family=='uncapped')&(W.weight_insufficient==w)]
        o=fam[fam.row=='J_v6S|v3a'].iloc[0]; b=fam[fam.row=='tab1.J_wf960|v3a'].iloc[0]
        lines.append(f'| {ds} | {w} | {ci(o,"G","G_lo","G_hi")} | {int(o.rank_G)}/{len(fam)} | '
                     f'{ci(b,"delta_G","delta_lo","delta_hi")} |')
lines += ['', 'Rank includes all explicitly listed uncapped controls and no-refiner rows; it is not the historical '
    '12×3 comparison count. AV2 DESP loses positive point-estimate G at ×10 and is below u960 + components '
    'with paired intervals excluding zero at ×2, ×5 and ×10 (unadjusted intervals). '
    'All sign reversals, including non-significant ones, remain in k40_2_weights.csv. '
    'The exploratory Holm columns use separate uncapped/nominal-cap × weight families across datasets; '
    'they must not be presented as the original primary family.', '',
    '| Dataset | Row | Required episodes | Missed episodes | Missed share [95% CI] | False episodes / 1000 [95% CI] |',
    '|---|---|---|---|---|---|']
for ds in DS:
    for row,label in ROWS.items():
        r=S[(S.dataset==ds)&(S.row==row)&(S.level==1)&(S.family=='uncapped')].iloc[0]
        lines.append(f'| {ds} | {label} | {r.episodes} | {r.missed} | '
          f'{ci(r,"missed_episode_rate","missed_episode_rate_lo","missed_episode_rate_hi")} | '
          f'{ci(r,"false_episodes_per_1000_frames","false_episodes_per_1000_frames_lo","false_episodes_per_1000_frames_hi")} |')
lines += ['', 'Original deployed thresholds are shown above; frame metrics, HARD-BRAKE episodes and paired differences '
    'are in the safety CSVs. The complete stored threshold sweep and original dev-selected operating points '
    'are separate CSVs, including infeasible dev constraints. No matched-miss operating point is invented. '
    'Point Pareto membership is descriptive, not statistical dominance. Latency/energy use different frame '
    'subsets and will be reported separately with their protocol.', '',
    'K40-1(c), valid matched-mean routing, tube variants and remaining K41/LAT tasks are still open.']
(OUT/'CPU_LOSS_SAFETY.md').write_text('\n'.join(lines)+'\n')
print('Rendered CPU_LOSS_SAFETY.md, copied source tables, PDF + PNG figures')
