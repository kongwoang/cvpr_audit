"""Rebuild the local all-row handoff; distinguish completed screen from live inference."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/paper/handoff/OPTIMIZATION_R2_20261023'
MIRROR=ROOT/'results/vuw_summaries/optimization_r2_20261023'

def main():
    s=pd.read_csv(OUT/'screen/summary.csv');c=pd.read_csv(OUT/'screen/contrasts.csv')
    e=pd.read_csv(OUT/'screen/episodes.csv');u=pd.read_csv(OUT/'screen/units.csv')
    baseline=e[e.pipeline=='v6-S'].set_index(['dataset','unit','start_frame'])
    changed=[];concentration=[]
    for name in e.pipeline.unique():
        x=e[e.pipeline==name].set_index(['dataset','unit','start_frame']).join(baseline[['missed']],rsuffix='_baseline')
        for ds,g in x.groupby(level=0):
            changed.append(dict(dataset=ds,pipeline=name,rescued=int((g.missed_baseline&~g.missed).sum()),
                                newly_missed=int((~g.missed_baseline&g.missed).sum())))
    for ds,g in u.groupby('dataset'):
        b=g[g.pipeline=='v6-S'].set_index('unit')
        for name,h in g.groupby('pipeline'):
            h=h.set_index('unit').reindex(b.index);delta=b.loss-h.loss
            gains=delta[delta>1e-12]
            concentration.append(dict(dataset=ds,pipeline=name,loss_improved_units=len(gains),
                loss_worsened_units=int((delta < -1e-12).sum()),net_loss_saved=float(delta.sum()),
                largest_unit_share_of_positive_savings=float(gains.max()/gains.sum()) if len(gains) else np.nan))
    pd.DataFrame(changed).to_csv(OUT/'screen/episode_transitions.csv',index=False)
    pd.DataFrame(concentration).to_csv(OUT/'screen/concentration.csv',index=False)
    rows=['camera anchored','median correction 3','half log correction','looming height 3',
          'looming width-height 3','looming width-height 2 missing']
    fig,axes=plt.subplots(2,3,figsize=(12.8,6.4),sharey=True)
    for di,ds in enumerate(['nusc','av2']):
        for mi,(metric,title,mult) in enumerate([('missed_rate','Missed episodes (percentage points)',100),
                ('false_per_1000','False episodes / 1,000 frames',1),('loss_per_frame','Insufficient-braking loss / frame',1)]):
            ax=axes[di,mi];g=c[(c.dataset==ds)&(c.metric==metric)].set_index('pipeline').reindex(rows)
            yy=np.arange(len(rows));point=g.difference.to_numpy()*mult;lo=g.ci_lo.to_numpy()*mult;hi=g.ci_hi.to_numpy()*mult
            ax.hlines(yy,lo,hi,color='#496b93',lw=2);ax.scatter(point,yy,color='#496b93',s=22,zorder=3)
            ax.axvline(0,color='#555',lw=.8,ls='--');ax.set_yticks(yy);ax.set_yticklabels(rows,fontsize=8)
            ax.invert_yaxis();ax.set_title(('nuScenes' if ds=='nusc' else 'AV2')+' — '+title,fontsize=10)
            ax.grid(axis='x',alpha=.2)
    fig.suptitle('Fixed CPU screen: candidate minus v6-S; pointwise paired unit 95% intervals\n'
        '152 scenes / 300 logs; 2,000 resamples, seed 0. Motion rows keep acquisition frozen.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(OUT/'screen_tradeoffs.png',dpi=180);fig.savefig(OUT/'screen_tradeoffs.pdf');plt.close(fig)
    lines=['# Optimization round 2 — case-led analysis and isolated experiments','',
        '**Completed:** direct review of nine actual three-frame cases and a full 452-unit / 29,782-frame CPU screen (20 rows).',
        '**Scope:** development exploration; main code, weights, thresholds and frozen K34 results unchanged. No method promoted.',
        '**Intervals:** paired unit bootstrap, 2,000, seed 0; pointwise and exploratory.','',
        '## Findings from the images and code','',
        '- A gap in verified observations can remove the C1 chain even when cheap detects the object. Fresh instrumentation also confirms a same-class ID switch in retrospective recovery (GT track 21 linked to 25). Restoring actual history in another case changes TTC from infinity to 6.10 s, still above GT 4.21 s.',
        '- Refined range can be closer to GT but cross the wrong action threshold: the consumer takes the maximum of an ego stopping term and a TTC-derived closing term. See the [measured case figure](range_decision_valley.pdf).',
        '- Episode-level flags must not be assigned to every frame. The cyclist case is matched through its person box at frames 37–40; the missing match occurs at frame 41. The initial visualization used overly strict class equality and was corrected to the frozen person/cyclist compatibility rule; aggregate evaluations were unaffected. A severe crossing truck and clipped edge object additionally expose spatial/visibility limitations.',
        '- Some cheap false alarms are correctly removed. Preserving cheap braking or weakening every correction has a measurable cost.',
        '', 'Evidence and all nine image links: [CASE_REVIEW.md](CASE_REVIEW.md).','',
        '## Full-development CPU screen','',
        'Refiner postprocessing rows are exact on unchanged acquisitions; motion rows are **frozen-query counterfactuals**, requiring fresh inference for actual method claims. No threshold fitting or GT enters a policy.',
        '', '| Dataset | Row | Missed / episodes | False / 1,000 | Insufficient loss / frame | G |',
        '|---|---|---:|---:|---:|---:|']
    for _,r in s.iterrows():
        lines.append(f'| {r.dataset} | {r.pipeline} | {r.missed}/{r.episodes} | {r.false_per_1000:.3f} | {r.loss_per_frame:.5f} | {r.G:.5f} |')
    lines += ['', '**Tradeoff, not a solved weakness:** half-log correction rescues 7 AV2 episodes, with no newly missed baseline episode, but adds 1.733 [0.760, 2.834] false episodes/1,000. Loss/frame falls 0.0420 [0.0326, 0.0523]; G difference +0.0116 [−0.0003, 0.0237] is uncertain. On nuScenes it rescues one episode, increases false alarms, and does not establish a loss/G improvement.',
        '', 'The three-observation width/height floor rescues two AV2 episodes with a very small loss effect; it changes no nuScenes actions. Camera anchoring loses one AV2 episode. Median smoothing rescues two and loses two. These negative and limited results are retained.',
        '', '[All summary rows](screen/summary.csv), [intervals](screen/intervals.csv), [paired contrasts](screen/contrasts.csv), [rescued/newly missed counts](screen/episode_transitions.csv), [unit concentration](screen/concentration.csv), [tradeoff figure](screen_tradeoffs.pdf).',
        '', '## Fresh in-loop experiment','',
        'Five acquisition rows, each with no-refiner/v3a/half-log controls: original v6-S, round-1 combined, proactive completion, proactive + recovery, and verified looming + recovery. Each owns its temporal history. Pending completion keeps one target for two future frames, coalesces existing coverage and admits only actual detections into C1 history. All bounded rows retain the two-batch cap; crop executions/pixels can increase.',
        '', 'Protocol and code: `experiments/optimization_r2_20261023/inloop_protocol.md`, `runner.py`, `proactive.py`, `estimators.py`. Smoke uses the full sequences of the nine diagnostic cases and checks baseline/round-1 parity. The coordinator then runs all 152/300 development units through the unchanged GPU wrapper on VUW/VUW1.']
    if (OUT/'dev/summary.csv').exists():
        d=pd.read_csv(OUT/'dev/summary.csv')
        assert len(d)==32 and set(d.units)=={152,300}
        lines += ['', '**Full in-loop development finished:** all 452 units and 32 rows. [In-loop report](INLOOP_REPORT.md), [summary](dev/summary.csv), [paired contrasts](dev/contrasts.csv).',
            '', '| Dataset | Acquisition (v3a) | Missed / episodes | False / 1,000 | Loss / frame | G | Requested px / native |',
            '|---|---|---:|---:|---:|---:|---:|']
        for _,r in d[d.pipeline.isin(['v6-S','round1 combined','proactive','proactive + recovery','looming + recovery'])].iterrows():
            lines.append(f'| {r.dataset} | {r.pipeline} | {r.missed}/{r.episodes} | {r.false_per_1000:.3f} | {r.insufficient_loss_per_frame:.5f} | {r.G:.5f} | {r.mean_network_px:.5f} |')
    else:
        lines += ['', '**In-loop inference is not yet complete.** [Live coordinator state](STATUS.json). The CPU screen above is completed evidence; queued or running jobs are not results.']
    if (OUT/'ASSOCIATION_REVIEW.md').exists():
        lines += ['', '## Mechanism checks and next decisions','',
            '**Small improvements do not yet justify a method change.** Proactive+recovery saves one more episode per dataset than round-1 combined, but its sole nuScenes rescue brakes because another object has an erroneous TTC. The actual decisive object remains unmatched. On AV2, it adds .423 [.169, .719] false episodes/1,000 and +1.55% requested pixels. The looming addition saves two AV2 episodes, with a tiny loss effect and no nuScenes change. No additional hard-demand episode is rescued by either addition.',
            '', 'A stronger association gate rejects a confirmed wrong identity but loses a useful early recovery: the fresh nine-unit pilot worsens nuScenes and leaves AV2 actions unchanged. See [identity audit and negative pilot](ASSOCIATION_REVIEW.md), [actual-image identity switch](recovery_identity_switch.pdf), and [coincidental rescue](coincidental_rescue.json).']
    if (OUT/'head_audit/reliability.csv').exists():
        lines += ['', '**C1 audit completed:** all 14,696 original validation examples; original MAE reproduced within 3e-7. C1 beats T4 overall. High-GT-rate examples tend to be underestimated, but AV2 is close to calibrated when grouped by its runtime prediction; a blanket upward correction is not supported. The original training pipeline labels current detections without checking the historical GT identity. [Full head review](HEAD_REVIEW.md).',
            '', '**Recommended next experiments:** identity-aware training-data audit, a temporal validity/uncertainty output, and joint range/motion decision-loss calibration; measured-motion queries for low ego speed need their own cost budget. [Feasibility, tradeoffs and falsifiers](NEXT_STEPS.md). These are proposals, not measured improvements.']
    lines += ['', 'No new Jetson timing claimed. New proposals will need actual padded-pixel/launch accounting and an exclusive timing window if their quality/cost tradeoff justifies it. GPU development elapsed time is not Jetson latency.',
        '', 'Verification: all stored baseline actions/costs replay exactly in the CPU screen; semantic tests cover ambiguous matches, stationary/shrinking motion, conservative growth, camera-ray coupling, pending expiry and genuine-chain completion. One initial CPU attempt failed on a pandas column-name access before producing aggregate rows; the failed status is retained as `screen/FAILED_attempt1.json`, corrected and rerun in full.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    MIRROR.mkdir(parents=True,exist_ok=True);manifest={}
    for folder in ('screen','smoke','dev','association','association_inloop','head_audit','training_audit'):
        for path in (OUT/folder).glob('*.csv'):
            if path.name in ('frames.csv','episodes.csv','changed_objects.csv','events.csv') or path.name.endswith('_predictions.csv'):continue
            dest=MIRROR/folder/path.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest)
            manifest[str(dest.relative_to(ROOT))]=hashlib.sha256(dest.read_bytes()).hexdigest()
    (MIRROR/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Rebuilt round-2 report, screen transitions/concentration, figures and summary mirrors')

if __name__=='__main__':main()
