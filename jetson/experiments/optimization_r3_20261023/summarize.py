"""Refresh reviewable report/figures from complete dataset reports only."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from common import HERE,save

def main():
    out=HERE/'reports/overview';out.mkdir(parents=True,exist_ok=True)
    sources=[];rows=[];contrasts=[]
    for stage in ['dev','learned','calibrated','lift','persist','iddev','low','compact']:
        for ds in ['nusc','av2']:
            if stage=='iddev' and ds=='av2':continue
            options=[HERE/'reports'/stage]
            options+=[HERE/'reports'/('nusc_complete' if stage=='dev' else 'nusc_'+stage+'_complete')] if ds=='nusc' else []
            for folder in options:
                file=folder/'summary.csv'
                if not file.exists():continue
                d=pd.read_csv(file);d=d[d.dataset==ds]
                if not len(d) or (d.units!={'nusc':152,'av2':300}[ds]).any():continue
                d=d.copy();d.insert(0,'suite',stage);rows.append(d)
                c=pd.read_csv(folder/'contrasts.csv');c=c[c.dataset==ds].copy();c.insert(0,'suite',stage);contrasts.append(c)
                sources.append(dict(suite=stage,dataset=ds,report=str(folder),rows=len(d)));break
    if rows:
        d=pd.concat(rows,ignore_index=True);d.to_csv(out/'all_complete_rows.csv',index=False)
        c=pd.concat(contrasts,ignore_index=True);c.to_csv(out/'all_complete_contrasts.csv',index=False)
    else:d=pd.DataFrame();c=pd.DataFrame()
    complete=len(sources)==15
    status='complete' if complete else 'interim'
    lines=['# Mechanism-repair experiments — round 3 ('+status+')',
        '',time.strftime('Updated %Y-%m-%d %H:%M:%S %Z'),'',
        'Main source/weights/thresholds and K34 remain frozen. All changes are sandbox-only.',
        'New policies use training/REF-VAL and CFG-DEV only. Multiple adaptive rounds: exploratory, not confirmatory.',
        'Paired unit bootstrap: 2,000, seed0, pointwise95% CI. G uses the prior cheap-action jerk convention.',
        'Requested detector pixels do not include encoder/head work and are not Jetson latency.','',
        '## Complete development datasets',
        '', 'Every executed row from a completed dataset is listed below, including no-refiner and weaker alternatives.',
        'Full comparisons/intervals: all_complete_rows.csv and all_complete_contrasts.csv. Partial units are excluded from this table.',
        '', '| Suite | Dataset | Pipeline | Missed / episodes | False episodes /1000 | Insufficient loss/frame | G | Detector px/native |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in d.itertuples():lines.append(f'| {r.suite} | {r.dataset} | {r.pipeline} | {r.missed}/{r.episodes} | {r.false_per_1000:.3f} | {r.insufficient_loss_per_frame:.5f} | {r.G:.3f} | {r.mean_network_px:.5f} |')
    lines+=['','## Training-data and model audit','',
        '785 original training/holdout units were regenerated (385 nuScenes,400 AV2), retaining partial and unmatched histories.',
        'Every finite-target original C1 example was uniquely matched to regenerated crops by exact image bytes: 15,360 nuScenes and81,605 AV2.',
        'Known full crop-chain GT identity switches: nuScenes train306/3077 andholdout115/609; AV2 train1210/21861 andholdout263/3973.',
        'These are annotation-matching diagnostics, not manually certified identity errors for every chain. The regenerated current-object matching population includes incomplete chains, unlike the original sampler; some matched targets differ. Alignment CSVs retain all differences.',
        'Masked adapters freeze the original image encoder. Geometric and decision-aware heads, raw/calibrated quantiles, a separate pair-identity classifier and every selection epoch are retained.',
        'Holdouts already influenced model/epoch selection. Quantile calibration is empirical; no conformal/safety guarantee. Model-audit results concern matched objects, not end-to-end coverage or episode performance.','',
        '## Mechanism and tradeoff checks','',
        'On completed nuScenes, point fallback rescued3 episodes and lost0 versus round1; all3 have decisive-object braking evidence. Upper-quantile fallback rescued9 and lost0, but one has no decisive-object evidence.',
        'The crossing-truck upper-quantile rescue brakes at the earlier matched frame but still misses the peak-demand frame: L1 episode rescue is not adequate hard braking.',
        'Auxiliary identity logits initially separated switches poorly. The separate classifier meets the predeclared feasibility gate on nuScenes only; AV2 was not deployed as an identity gate.',
        'Support-only adapter refinement and joint estimation inside the loop are distinct experiments and are both retained.',
        'The low-tradeoff suite tests temporal-evidence-only fallback, corroborated upper estimates and point+extent persistence.','',
        '## Execution and reproducibility','',
        'Authoritative scripts/protocols: experiments/optimization_r3_20261023/. Each worker stores hashes, actual exit status and atomic unit files; only original import-path failures were repaired/retried.',
        'Initial AV2 import failure and a CPU audit duplicate-key error are documented in execution_notes.md; original attempts/logs are preserved.',
        'GPU launches use the unchanged run_when_gpu_free.sh, SHARE=1 MIN_FREE_MB=10000, two workers per group. VUW/VUW1 share one workspace.',
        'Jetson feasibility is separately isolated in experiments/optimization_r3_jetson_20261024, using the prior fixed development manifests and existing engines.',
        'No new test-set selection, main-code promotion or closed-loop safety claim.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if len(d):
        import matplotlib
        matplotlib.use('Agg');import matplotlib.pyplot as plt
        chosen=['cheap','v6-S','round1 combined','learned missing history','upper-quantile missing history','uncertainty query',
                'critical extent retention','corroborated upper fallback','observed-history point fallback','point fallback + extent retention','compact point fallback']
        fig,axs=plt.subplots(1,2,figsize=(13,5))
        for ax,ds in zip(axs,['nusc','av2']):
            sub=d[(d.dataset==ds)&d.pipeline.isin(chosen)].drop_duplicates('pipeline')
            for i,r in enumerate(sub.itertuples()):
                ax.scatter(r.false_per_1000,100*r.all_missed_rate,s=42)
                ax.annotate(r.pipeline,(r.false_per_1000,100*r.all_missed_rate),xytext=(4,4+(i%2)*6),textcoords='offset points',fontsize=7)
            ax.set_title(ds+' — complete development datasets only');ax.set_xlabel('False episodes /1,000 frames');ax.set_ylabel('Missed required episodes (%)');ax.grid(alpha=.2)
        fig.suptitle('R3 development risk / false-alarm tradeoff (lower-left is better)');fig.tight_layout()
        fig.savefig(out/'tradeoff.pdf');fig.savefig(out/'tradeoff.png',dpi=160);plt.close(fig)
    save(out/'MANIFEST.json',dict(status=status,sources=sources,complete_dataset_rows=len(d),updated=time.time()))
    print(status,len(d),'complete-dataset rows',flush=True)
if __name__=='__main__':main()
