"""Measure Jetson/VUW method parity on every common dev-prefix frame."""
import json
import time
from pathlib import Path

ROOT=Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR/experiments/optimization_20261023/report_bundle')
H=ROOT/'docs/paper/handoff'
OUT=H/'OPTIMIZATION_20261023'


def main():
    while not (OUT/'DONE.json').exists():time.sleep(20)
    # CPU-only reporting is offloaded; the Jetson window is on another host.
    import numpy as np
    import pandas as pd
    v=pd.read_csv(H/'W1_BOUNDED_20261023/dev/frames.csv').set_index(['dataset','unit','frame']).sort_index()
    state=json.loads((ROOT/'experiments/temporal_jetson_20261023/logs/state.json').read_text())
    mapping={'ours':('v6S','v6-S'),'ours_retro':('v6S_retro','v6-S + recovery'),
             'ours_b2A':('v6S_b2A','B2K-A'),'ours_b2A_retro':('v6S_b2A_retro','B2K-A + recovery')}
    summaries=[];disagreements=[];ground_truth=[];effect_rows=[]
    for run in state['runs']:
        if run['smoke']:continue
        ds='nusc' if run['dataset']=='nusc_val' else 'av2'
        d=pd.read_pickle(Path(run['run'])/'frames.pkl').set_index(['unit','frame']).sort_index()
        reference=v.loc[ds].reindex(d.index)
        assert reference['act_gt'].notna().all() and len(d)==712
        gt_bad=d.act_gt.to_numpy()!=reference.act_gt.to_numpy()
        ground_truth.append(dict(dataset=ds,rep=run['rep'],common_frames=len(d),
            gt_action_differences=int(gt_bad.sum()),cheap_action_differences=int((d.a_cheap.to_numpy()!=reference.a_cheap.to_numpy()).sum()),
            max_gt_demand_absolute_difference=float(np.max(np.abs(d.a_gt.to_numpy()-reference.a_gt.to_numpy())))))
        for jet,(vu,label) in mapping.items():
            j=d['a_'+jet+'@trt32'].to_numpy();r=reference['a_'+vu+'|v3a'].to_numpy()
            different=j!=r
            for population,mask in [('all chronological',np.ones(len(d),bool)),('timed',~d.warm.to_numpy())]:
                dd=pd.DataFrame(dict(unit=d.index.get_level_values('unit')[mask],different=different[mask]))
                count=dd.groupby('unit').different.agg(['sum','count']);n=len(count)
                draws=np.random.default_rng(0).integers(0,n,(2000,n))
                weights=np.array([np.bincount(x,minlength=n) for x in draws],float)
                agreement=1-(weights@count['sum'].to_numpy())/(weights@count['count'].to_numpy())
                lo,hi=np.percentile(agreement,[2.5,97.5])
                summaries.append(dict(dataset=ds,rep=run['rep'],pipeline=label,population=population,
                    frames=int(mask.sum()),differing_frames=int(different[mask].sum()),
                    agreement=float(np.mean(~different[mask])),ci_lo=lo,ci_hi=hi))
            for idx in np.flatnonzero(different):
                unit,frame=d.index[idx]
                disagreements.append(dict(dataset=ds,unit=unit,frame=frame,rep=run['rep'],pipeline=label,warm=bool(d.warm.iloc[idx]),
                    jetson_action=int(j[idx]),vuw_action=int(r[idx]),gt_action_jetson=int(d.act_gt.iloc[idx]),
                    gt_action_vuw=int(reference.act_gt.iloc[idx]),
                    jetson_retro_calls=int(d['retro_calls_'+jet+'@trt32'].iloc[idx]),
                    vuw_retro_calls=int(reference.get('retro_calls_'+vu,pd.Series(np.zeros(len(reference)),index=reference.index)).iloc[idx])))
            if jet!='ours':
                jet_effect=j-d['a_ours@trt32'].to_numpy()
                vuw_effect=r-reference['a_v6S|v3a'].to_numpy()
                timed=~d.warm.to_numpy()
                effect_rows.append(dict(dataset=ds,rep=run['rep'],pipeline=label,frames=int(timed.sum()),
                    action_effect_differences=int((jet_effect[timed]!=vuw_effect[timed]).sum())))
    pd.DataFrame(summaries).to_csv(OUT/'backend_parity.csv',index=False)
    columns=['dataset','unit','frame','rep','pipeline','warm','jetson_action','vuw_action','gt_action_jetson','gt_action_vuw','jetson_retro_calls','vuw_retro_calls']
    pd.DataFrame(disagreements,columns=columns).to_csv(OUT/'backend_disagreements.csv',index=False)
    pd.DataFrame(ground_truth).to_csv(OUT/'backend_ground_truth.csv',index=False)
    pd.DataFrame(effect_rows).to_csv(OUT/'backend_action_effects.csv',index=False)
    table=pd.DataFrame(summaries);table=table[(table.population=='timed')&(table.rep==1)]
    lines=['# Jetson–VUW parity on common development frames','',
        'Every timed frame has a VUW reference. Both populations (712 chronological / 512 timed per run) are retained.',
        'Per-repetition agreement intervals use unit bootstrap 2,000 seed0; repeated discrepancies are not independent evidence.','',
        '| Dataset | Pipeline | Timed common frames, rep 1 | Differences, rep 1 | Agreement |',
        '|---|---|---:|---:|---:|']
    for row in table.to_dict('records'):
        lines.append(f"| {row['dataset']} | {row['pipeline']} | {row['frames']} | {row['differing_frames']} | {100*row['agreement']:.3f}% |")
    lines+=['','Every repetition is in backend_parity.csv. Individual disagreements: backend_disagreements.csv.',
        'GT/cheap comparisons: backend_ground_truth.csv. Candidate action-effect differences: backend_action_effects.csv.',
        'VUW uses individual crop canvases; bounded Jetson batches use a common padded canvas. Detector backends',
        'and resulting histories can differ. This audit measures disagreements; it does not isolate their causes.',
        'A zero disabled-port mismatch within Jetson does not imply cross-backend equivalence.',
        'All measurements remain included and no method is chosen from favorable backend outcomes.']
    (OUT/'BACKEND_PARITY.md').write_text('\n'.join(lines)+'\n')
    with (OUT/'FINAL_REPORT.md').open('a') as f:
        f.write('\nActual cross-backend counts and every differing frame: [BACKEND_PARITY.md](BACKEND_PARITY.md).\n')
    (OUT/'BACKEND_DONE.json').write_text(json.dumps(dict(status='complete',timing_runs=6,comparison_rows=len(summaries),
        disagreements=len(disagreements),all_timed_frames_have_reference=True),indent=2)+'\n')


if __name__=='__main__':main()
