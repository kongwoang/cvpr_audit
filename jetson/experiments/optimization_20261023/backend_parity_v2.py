"""Audit all matching frames and explicitly retain legacy non-dev frames.

The initial strict audit found legacy K29 REF-TRAIN logs without dev references.
Verified CFG-DEV and nuScenes must have complete reference coverage; legacy
non-CFG frames are listed, never silently treated as agreement/disagreement.
"""
import json
import csv
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
H=ROOT/'docs/paper/handoff'
OUT=H/'OPTIMIZATION_20261023'


def main():
    while not (H/'VERIFIED_JETSON_20261023/DONE.json').exists():time.sleep(20)
    while (Path.home()/'.jetson_pause').exists():time.sleep(20)
    import numpy as np
    import pandas as pd
    v=pd.read_csv(H/'W1_BOUNDED_20261023/dev/frames.csv').set_index(['dataset','unit','frame']).sort_index()
    roles={r['log']:r['role'] for r in csv.DictReader((ROOT/'experiments/optimization_20261023/current_av2_split.csv').open())}
    mapping={'ours':('v6S','v6-S'),'ours_retro':('v6S_retro','v6-S + recovery'),
             'ours_b2A':('v6S_b2A','B2K-A'),'ours_b2A_retro':('v6S_b2A_retro','B2K-A + recovery')}
    summaries=[];diffs=[];gt=[];effects=[];missing=[]
    for source,experiment in [('legacy_k29','temporal_jetson_20261023'),('verified_cfg','verified_jetson_20261023')]:
        state=json.loads((ROOT/'experiments'/experiment/'logs/state.json').read_text())
        for run in state['runs']:
            if run['smoke']:continue
            ds='nusc' if run['dataset']=='nusc_val' else 'av2'
            d=pd.read_pickle(Path(run['run'])/'frames.pkl').set_index(['unit','frame']).sort_index()
            ref=v.loc[ds].reindex(d.index);available=ref.act_gt.notna().to_numpy()
            assert len(d)==712 and (~d.warm).sum()==512
            if source=='verified_cfg' or ds=='nusc':assert available.all(),(source,ds)
            for i in np.flatnonzero(~available):
                unit,frame=d.index[i];role=roles.get(unit,'UNRECOGNIZED')
                assert role!='CFG-DEV',(unit,frame,'unexpected missing CFG reference')
                missing.append(dict(source=source,dataset=ds,rep=run['rep'],unit=unit,frame=frame,warm=bool(d.warm.iloc[i]),role=role))
            bad_gt=d.act_gt.to_numpy()[available]!=ref.act_gt.to_numpy()[available]
            gt.append(dict(source=source,dataset=ds,rep=run['rep'],expected_chronological_frames=len(d),
                reference_frames=int(available.sum()),missing_references=int((~available).sum()),
                gt_action_differences=int(bad_gt.sum()),
                cheap_action_differences=int((d.a_cheap.to_numpy()[available]!=ref.a_cheap.to_numpy()[available]).sum()),
                max_gt_demand_absolute_difference=float(np.max(np.abs(d.a_gt.to_numpy()[available]-ref.a_gt.to_numpy()[available])))))
            for jet,(vu,label) in mapping.items():
                j=d['a_'+jet+'@trt32'].to_numpy();r=ref['a_'+vu+'|v3a'].to_numpy()
                different=(j!=r)&available
                for population,original_mask in [('chronological',np.ones(len(d),bool)),('timed',~d.warm.to_numpy())]:
                    mask=original_mask&available
                    cc=pd.DataFrame(dict(unit=d.index.get_level_values('unit')[mask],different=different[mask])).groupby('unit').different.agg(['sum','count'])
                    n=len(cc);draws=np.random.default_rng(0).integers(0,n,(2000,n))
                    w=np.array([np.bincount(x,minlength=n) for x in draws],float)
                    boot=1-(w@cc['sum'].to_numpy())/(w@cc['count'].to_numpy())
                    lo,hi=np.percentile(boot,[2.5,97.5])
                    summaries.append(dict(source=source,dataset=ds,rep=run['rep'],pipeline=label,population=population,
                        expected_frames=int(original_mask.sum()),common_frames=int(mask.sum()),
                        missing_references=int((original_mask&~available).sum()),differing_frames=int(different[mask].sum()),
                        agreement=float(np.mean(~different[mask])),ci_lo=lo,ci_hi=hi))
                for i in np.flatnonzero(different):
                    unit,frame=d.index[i]
                    diffs.append(dict(source=source,dataset=ds,rep=run['rep'],pipeline=label,unit=unit,frame=frame,warm=bool(d.warm.iloc[i]),
                        jetson_action=int(j[i]),vuw_action=int(r[i]),gt_jetson=int(d.act_gt.iloc[i]),gt_vuw=int(ref.act_gt.iloc[i]),
                        jetson_retro_calls=int(d['retro_calls_'+jet+'@trt32'].iloc[i]),
                        vuw_retro_calls=int(ref['retro_calls_'+vu].iloc[i])))
                if jet!='ours':
                    mask=(~d.warm.to_numpy())&available
                    effects.append(dict(source=source,dataset=ds,rep=run['rep'],pipeline=label,common_frames=int(mask.sum()),
                        action_effect_differences=int(((j-d['a_ours@trt32'].to_numpy())[mask]!=(r-ref['a_v6S|v3a'].to_numpy())[mask]).sum())))
    pd.DataFrame(summaries).to_csv(OUT/'backend_parity.csv',index=False)
    pd.DataFrame(diffs,columns=['source','dataset','rep','pipeline','unit','frame','warm','jetson_action','vuw_action','gt_jetson','gt_vuw','jetson_retro_calls','vuw_retro_calls']).to_csv(OUT/'backend_disagreements.csv',index=False)
    pd.DataFrame(missing,columns=['source','dataset','rep','unit','frame','warm','role']).to_csv(OUT/'backend_missing_references.csv',index=False)
    pd.DataFrame(gt).to_csv(OUT/'backend_ground_truth.csv',index=False)
    pd.DataFrame(effects).to_csv(OUT/'backend_action_effects.csv',index=False)
    table=pd.DataFrame(summaries);table=table[(table.population=='timed')&(table.rep==1)]
    lines=['# Jetson–VUW decision parity and dataset audit','',
        'All verified CFG-DEV and nuScenes frames have VUW references. Legacy K29 AV2 contains REF-TRAIN logs;',
        'these missing dev references are explicitly retained in backend_missing_references.csv.',
        'Only matching-frame agreement is reported; missing references are never scored as agreement.',
        'Per-repetition 95% unit bootstrap, 2000 seed0; repeated discrepancies are not independent evidence.','',
        '| Source | Dataset | Policy | Common / expected timed frames | Differences, rep 1 | Agreement |',
        '|---|---|---|---:|---:|---:|']
    for r in table.to_dict('records'):
        lines.append(f"| {r['source']} | {r['dataset']} | {r['pipeline']} | {r['common_frames']}/{r['expected_frames']} | {r['differing_frames']} | {100*r['agreement']:.3f}% |")
    lines+=['','Every repetition and every differing frame remain in the CSVs. Ground-truth and cheap-action differences',
        'are reported separately. VUW individual crop canvases, Jetson common padded batches and detector backends',
        'can produce different detections/histories. These counts do not isolate the cause of every disagreement.',
        'Disabled/fast-helper parity within Jetson is not a cross-backend equivalence claim.',
        'The full VUW quality screen has no overlap with the available training/selection unit IDs, and its',
        'sample counts/checkpoint normalization statistics match the frozen training reports exactly.']
    (OUT/'BACKEND_PARITY.md').write_text('\n'.join(lines)+'\n')
    with (OUT/'FINAL_REPORT.md').open('a') as f:f.write('\nCorrected AV2 CFG-DEV timings: [VERIFIED_JETSON_20261023](../VERIFIED_JETSON_20261023/REPORT.md). Actual cross-backend coverage and differences: [BACKEND_PARITY.md](BACKEND_PARITY.md).\n')
    (OUT/'BACKEND_DONE.json').write_text(json.dumps(dict(status='complete',comparison_rows=len(summaries),
        differences=len(diffs),verified_reference_coverage_complete=True,legacy_missing_rows=len(missing)),indent=2)+'\n')


if __name__=='__main__':main()
