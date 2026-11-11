"""Episode-level residual mechanisms on completed dev records; no inference."""
import csv
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

B=Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR')
E=B/'experiments/w1_dev_20261023'
sys.path.insert(0,str(B/'scripts/A_method'))
from k31_episodes import runs


def flags(r):
    verified=bool(r.get('dec_found_v6S',False) and r.get('dec_from_crop_v6S',False))
    chain=bool(r.get('diag_dec_chain_v6S',False))
    return dict(
        decisive_not_found=not bool(r.get('dec_found_v6S',False)),
        decisive_geometrically_uncovered=not bool(r.get('dec_in_crop_v6S',False)),
        decisive_verified_missing_chain=verified and not chain,
        decisive_verified_has_chain=verified and chain,
        decisive_verified_chain_but_infinite_ttc=verified and chain and not math.isfinite(r.get('dec_ttc_v6S',float('inf'))),
        refiner_turns_braking_to_keep=r['a_v6S|none']>=1 and r['a_v6S|v3a']==0,
        raw_loop_not_converged=not bool(r.get('conv_v6S',True)),
        post_refiner_unmet=bool(r.get('diag_post_field_v6S') or r.get('diag_post_support_v6S')),
        recovery_attempted=bool(r.get('retro_attempted_v6S_retro',0)),
        recovery_succeeded=bool(r.get('retro_success_v6S_retro',0)))


def main():
    out=E/'reports/residual_audit';out.mkdir(exist_ok=True)
    episodes=[];framecounts=[];reference=pd.read_csv(E/'reports/dev/episodes.csv')
    for ds,expected in [('nusc',152),('av2',300)]:
        files=sorted((E/'results/dev'/ds).glob('*.pkl'));assert len(files)==expected
        counts=Counter()
        for file in files:
            records=[r for r in pickle.load(file.open('rb')) if r['gt']]
            gt=np.array([r['act_gt'] for r in records])
            for r in records:
                if r['act_gt']>=1 and r['a_v6S|v3a']==0:
                    counts['required_keep_frames']+=1
                    counts.update({k:int(v) for k,v in flags(r).items()})
            for i,(start,length) in enumerate(runs(gt>=1)):
                rr=records[start:start+length]
                f=[flags(r) for r in rr if r['act_gt']>=1]
                row=dict(dataset=ds,unit=file.stem,episode=i,start_frame=rr[0]['frame'],end_frame=rr[-1]['frame'],
                    severity='hard' if max(r['a_gt'] for r in rr)>=3.5 else 'below_hard',
                    baseline_missed=not any(r['a_v6S|v3a']>=1 for r in rr),
                    cheap_missed=not any(r['a_cheap']>=1 for r in rr),
                    raw_missed=not any(r['a_v6S|none']>=1 for r in rr),
                    recovery_missed=not any(r['a_v6S_retro|v3a']>=1 for r in rr),
                    recovery_raw_missed=not any(r['a_v6S_retro|none']>=1 for r in rr))
                row.update({k:any(x[k] for x in f) for k in f[0]})
                episodes.append(row)
        framecounts.append(dict(dataset=ds,units=expected,**counts))
    d=pd.DataFrame(episodes)
    ref=reference[reference.pipeline=='v6-S'].set_index(['dataset','unit','start_frame','end_frame']).sort_index()
    indexed=d.set_index(['dataset','unit','start_frame','end_frame']).sort_index()
    assert indexed.index.equals(ref.index)
    assert np.array_equal(indexed.baseline_missed,ref.missed)
    summary=[];mechanisms=[]
    for ds,g in d.groupby('dataset'):
        summary.append(dict(dataset=ds,episodes=len(g),baseline_missed=int(g.baseline_missed.sum()),
            cheap_missed=int(g.cheap_missed.sum()),recovery_missed=int(g.recovery_missed.sum()),
            missed_by_baseline_but_not_cheap=int((g.baseline_missed&~g.cheap_missed).sum()),
            missed_by_cheap_but_not_baseline=int((g.cheap_missed&~g.baseline_missed).sum()),
            recovered=int((g.baseline_missed&~g.recovery_missed).sum()),
            introduced_by_recovery=int((~g.baseline_missed&g.recovery_missed).sum()),
            refiner_lost_episodes=int((~g.raw_missed&g.baseline_missed).sum()),
            refiner_gained_episodes=int((g.raw_missed&~g.baseline_missed).sum())))
        for population,mask in [('all baseline misses',g.baseline_missed),
                ('baseline misses where cheap detects',g.baseline_missed&~g.cheap_missed),
                ('remaining after recovery',g.recovery_missed)]:
            for key in flags({**{'a_v6S|none':0,'a_v6S|v3a':0}}):
                mechanisms.append(dict(dataset=ds,population=population,mechanism_any_frame=key,
                    episodes=int(mask.sum()),episodes_with_flag=int(g.loc[mask,key].sum())))
    for name,rows in [('episodes',d),('summary',summary),('mechanism_counts',mechanisms),('required_keep_frames',framecounts)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    text=['# Remaining missed episodes — development attribution','',
        'All 152 nuScenes/300 AV2 units. Required episode definitions reproduce the completed screen exactly.',
        'Flags describe the baseline scene and can overlap. A flag in any required frame is not the cause of an episode miss.',
        'GT matching identifies the annotated decisive object in the scene; there is no GT track-ID identity audit.',
        'The remaining-after-recovery population still uses BASELINE scene flags, not unrecorded recovery-scene geometry.','',
        '| Dataset | Episodes | Baseline miss | Recovery miss | Recovered | Introduced | Baseline-only vs cheap | Cheap-only vs baseline |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        text.append(f"| {r['dataset']} | {r['episodes']} | {r['baseline_missed']} | {r['recovery_missed']} | {r['recovered']} | {r['introduced_by_recovery']} | {r['missed_by_baseline_but_not_cheap']} | {r['missed_by_cheap_but_not_baseline']} |")
    text+=['','These counts distinguish net improvement from recovery/harm transitions. No new method is selected.',
        'Detailed overlapping mechanisms: mechanism_counts.csv. Required-frame counts: required_keep_frames.csv.']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    (out/'DONE.json').write_text(json.dumps(dict(status='complete',episodes=len(d)),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
