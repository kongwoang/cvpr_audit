"""Freeze diagnostic cases from full-development outcomes, before image review."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
H=ROOT/'docs/paper/handoff'
HERE=Path(__file__).resolve().parent

def main():
    ep=pd.read_csv(H/'W1_DEV_20261023/residual_audit/episodes.csv')
    fr=pd.read_csv(H/'W1_BOUNDED_20261023/dev/frames.csv')
    cases=[]
    for ds in ('nusc','av2'):
        episodes=ep[(ep.dataset==ds)&ep.recovery_missed&~ep.cheap_missed]
        used=set()
        for flag in ['refiner_turns_braking_to_keep','decisive_verified_missing_chain','decisive_not_found']:
            eligible=episodes[episodes[flag]].sort_values(['unit','start_frame'])
            chosen=next((r for r in eligible.itertuples() if (r.unit,r.start_frame) not in used),None)
            if chosen is None:continue
            used.add((chosen.unit,chosen.start_frame))
            d=fr[(fr.dataset==ds)&(fr.unit==chosen.unit)&fr.frame.between(chosen.start_frame,chosen.end_frame)]
            relevant=d[(d.act_gt>=1)&(d.a_cheap>=1)]
            if len(relevant):d=relevant
            r=d.sort_values(['a_gt','frame'],ascending=[False,True]).iloc[0]
            cases.append(dict(dataset=ds,unit=chosen.unit,frame=int(r.frame),category=flag,
                              episode_start=int(chosen.start_frame),episode_end=int(chosen.end_frame),
                              required_peak=float(d.a_gt.max()),selection='first unused unit/episode with flag; peak required demand frame'))
        # Control: longest run of cheap braking correctly suppressed when GT is KEEP.
        d=fr[fr.dataset==ds];controls=[]
        for unit,g in d.groupby('unit',sort=True):
            mask=(g.act_gt==0)&(g.a_cheap>=1)&(g['a_v6S|v3a']==0)
            active=[]
            for row,ok in zip(g.itertuples(),mask):
                if ok:active.append(row.frame)
                elif active:
                    controls.append((len(active),unit,active[len(active)//2],active[0],active[-1]));active=[]
            if active:controls.append((len(active),unit,active[len(active)//2],active[0],active[-1]))
        length,unit,frame,start,end=sorted(controls,key=lambda x:(-x[0],x[1],x[2]))[0]
        cases.append(dict(dataset=ds,unit=unit,frame=int(frame),category='correctly_removed_cheap_false_alarm',
                          episode_start=int(start),episode_end=int(end),selection='longest consecutive false-braking run; middle frame'))
    for i,r in enumerate(cases):r['case_id']=f'{i+1:02d}_{r["dataset"]}_{r["category"]}'
    result=dict(cases=cases,selection_scope='diagnostic, outcome-conditioned; not unbiased evaluation',
                source='completed round-1 development CSVs')
    out=HERE/'case_manifest.json'
    assert not out.exists(),'case manifest already frozen'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
