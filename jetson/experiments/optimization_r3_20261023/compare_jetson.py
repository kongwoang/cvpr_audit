"""Compare common-frame VUW/Torch and Jetson/TRT decisions; do not assume parity."""
import json
import argparse
import pickle
import numpy as np
import pandas as pd
from common import HERE,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['point','tradeoff'],default='point');a=ap.parse_args()
    out=HERE/'reports'/('jetson_backend' if a.variant=='point' else 'jetson_tradeoff_backend');out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(HERE/('jetson_'+a.variant+'_frames.csv'));d=d[d.rep==1]
    names={'v6-S':'v6S','round1 combined':'v6S_b2A_retro','learned missing history':'v6S_b2A_missing_retro'}
    stages={name:'learned' for name in names}
    if a.variant=='tradeoff':
        names.update({'compact point fallback':'v6S_b2A_missing_compact_retro','critical extent retention':'v6S_b2A_extent_retro','corroborated upper fallback':'v6S_b2A_missing_corroborated_retro'})
        stages.update({'compact point fallback':'compact','critical extent retention':'dev','corroborated upper fallback':'low'})
    rows=[];summary=[]
    for ds,g in d.groupby('dataset'):
        source={'nusc_val':'nusc','av2_dev':'av2'}[ds];cache={}
        for unit in g.unit.unique():
            for stage in set(stages.values()):
                file=HERE/'results'/stage/source/(unit+'.pkl')
                assert file.exists(),('not finished',file)
                cache[stage,unit]={int(r['frame']):r for r in pickle.load(file.open('rb'))}
        for r in g.itertuples():
            lookup=cache[stages[r.pipeline],r.unit]
            assert int(r.frame) in lookup
            v=lookup[int(r.frame)];key=names[r.pipeline]+'|v3a'
            rows.append(dict(dataset=ds,pipeline=r.pipeline,unit=r.unit,frame=int(r.frame),
                jetson_action=int(r.action),vuw_action=int(v['a_'+key]),
                jetson_gt_action=int(r.act_gt),vuw_gt_action=int(v['act_gt']),
                jetson_gt_demand=float(r.a_gt),vuw_gt_demand=float(v['a_gt']),
                action_equal=int(r.action)==int(v['a_'+key]),
                gt_equal=int(r.act_gt)==int(v['act_gt']) and np.isclose(r.a_gt,v['a_gt'],atol=1e-7,rtol=1e-7)))
    rows=pd.DataFrame(rows);rows.to_csv(out/'common_frames.csv',index=False)
    for (ds,pipe),g in rows.groupby(['dataset','pipeline']):
        summary.append(dict(dataset=ds,pipeline=pipe,common_frames=len(g),units=g.unit.nunique(),
            differing_actions=int((~g.action_equal).sum()),differing_gt=int((~g.gt_equal).sum()),
            jetson_higher_actions=int((g.jetson_action>g.vuw_action).sum()),jetson_lower_actions=int((g.jetson_action<g.vuw_action).sum())))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    lines=['# R3 cross-backend common-frame check','',
        '512 timed frames per dataset; first repetition only. Original timeline/history prefixes retained in both implementations.',
        'VUW Torch detector and requested crop canvases differ from Jetson TensorRT detector/padded canvases. This measures differences, not equivalence.',
        'The within-Jetson port/encoder checks and three-repetition decision check are separate. Ground-truth equality is checked explicitly.','',
        '| '+' | '.join(summary[0])+' |','| '+' | '.join(['---']*len(summary[0]))+' |',
        *['| '+' | '.join(str(v) for v in r.values())+' |' for r in summary],'',
        'Use full development VUW results for quality and Jetson runs for runtime feasibility. Changes in this prefix cannot be attributed solely to the new readout if baseline backend decisions also differ.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(out/'DONE.json',dict(status='complete',rows=len(rows),gt_differences=int((~rows.gt_equal).sum())))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
if __name__=='__main__':main()
