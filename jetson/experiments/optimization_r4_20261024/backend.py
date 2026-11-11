"""Compare common timed development frames with VUW records, CPU on VUW."""
import pickle
import numpy as np
import pandas as pd
from common import HERE,save

LABELS={'v6-S':('dev','v6S'),'round1 combined':('dev','v6S_b2A_retro'),
 'critical extent retention':('dev','v6S_b2A_extent_retro'),
 'objectwise median full':('readout','v6S_b2A_median'),'objectwise median skip':('readout','v6S_b2A_median')}

def main():
    records=[];summary=[]
    for variant in ['jetson','jetson_fast']:
        d=pd.read_csv(HERE/'jetson_exports'/variant/'frames.csv');d=d[d.rep==1]
        cache={}
        for (dataset,pipeline),g in d.groupby(['dataset','pipeline']):
            ds='nusc' if dataset=='nusc_val' else 'av2';stage,prefix=LABELS[pipeline];own=[]
            for row in g.itertuples():
                key=(ds,stage,row.unit)
                if key not in cache:cache[key]={int(r['frame']):r for r in pickle.load((HERE/'results'/stage/ds/(row.unit+'.pkl')).open('rb'))}
                r=cache[key][row.frame];delta=abs(float(r['a_gt'])-row.a_gt)
                assert delta<1e-8,(ds,row.unit,row.frame,delta)
                item=dict(variant=variant,dataset=ds,pipeline=pipeline,unit=row.unit,frame=row.frame,gt_demand_difference=delta,
                    jetson_action=row.action,vuw_action=r['a_'+prefix+'|v3a'],action_difference=row.action!=r['a_'+prefix+'|v3a'])
                own.append(item);records.append(item)
            summary.append(dict(variant=variant,dataset=ds,pipeline=pipeline,frames=len(own),gt_differences=0,
                action_differences=sum(x['action_difference'] for x in own)))
    out=HERE/'reports/jetson_backend';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(out/'frames.csv',index=False);pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    (out/'REPORT.md').write_text('# Cross-backend decision audit\n\nAll common timed frames, repetition 1 (repetition stability audited separately). GT demands must match exactly within 1e-8. Action differences are retained, not silently treated as parity. VUW uses sequential logical crop canvases and PyTorch; Jetson uses common TensorRT canvases and existing fast encoders. Runtime prefix quality cannot replace complete VUW quality.\n')
    save(out/'DONE.json',dict(status='complete',rows=len(summary),gt_differences=0))
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':main()
