#!/usr/bin/env python3
"""K40-3 dev mean-budget diagnostics. Cached router histories are explicitly non-deployment-exact."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from k40r_cpu import interval,delta,read

B=Path(os.environ['B']);OUT=B/'results/k40r/matched';K=B/'results/k33'
s=importlib.util.spec_from_file_location('rc',B/'scripts/A_method/k33_rc.py');RC=importlib.util.module_from_spec(s);s.loader.exec_module(RC)
from analysis.common import Bootstrap

def threshold(scores,cost,target,cheap):
    order=np.argsort(-scores,kind='stable');so=scores[order];co=cost[order];ends=np.r_[np.flatnonzero(np.diff(so)!=0),len(so)-1]
    th=np.r_[np.inf,so[ends]];bud=np.r_[cheap,cheap+np.cumsum(co)[ends]/len(so)]
    i=min(range(len(th)),key=lambda j:(abs(bud[j]-target),bud[j],th[j]))
    return float(th[i]),float(bud[i])

def policy(t,ds):
    import torch
    ck=torch.load(K/'rc'/f'chanakya_ch128B_csc_{ds}.pt',map_location='cpu',weights_only=False)
    n=len(ck['mu']);h=ck['hidden'];net=torch.nn.Sequential(torch.nn.Linear(n,h),torch.nn.ReLU(),torch.nn.Linear(h,h),torch.nn.ReLU(),torch.nn.Linear(h,4))
    net.load_state_dict(ck['state']);net.eval()
    x=RC.feats(t,ck['fs'])
    with torch.no_grad():logits=net(torch.from_numpy((x-ck['mu'])/ck['sd'])).numpy()
    strong=logits[:,1:].argmax(1)+1
    return logits[np.arange(len(t)),strong]-logits[:,0],np.asarray(RC.ACTS)[strong]

def main():
    OUT.mkdir(parents=True,exist_ok=True);verified=set(json.loads((Path(__file__).parent/'k40r_verified_av2.json').read_text())['logs'])
    choices={};audit=[];out=[];missing=[]
    for ds in ['nusc','av2']:
        t=RC.frame_table(ds,'dev');p=RC.recs(K/f'{ds}_pub_dev');v=RC.recs(K/f'{ds}_de_dev')
        p=p[p['gt'].astype(bool)];v=v[v['gt'].astype(bool)]
        before=t.unit.nunique()
        if ds=='av2':
            t=t[t.unit.isin(verified)].reset_index(drop=True);p=p[p.unit.isin(verified)];v=v[v.unit.isin(verified)]
            assert set(t.unit)==verified,(len(set(t.unit)),len(verified))
        key=['unit','frame'];q=t[key].merge(v[key+['px_v6S']],on=key,validate='1:1');target=float(q.px_v6S.mean())
        audit.append(dict(dataset=ds,original_dev_units=before,used_dev_units=t.unit.nunique(),frames=len(t),target_px=target,
          manifest='verified_log_manifest.json (58 disjoint CFG-DEV logs)' if ds=='av2' else 'K33 nuScenes dev scenes'))
        choices[ds]={}
        for label,score in [('BAR','s_bar'),('Router-G','s_dvr')]:
            th,px=threshold(t[score].to_numpy(),t.px_1280.to_numpy(),target,float(t.px_cheap.mean()))
            choices[ds][label]=dict(threshold=th,dev_px=px,target_px=target,score=score)
        sc,sz=policy(t,ds);cost=np.array([t.loc[i,f'px_{s}'] for i,s in enumerate(sz)])
        th,px=threshold(sc,cost,target,float(t.px_cheap.mean()))
        choices[ds]['Chanakya']=dict(threshold=th,dev_px=px,target_px=target,policy='ch128B_csc; preserve argmax among strong arms, gate by strong-minus-cheap logit')
        candidates=[c for c in p if c.startswith('px_ada') and '@' not in c]
        candidates=sorted(candidates)
        best=min(candidates,key=lambda c:(abs(float(p[c].mean())-target),float(p[c].mean()),c))
        choices[ds]['AdaScale']=dict(row=best[3:],dev_px=float(p[best].mean()),target_px=target,candidates=candidates)
    (OUT/'dev_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
    pd.DataFrame(audit).to_csv(OUT/'dev_audit.csv',index=False)
    for ds,name in [('nusc','nuscenes'),('av2','av2'),('ahd','argoverse_hd')]:
        t=RC.frame_table(ds,'test');p=RC.recs(K/f'{ds}_pub_test');p=p[p['gt'].astype(bool)]
        d=read(name);t=d[['unit','frame']].merge(t,on=['unit','frame'],validate='1:1');p=d[['unit','frame']].merge(p,on=['unit','frame'],validate='1:1')
        assert len(t)==len(d) and len(p)==len(d)
        bt=Bootstrap(d.unit);dev=ds if ds!='ahd' else 'av2';L=d.J_cheap.to_numpy();ours=d['J_v6S|v3a'].to_numpy()
        for label in ['AdaScale','BAR','Router-G','Chanakya']:
            c=choices[dev][label]
            if label=='AdaScale':
                row=c['row']
                if 'J_'+row+'|v3a' not in p or 'px_'+row not in p:
                    missing.append(dict(dataset=name,row=label,selected_policy=row,
                        reason='Dev-selected stored policy absent from test records; do not substitute a test-selected policy',
                        dev_px=c['dev_px'],dev_DESP_px=c['target_px']))
                    continue
                J=p['J_'+row+'|v3a'].to_numpy();px=p['px_'+row].to_numpy();history='stored own-row C1 history; chosen among already run policies'
            else:
                if label=='Chanakya':score,sz=policy(t,dev)
                else:score=t[c['score']].to_numpy();sz=np.full(len(t),1280)
                sz=np.where(score>=c['threshold'],sz,0);J,px=RC.apply_sizes(t,sz)
                history='cached continuous strong-arm history; NOT exact routed history; no corrected 36-family claim'
            g,lo,hi,_=interval(bt,L-J,L);pm,pl,ph,_=interval(bt,px,np.ones(len(d)))
            out.append(dict(task='K40-3',dataset=name,row=label,frames=len(d),units=d.unit.nunique(),dev_dataset=dev,
                dev_px=c['dev_px'],dev_DESP_px=c['target_px'],G=g,G_lo=lo,G_hi=hi,px_mean=pm,px_lo=pl,px_hi=ph,
                DESP_px=float(d.px_v6S.mean()),**delta(bt,L,ours,J),history=history))
        print(ds,'cached budget diagnostics complete',flush=True)
    pd.DataFrame(out).to_csv(OUT/'k40_3_cached_mean_budget.csv',index=False)
    pd.DataFrame(missing).to_csv(OUT/'missing_test_policies.csv',index=False)
    (OUT/'run.json').write_text(json.dumps(dict(md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),new_detector=False,new_training=False,
      corrected_primary_family='blocked: own routed histories absent; AdaScale closest stored discrete policy need not exactly match pixels'),indent=2)+'\n')

if __name__=='__main__':main()
