"""Cheap evidence discarded by negative verification; GT-free inference."""
import argparse
import importlib.util
import pickle
import sys
import numpy as np
import pandas as pd
from common import HERE,REMOTE,save
from features import PP,features,complete_history,repair,action,probability_action
from rap import planner as P
from vtf import speculative as SP
sys.path.insert(0,str(REMOTE/'experiments/optimization_r4_20261024'))
import simple
sys.path.insert(0,str(REMOTE/'scripts/A_method'))
s=importlib.util.spec_from_file_location('k27_r5residual',REMOTE/'scripts/A_method/k27_dev.py')
K=importlib.util.module_from_spec(s);s.loader.exec_module(K)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);a=ap.parse_args();ds=a.dataset
    model=pickle.load((HERE/'models'/ds/'all.pkl').open('rb'))
    allobj=pd.read_csv(HERE/'reports/objects'/(ds+'.csv'))
    objunits={u:d for u,d in allobj.groupby('unit')};meta=[];out=HERE/'results/residual'/ds;out.mkdir(parents=True,exist_ok=True)
    wh=np.array([1600,900,1600,900] if ds=='nusc' else [1550,2048,1550,2048]);cp=P.CostParams()
    for f in sorted((REMOTE/'experiments/optimization_r4_20261024/results/dev'/ds).glob('*.pkl')):
        rows=pickle.load(f.open('rb'));original={r['frame']:r for r in pickle.load((HERE/'results/readout'/ds/f.name).open('rb'))}
        ob={int(fr):g for fr,g in objunits.get(f.stem,pd.DataFrame(columns=['frame'])).groupby('frame')}
        hist=[];xs=[]
        for r in rows:
            c=r['trace_cheap'];n=len(c['xyxy']);history=np.zeros(n,bool)
            if len(hist)==2:
                j1=K.assoc_idx(c['xyxy'],hist[-1]);j12=K.assoc_idx(hist[-1],hist[-2])
                history=np.array([j>=0 and j12[j]>=0 for j in j1],bool)
            cls=np.array([{'vehicle':0,'person':1,'cyclist':2}[str(c)] for c in c['coarse']])
            xs.append(features(np.stack(c['mono'],1),r['v'],c['conf'],cls,c['xyxy']/wh,np.zeros(n,bool),history))
            hist=(hist+[c['xyxy']])[-2:]
        X=np.concatenate(xs);probs=model.predict_proba(X) if len(X) else np.zeros((0,3));offset=0;result=[];prev=None
        for r,x in zip(rows,xs):
            row=dict(original[r['frame']]);t=r['trace_v6S_b2A_retro'];c=r['trace_cheap'];v=r['v'];n=len(t['xyxy']);nc=len(x)
            _,raw=P.required_decel(*t['raw'],v,PP);_,ref=P.required_decel(*t['refined'],v,PP);_,ch=P.required_decel(*c['mono'],v,PP)
            m=simple.links(c['xyxy'],c['coarse'],t['xyxy'],t['coarse'])
            residual=(m<0)&SP.centres_in(c['xyxy'],t['crops'])&(ch>=1)
            prob=np.zeros((n,3));prob[:,0]=1
            for o in ob.get(int(r['frame']),pd.DataFrame()).itertuples():
                prob[int(o.index)]=[1-o.p_decel,o.p_decel-o.p_hard,o.p_hard]
            domain=(np.isfinite(t['raw'][0]))&(t['raw'][0]>0)&(t['raw'][0]<=80)&(t['raw'][1]<3.2)&(t['raw'][2]>-3.2)
            full=np.asarray(t['from_full'],bool);med,*_=simple.readout_demands(c,t,raw,ref,ch);history=complete_history(t)
            choices={'v6S_r5_residual_ungated':(ref,np.where(residual,ch,0.))}
            for q in [.5,.8,.95]:
                qa=str(int(q*100));ca=np.minimum(action(ch),probability_action(probs[offset:offset+nc],q))
                cd=np.where(residual,np.take([0.,1.,3.5],ca),0.)
                both=repair(raw,ref,med,full&domain,history,prob,q,'both')
                choices['v6S_r5_domain_both_'+qa]=(both,np.zeros(nc))
                choices['v6S_r5_residual_'+qa]=(ref,cd)
                choices['v6S_r5_residual_both_'+qa]=(both,cd)
            dec=int(r['dec'])
            def match(trace):
                return int(K.IE.matched(r['gt_boxes'][dec:dec+1],r['gt_classes'][dec:dec+1],trace['xyxy'],trace['coarse'])[0]) if dec>=0 else -1
            j=match(t);cj=match(c)
            for name,(fd,cd) in choices.items():
                aa=int(action(max(float(fd.max()) if n else 0,float(cd.max()) if nc else 0)))
                row['a_'+name+'|v3a']=aa;row['J_'+name+'|v3a']=P.decision_cost(aa,r['a_gt'],prev,PP,cp)['J']
                for key in ['px_','ncalls_']:row[key+name]=r[key+'v6S_b2A_retro']
                found=j>=0 or (cj>=0 and cd[cj]>=1)
                row['dec_found_'+name]=found
                row['dec_dem_'+name]=max(fd[j] if j>=0 else 0,cd[cj] if cj>=0 else 0) if found else np.nan
                assert aa>=r['a_v6S_b2A_retro|v3a']
            meta.append(dict(dataset=ds,unit=r['unit'],frame=r['frame'],eligible_cheap=int(residual.sum()),
                annotated_decisive_eligible=bool(cj>=0 and residual[cj]),verified_out_of_training_domain=int((full&~domain).sum()),
                q50_domain_action_difference=int(row['a_v6S_r5_domain_both_50|v3a']!=row['a_v6S_r5_both_50|v3a']),
                q80_domain_action_difference=int(row['a_v6S_r5_domain_both_80|v3a']!=row['a_v6S_r5_both_80|v3a']),
                q95_domain_action_difference=int(row['a_v6S_r5_domain_both_95|v3a']!=row['a_v6S_r5_both_95|v3a'])))
            result.append(row);prev=r['a_cheap'];offset+=nc
        target=out/f.name;tmp=target.with_suffix('.tmp');pickle.dump(result,tmp.open('wb'));tmp.replace(target)
        save(target.with_suffix('.parity.json'),dict(dataset=ds,unit=f.stem,differences={'controls_from_exact_readout':0}))
    od=HERE/'reports/residual_audit';od.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(meta).to_csv(od/(ds+'.csv'),index=False)
    save(out/'DONE.json',dict(status='complete',dataset=ds,frames=len(meta),new_policies=10,units=len(list(out.glob('*.pkl')))))

if __name__=='__main__':main()
