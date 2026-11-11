"""Frozen conditional gates on round1 and prior point+extent final scenes."""
import argparse
import importlib.util
import pickle
import subprocess
import sys
import numpy as np
import pandas as pd
from common import HERE,REMOTE,save
from features import PP,complete_history,repair,action,probability_action
from conditional import trace_input
from rap import planner as P
sys.path.insert(0,str(REMOTE/'scripts/A_method'))
s=importlib.util.spec_from_file_location('k27_r5conditional',REMOTE/'scripts/A_method/k27_dev.py');K=importlib.util.module_from_spec(s);s.loader.exec_module(K)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);a=ap.parse_args();ds=a.dataset;folder=HERE/'conditional_models'/ds
    models={name:pickle.load((folder/(name+'.pkl')).open('rb')) for name in ['pooled','conditional'] if (folder/(name+'.pkl')).exists()}
    assert models,'No feasible conditional model'
    out=HERE/'results/residual'/ds;cp=P.CostParams();obj=[];frames=[]
    for f in sorted(out.glob('*.pkl')):
        rows=pickle.load(f.open('rb'));originals={}
        for parent,source,prefix in [('round1','optimization_r4_20261024/results/dev','v6S_b2A_retro'),
                                     ('point','optimization_r3_20261023/results/low','v6S_b2A_missing_extent_retro')]:
            old=pickle.load((REMOTE/'experiments'/source/ds/f.name).open('rb'));assert len(old)==len(rows)
            xs=[trace_input(r['trace_'+prefix],r['v'],ds) for r in old];X=np.concatenate(xs)
            probs={k:m.predict_proba(X) if len(X) else np.zeros((0,3)) for k,m in models.items()}
            offset=0;prev=None
            for row,r,x in zip(rows,old,xs):
                assert r['frame']==row['frame'] and r['a_gt']==row['a_gt'];t=r['trace_'+prefix];n=len(x);v=r['v']
                _,raw=P.required_decel(*t['raw'],v,PP);_,ref=P.required_decel(*t['refined'],v,PP)
                assert int(action(ref.max() if n else 0))==row['a_'+prefix+'|v3a']
                domain=(np.isfinite(t['raw'][0]))&(t['raw'][0]>0)&(t['raw'][0]<=80)&(t['raw'][1]<3.2)&(t['raw'][2]>-3.2)
                full=np.asarray(t['from_full'],bool)&domain;history=complete_history(t)
                dec=int(r['dec']);j=int(K.IE.matched(r['gt_boxes'][dec:dec+1],r['gt_classes'][dec:dec+1],t['xyxy'],t['coarse'])[0]) if dec>=0 else -1
                for model in models:
                    prob=probs[model][offset:offset+n]
                    for q in [.5,.8,.95]:
                        name='v6S_r5_cond_'+parent+'_'+model+'_'+str(int(q*100))
                        d=repair(raw,ref,ref,full,history,prob,q,'rollback');aa=int(action(d.max() if n else 0))
                        assert aa>=row['a_'+prefix+'|v3a']
                        row['a_'+name+'|v3a']=aa;row['J_'+name+'|v3a']=P.decision_cost(aa,r['a_gt'],prev,PP,cp)['J']
                        for key in ['px_','ncalls_']:row[key+name]=row[key+prefix]
                        row['dec_found_'+name]=j>=0;row['dec_dem_'+name]=float(d[j]) if j>=0 else np.nan
                    if parent=='round1':
                        for i in np.flatnonzero(full&(action(raw)>action(ref))):
                            obj.append(dict(dataset=ds,unit=r['unit'],frame=r['frame'],index=int(i),model=model,
                                p_decel=float(prob[i,1:].sum()),p_hard=float(prob[i,2]),raw_demand=raw[i],ref_demand=ref[i]))
                prev=r['a_cheap'];offset+=n
        tmp=f.with_suffix('.tmp');pickle.dump(rows,tmp.open('wb'));tmp.replace(f)
    od=HERE/'reports/conditional'/ds;od.mkdir(parents=True,exist_ok=True)
    objects=pd.DataFrame(obj);truth=pd.read_csv(HERE/'reports/objects'/(ds+'.csv'))[['unit','frame','index','gt_object_action','gt_object_demand','annotation_matched']]
    objects=objects.merge(truth,on=['unit','frame','index'],validate='many_to_one');objects.to_csv(od/'disagreement_predictions.csv',index=False)
    cal=[]
    for model,g in objects.groupby('model'):
        for q in [.5,.8,.95]:
            for level,col in [(1,'p_decel'),(2,'p_hard')]:
                pred=g[col]>=q;true=g.gt_object_action>=level;tp=int((pred&true).sum())
                cal.append(dict(dataset=ds,model=model,q=q,level=level,samples=len(g),predicted=int(pred.sum()),required=int(true.sum()),correct=tp,
                    precision=tp/pred.sum() if pred.any() else np.nan,recall=tp/true.sum() if true.any() else np.nan))
    pd.DataFrame(cal).to_csv(od/'dev_disagreement_thresholds.csv',index=False)
    report='final_'+ds
    subprocess.run([sys.executable,str(HERE/'analyze.py'),'--stage','residual','--dataset-only',ds,'--out-name',report],check=True)
    subprocess.run([sys.executable,str(HERE/'attribution.py'),'--stage','residual','--report',report],check=True)
    save(od/'DONE.json',dict(status='complete',dataset=ds,units=len(list(out.glob('*.pkl'))),models=list(models),all_q_reported=True))

if __name__=='__main__':main()
