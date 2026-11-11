"""Incremental cheap-evidence retention on two fixed prior acquisitions."""
import argparse
import importlib.util
import pickle
import subprocess
import sys
import numpy as np
from common import HERE,REMOTE,save
from features import PP,features,action,probability_action
from rap import planner as P
from vtf import speculative as SP
sys.path.insert(0,str(REMOTE/'experiments/optimization_r4_20261024'))
import simple
sys.path.insert(0,str(REMOTE/'scripts/A_method'))
s=importlib.util.spec_from_file_location('k27_r5augment',REMOTE/'scripts/A_method/k27_dev.py');K=importlib.util.module_from_spec(s);s.loader.exec_module(K)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);a=ap.parse_args();ds=a.dataset
    model=pickle.load((HERE/'models'/ds/'all.pkl').open('rb'));cp=P.CostParams();out=HERE/'results/residual'/ds
    wh=np.array([1600,900,1600,900] if ds=='nusc' else [1550,2048,1550,2048])
    for f in sorted(out.glob('*.pkl')):
        rows=pickle.load(f.open('rb'))
        old=pickle.load((REMOTE/'experiments/optimization_r3_20261023/results/low'/ds/f.name).open('rb'))
        r4=pickle.load((REMOTE/'experiments/optimization_r4_20261024/results/dev'/ds/f.name).open('rb'))
        assert len(rows)==len(old)==len(r4)
        xs=[];hist=[]
        for r in old:
            c=r['trace_cheap'];n=len(c['xyxy']);history=np.zeros(n,bool)
            if len(hist)==2:
                j1=K.assoc_idx(c['xyxy'],hist[-1]);j12=K.assoc_idx(hist[-1],hist[-2]);history=np.array([j>=0 and j12[j]>=0 for j in j1])
            cls=np.array([{'vehicle':0,'person':1,'cyclist':2}[str(c)] for c in c['coarse']])
            xs.append(features(np.stack(c['mono'],1),r['v'],c['conf'],cls,c['xyxy']/wh,np.zeros(n,bool),history))
            hist=(hist+[c['xyxy']])[-2:]
        X=np.concatenate(xs);ps=model.predict_proba(X) if len(X) else np.zeros((0,3));offset=0;prev=None
        for row,r,s4,x in zip(rows,old,r4,xs):
            assert row['frame']==r['frame']==s4['frame']
            assert row['a_gt']==r['a_gt']==s4['a_gt']
            c=r['trace_cheap'];v=r['v'];nc=len(x);_,cd=P.required_decel(*c['mono'],v,PP)
            for label,prefix,src in [('point','v6S_b2A_missing_extent_retro',r),('extent','v6S_b2A_extent_retro',s4)]:
                t=src['trace_'+prefix];n=len(t['xyxy']);_,ref=P.required_decel(*t['refined'],v,PP)
                assert int(action(ref.max() if n else 0))==src['a_'+prefix+'|v3a']
                assert abs(P.decision_cost(src['a_'+prefix+'|v3a'],r['a_gt'],prev,PP,cp)['J']-row['J_'+prefix+'|v3a'])<1e-8
                links=simple.links(c['xyxy'],c['coarse'],t['xyxy'],t['coarse'])
                eligible=(links<0)&SP.centres_in(c['xyxy'],t['crops'])&(cd>=1)
                dec=int(r['dec'])
                def match(trace):
                    return int(K.IE.matched(r['gt_boxes'][dec:dec+1],r['gt_classes'][dec:dec+1],trace['xyxy'],trace['coarse'])[0]) if dec>=0 else -1
                j=match(t);cj=match(c)
                for q in [.5,.8,.95]:
                    name='v6S_r5_'+label+'_residual_'+str(int(q*100))
                    aa=np.minimum(action(cd),probability_action(ps[offset:offset+nc],q));d=np.where(eligible,np.take([0.,1.,3.5],aa),0.)
                    act=int(action(max(float(ref.max()) if n else 0,float(d.max()) if nc else 0)))
                    assert act>=row['a_'+prefix+'|v3a']
                    row['a_'+name+'|v3a']=act;row['J_'+name+'|v3a']=P.decision_cost(act,r['a_gt'],prev,PP,cp)['J']
                    for key in ['px_','ncalls_']:row[key+name]=row[key+prefix]
                    found=j>=0 or (cj>=0 and d[cj]>=1);row['dec_found_'+name]=found
                    row['dec_dem_'+name]=max(ref[j] if j>=0 else 0,d[cj] if cj>=0 else 0) if found else np.nan
            prev=r['a_cheap'];offset+=nc
        tmp=f.with_suffix('.tmp');pickle.dump(rows,tmp.open('wb'));tmp.replace(f)
    report='final_'+ds
    subprocess.run([sys.executable,str(HERE/'analyze.py'),'--stage','residual','--dataset-only',ds,'--out-name',report],check=True)
    subprocess.run([sys.executable,str(HERE/'attribution.py'),'--stage','residual','--report',report],check=True)
    save(HERE/'reports'/report/'augment_DONE.json',dict(status='complete',dataset=ds,units=len(list(out.glob('*.pkl'))),
        original_point_extent_v3a_trace_action_differences=0,original_extent_v3a_trace_action_differences=0,new_policies=6))

if __name__=='__main__':main()
