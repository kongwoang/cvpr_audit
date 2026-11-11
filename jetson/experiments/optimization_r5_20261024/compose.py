"""Fixed final-readout composition with already executed extent acquisition."""
import argparse
import importlib.util
import pickle
import subprocess
import sys
import numpy as np
from common import HERE,REMOTE,save
from features import PP,trace_features,complete_history,repair,action
from rap import planner as P
sys.path.insert(0,str(REMOTE/'scripts/A_method'))
s=importlib.util.spec_from_file_location('k27_r5compose',REMOTE/'scripts/A_method/k27_dev.py');K=importlib.util.module_from_spec(s);s.loader.exec_module(K)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);a=ap.parse_args();ds=a.dataset
    model=pickle.load((HERE/'models'/ds/'all.pkl').open('rb'));cp=P.CostParams();prefix='v6S_b2A_extent_retro'
    out=HERE/'results/residual'/ds
    for f in sorted(out.glob('*.pkl')):
        rows=pickle.load(f.open('rb'));src=pickle.load((REMOTE/'experiments/optimization_r4_20261024/results/dev'/ds/f.name).open('rb'))
        assert len(src)==len(rows)
        xs=[trace_features(r['trace_'+prefix],r['v'],ds) for r in src];X=np.concatenate(xs)
        ps=model.predict_proba(X) if len(X) else np.zeros((0,3));offset=0;prev=None
        for row,r,x in zip(rows,src,xs):
            assert row['frame']==r['frame'] and row['a_gt']==r['a_gt']
            t=r['trace_'+prefix];n=len(x);v=r['v'];_,raw=P.required_decel(*t['raw'],v,PP);_,ref=P.required_decel(*t['refined'],v,PP)
            assert int(action(ref.max() if n else 0))==r['a_'+prefix+'|v3a']
            for key in ['a_','J_']:row[key+prefix+'|v3a']=r[key+prefix+'|v3a']
            for key in ['px_','ncalls_','dec_found_','dec_dem_']:
                if key+prefix in r:row[key+prefix]=r[key+prefix]
            domain=(np.isfinite(t['raw'][0]))&(t['raw'][0]>0)&(t['raw'][0]<=80)&(t['raw'][1]<3.2)&(t['raw'][2]>-3.2)
            full=np.asarray(t['from_full'],bool)&domain;history=complete_history(t)
            dec=int(r['dec']);j=int(K.IE.matched(r['gt_boxes'][dec:dec+1],r['gt_classes'][dec:dec+1],t['xyxy'],t['coarse'])[0]) if dec>=0 else -1
            for q in [.5,.8,.95]:
                name='v6S_r5_extent_both_'+str(int(q*100));d=repair(raw,ref,ref,full,history,ps[offset:offset+n],q,'both')
                aa=int(action(d.max() if n else 0));row['a_'+name+'|v3a']=aa
                row['J_'+name+'|v3a']=P.decision_cost(aa,r['a_gt'],prev,PP,cp)['J']
                for key in ['px_','ncalls_']:row[key+name]=r[key+prefix]
                row['dec_found_'+name]=j>=0;row['dec_dem_'+name]=float(d[j]) if j>=0 else np.nan
            prev=r['a_cheap'];offset+=n
        tmp=f.with_suffix('.tmp');pickle.dump(rows,tmp.open('wb'));tmp.replace(f)
    report='final_'+ds
    subprocess.run([sys.executable,str(HERE/'analyze.py'),'--stage','residual','--dataset-only',ds,'--out-name',report],check=True)
    subprocess.run([sys.executable,str(HERE/'attribution.py'),'--stage','residual','--report',report],check=True)
    save(HERE/'reports'/report/'DONE.json',dict(status='complete',dataset=ds,units=len(list(out.glob('*.pkl'))),all_rows_reported=True))

if __name__=='__main__':main()
