"""Replay all fixed R5 policies on exact R4 final scenes; no acquisition change."""
import argparse
import importlib.util
import pickle
import sys
import numpy as np
import pandas as pd
from common import HERE, REMOTE, save
from features import PP, trace_features, complete_history, repair, action
from rap import planner as P
sys.path.insert(0,str(REMOTE/'experiments/optimization_r4_20261024'))
import simple
sys.path.insert(0,str(REMOTE/'scripts/A_method'))
s=importlib.util.spec_from_file_location('k27_r5readout',REMOTE/'scripts/A_method/k27_dev.py')
K=importlib.util.module_from_spec(s);s.loader.exec_module(K)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);a=ap.parse_args()
    models={k:pickle.load((HERE/'models'/a.dataset/(k+'.pkl')).open('rb')) for k in ['all','matched']}
    out=HERE/'results/readout'/a.dataset;out.mkdir(parents=True,exist_ok=True)
    objout=HERE/'reports/objects';objout.mkdir(parents=True,exist_ok=True);objects=[];stats=[]
    cp=P.CostParams();controls=['v6S|none','v6S|v3a','v6S_b2A_retro|none','v6S_b2A_retro|v3a']
    src=REMOTE/'experiments/optimization_r4_20261024/results/dev'/a.dataset
    expected={p.stem for p in src.glob('*.pkl')}
    generators={u:g for u,cam,g in K.units(a.dataset,0)}
    assert set(generators)==expected
    for f in sorted(src.glob('*.pkl')):
        rows=pickle.load(f.open('rb'));xs=[trace_features(r['trace_v6S_b2A_retro'],r['v'],a.dataset) for r in rows]
        X=np.concatenate(xs);probs={k:m.predict_proba(X) if len(X) else np.zeros((0,3)) for k,m in models.items()}
        # GT is loaded only after predictions, solely for the evaluator below.
        truth={int(fr):(vv,g) for fr,path,vv,g,cached in generators[f.stem]()}
        offset=0;result=[];prev=None
        for r,x in zip(rows,xs):
            row={k:r[k] for k in ['unit','frame','v','dt','gt','a_gt','act_gt','dec','a_cheap','J_cheap','px_cheap']}
            for p in controls:
                for key in ['a_','J_']:row[key+p]=r[key+p]
                prefix=p.split('|')[0]
                for key in ['px_','ncalls_','dec_found_','dec_dem_']:
                    if key+prefix in r:row[key+prefix]=r[key+prefix]
            t=r['trace_v6S_b2A_retro'];c=r['trace_cheap'];v=r['v'];n=len(x)
            vg,g=truth[int(r['frame'])]
            gtmono=tuple(np.asarray(g[k],float) for k in ['long_near','lat_min','lat_max','ttc'])
            gd,go=P.required_decel(*gtmono,vg,PP)
            assert abs(v-vg)<1e-8 and abs(gd-r['a_gt'])<1e-8
            assert len(g)==len(r['gt_boxes'])
            if len(g):assert np.array_equal(np.stack([g[k] for k in ['x1','y1','x2','y2']],1),r['gt_boxes'])
            _,raw=P.required_decel(*t['raw'],v,PP);_,ref=P.required_decel(*t['refined'],v,PP)
            _,ch=P.required_decel(*c['mono'],v,PP)
            assert int(action(raw.max() if n else 0))==r['a_v6S_b2A_retro|none']
            assert int(action(ref.max() if n else 0))==r['a_v6S_b2A_retro|v3a']
            assert abs(P.decision_cost(r['a_v6S_b2A_retro|v3a'],r['a_gt'],prev,PP,cp)['J']-r['J_v6S_b2A_retro|v3a'])<1e-8
            median,*_=simple.readout_demands(c,t,raw,ref,ch)
            choices={'v6S_b2A_median':median};history=complete_history(t)
            for q in [.5,.8,.95]:
                for mode in ['rollback','missing','both','median_gate']:
                    name='v6S_r5_'+mode+'_'+str(int(q*100))
                    choices[name]=repair(raw,ref,median,t['from_full'],history,probs['all'][offset:offset+n],q,mode)
            choices['v6S_r5_matched_both_80']=repair(raw,ref,median,t['from_full'],history,probs['matched'][offset:offset+n],.8,'both')
            dec=int(r['dec']);j=int(K.IE.matched(r['gt_boxes'][dec:dec+1],r['gt_classes'][dec:dec+1],t['xyxy'],t['coarse'])[0]) if dec>=0 else -1
            for name,d in choices.items():
                aa=int(action(d.max() if n else 0));row['a_'+name+'|v3a']=aa
                row['J_'+name+'|v3a']=P.decision_cost(aa,r['a_gt'],prev,PP,cp)['J']
                for key in ['px_','ncalls_']:row[key+name]=r[key+'v6S_b2A_retro']
                row['dec_found_'+name]=j>=0;row['dec_dem_'+name]=float(d[j]) if j>=0 else np.nan
                if name!='v6S_b2A_median':assert aa>=r['a_v6S_b2A_retro|v3a']
            # Per-object annotation matches are used ONLY below for diagnostics.
            matched=K.IE.matched(r['gt_boxes'],r['gt_classes'],t['xyxy'],t['coarse']) if len(r['gt_boxes']) else []
            gt_indices={int(i):gi for gi,i in enumerate(matched) if i>=0}
            for i in range(n):
                if not t['from_full'][i]:continue
                p=probs['all'][offset+i];objects.append(dict(dataset=a.dataset,unit=r['unit'],frame=r['frame'],index=i,
                    annotation_matched=i in gt_indices,decisive=i==j,history=bool(history[i]),finite_ttc=bool(np.isfinite(t['raw'][3][i])),
                    gt_object_demand=float(go[gt_indices[i]]) if i in gt_indices else 0.,
                    gt_object_action=int(action(go[gt_indices[i]])) if i in gt_indices else 0,
                    gt_frame_action=r['act_gt'],gt_frame_demand=r['a_gt'],raw_demand=raw[i],ref_demand=ref[i],median_demand=median[i],
                    p_decel=float(p[1]+p[2]),p_hard=float(p[2]),z=float(t['raw'][0][i]),lmin=float(t['raw'][1][i]),lmax=float(t['raw'][2][i])))
            offset+=n;prev=r['a_cheap'];result.append(row)
        target=out/f.name;tmp=target.with_suffix('.tmp');pickle.dump(result,tmp.open('wb'));tmp.replace(target)
        save(target.with_suffix('.parity.json'),dict(dataset=a.dataset,unit=f.stem,frames=len(result),differences={'raw':0,'v3a':0,'cost':0}))
        stats.append(dict(unit=f.stem,frames=len(result),objects=offset));print(a.dataset,f.stem,'replayed',len(result),flush=True)
    pd.DataFrame(objects).to_csv(objout/(a.dataset+'.csv'),index=False)
    save(out/'DONE.json',dict(status='complete',units=len(stats),frames=sum(x['frames'] for x in stats),objects=sum(x['objects'] for x in stats),
        policies=13,baseline_parity_differences=0,scope='Final readout only; same current detections, raw states, future histories and queries.'))

if __name__=='__main__':main()
