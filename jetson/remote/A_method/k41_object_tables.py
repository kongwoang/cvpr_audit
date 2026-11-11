#!/usr/bin/env python3
"""K40-1c and K41-3/4 post-hoc summaries of parity-gated cached object records."""
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from k40r_cpu import interval, delta, read
from k41_cached_objects import NAMES, OUT as OBJ

B=Path(os.environ['B']); OUT=B/'results/k41/object_tables'
sys.path.insert(0,str(B/'results/k40r/vendor'))
from analysis.common import Bootstrap

def est(bt,num,den):
    pt,lo,hi,valid=interval(bt,np.asarray(num,float),np.asarray(den,float))
    return dict(value=pt,lo=lo,hi=hi,valid_draws=valid,numerator=float(np.sum(num)),denominator=float(np.sum(den)))

def bins(x):
    y=np.asarray(x,float);valid=np.isfinite(y)
    q=np.quantile(y[valid],[1/3,2/3]) if valid.any() else [np.nan,np.nan]
    out=np.full(len(y),'missing',dtype=object)
    out[valid&(y<=q[0])]='low';out[valid&(y>q[0])&(y<=q[1])]='middle';out[valid&(y>q[1])]='high'
    return out,q

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    strata=[];cuts=[];local=[];coverage=[];cond=[];tax=[];missing=[]
    for ds,name in NAMES.items():
        parts=[pickle.loads(f.read_bytes()) for f in sorted((OBJ/ds).glob('*.pkl'))]
        F=pd.DataFrame([r for p in parts for r in p['frames']]);O=pd.DataFrame([r for p in parts for r in p['objects']]);T=pd.DataFrame([r for p in parts for r in p['taxonomy']])
        base=read(name);base.unit=base.unit.astype(str)
        missing.append(dict(dataset=name,expected_frames=len(base),extracted_frames=len(F),missing_frames=len(base)-len(F),units_expected=base.unit.nunique(),units_extracted=F.unit.nunique()))
        assert not F.duplicated(['unit','frame']).any()
        d=base.merge(F.drop(columns=['dataset','v']),on=['unit','frame'],how='left',validate='1:1');bt=Bootstrap(d.unit)
        rows=[('DESP','J_v6S|v3a','px_v6S'),('u960 + comp.','tab1.J_wf960|v3a','px_wf960'),
          ('hybrid','J_tsp8@S+C1|v3a','px_tsp8@S+C1'),('uncertainty','J_unc2@S+C1|v3a','px_unc2@S+C1'),
          ('AnA iterated','J_ana05hinfi|v3a','px_ana05hinfi')]
        for feature in ['target_count','corridor_count','support_size','decisive_height','decisive_closing','decisive_range']:
            groups,q=bins(d[feature]);cuts.append(dict(dataset=name,feature=feature,q1=q[0],q2=q[1],valid=int(np.isfinite(d[feature]).sum()),scope='descriptive test quantiles'))
            for group in ['low','middle','high','missing']:
                m=groups==group
                if not m.any():continue
                for label,j,px in rows:
                    for ref in ('v3a','none'):
                        jj=j.replace('|v3a','|'+ref);ac=jj.replace('J_','a_',1)
                        if jj not in d or ac not in d:
                            missing.append(dict(dataset=name,row=jj,reason='missing stored row/action'));continue
                        meta=dict(task='K41-3',dataset=name,feature=feature,stratum=group,row=label,refiner=ref,frames=int(m.sum()),units=d.loc[m,'unit'].nunique())
                        L=d.J_cheap.to_numpy();J=d[jj].to_numpy();A=d[ac].to_numpy();F0=d['a_wf960+C1|none'].to_numpy();ch=d.a_cheap.to_numpy()
                        for metric,n,de in [('G',m*(L-J),m*L),('agreement',m*(A==F0),m),('repair',m*(ch!=F0)*(A==F0),m*(ch!=F0)),('px',m*d[px].to_numpy(),m)]:
                            strata.append(dict(meta,metric=metric,**est(bt,n,de)))
                        ours=d['J_v6S|'+ref].to_numpy();strata.append(dict(meta,metric='DESP_minus_row_G',**est(bt,m*(J-ours),m*L)))
                        if label=='DESP':
                            valid=d.decisive_covered.notna().to_numpy()&m
                            strata.append(dict(meta,metric='target_decisive_coverage',**est(bt,np.where(valid,d.decisive_covered.fillna(False).to_numpy(float),0),valid)))
        # Target object locality. Masked or infinite ranges do not become finite range errors.
        O['height_bin'],q=bins(O.height);cuts.append(dict(dataset=name,feature='target_object_height',q1=q[0],q2=q[1],scope='descriptive all target objects; duplicated identically at two IoU thresholds'))
        O['edge_bin']=np.select([O.edge_ratio.isna(),O.edge_ratio<=.5,O.edge_ratio<=1],['outside','edge<=0.5','edge(0.5,1]'],default='interior>1')
        # Include zero-object units in exactly the same unit bootstrap as frames.
        # Dummy rows have no eligible object denominator at either IoU threshold.
        absent=sorted(set(d.unit)-set(O.unit))
        if absent:
            dummy=pd.DataFrame({'unit':absent,'iou_threshold':0.,'covered':False,
                'relevant':False,'decisive':False})
            O=pd.concat([O,dummy],ignore_index=True)
            for col in ['covered','relevant','decisive']:
                O[col]=O[col].fillna(False).astype(bool)
        ob=Bootstrap(O.unit)
        assert np.array_equal(ob.idx,bt.idx)
        for th in (.5,.7):
            tt=O.iou_threshold==th
            allgroups=[('all','all',tt)]
            for col in ['height_bin','edge_bin','containing_guard','class_name']:
                allgroups.extend((col,str(v),tt&(O[col]==v)) for v in sorted(O[col].unique(),key=str))
            for feature,group,m0 in allgroups:
                m=(m0&O.covered).to_numpy()
                if not m.any():continue
                meta=dict(task='K41-4a',dataset=name,iou_threshold=th,feature=feature,stratum=group,objects=int(m.sum()),units=O.loc[m,'unit'].nunique())
                for metric,col in [('match_rate','matched'),('range_abs_error','range_abs_error'),('range_rel_error','range_rel_error'),
                  ('ttc_abs_error','ttc_abs_error'),('ttc_rel_error','ttc_rel_error'),('ttc_finite_mismatch','ttc_finite_mismatch'),('class_error','class_mismatch')]:
                    vals=pd.to_numeric(O[col],errors='coerce').to_numpy(float);valid=m&np.isfinite(vals)
                    local.append(dict(meta,metric=metric,**est(ob,np.where(valid,vals,0),valid)))
            for feature in ['draft_action','final_rounds']:
                for value in sorted(O[feature].unique()):
                    for kind,sel in [('all_demand_ge_tau1',O.relevant),('decisive',O.decisive)]:
                        m=(tt&(O[feature]==value)&sel).to_numpy()
                        coverage.append(dict(task='K41-4b',dataset=name,feature=feature,stratum=value,objects=kind,
                          round_semantics='final number of rounds, not first covered round',**est(ob,m*O.covered.to_numpy(),m)))
        # Frame conditional probabilities and explicit complete-case denominators.
        fb=Bootstrap(d.unit);dis=(d['a_v6S|none']!=d['a_wf960+C1|none']).to_numpy()
        for th in (.5,.7):
            col=f'locality_violation_{th}';valid=d[col].notna().to_numpy()&d.coverage_miss.notna().to_numpy()
            lv=d[col].fillna(False).to_numpy(bool);cv=d.coverage_miss.fillna(False).to_numpy(bool)
            for label,m in [('locality_violation',valid&lv),('coverage_miss',valid&cv),('neither',valid&~lv&~cv)]:
                cond.append(dict(task='K41-4c',dataset=name,iou_threshold=th,condition=label,complete_frames=int(valid.sum()),**est(fb,m*dis,m)))
        # Taxonomy contributions use the total dataset cheap loss denominator, not disagreement-only loss.
        for setting in ['no_refiner','full_DESP_vs_noref_target']:
            for cat in ['i','ii','iii','iv']:
                z=T[(T.setting==setting)&(T.category==cat)] if len(T) else T
                q=d[['unit','frame','J_cheap']].merge(z.drop(columns=['dataset','cheap_loss'],errors='ignore'),on=['unit','frame'],how='left',validate='1:1')
                m=q.category.notna().to_numpy() if 'category' in q else np.zeros(len(q),bool)
                tb=Bootstrap(q.unit);meta=dict(task='K40-1c',dataset=name,setting=setting,category=cat,frames=int(m.sum()))
                for metric,col in [('DESP_correct','desp_correct'),('target_correct','target_correct'),
                    ('target_decisive_annotation_unmatched','target_decisive_unmatched_to_annotation'),('desp_decisive_annotation_unmatched','desp_decisive_unmatched_to_annotation')]:
                    values=q[col].fillna(0).to_numpy(float) if col in q else np.zeros(len(q))
                    tax.append(dict(meta,metric=metric,**est(tb,values,m)))
                values=q.loss_advantage.fillna(0).to_numpy(float) if 'loss_advantage' in q else np.zeros(len(q))
                tax.append(dict(meta,metric='contribution_delta_G',**est(tb,values,q.J_cheap)))
        print(name,'parity-gated frames',len(F),'/',len(base),flush=True)
    for name,data in [('k41_3_strata',strata),('k41_3_cutoffs',cuts),('k41_4_locality',local),('k41_4_coverage',coverage),('k41_4_conditional',cond),('k40_1c_taxonomy',tax),('missingness',missing)]:
        pd.DataFrame(data).to_csv(OUT/(name+'.csv'),index=False)
    (OUT/'run.json').write_text(json.dumps(dict(md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),source=str(OBJ),draws=2000,seed=0),indent=2)+'\n')

if __name__=='__main__':main()
