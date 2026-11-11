"""K34-1(b): all threshold points, fixed deployed GT and separate K30 GT sensitivity."""
import argparse
from dataclasses import replace
import numpy as np
import pandas as pd
from rap import planner as P
from k31_episodes import unit_stats
from k34_common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);args=ap.parse_args();ds=args.dataset
    od=OUT/'operating'/ds;od.mkdir(parents=True,exist_ok=True);manifest=[]
    d=load_base(ds,manifest);units,C=bootstrap(d.unit);idx=pd.MultiIndex.from_frame(d[['unit','frame']])
    replay=[];checks=[]
    for f in sorted((OUT/'replay'/ds).glob('*.pkl')):
        q=pickle.load(f.open('rb'));replay+=q['rows'];checks+=q['checks']
        manifest.append({'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size})
    r=pd.DataFrame(replay).set_index(['unit','frame']);assert r.index.is_unique and len(r)==len(d)
    assert set(r.index)==set(idx);r=r.loc[idx].reset_index()
    cons=read_rows(B/'results/k30/cons'/ds,manifest).set_index(['unit','frame']).loc[idx].reset_index()
    assert np.all(cons.act_gt.to_numpy()==d.act_gt.to_numpy())
    points=[];validation=[];pp=P.PlannerParams()
    for name,col in [('cheap','a_cheap'),('v6-S','a_v6S|v3a'),('u960 + C1 + v3a','a_wf960+C1|v3a'),
                     ('native + C1 + v3a',f'a_wf{NWS[ds]}+C1|v3a')]:
        validation.append({'dataset':ds,'check':'K30_cons_vs_K28_base','pipeline':name,'different_frames':int((cons[col]!=d[col]).sum())})
    for label,key in [('cheap','cheap'),('u960 + C1 without refiner','wf960+C1|none'),('native + C1 without refiner',f'wf{NWS[ds]}+C1|none')]:
        for th in (.6,.8,1.,1.2,1.5):
            par=replace(pp,thr_decel=th);a=np.array([P.discrete_action(x,par) for x in r['demand_'+key]])
            points.append((label,th,3.5,'fixed_deployed_GT','CPU_cached_C1_or_cheap_geometry','exact_threshold_independent_scene',a,d.act_gt))
    for label,key in [('u960 + C1 + v3a','wf960+C1'),('native + C1 + v3a',f'wf{NWS[ds]}+C1')]:
        q=pd.DataFrame(checks);nd=int((~q[q.pipeline==key].ref_matches_apdump).sum())
        validation.append({'dataset':ds,'check':'CPU_refiner_vs_K30_apdump_base','pipeline':label,'different_frames':nd})
        validation.append({'dataset':ds,'check':'CPU_no_refiner_vs_K28_base','pipeline':label.replace(' + v3a',' without refiner'),
                           'different_frames':int((np.array([P.discrete_action(x,pp) for x in r['demand_'+key+'|none']])!=d['a_'+key+'|none']).sum())})
        for th in (.6,.8,1.,1.2,1.5):
            points.append((label,th,3.5,'fixed_deployed_GT','CPU_support_refiner_replay',
                           'threshold_specific_support; base_refiner_differences='+str(nd),r[f'a_{key}|v3a@{th:g}'],d.act_gt))
    # Deployed point remains the original K28 action; K30 base is separately identified.
    points.append(('v6-S',1.,3.5,'fixed_deployed_GT','K28_deployed','stored_deployed',d['a_v6S|v3a'],d.act_gt))
    for label,key in [('v6-S','v6S'),('u960 + C1 + v3a','wf960+C1'),('native + C1 + v3a',f'wf{NWS[ds]}+C1')]:
        for th,hard in ((1.,3.5),(.8,2.8),(1.2,4.2)):
            suffix='' if th==1 else f'@tau{th:g}'
            for protocol,gt in [('fixed_deployed_GT',d.act_gt),('K30_threshold_specific_GT',cons['act_gt'+suffix])]:
                points.append((label,th,hard,protocol,'K30_cons'+suffix,'stored_joint_threshold_sensitivity',cons['a_'+key+'|v3a'+suffix],gt))
    output=[];ints=[];dist={};pointvals={};pointmeta={}
    for pi,(name,th,hard,protocol,source,exactness,actions,gt) in enumerate(points):
        key=f'{ds}_p{pi:02d}';q=d[['unit','frame']].copy();q['act_gt']=np.asarray(gt);q['pred']=np.asarray(actions)
        counts=[]
        for u,g in q.groupby('unit',sort=True):
            eps,fa,fr,_,_=unit_stats(g,'pred',1)
            counts.append([len(eps),sum(x[2] for x in eps),fa,fr])
        counts=np.array(counts,float);ne,nm,fa,fr=counts.T
        pv={'false_episodes_per_1000_frames':float(1000*ratio(fa.sum(),fr.sum())), 'missed_rate':float(ratio(nm.sum(),ne.sum()))}
        bd={'false_episodes_per_1000_frames':1000*ratio(C@fa,C@fr),'missed_rate':ratio(C@nm,C@ne)}
        meta={'dataset':ds,'point_id':key,'pipeline':name,'thr_decel_mps2':th,'thr_hard_mps2':hard,'gt_protocol':protocol,
              'source':source,'exactness':exactness,'frames':int(fr.sum()),'units':len(units),'required_episodes':int(ne.sum()),
              'false_episodes':int(fa.sum()),'missed_episodes':int(nm.sum()),'status':'available'}
        output.append({**meta,**pv});dist[key]=bd;pointvals[key]=pv;pointmeta[key]=meta
        for metric in pv:ints.append({**meta,'metric':metric,'estimate':pv[metric],**interval(bd[metric])})
    pairs=[]
    own=[k for k,m in pointmeta.items() if m['pipeline']=='v6-S' and m['gt_protocol']=='fixed_deployed_GT']
    whole=[k for k,m in pointmeta.items() if m['pipeline'].startswith(('u960','native')) and m['gt_protocol']=='fixed_deployed_GT']
    for a in own:
        for b in whole:
            dif={m:pointvals[a][m]-pointvals[b][m] for m in pointvals[a]}
            for metric in dif:
                pairs.append({'dataset':ds,'v6S_point':a,'whole_frame_point':b,'metric':metric,'v6S_minus':dif[metric],
                              'v6S_point_dominates':all(v<=0 for v in dif.values()) and any(v<0 for v in dif.values()),
                              'whole_frame_point_dominates':all(v>=0 for v in dif.values()) and any(v>0 for v in dif.values()),
                              **interval(dist[a][metric]-dist[b][metric])})
    for name,rows in [('operating_points',output),('operating_intervals',ints),('operating_pairwise',pairs),
                     ('validation',validation),('source_manifest',manifest)]:pd.DataFrame(rows).to_csv(od/(name+'.csv'),index=False)
    r.to_csv(od/'continuous_demands.csv',index=False)
    done(od/'DONE.json',dataset=ds,frames=len(d),code_sha256=sha(__file__))
    print('K34 OPERATING DONE',ds,len(output),'points',flush=True)

if __name__=='__main__':main()
