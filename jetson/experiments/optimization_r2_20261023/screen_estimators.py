"""Full-development CPU estimator screen on frozen query traces."""
import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rap import planner as P
from vtf import speculative as SP
from estimators import unique_links,motion_floors,add_floor,scale_refinement

B=Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR')
HERE=B/'experiments/optimization_r2_20261023'
sys.path.insert(0,str(B/'scripts/A_method'))
from k31_episodes import runs

ROWS=['v6-S','cheap','no refiner','round1 recovery','camera anchored','median correction 3',
      'half log correction','looming height 3','looming width-height 3','looming width-height 2 missing']

def main():
    out=HERE/'screen';out.mkdir(exist_ok=True)
    pp,cp=P.PlannerParams(),P.CostParams();cmd=np.array([P.commanded_decel(i,pp) for i in range(3)])
    metrics=[];frames=[];episodes=[];changes=[]
    state={'status':'running','pid':os.getpid()}
    def save(**kw):
        state.update(kw);(out/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        for ds,count in [('nusc',152),('av2',300)]:
            files=sorted((B/'experiments/w1_dev_20261023/results/dev'/ds).glob('*.pkl'));assert len(files)==count
            for ui,file in enumerate(files):
                if ds=='nusc':
                    from rap.cache import Camera
                    trans=Camera('nuScenes',file.stem).transform('ego')[1]
                else:
                    ext=pd.read_feather(B/'datasets/av2/sensor/train'/file.stem/'calibration/egovehicle_SE3_sensor.feather')
                    e=ext.set_index('sensor_name').loc['ring_front_center'];trans=np.array([e.tx_m,e.ty_m,e.tz_m])
                records=pickle.load(file.open('rb'));history=[];factor_history=[];prev=None;unit=[]
                for record in records:
                    tr=record['trace_v6S'];raw=tuple(np.asarray(x,float) for x in tr['raw'])
                    ref=tuple(np.asarray(x,float) for x in tr['refined']);x=np.asarray(tr['xyxy']);cl=np.asarray(tr['coarse'])
                    full=np.asarray(tr['from_full'],bool);factor=np.ones(len(x));valid=np.isfinite(raw[0])&(raw[0]>0)&np.isfinite(ref[0])
                    factor[valid]=ref[0][valid]/raw[0][valid]
                    support=np.zeros(len(x),bool);support[SP.support(raw,record['v'],pp)]=True;support&=full&valid
                    median=factor.copy()
                    if len(factor_history)==2:
                        p1,p2=factor_history[-1],factor_history[-2]
                        j1=unique_links(x,cl,p1['x'],p1['c']);j2=unique_links(p1['x'],p1['c'],p2['x'],p2['c'])
                        for i in np.flatnonzero(support):
                            j=j1[i];k=j2[j] if j>=0 else -1
                            if k>=0 and p1['support'][j] and p2['support'][k]:
                                median[i]=np.exp(np.median(np.log([factor[i],p1['factor'][j],p2['factor'][k]])))
                    states={'v6-S':ref,'no refiner':raw,'camera anchored':scale_refinement(raw,factor,trans),
                        'median correction 3':scale_refinement(raw,median),
                        'half log correction':scale_refinement(raw,np.sqrt(factor))}
                    h1=history[-1] if history else None;h2=history[-2] if len(history)>1 else None
                    rh,rhw,r2=motion_floors(x,cl,h1,h2,.6 if ds=='nusc' else 960/2048,record['dt'])
                    missing=full.copy();missing[np.asarray(tr['chain_ids'],int)]=False
                    for label,rates,eligible in [('looming height 3',rh,full),('looming width-height 3',rhw,full),
                                                  ('looming width-height 2 missing',r2,missing)]:
                        states[label],mask=add_floor(ref,rates,eligible)
                        if mask.any():changes.append(dict(dataset=ds,unit=file.stem,frame=record['frame'],pipeline=label,
                            changed_objects=int(mask.sum()),gt_action=record['act_gt']))
                    actions={name:P.discrete_action(P.required_decel(*mono,record['v'],pp)[0],pp) for name,mono in states.items()}
                    actions.update(cheap=record['a_cheap'],**{'round1 recovery':record['a_v6S_retro|v3a']})
                    assert actions['v6-S']==record['a_v6S|v3a'] and actions['no refiner']==record['a_v6S|none']
                    if record['gt']:
                        row=dict(dataset=ds,unit=file.stem,frame=record['frame'],gt=record['act_gt'],demand=record['a_gt'],
                                 cheap_cost=record['J_cheap'],dt=record['dt'])
                        for name,a in actions.items():
                            cost=P.decision_cost(a,record['a_gt'],prev,pp,cp)['J']
                            if name=='v6-S':assert abs(cost-record['J_v6S|v3a'])<1e-8
                            row['a_'+name]=a;row['J_'+name]=cost;row['loss_'+name]=max(0,record['a_gt']-cmd[a])**2
                        unit.append(row);frames.append(row)
                    prev=record['a_cheap']
                    history=(history+[(x[full].copy(),cl[full].copy())])[-2:]
                    factor_history=(factor_history+[dict(x=x.copy(),c=cl.copy(),factor=factor.copy(),support=support.copy())])[-2:]
                d=pd.DataFrame(unit);required=runs(d['gt'].to_numpy()>=1)
                for name in ROWS:
                    act=d['a_'+name].to_numpy();gt=d['gt'].to_numpy();miss=delay=hard_n=hard_miss=0
                    for start,length in required:
                        hit=np.flatnonzero(act[start:start+length]>=1);m=not len(hit)
                        dt=float(d.dt.iloc[start]);de=(hit[0] if len(hit) else length)*dt
                        hard=d.demand.iloc[start:start+length].max()>=pp.thr_hard
                        miss+=m;delay+=de;hard_n+=hard;hard_miss+=hard and m
                        episodes.append(dict(dataset=ds,unit=file.stem,pipeline=name,start_frame=int(d.frame.iloc[start]),
                            end_frame=int(d.frame.iloc[start+length-1]),missed=bool(m),delay_s=de,hard=bool(hard)))
                    false=sum((gt[s:s+l]<1).all() for s,l in runs(act>=1))
                    metrics.append(dict(dataset=ds,unit=file.stem,pipeline=name,frames=len(d),episodes=len(required),missed=miss,
                        false=false,onset_delay_sum=delay,hard_episodes=hard_n,hard_missed=hard_miss,
                        loss=d['loss_'+name].sum(),cost=d['J_'+name].sum(),cheap_cost=d.cheap_cost.sum()))
                if ui%25==0:save(dataset=ds,units=ui+1);print(ds,ui+1,'/',count,flush=True)
        u=pd.DataFrame(metrics);summary=[];intervals=[];contrasts=[]
        for ds,d in u.groupby('dataset'):
            units=sorted(d.unit.unique());n=len(units);draws=np.random.default_rng(0).integers(0,n,(2000,n))
            weights=np.array([np.bincount(x,minlength=n) for x in draws],float);points={};boots={}
            for name in ROWS:
                g=d[d.pipeline==name].set_index('unit').reindex(units)
                num={'G':g.cheap_cost-g.cost,'missed_rate':g.missed,'false_per_1000':1000*g.false,
                     'loss_per_frame':g.loss,'onset_delay_s':g.onset_delay_sum,'hard_missed_rate':g.hard_missed}
                den={'G':g.cheap_cost,'missed_rate':g.episodes,'false_per_1000':g.frames,
                     'loss_per_frame':g.frames,'onset_delay_s':g.episodes,'hard_missed_rate':g.hard_episodes}
                points[name]={};boots[name]={}
                for metric in num:
                    nn,dd=np.asarray(num[metric],float),np.asarray(den[metric],float)
                    estimate=nn.sum()/dd.sum();bnum,bden=weights@nn,weights@dd
                    boot=np.divide(bnum,bden,out=np.full(2000,np.nan),where=bden!=0)
                    points[name][metric]=estimate;boots[name][metric]=boot
                    lo,hi=np.nanpercentile(boot,[2.5,97.5]);intervals.append(dict(dataset=ds,pipeline=name,metric=metric,estimate=estimate,ci_lo=lo,ci_hi=hi))
                summary.append(dict(dataset=ds,pipeline=name,units=n,frames=int(g.frames.sum()),episodes=int(g.episodes.sum()),
                                    missed=int(g.missed.sum()),**points[name]))
            for name in ROWS:
                if name=='v6-S':continue
                for metric in points[name]:
                    lo,hi=np.nanpercentile(boots[name][metric]-boots['v6-S'][metric],[2.5,97.5])
                    contrasts.append(dict(dataset=ds,pipeline=name,comparator='v6-S',metric=metric,
                        difference=points[name][metric]-points['v6-S'][metric],ci_lo=lo,ci_hi=hi))
        for name,data in [('summary',summary),('intervals',intervals),('contrasts',contrasts),('units',metrics),
                          ('frames',frames),('episodes',episodes),('changed_objects',changes)]:
            pd.DataFrame(data).to_csv(out/(name+'.csv'),index=False)
        save(status='complete',units=452,frames=len(frames),rows=len(summary),baseline_parity=True)
        print(pd.DataFrame(summary).to_string(index=False),flush=True)
    except BaseException as exc:save(status='failed',error=repr(exc));raise

if __name__=='__main__':main()
