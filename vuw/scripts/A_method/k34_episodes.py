"""K34-1(a,c): severity and insufficient braking, stored frozen actions only."""
import argparse
import numpy as np
import pandas as pd
from rap import planner as P
from k31_episodes import runs, unit_stats, DSL
from k34_common import *

SEV = ['below_hard','at_or_above_hard']
SHORT = ['zero','(0,1)','[1,3.5)','[3.5,8)','[8,inf)']

def shortbin(x):
    return SHORT[0 if x==0 else 1 if x<1 else 2 if x<3.5 else 3 if x<8 else 4]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);args=ap.parse_args();ds=args.dataset
    od=OUT/'episodes'/ds;od.mkdir(parents=True,exist_ok=True)
    manifest=[];d=load_base(ds,manifest); pp,cp=P.PlannerParams(),P.CostParams()
    units,C=bootstrap(d.unit); n=len(units); checks=[];eps=[]
    cmds=np.array([P.commanded_decel(a,pp) for a in range(3)])
    for name,col in pipes(ds).items():
        costs=np.array([P.decision_cost(int(a),gt,None if pd.isna(pv) else int(pv),pp,cp)['J']
                        for a,gt,pv in zip(d[col],d.a_gt,d.prev_cheap)])
        delta=float(np.max(np.abs(costs-d['J'+col[1:]])))
        checks.append({'dataset':ds,'check':'stored_cost','pipeline':name,'max_abs_error':delta})
        assert delta<1e-5,(name,delta)
    for u,g in d.groupby('unit',sort=True):
        gt=g.a_gt.to_numpy(float);cheap=g.a_cheap.to_numpy(int)
        for ei,(s,l) in enumerate(runs(g.act_gt.to_numpy(int)>=1)):
            peak=float(gt[s:s+l].max());sf=float(np.maximum(0,gt[s:s+l]-cmds[cheap[s:s+l]]).max())
            for name,col in pipes(ds).items():
                a=g[col].to_numpy(int)[s:s+l];hit=np.flatnonzero(a>=1)
                loss=cp.lam_risk*np.maximum(0,gt[s:s+l]-cmds[a])**2
                eps.append({'dataset':ds,'unit':u,'episode':ei,'start_frame':int(g.frame.iloc[s]),'end_frame':int(g.frame.iloc[s+l-1]),
                            'episode_frames':l,'peak_required_mps2':peak,'peak_cheap_shortfall_mps2':sf,
                            'severity':SEV[int(peak>=pp.thr_hard)],'shortfall_bin':shortbin(sf),'pipeline':name,
                            'missed':not len(hit),'delay_frames':int(hit[0]) if len(hit) else l,
                            'insufficient_loss':float(loss.sum())})
    e=pd.DataFrame(eps);e.to_csv(od/'episodes.csv',index=False)
    L=d.groupby('unit').J_cheap.sum().reindex(units).to_numpy(); summary=[];ints=[];contrasts=[];distributions={}
    strata=[('all','all',np.ones(len(e),bool))]
    strata += [('severity',s,e.severity==s) for s in SEV]
    strata += [('shortfall',b,e.shortfall_bin==b) for b in SHORT]
    strata += [('severity_x_shortfall',s+' / '+b,(e.severity==s)&(e.shortfall_bin==b)) for s in SEV for b in SHORT]
    for family,label,mask in strata:
        for name in pipes(ds):
            q=e[mask&(e.pipeline==name)].copy()
            q['det_delay']=q.delay_frames*(~q.missed);q['detected']=~q.missed
            q['missed_loss']=q.insufficient_loss*q.missed;q['detected_loss']=q.insufficient_loss*(~q.missed)
            group=q.groupby('unit'); agg=group[['missed','delay_frames','det_delay','detected','insufficient_loss','missed_loss','detected_loss']].sum().reindex(units,fill_value=0)
            ne=group.size().reindex(units,fill_value=0).to_numpy(float)
            numer={'missed_rate':agg.missed.to_numpy(),'mean_onset_delay_s':DT[ds]*agg.delay_frames.to_numpy(),
                   'detected_only_onset_delay_s':DT[ds]*agg.det_delay.to_numpy(),
                   'loss_share_cheap_total':agg.insufficient_loss.to_numpy(),'missed_loss_share_cheap_total':agg.missed_loss.to_numpy(),
                   'detected_loss_share_cheap_total':agg.detected_loss.to_numpy()}
            point={};dist={}
            for metric,num in numer.items():
                den=L if 'loss_share' in metric else agg.detected.to_numpy() if metric.startswith('detected_only') else ne
                point[metric]=float(ratio(num.sum(),den.sum()));dist[metric]=ratio(C@num,C@den)
                ints.append({'dataset':ds,'stratification':family,'stratum':label,'pipeline':name,'metric':metric,
                             'estimate':point[metric],**interval(dist[metric])})
            summary.append({'dataset':ds,'stratification':family,'stratum':label,'pipeline':name,'episodes':len(q),
                            'contributing_units':q.unit.nunique(),'all_units':n,'episode_frames':int(q.episode_frames.sum()),
                            'missed_episodes':int(q.missed.sum()),'raw_insufficient_loss':q.insufficient_loss.sum(),
                            'cheap_total_decision_loss':L.sum(),**point})
            distributions[(family,label,name)]=(point,dist)
    for family,label,_ in strata:
        p0,b0=distributions[(family,label,'v6-S')]
        for name in pipes(ds):
            if name=='v6-S':continue
            p1,b1=distributions[(family,label,name)]
            for metric in p0:
                contrasts.append({'dataset':ds,'stratification':family,'stratum':label,'comparator':name,'metric':metric,
                                  'v6S_minus':p0[metric]-p1[metric],**interval(b0[metric]-b1[metric])})
    summ=pd.DataFrame(summary)
    old=pd.read_csv(B/'results/k31/tables/ep_summary.csv')
    for name in pipes(ds):
        q=summ[(summ.stratification=='all')&(summ.pipeline==name)].iloc[0]
        oldq=old[(old.dataset==DSL[ds])&(old.level=='required (>= DECELERATE)')&(old.pipeline==name)].iloc[0]
        assert q.episodes==oldq.episodes and np.isclose(q.missed_rate,oldq.missed_rate)
        assert np.isclose(q.mean_onset_delay_s,oldq.mean_delay_s)
        cross=summ[(summ.stratification=='severity_x_shortfall')&(summ.pipeline==name)]
        assert cross.episodes.sum()==q.episodes and np.isclose(cross.raw_insufficient_loss.sum(),q.raw_insufficient_loss)
        checks.append({'dataset':ds,'check':'K31_episodes_and_cross_table','pipeline':name,'max_abs_error':0})
    loss={name:cp.lam_risk*np.maximum(0,d.a_gt.to_numpy()-cmds[d[col].to_numpy(int)])**2 for name,col in pipes(ds).items()}
    dif=loss['v6-S']-loss['native + C1 + v3a'];sums=pd.Series(dif).groupby(d.unit).sum().reindex(units).to_numpy()
    nf=d.groupby('unit').size().reindex(units).to_numpy(); out=[]
    for metric,den in [('mean_insufficient_loss_per_frame',nf),('loss_share_cheap_total',L)]:
        out.append({'dataset':ds,'metric':metric,'frames':len(d),'units':n,'v6S_sum':loss['v6-S'].sum(),
                    'native_sum':loss['native + C1 + v3a'].sum(),'v6S_minus':float(ratio(sums.sum(),den.sum())),
                    **interval(ratio(C@sums,C@den))})
    for filename,rows in [('severity_summary',summary),('severity_intervals',ints),('severity_contrasts',contrasts),
                          ('loss_vs_native',out),('validation',checks),('source_manifest',manifest)]:
        pd.DataFrame(rows).to_csv(od/(filename+'.csv'),index=False)
    done(od/'DONE.json',dataset=ds,frames=len(d),code_sha256=sha(__file__))
    print('K34 EPISODES DONE',ds,len(d),flush=True)

if __name__=='__main__':main()
