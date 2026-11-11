"""Severity, rescue evidence, and unit concentration; all rows, no selection."""
import argparse,pickle,sys
import numpy as np
import pandas as pd
from common import HERE,REMOTE,save
from policies import NAMES,PP
NAMES=NAMES+['track_agree1_r5','track_agree2_r5']
from rap import planner as P
sys.path.insert(0,str(REMOTE/'scripts/A_method'))
from k31_episodes import runs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);a=ap.parse_args();ds=a.dataset;folder=HERE/'reports'/('followup_'+ds)
    rows=[]
    for f in sorted((HERE/'results/followup'/ds).glob('*.pkl')):rows.extend(pickle.load(f.open('rb')))
    d=pd.DataFrame(rows);d=d[d['gt'].astype(bool)].sort_values(['unit','frame'])
    units=sorted(d.unit.unique());n=len(units);idx={u:i for i,u in enumerate(units)}
    C=np.array([np.bincount(x,minlength=n) for x in np.random.default_rng(0).integers(0,n,(2000,n))])
    cmd=np.array([P.commanded_decel(i,PP) for i in range(3)]);cp=P.CostParams();rescue=[];severe=[];ud=[]
    comps=['v6S','v6S_b2A_missing_extent_retro','v6S_r5_point_residual_50']
    for name in NAMES:
        pre='v6S_r6_'+name;col='a_'+pre+'|v3a'
        for comp in comps:
            bcol='a_'+comp+'|v3a';nums={scope:np.zeros((n,5)) for scope in ['required','hard_required']};dens={scope:np.zeros(n) for scope in nums}
            for u,g in d.groupby('unit',sort=True):
                gt=g.act_gt.to_numpy(int);dem=g.a_gt.to_numpy(float);act=g[col].to_numpy(int);base=g[bcol].to_numpy(int)
                loss=cp.lam_risk*np.maximum(0,dem-cmd[act])**2;bloss=cp.lam_risk*np.maximum(0,dem-cmd[base])**2
                add=g['dec_dem_'+pre].fillna(0).to_numpy(float)
                ud.append(dict(dataset=ds,pipeline=name,comparator=comp,unit=u,frames=len(g),loss_delta_sum=float((loss-bloss).sum()),J_delta_sum=float((g['J_'+pre+'|v3a']-g['J_'+comp+'|v3a']).sum())))
                for scope,sel in [('required',gt>=1),('hard_required',gt>=2)]:
                    nums[scope][idx[u]]=[(loss-bloss)[sel].sum(),loss[sel].sum(),((act<gt)&sel).sum()-((base<gt)&sel).sum(),((act>base)&(act>=gt)&sel).sum(),((act>base)&(act>=gt)&(add>=np.where(gt>=2,3.5,1))&sel).sum()]
                    dens[scope][idx[u]]=sel.sum()
                for ei,(s,l) in enumerate(runs(gt>=1)):
                    hit=(act[s:s+l]>=1).any();bhit=(base[s:s+l]>=1).any()
                    if hit==bhit:continue
                    peak=float(dem[s:s+l].max());dd=add[s:s+l];aa=act[s:s+l];gg=gt[s:s+l]
                    rescue.append(dict(dataset=ds,pipeline=name,comparator=comp,unit=u,episode=ei,start_frame=int(g.frame.iloc[s]),end_frame=int(g.frame.iloc[s+l-1]),peak_demand=peak,hard_peak=peak>=3.5,rescued=bool(hit),lost=bool(bhit),
                        decisive_braking_evidence=bool(((aa>=1)&(dd>=1)).any()),adequate_hard_decisive_evidence=bool(((gg>=2)&(aa>=2)&(dd>=3.5)).any())))
            for scope,A in nums.items():
                den=dens[scope];btden=C@den
                for j,metric in enumerate(['loss_delta_per_frame','candidate_loss_per_frame','underaction_rate_delta','adequate_action_gain_rate','adequate_decisive_gain_rate']):
                    values=np.divide(C@A[:,j],btden,out=np.full(2000,np.nan),where=btden!=0);valid=values[np.isfinite(values)]
                    severe.append(dict(dataset=ds,pipeline=name,comparator=comp,scope=scope,metric=metric,estimate=A[:,j].sum()/den.sum() if den.sum() else np.nan,ci_lo=np.percentile(valid,2.5) if len(valid) else np.nan,ci_hi=np.percentile(valid,97.5) if len(valid) else np.nan,frames=int(den.sum()),valid_draws=len(valid)))
    pd.DataFrame(rescue).to_csv(folder/'rescued_episodes.csv',index=False);pd.DataFrame(severe).to_csv(folder/'severity.csv',index=False);pd.DataFrame(ud).to_csv(folder/'unit_deltas.csv',index=False)
    concentrations=[];uu=pd.DataFrame(ud)
    for (name,comp),g in uu.groupby(['pipeline','comparator']):
        gain=-g.loss_delta_sum;positive=gain[gain>1e-10];total=positive.sum();top=g.iloc[int(np.argmax(gain.to_numpy()))]
        concentrations.append(dict(dataset=ds,pipeline=name,comparator=comp,improved_units=len(positive),worse_units=int((gain< -1e-10).sum()),largest_gain_unit=top.unit,largest_positive_gain_share=float(positive.max()/total) if total else np.nan,net_loss_reduction=float(gain.sum())))
    pd.DataFrame(concentrations).to_csv(folder/'concentration.csv',index=False)
    pa=[]
    for f in sorted((HERE/'proposals'/ds).glob('*.json')):
        if f.name.startswith('DONE') or '.audit.' in f.name:continue
        import json
        obj=json.loads(f.read_text());aud=json.loads(f.with_name(f.stem+'.audit.json').read_text());pa.append({**obj,**{k:v for k,v in aud.items() if k!='bad'}})
    pd.DataFrame(pa).to_csv(folder/'proposal_audit.csv',index=False)
    # Weighted bootstrap quantiles of unchanged acquisition, once for the parent.
    px=d['px_v6S_b2A_missing_extent_retro'].to_numpy(float);codes=np.array([idx[u] for u in d.unit]);order=np.argsort(px);xx=px[order];ii=codes[order];samples=[]
    for count in C:
        cum=np.cumsum(count[ii]);target=.95*(cum[-1]-1);lo=int(np.floor(target));hi=int(np.ceil(target));f=target-lo
        samples.append((1-f)*xx[np.searchsorted(cum,lo,side='right')]+f*xx[np.searchsorted(cum,hi,side='right')])
    qlo,qhi=np.percentile(samples,[2.5,97.5]);pd.DataFrame([dict(dataset=ds,pipeline=name,p95_network_px=np.percentile(px,95),ci_lo=qlo,ci_hi=qhi,extra_detector_calls=0,extra_detector_pixels=0) for name in NAMES]).to_csv(folder/'detector_p95.csv',index=False)
    save(folder/'audit_DONE.json',dict(status='complete',dataset=ds,bootstrap=2000,seed=0,policies=len(NAMES)))
if __name__=='__main__':main()
