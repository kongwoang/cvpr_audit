"""GT-free bounded observation rules. Only current observed boxes may add action."""
import sys
import numpy as np
from common import REMOTE
from rap import planner as P
from vtf import speculative as SP
sys.path.insert(0,str(REMOTE/'experiments/optimization_r4_20261024'))
from simple import links
PP=P.PlannerParams()
RESIDUAL=['ungated','seen1','seen2','agree1','agree2','low','lowgeom','low_or_seen1','low_and_seen1','low_or_seen2','low_and_seen2','proposal_any']
LOW=['observed1','observed2','agree1','agree2']
NAMES=['res_'+x for x in RESIDUAL]+['track_'+x for x in LOW]+['combo1','combo2','combo1_r5','combo2_r5']
def action(x):return (np.asarray(x)>=1).astype(int)+(np.asarray(x)>=3.5).astype(int)
def demand(t,v):return P.required_decel(*t,v,PP)[1]
def mx(x):return float(np.max(x)) if len(x) else 0.
def acts_as_demands(a):return np.take([0.,1.,3.5],a)
def match(a,b):return links(a['xyxy'],a['coarse'],b['xyxy'],b['coarse'])
def subset(t,ix):return {k:np.asarray(t[k])[ix] for k in ('xyxy','coarse','conf')}
def apply(cheap,scene,proposal,history,v,dt,scale):
    """Return per-current-cheap and per-current-proposal demands, no GT input."""
    nc,nl=len(cheap['xyxy']),len(proposal['xyxy']);cd=demand(cheap['mono'],v);ca=action(cd)
    remaining=(match(cheap,scene)<0)&SP.centres_in(cheap['xyxy'],scene['crops'])&(cd>=1)
    prev1=np.full(nc,-1,int);prev2=np.full(nc,-1,int);pa1=np.zeros(nc,int);pa2=np.zeros(nc,int)
    if history:
        h=history[-1];prev1=match(cheap,h['cheap']);ok=prev1>=0
        pa1[ok]=action(demand(h['cheap']['mono'],h['v']))[prev1[ok]]
        if len(history)>=2:
            h2=history[-2];j12=match(h['cheap'],h2['cheap']);prev2[ok]=j12[prev1[ok]];ok2=prev2>=0
            pa2[ok2]=action(demand(h2['cheap']['mono'],h2['v']))[prev2[ok2]]
    seen1=prev1>=0;seen2=prev2>=0
    # High-quality current final scene gets first claim; only unmatched proposals
    # can corroborate residual cheap objects or continue a lost full observation.
    free=match(proposal,scene)<0
    li=np.flatnonzero(free&(proposal['conf']<proposal['threshold']))
    low=subset(proposal,li);lj=match(cheap,low);lowok=lj>=0
    ai=np.flatnonzero(free);aj=match(cheap,subset(proposal,ai));anyok=aj>=0
    ego=demand(proposal['mono'],v);lega=np.zeros(nc,int)
    lega[lowok]=action(ego[li[lj[lowok]]])
    masks=dict(ungated=np.ones(nc,bool),seen1=seen1,seen2=seen2,agree1=seen1,agree2=seen2,low=lowok,lowgeom=lowok,
        low_or_seen1=lowok|seen1,low_and_seen1=lowok&seen1,low_or_seen2=lowok|seen2,low_and_seen2=lowok&seen2,proposal_any=anyok)
    out={}
    for name,m in masks.items():
        a=ca.copy()
        if name=='agree1':a=np.minimum(a,pa1)
        if name=='agree2':a=np.minimum(np.minimum(a,pa1),pa2)
        if name=='lowgeom':a=np.minimum(a,lega)
        out['res_'+name]=(np.where(remaining&m,acts_as_demands(a),0.),np.zeros(nl))
    full1=full2=None;j1=np.full(nl,-1,int);j2=np.full(nl,-1,int)
    fa1=np.zeros(nl,int);fa2=np.zeros(nl,int);ttc=np.full(nl,np.inf)
    if history:
        hs=history[-1]['scene'];ix=np.flatnonzero(hs['from_full']);full1=subset(hs,ix);j1=match(proposal,full1);ok=j1>=0
        fa1[ok]=action(demand(hs['refined'],history[-1]['v']))[ix[j1[ok]]]
        if len(history)>=2:
            hs2=history[-2]['scene'];ix2=np.flatnonzero(hs2['from_full']);full2=subset(hs2,ix2);j12=match(full1,full2);j2[ok]=j12[j1[ok]];ok2=j2>=0
            fa2[ok2]=action(demand(hs2['refined'],history[-2]['v']))[ix2[j2[ok2]]]
            h=proposal['xyxy'][:,3]-proposal['xyxy'][:,1];h1=np.full(nl,np.nan);h2=np.full(nl,np.nan)
            h1[ok]=full1['xyxy'][j1[ok],3]-full1['xyxy'][j1[ok],1]
            h2[ok2]=full2['xyxy'][j2[ok2],3]-full2['xyxy'][j2[ok2],1]
            with np.errstate(invalid='ignore',divide='ignore'):
                rate=np.median(np.stack([(h1/h2-1)/dt,(h/h1-1)/dt,(h/h2-1)/(2*dt)]),axis=0)
                valid=ok2&(h>1)&(h1>1)&(h2>1)&(rate>1e-3/dt)&((h-h2)*scale>1.)
                ttc[valid]=np.minimum(1/rate[valid],1e3)
    mono=(*proposal['mono'][:3],ttc);la=action(demand(mono,v));lowfree=free&(proposal['conf']<proposal['threshold'])
    for name in LOW:
        allowed=lowfree&(j1>=0)
        if name in ('observed2','agree2'):allowed&=j2>=0
        a=la.copy()
        if name=='agree1':a=np.minimum(a,fa1)
        if name=='agree2':a=np.minimum(np.minimum(a,fa1),fa2)
        out['track_'+name]=(np.zeros(nc),np.where(allowed,acts_as_demands(a),0.))
    for n in (1,2):
        cd1=out['res_low_or_seen1'][0];ld1=out['track_observed'+str(n)][1]
        out['combo'+str(n)]=(cd1.copy(),ld1.copy())
        out['combo'+str(n)+'_r5']=(cd1.copy(),ld1.copy())
    assert set(out)==set(NAMES)
    assert all(len(a)==nc and len(b)==nl for a,b in out.values())
    return out,dict(eligible=int(remaining.sum()),low=int(len(li)),low_cheap=int((remaining&lowok).sum()),seen1=int((remaining&seen1).sum()),seen2=int((remaining&seen2).sum()),low_track1=int((lowfree&(j1>=0)).sum()),low_track2=int((lowfree&(j2>=0)).sum()))
