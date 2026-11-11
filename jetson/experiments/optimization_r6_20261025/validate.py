"""Causality/observation boundary checks, independent of dev quality labels."""
import argparse
import numpy as np
from common import save,HERE
from policies import apply

def obj(boxes,cls='vehicle',z=10.,confidence=.9):
    b=np.asarray(boxes,float).reshape(-1,4);n=len(b)
    return dict(xyxy=b,coarse=np.array([cls]*n),conf=np.full(n,confidence),mono=(np.full(n,z),np.full(n,-.5),np.full(n,.5),np.full(n,np.inf)))
def scene(d):return dict(**{k:d[k] for k in ('xyxy','coarse','conf')},from_full=np.ones(len(d['xyxy']),bool),raw=d['mono'],refined=d['mono'],crops=[[0,0,200,200]])
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='all');p.add_argument('--shard',default='0/1');a=p.parse_args()
    empty=obj([]);s=scene(empty);current=obj([[30,30,70,100]]);prev=obj([[30,30,70,98]]);prop=obj([[30,30,70,100]],confidence=.15);prop['threshold']=.25
    history=[dict(cheap=prev,scene=scene(prev),v=8.)]
    got,_=apply(current,s,prop,[],8.,.5,.6)
    assert all(np.max(got['track_'+k][1],initial=0)==0 for k in ['observed1','observed2','agree1','agree2']), 'low score must not create unanchored birth'
    got,_=apply(empty,s,prop,history,8.,.5,.6)
    assert got['track_observed1'][1].max()>0
    assert got['track_observed2'][1].max()==0, 'two observations are not a three-frame chain'
    pp=obj([]);pp['threshold']=.25;got,_=apply(empty,s,pp,history,8.,.5,.6)
    assert all(not a.any() and not b.any() for a,b in got.values()), 'no stale box may generate braking'
    mismatch=obj([[30,30,70,100]],cls='person',confidence=.15);mismatch['threshold']=.25
    got,_=apply(empty,s,mismatch,history,8.,.5,.6)
    assert all(not b.any() for _,b in got.values()), 'vehicle history cannot confirm VRU'
    prior=obj([[30,30,70,98]],z=70.);h=[dict(cheap=prior,scene=scene(prior),v=8.)]
    got,_=apply(current,s,prop,h,8.,.5,.6)
    assert got['track_agree1'][1].max()==0, 'existence is not corroborated demand'
    assert got['res_agree1'][0].max()==0
    got,_=apply(current,scene(current),prop,history,8.,.5,.6)
    assert all(not aa.any() and not bb.any() for aa,bb in got.values()), 'already verified object must not be added again'
    save(HERE/'reports/validation.json',dict(status='complete',checks=['no low-score births','no two-frame chain miscount','no stale-object brake','no cross vehicle-VRU identity','existence does not imply demand','no duplicate verified-object additions'],gt_inputs=False))
if __name__=='__main__':main()
