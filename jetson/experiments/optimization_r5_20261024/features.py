"""GT-free features and fixed upward-only readout policies."""
import numpy as np
from rap import planner as P

PP = P.PlannerParams()
FEATURES = ['z', 'lat_min', 'lat_max', 'rate', 'v', 'demand', 'ego_demand',
            'confidence', 'vehicle', 'person', 'cyclist', 'x1', 'y1', 'x2', 'y2',
            'width', 'height', 'full', 'complete_history', 'finite_ttc']

def demands(mono, v):
    """Batched equivalent of the frozen consumer, with per-object ego speed."""
    x=np.asarray(mono,float).reshape(-1,4);v=np.broadcast_to(v,(len(x),))
    z,lo,hi,tt=x.T
    ego=v*v/(2*np.maximum(z-PP.standoff-v*PP.reaction_t,.05))
    rel=np.clip(z/np.clip(tt,.1,1e3),0,60)
    close=rel*rel/(2*np.maximum(z-PP.standoff,.05))
    sel=(lo<PP.corridor_half_w)&(hi>-PP.corridor_half_w)&(z<=PP.max_range)&(v>.1)
    return np.where(sel,np.clip(np.maximum(ego,close),0,PP.a_cap),0.)

def action(d):
    return (np.asarray(d)>=PP.thr_decel).astype(int)+(np.asarray(d)>=PP.thr_hard).astype(int)

def features(raw, v, conf, cls, boxes_normalized, full, history):
    raw=np.asarray(raw,float).reshape(-1,4);n=len(raw)
    box=np.asarray(boxes_normalized,float).reshape(-1,4)
    v=np.broadcast_to(v,(n,));full=np.broadcast_to(full,(n,));history=np.broadcast_to(history,(n,))
    rate=np.where(np.isfinite(raw[:,3]),1/np.clip(raw[:,3],.1,1e3),0.)
    ego=raw.copy();ego[:,3]=np.inf
    cl=np.asarray(cls)
    x=np.c_[raw[:,:3],rate,v,demands(raw,v),demands(ego,v),conf,
            cl==0,cl==1,cl==2,box,box[:,2:]-box[:,:2],full,history,np.isfinite(raw[:,3])]
    assert x.shape==(n,len(FEATURES))
    return np.nan_to_num(x,nan=0.,posinf=1e3,neginf=-1e3).astype(np.float32)

def complete_history(trace):
    """Original stored chains plus successful real two-frame retro recovery."""
    n=len(trace['xyxy']);have=np.zeros(n,bool)
    have[np.asarray(trace.get('chain_ids',[]),int)]=True
    events=trace.get('retro_events',[])
    for e1,e2 in zip(events,events[1:]):
        if e1['lag']==1 and e2['lag']==2 and e1['match']>=0 and e2['match']>=0:
            have|=np.all(np.asarray(trace['xyxy'])==np.asarray(e1['target']),axis=1)
    return have

def trace_features(trace, v, dataset):
    wh=np.array([1600,900,1600,900] if dataset=='nusc' else [1550,2048,1550,2048])
    cl=np.array([{'vehicle':0,'person':1,'cyclist':2}[str(c)] for c in trace['coarse']])
    return features(np.stack(trace['raw'],1),v,trace['conf'],cl,
                    np.asarray(trace['xyxy'])/wh,trace['from_full'],complete_history(trace))

def probability_action(prob,q):
    p=np.asarray(prob);out=(p[:,1:].sum(1)>=q).astype(int)
    out[p[:,2]>=q]=2
    return out

def repair(raw, refined, median, full, history, prob, q, mode):
    """Scalar per-object decision evidence; never GT-conditioned."""
    raw=np.asarray(raw);refined=np.asarray(refined);out=refined.copy()
    ra,fa=action(raw),action(refined);pa=probability_action(prob,q)
    full=np.asarray(full,bool)
    if mode in ('rollback','both'):
        # Clamp to the classifier's supported action. Do not accidentally promote
        # to raw HARD merely because DECELERATE was corroborated.
        proposed=np.minimum(ra,pa)
        for act,thr in [(1,PP.thr_decel),(2,PP.thr_hard)]:
            k=full&(proposed>=act)&(fa<act)
            out[k]=np.maximum(out[k],thr)
    if mode in ('missing','both'):
        for act,thr in [(1,PP.thr_decel),(2,PP.thr_hard)]:
            k=full&~np.asarray(history,bool)&(pa>=act)
            out[k]=np.maximum(out[k],thr)
    if mode=='median_gate':
        proposed=np.minimum(action(median),pa)
        for act,thr in [(1,PP.thr_decel),(2,PP.thr_hard)]:
            k=full&(proposed>=act)&(fa<act)
            out[k]=np.maximum(out[k],thr)
    assert np.all(action(out)>=fa)
    return out
