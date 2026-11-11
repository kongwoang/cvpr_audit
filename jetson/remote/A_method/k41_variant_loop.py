"""Isolated experimental request loop; reference merge/guard order, explicit cap semantics.

The frozen method is never replaced. request_fn(scene, crops, grown, round_idx)
supplies ordered requests. Nominal budgets count ordinary crop canvases before
guard; strict budgets count every requested detector canvas including guards.
The strict/strict_observed modes exclude denied ordinary crops from the merge.
k39_strict retains K39's selected-but-denied merge regions explicitly; trace is
the authoritative actually observed union. Skipped guards keep pending boxes.
"""
import numpy as np
from rap.mono import box_iou
from vtf import speculative as SP


def empty():
    return dict(xyxy=np.zeros((0,4)),conf=np.zeros(0),coarse=np.zeros(0,'<U8'))


def guarded(detect,crop,W,H,canvas,budget,used,trace,round_idx):
    def call(c,depth):
        pixels=canvas(c)
        if used[0]+pixels>budget+1e-6:
            trace.append(dict(round=round_idx,guard_depth=depth,crop=np.asarray(c).copy(),pixels=pixels,executed=False))
            return None
        used[0]+=pixels;d=detect(c)
        trace.append(dict(round=round_idx,guard_depth=depth,crop=np.asarray(c).copy(),pixels=pixels,executed=True,
            detections={k:np.asarray(v).copy() for k,v in d.items()}))
        return {k:np.asarray(v) for k,v in d.items()}
    d=call(crop,0)
    if d is None:return empty(),[],False
    if not len(d['conf']):return d,[],True
    top,other=SP.truncated(d['xyxy'],crop,W,H,top_only=True);tr=top|other
    out={k:x[~tr] for k,x in d.items()};extra=[]
    live=[({k:d[k][i:i+1] for k in d},crop) for i in np.flatnonzero(tr)]
    for step in range(2):
        nxt=[]
        for row,c in live:
            b=row['xyxy'][0];g=SP.grow(b,c,W,H);dg=call(g,step+1)
            if dg is None:
                out={k:np.concatenate([out[k],row[k]]) for k in out};continue
            extra.append(g)
            if not len(dg['conf']):continue
            ix=np.clip(np.minimum(dg['xyxy'][:,2],b[2])-np.maximum(dg['xyxy'][:,0],b[0]),0,None)
            iy=np.clip(np.minimum(dg['xyxy'][:,3],b[3])-np.maximum(dg['xyxy'][:,1],b[1]),0,None)
            io=ix*iy/max((b[2]-b[0])*(b[3]-b[1]),1e-6);j=int(np.argmax(io))
            if io[j]<.5:continue
            rj={k:dg[k][j:j+1] for k in dg}
            if not SP.truncated(dg['xyxy'][j:j+1],g,W,H)[0] or step==1:
                out={k:np.concatenate([out[k],rj[k]]) for k in out}
            else:nxt.append((rj,g))
        live=nxt
        if not live:break
    return out,extra,True


def run(cheap,request_fn,detect,lift,W,H,canvas,budget=np.inf,cap='none',max_rounds=3):
    assert cap in ('none','nominal','strict','strict_observed','k39_strict')
    crops,grown,full,trace=[],[],empty(),[];scene=dict(cheap);used=[0.];ordinary=0.;rounds=0;converged=False
    for step in range(max_rounds):
        new=request_fn(scene,crops,grown,step)
        if not new:converged=True;break
        accepted=[]
        selected=[];selected_pixels=ordinary
        for c in new:
            px=canvas(c)
            if cap in ('nominal','strict_observed','k39_strict') and selected_pixels+px>budget+1e-6:
                trace.append(dict(round=step+1,guard_depth=0,crop=np.asarray(c).copy(),pixels=px,executed=False,reason='nominal_selection'))
                continue
            selected.append(c);selected_pixels+=px
        for c in selected:
            px=canvas(c)
            d,ex,executed=guarded(detect,c,W,H,canvas,budget if cap in ('strict','strict_observed','k39_strict') else np.inf,used,trace,step+1)
            if not executed and cap!='k39_strict':continue
            accepted.append(c);ordinary+=px;grown.extend(ex)
            full={k:np.concatenate([full[k],np.asarray(d[k])]) for k in full}
        if not accepted:break
        rounds+=1;crops.extend(accepted)
        if len(full['conf'])>1:
            order=np.argsort(-full['conf'],kind='stable');iou=box_iou(full['xyxy'],full['xyxy']);keep=[]
            for i in order:
                if not keep or iou[i,keep].max()<.5:keep.append(i)
            full={k:full[k][np.sort(np.asarray(keep,int))] for k in full}
        out=SP.centres_in(cheap['xyxy'],crops)
        if grown and len(full['conf']) and len(cheap['xyxy']):out|=box_iou(np.asarray(cheap['xyxy'],float),full['xyxy']).max(1)>=.5
        fg=lift(full) if len(full['conf']) else tuple(np.zeros(0) for _ in range(4))
        scene=dict(xyxy=np.r_[cheap['xyxy'][~out],full['xyxy']],conf=np.r_[cheap['conf'][~out],full['conf']],
            coarse=np.r_[cheap['coarse'][~out],full['coarse']],mono=tuple(np.r_[a[~out],b] for a,b in zip(cheap['mono'],fg)),
            from_full=np.r_[np.zeros(int((~out).sum()),bool),np.ones(len(full['conf']),bool)])
    if not converged:converged=not request_fn(scene,crops,grown,max_rounds)
    scene.setdefault('from_full',np.zeros(len(scene['conf']),bool))
    return scene,crops,grown,rounds,converged,trace
