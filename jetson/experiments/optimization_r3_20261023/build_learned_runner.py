"""Additional learned-head runner; bridge-screen runner stays immutable in flight."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
def once(s,a,b):
    assert s.count(a)==1,(s.count(a),a[:100]);return s.replace(a,b)
def main():
    s=(HERE/'runner.py').read_text()
    s=once(s,'import bridge','import bridge\nfrom inference import Runtime')
    s=once(s,'    hmu, hsd = ckh["fmu"], ckh["fsd"]',
        '    hmu, hsd = ckh["fmu"], ckh["fsd"]\n    runtime=Runtime(ds,head,crop64,K27.assoc_idx)')
    s=once(s,'        st = {}','        st = {}\n        runtime.bank={}')
    s=once(s,'            rec = {}  # recompute controls; do not import outcomes into policies',
        '            runtime.frame(int(fr));learned_maps={}\n            rec = {}  # recompute controls; do not import outcomes into policies')
    marker='''                def f(full):
                    h = dict(original_h)'''
    replacement='''                def apply_adapter(full,h,chn,mono):
                    if not any(tag in name for tag in ('_missing','_allgeo','_alldec','_upper','_query')):return finish(full,mono)
                    objective='decision' if '_alldec' in name else 'geometry'
                    zz=predicted_geometry({'xyxy':full['xyxy'],'coarse':full['coarse']},None,cam.calib,cam.cam_h,dt,cam_to_ego=(R,t))
                    pred,mask=runtime.infer(objective,full,mono,zz,h,chn,st.get('cheap_history',{}),imgs,img,W,H,v,dt)
                    out=[np.asarray(q).copy() for q in mono];known={i for i,_,_ in chn}
                    for i in range(len(full['xyxy'])):
                        if not np.isfinite(pred[i,0]):continue
                        learned_maps.setdefault(name,{})[tuple(full['xyxy'][i])]=pred[i].copy()
                        if ('_missing' in name or '_upper' in name) and i in known:continue
                        rate=float(pred[i,4] if '_upper' in name else pred[i,0])
                        out[3][i]=min(1/rate,1e3) if rate>1e-3/dt else np.inf
                    rec['adapter_items_'+name]=rec.get('adapter_items_'+name,0)+int(np.isfinite(pred[:,0]).sum())
                    rec['adapter_missing_'+name]=rec.get('adapter_missing_'+name,0)+int((mask.sum(1)<3).sum())
                    return tuple(out)

                def f(full):
                    h = dict(original_h)'''
    s=once(s,marker,replacement)
    s=once(s,'                    if not chn:\n                        return finish(full,(z,lo,hi,ttc))',
        '                    if not chn:\n                        return apply_adapter(full,h,chn,(z,lo,hi,ttc))')
    s=once(s,'                    return finish(full,(z,lo,hi,ttc))\n                return f',
        '                    return apply_adapter(full,h,chn,(z,lo,hi,ttc))\n                return f')
    s=once(s,'                if name in ("none", "closed"):',
        '''                if name=='adapt':
                    factor=np.ones(len(tt[0]));sup=sup_fn(scene['mono'])
                    for i in sup:
                        if scene['from_full'][i]:factor[i]=scene.get('_adapter_factors',np.ones(len(tt[0])))[i]
                    return scale_refinement(tt,factor)
                if name in ("none", "closed"):''')
    s=once(s,"                rec['probe_vclose_'+nm]=vc",
        '''                rec['probe_vclose_'+nm]=vc
                if '_query' in nm:
                    chi=st.get('cheap_history',{})
                    chains=KTD.chains(base['xyxy'],chi.get('s1'),chi.get('s2')) if len(imgs)==2 else []
                    zz=predicted_geometry({'xyxy':base['xyxy'],'coarse':base['coarse']},None,cam.calib,cam.cam_h,dt,cam_to_ego=(R,t))
                    pred,mask=runtime.infer('geometry',base,base['mono'],zz,chi,chains,chi,imgs,img,W,H,v,dt,kind='cheap')
                    factor=np.where(np.isfinite(pred[:,1]),pred[:,1],1.)
                    lower=list(scale_refinement(base['mono'],factor));upper=[q.copy() for q in lower]
                    for i in range(len(pred)):
                        if np.isfinite(pred[i,0]):
                            lower[3][i]=1/max(float(pred[i,2]),.001);upper[3][i]=1/max(float(pred[i,4]),.001)
                    _,dl=P.required_decel(*lower,v,PP0);_,du=P.required_decel(*upper,v,PP0)
                    candidates=[i for i in range(len(pred)) if np.isfinite(pred[i,0]) and P.discrete_action(dl[i],PP0)!=P.discrete_action(du[i],PP0)]
                    if candidates:
                        i=max(candidates,key=lambda j:(du[j]-dl[j],-j));seed.append(KL.box_crop_ctx(base['xyxy'][i],W,H,2.))
                    rec['uncertain_query_'+nm]=int(bool(candidates))''')
    s=once(s,'                fl=list(FLOG)',
        '''                fl=list(FLOG)
                adapted=any(tag in nm for tag in ('_allgeo','_alldec','_query'))
                if adapted:
                    estimates=learned_maps.get(nm,{})
                    sc['_adapter_factors']=np.array([estimates[tuple(b)][1] if tuple(b) in estimates else 1. for b in sc['xyxy']])
                    refs=('none','v3a','adapt')''')
    s=once(s,"                tr = sc['mono'] if close else refine(sc,'v3a',brake_sup(PP0))",
        "                tr = sc['mono'] if close else refine(sc,'adapt' if adapted else 'v3a',brake_sup(PP0))")
    s=once(s,"            for nm in ('v6S','v6S_b2A_retro','v6S_b2A_bridge','v6S_b2A_bridge_retro','v6S_b2A_transfer_retro','v6S_b2A_extent_retro','v6S_b2A_motion_retro'):",
        "            for nm in ('v6S','v6S_b2A_retro','v6S_b2A_missing_retro','v6S_b2A_allgeo_retro','v6S_b2A_alldec_retro','v6S_b2A_upper_retro','v6S_b2A_query_retro'):")
    (HERE/'runner_learned.py').write_text(s)
    print('Built learned sandbox runner')
if __name__=='__main__':main()
