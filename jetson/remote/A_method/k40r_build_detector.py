"""Build K30-derived K40R tube / K41 grid driver on KIT; no detector execution."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def once(s,a,b):
    assert s.count(a)==1,(s.count(a),a[:100])
    return s.replace(a,b)


PART='''            elif part in ('tubes','grid'):
                from k41_variant_loop import run as variant_run
                from vtf import dsf
                crop6 = loop('v6S',2.0,960,dec=True)
                rec['crops_v6S'] = [np.asarray(c).copy() for c in last['v6S'][0]+last['v6S'][1]]
                rec['scene_v6S'] = last['v6S'][2]
                rec['history_v6S'] = {k:st['v6S'].get(k) for k in ('s1','s2')}
                variants = []
                if part == 'tubes':
                    variants = [('speed_tau1',1.,{}),('speed_tau2',3.5,{})]
                else:
                    import itertools
                    for idx,(rm,lm,pn,vc) in enumerate(itertools.product((.5,1.,2.),(.5,1.,2.),('ped','car','truck'),(0.,5.,10.))):
                        if args.grid_index>=0 and idx!=args.grid_index:continue
                        rg=np.sort(np.r_[SP.R_GRID,(SP.R_GRID[:-1]+SP.R_GRID[1:])/2]) if rm==.5 else (SP.R_GRID if rm==1 else SP.R_GRID[[0,2,4,6,7]])
                        yg=np.arange(-6,6+1e-9,1.5*lm)
                        variants.append((f'grid{idx:02d}',None,dict(r_grid=rg,y_grid=yg,probes=(dict(ped=KL.PED,car=KL.CAR,truck=KL.TRUCK)[pn],),v_closes=(vc,))))
                for label,tau,kwargs in variants:
                    for cap in (('nominal','k39_strict','strict_observed') if part=='tubes' else ('none',)):
                        name=f'{label}@{cap}';tpx=[0.];field_us=[0.]
                        active[0]=name
                        def request(scene,crops,grown,step):
                            import time
                            round_id[0]=step+1
                            start=time.perf_counter()
                            if tau is None:
                                fc=_fcg(scene['mono'],v,cam,W,H,PP0,kwargs['v_closes'],kwargs['r_grid'],kwargs['y_grid'],kwargs['probes'])
                            else:
                                from k40r_tube_geometry import tube as speed_tube
                                fc, tube_info=speed_tube(v,tau,cam,W,H,PP0)
                                rec['tube_'+name]=tube_info
                            new=[]
                            if fc is not None and not SP.covered(fc,crops):new.append(fc)
                            for k in SP.support(scene['mono'],v,PP0):
                                bc=KL.box_crop_ctx(scene['xyxy'][k],W,H,2.)
                                if not SP.covered(bc,crops+grown+new):new.append(bc)
                            field_us[0]+=(time.perf_counter()-start)*1e6
                            return new
                        sc,cs,gs,rd,conv,tr=variant_run(base,request,make_detect(tpx,scale9),lift_c(name,'C1',scale9),W,H,canvas,budget=crop6,cap=cap)
                        assert abs(tpx[0]-sum(t['pixels'] for t in tr if t['executed']))<1e-6
                        if cap in ('k39_strict','strict_observed'):assert tpx[0]<=crop6+1e-6
                        observed=[t['crop'] for t in tr if t['executed']]
                        score(name,sc,('none','v3a'),c2=False,crops=observed,dec=True)
                        rec.update({f'px_{name}':(cheap_px+tpx[0])/NATIVE,f'rounds_{name}':rd,f'converged_{name}':conv,
                            f'crops_{name}':cs,f'grown_{name}':gs,f'field_us_{name}':field_us[0],f'trace_{name}':tr,
                            f'observed_crops_{name}':observed,
                            f'scene_{name}':{k:sc[k] for k in ('xyxy','conf','coarse','mono','from_full')}})
                        push(name,sc)
                        rec['history_'+name]={k:st[name].get(k) for k in ('s1','s2')}
            elif part == "cons":
                for sn_ in SETTINGS:'''


def main():
    source=HERE/'k30_eval.py';s=source.read_text()
    old_doc=ast.get_docstring(ast.parse(s),clean=False)
    s=once(s,old_doc,'Isolated K40R speed-tube / K41 grid rerun; generated from read-only k30_eval.py.\nFrozen v6-S is a checked control. New outputs only in results/k40r or results/k41.')
    s=once(s,'choices=["match", "axis", "idm", "cons"]','choices=["tubes", "grid"]')
    s=once(s,'    args = ap.parse_args()','    ap.add_argument("--grid_index",type=int,default=-1)\n    ap.add_argument("--max_frames",type=int,default=0)\n    args = ap.parse_args()')
    s=once(s,'REAL / "results" / "k30" / (part + (f"_{args.tag}" if args.tag else "")) / ds',
        'REAL / "results" / ("k40r" if part=="tubes" else "k41") / (part + (f"_{args.grid_index:02d}" if part=="grid" else "") + (f"_{args.tag}" if args.tag else "")) / ds')
    s=once(s,'            elif part == "cons":\n                for sn_ in SETTINGS:',PART)
    s=once(s,'    FLOG = []','''    from k40r_runcheck import room, check_and_save
    room(REAL)
    active=[None];round_id=[0];detector_trace=[];guard_context={'ordinary':None,'ordinal':0}
    original_guarded=SP.guarded
    def traced_guarded(detect,W,H,*a,**kw):
        def watched(c):
            guard_context['ordinal']+=1
            return detect(c)
        original=original_guarded(watched,W,H,*a,**kw)
        def call(c):
            before=dict(guard_context)
            guard_context.update(ordinary=np.asarray(c).copy(),ordinal=0)
            try:return original(c)
            finally:guard_context.clear();guard_context.update(before)
        return call
    SP.guarded=traced_guarded
    FLOG = []''')
    s=once(s,'        r_ = _fcg(*a, **k)','        round_id[0]+=1\n        r_ = _fcg(*a, **k)')
    s=once(s,'        if fo.exists():\n            continue','''        if fo.exists():
            assert fo.with_suffix('.done.json').exists(), 'Unvalidated output exists: '+str(fo)
            continue
        room(REAL)''')
    s=once(s,'            if img is None:\n                continue','            assert img is not None, (unit, fr, path)')
    s=once(s,'            cache, rcache, last = {}, {}, {}','            cache, rcache, last = {}, {}, {}\n            detector_trace.clear()')
    s=once(s,'                    if ck_ not in cache:\n', '                    hit=ck_ in cache\n                    if ck_ not in cache:\n')
    s=once(s,'                    out = {k: x.copy() for k, x in cache[ck_].items()}','''                    out = {k: x.copy() for k, x in cache[ck_].items()}
                    detector_trace.append(dict(row=active[0],round=round_id[0],crop=np.asarray(cr).copy(),canvas=key,
                        cache_hit=hit,gpu_executed=not hit,charged_pixels=key[4]*key[5],
                        ordinary_group=guard_context['ordinary'],guard_call=guard_context['ordinal']>1,
                        detections={k:np.asarray(x).copy() for k,x in out.items()}))''')
    s=once(s,'                FLOG.clear()','                FLOG.clear()\n                active[0]=key;round_id[0]=0')
    s=once(s,'                    a1 = act(tr_, pp_)','''                    a1 = act(tr_, pp_)
                    rec[f'geometry_{prefix}|{r_}{sfx}']=tuple(np.asarray(x).copy() for x in tr_)''')
    s=once(s,'            rows.append(rec)','''            rec['detector_trace']=list(detector_trace)
            rec['gt_boxes']=gtb;rec['gt_coarse']=gtc;rec['gt_geometry']=gt_t
            rec['draft_scene']=base
            rows.append(rec)
            if len(rows)%20==0:print(f'  {unit} {len(rows)} frames',flush=True)
            if args.max_frames and len(rows)>=args.max_frames:break''')
    s=once(s,'        with open(fo, "wb") as fh:\n            pickle.dump(rows, fh)', '''        check_and_save(rows,fo,REAL,ds,__file__)''')
    s=once(s,'print(f"K30 {part.upper()} {ds} DONE", flush=True)','print(f"K40R/K41 {part.upper()} {ds} DONE", flush=True)')
    # Read-only wrapper import paths still use B/k10_masked for AV2 as original.
    ast.parse(s);dest=HERE/'k40r_detector.py';dest.write_text(s)
    manifest={p.name:hashlib.md5(p.read_bytes()).hexdigest() for p in [source,dest,Path(__file__).resolve(),
        HERE/'k41_variant_loop.py',HERE/'k40r_tube_geometry.py',HERE/'k40r_runcheck.py',
        HERE/'k41_k39_cap_reference.py',HERE/'k40r_d02_tests.py',HERE/'k40r_d02_dispatch.py',HERE/'k41_job_exec.py']}
    (HERE/'k40r_detector_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Built',dest,'not run')


if __name__=='__main__':main()
