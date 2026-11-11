"""Build an isolated D01 driver from the registered D02 copy of K30."""
import ast
import hashlib
import json
from pathlib import Path
from k40r_build_detector import once

HERE=Path(__file__).resolve().parent
PART='''            elif part == 'mean':
                from k40r_mean_policy import canvas as wf_canvas,eligible_history
                loop('v6S',2.0,960,dec=True)
                rec['scene_v6S']=last['v6S'][2]
                route=ROUTES[str(int(fr))]
                assert route['gt']==rec['gt']
                rec['routing']=route
                whole_cache={}
                for name,decision in route['routes'].items():
                    side=decision['side'];active[0]=name;round_id[0]=0
                    old=st.get(name,{})
                    old_scales=old.get('scales',(None,None))
                    rec['history_input_'+name]=dict(s1=old.get('s1'),s2=old.get('s2'),scales=old_scales)
                    if side==0:
                        # Routers' cheap arm is the frozen draft, with no invented C1/refiner.
                        sc=dict(base,from_full=np.zeros(len(base['conf']),bool));pixels=cheap_px
                    else:
                        if name=='AdaScale' and side==320:
                            dd=ch;pixels=cheap_px
                        else:
                            nh,nw=wf_canvas(ds,side,ada=name=='AdaScale')
                            key=(nh,nw);hit=key in whole_cache
                            if not hit:whole_cache[key]=keep(det.detect(img,Mode(f'mean{nh}x{nw}',nh,nw))[0])
                            dd=whole_cache[key];pixels=cheap_px+nh*nw
                            detector_trace.append(dict(row=name,round=0,crop=np.array([0,0,W,H]),canvas=(0,0,W,H,nh,nw),
                                cache_hit=hit,gpu_executed=not hit,charged_pixels=nh*nw,ordinary_group=None,guard_call=False,
                                detections={k:np.asarray(dd[k]).copy() for k in ('xyxy','conf','coarse')}))
                        dd={k:np.asarray(dd[k]) for k in ('xyxy','conf','coarse')}
                        dd['xyxy']=dd['xyxy'].astype(float).reshape(-1,4);dd['conf']=dd['conf'].astype(float)
                        st[name]=eligible_history(old,side)
                        try:
                            mono=lift_c(name,'C1',side/NW)(dd) if len(dd['conf']) else tuple(np.zeros(0) for _ in range(4))
                        finally:st[name]=old
                        sc=scene_dict(dd,mono)
                    score(name,sc,('none','v3a'),c2=False,dec=True,crops=[np.array([0,0,W,H])] if side else [])
                    if side==0:
                        for ref in ('none','v3a'):
                            assert rec[f'a_{name}|{ref}']==rec['a_cheap'] and rec[f'J_{name}|{ref}']==rec['J_cheap']
                    rec[f'px_{name}']=pixels/NATIVE;rec['scene_'+name]=sc
                    push(name,sc);st[name]['scales']=(side,old_scales[0])
                    rec['history_'+name]={k:st[name].get(k) for k in ('s1','s2','scales')}
            elif part in ('tubes','grid'):'''


def main():
    source=HERE/'k40r_detector.py';s=source.read_text()
    s=once(s,'choices=["tubes", "grid"]','choices=["mean"]')
    s=once(s,'REAL / "results" / ("k40r" if part=="tubes" else "k41") / (part + (f"_{args.grid_index:02d}" if part=="grid" else "") + (f"_{args.tag}" if args.tag else "")) / ds',
           'REAL / "results/k40r" / ("mean_history" + (f"_{args.tag}" if args.tag else "")) / ds')
    s=once(s,"            elif part in ('tubes','grid'):",PART)
    s=once(s,'        R, t = cam.transform("ego")','''        import json
        route_path=REAL/'results/k40r/routes'/ds/(unit+'.json')
        ROUTES=json.loads(route_path.read_text())
        R, t = cam.transform("ego")''')
    old=ast.get_docstring(ast.parse(s),clean=False)
    s=once(s,old,'D01 own routed, fidelity-consistent histories, locked dev choices and cached scores.\nIsolated K30-derived control; results/k40r/mean_history only. Generated on KIT.')
    ast.parse(s);dest=HERE/'k40r_mean_history.py';dest.write_text(s)
    dispatch=(HERE/'k40r_d02_dispatch.py').read_text()
    dispatch=once(dispatch,"('tubes_'+tag if tag else 'tubes')","('mean_history_'+tag if tag else 'mean_history')")
    dispatch=dispatch.replace('D02','D01').replace('cuda13','cuda12').replace('VUW1','VUW')
    dispatch=dispatch.replace('k40r_detector_manifest.json','k40r_mean_manifest.json').replace('k40r_detector.py','k40r_mean_history.py')
    dispatch=dispatch.replace("'--part','tubes'","'--part','mean'").replace('ceiling_gpu_hours=60','ceiling_gpu_hours=40').replace('60*3600-consumed','40*3600-consumed')
    dispatch=once(dispatch,"    refs={}",'''    route_manifest=json.loads((b/'results/k40r/routes/manifest.json').read_text())
    for rel,expected in route_manifest['routing_hashes'].items():
        assert hashlib.md5((b/'results/k40r/routes'/rel).read_bytes()).hexdigest()==expected,rel
    save(out/'routing_manifest.json',route_manifest)
    refs={}''')
    dispatcher=HERE/'k40r_d01_dispatch.py';ast.parse(dispatch);dispatcher.write_text(dispatch)
    files=[source,dest,dispatcher,Path(__file__).resolve(),HERE/'k40r_mean_policy.py',HERE/'k40r_mean_routes.py',HERE/'k40r_runcheck.py',HERE/'k41_job_exec.py',HERE/'k40r_matched.py',HERE/'k33_rc.py']
    manifest={p.name:hashlib.md5(p.read_bytes()).hexdigest() for p in files}
    (HERE/'k40r_mean_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Built',dest,'not run')


if __name__=='__main__':main()
