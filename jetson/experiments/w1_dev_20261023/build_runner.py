"""Generate sandbox forks; never edit the imported frozen sources."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def replace_once(s, old, new):
    assert s.count(old)==1, (old[:100],s.count(old))
    return s.replace(old,new)

def main():
    origin=ROOT/'remote/A_method/k30_dev.py'
    loop=ROOT/'remote/A_method/k_loop.py'
    s=origin.read_text()
    s=replace_once(s,'import pickle\n','import pickle\nimport json\nimport time\n')
    s=replace_once(s,'IE, KL, KT = K27.IE, K27.KL, K27.KT', '''IE, KL, KT = K27.IE, K27.KL, K27.KT
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import Retro, RefineClosure, obligations, draft_recheck
_kl = importlib.util.spec_from_file_location('sandbox_loop', Path(__file__).with_name('sandbox_loop.py'))
KL = importlib.util.module_from_spec(_kl); _kl.loader.exec_module(KL)''')
    s=replace_once(s,'choices=["nusc", "av2", "ahd"]','choices=["nusc", "av2"]')
    s=replace_once(s,'ap.add_argument("--part", required=True, choices=["match", "axis", "idm", "cons", "devref"])',
                   'ap.add_argument("--part", default="devref", choices=["devref"])')
    s=replace_once(s,'od = REAL / "results" / "k30" / ("dev_" + part + (f"_{args.tag}" if args.tag else "")) / ds',
                   'od = REAL / "experiments/w1_dev_20261023/results" / (args.tag or "dev") / ds')
    s=replace_once(s,'    UNITS = {u for u in args.units.split(",") if u}', '''    UNITS = {u for u in args.units.split(",") if u}
    reference_dir = REAL / 'results/k30/dev_devref' / ds
    allowed = {p.stem for p in reference_dir.glob('*.pkl')}
    assert len(allowed) == {'nusc':152,'av2':300}[ds], (ds,len(allowed))
    if ds == 'av2':
        assert os.environ.get('K10_MASK_Y0') == '1830', 'fixed AV2 mask required'
        import pandas as pd
        split = pd.read_csv(REAL/'results/k23/av2_split.csv')
        assert allowed == set(split[split.role=='CFG-DEV'].log)
        assert not allowed.intersection(split[split.role.isin(['REF-TRAIN','REF-VAL'])].log)
    else:
        import pandas as pd
        split = pd.read_csv(REAL/'results/nusc_full/split.csv')
        train = set(split[(split['set']=='new') & (split.split=='train')].seq)
        assert allowed == set(split[(split['set']=='new') & (split.split=='val')].seq)
        assert not allowed.intersection(train), 'nuScenes dev/train leakage'
    (od.parent / (ds + '_split_audit.json')).write_text(json.dumps({
        'dataset':ds,'allowed_units':sorted(allowed),'training_disjoint':True},indent=2))''')
    s=replace_once(s,'        fo = od / f"{unit}.pkl"', '''        assert unit in allowed, ('non-dev unit',unit)
        fo = od / f"{unit}.pkl"''')
    s=replace_once(s,'            rec = {"unit": unit, "frame": fr, "v": v, "gt": not isinstance(cached, str)}',
                   '            rec = {"unit": unit, "frame": fr, "v": v, "dt":dt, "gt": not isinstance(cached, str)}')
    s=replace_once(s,'            cache, rcache, last = {}, {}, {}','            cache, rcache, last, RETRO = {}, {}, {}, {}')
    s=replace_once(s,'                h = st.get(name, {})\n\n                def f(full):\n                    X =',
                   '                original_h = st.get(name, {})\n\n                def f(full):\n                    h = dict(original_h)\n                    X =')
    s=replace_once(s,'                    if not chn:\n                        return z, lo, hi, ttc', '''                    if name in RETRO:
                        h, chn = RETRO[name].augment(full,(z,lo,hi,ttc),h)
                    if not chn:
                        return z, lo, hi, ttc''')
    s=replace_once(s,'                if name == "none":','                if name in ("none", "closed"):')
    # Insert counter and geometry audit before scoring. It does not drive any decision.
    pos='            def score(prefix, scene, refs=("none", "v3a"), sn_="base", sup_fn=None, c2=True, crops=None, field=None, dec=False):'
    audit='''            def audit(nm, sc, crops, grown, tr):
                h = st.get(nm,{})
                full_ids = np.flatnonzero(sc['from_full'])
                chains = KTD.chains(sc['xyxy'][full_ids],h.get('s1'),h.get('s2')) if len(imgs)==2 else []
                observed_ids = {int(full_ids[i]) for i,_,_ in chains}
                fc,ns = obligations(dict(sc,mono=tr),crops,grown,
                    lambda tt: KL.field_crop_g(tt,v,cam,W,H,PP0,(0.0,),SP.R_GRID,SP.Y_GRID,(KL.CAR,)),W,H,v,PP0)
                raw = sc.get('_raw_mono',sc['mono'])
                rec.update({f'diag_post_field_{nm}':fc, f'diag_post_support_{nm}':ns,
                    f'diag_action_drop_{nm}':int(act(tr)<act(raw)),
                    f'diag_full_objects_{nm}':len(full_ids), f'diag_chain_objects_{nm}':len(observed_ids)})
                if GT['base'][2] >= 0:
                    di=GT['base'][2]
                    m=int(IE.matched(gtb[di:di+1],gtc[di:di+1],sc['xyxy'],sc['coarse'])[0])
                    rec[f'diag_dec_chain_{nm}'] = m in observed_ids
                if nm == 'v6S':
                    rec['trace_v6S'] = dict(xyxy=sc['xyxy'],coarse=sc['coarse'],from_full=sc['from_full'],
                        raw=raw,refined=tr,crops=list(crops)+list(grown),chain_ids=sorted(observed_ids))

'''
    s=replace_once(s,pos,audit+pos)
    a=s.index('            def loop(nm, ctx, side,')
    b=s.index('            # ---- draft',a)
    s=s[:a]+'''            def loop(nm, ctx, side, sn_="base", refs=("none", "v3a"), dec=False, score_it=True, t4=False, **kw):
                scale = side / NW
                key = nm
                tpx, log, cnt = [0.0], {}, [0]
                FLOG.clear()
                if nm.endswith('_retro'):
                    RETRO[nm] = Retro(base,imgs,log,lambda im,k: run_det(det,im,k),KTD.chains,v,PP0)
                lf = lift_pol(key, parse(TPOL), scale) if t4 else lift_c(key, "C1", scale)
                close = RefineClosure(refine,v,PP0) if nm.endswith('_close') else None
                recheck = (lambda cr,gr,sc: draft_recheck(base,sc,cr,gr,W,H,v,PP0,KL.box_crop_ctx)) if nm.endswith('_recheck') else None
                sc,crops,grown,rounds,conv = KL.run_g(base,v,cam,W,H,make_detect(tpx,scale,log,None,cnt),lf,
                    PP0,v_closes=(0.0,),ctx=ctx,max_calls=None,post=close,post_rounds=1,recheck=recheck)
                fl=list(FLOG)
                if close is not None: refs=('closed',)
                score(nm,sc,refs,'base',crops=list(crops)+list(grown),field=fl,dec=True)
                tr = sc['mono'] if close else refine(sc,'v3a',brake_sup(PP0))
                audit(nm,sc,crops,grown,tr)
                r=RETRO.get(nm)
                rpx=r.pixels if r else 0
                rec.update({f'px_{nm}':(cheap_px+tpx[0]+rpx)/NATIVE,
                    f'ncalls_{nm}':cnt[0]+(r.calls if r else 0),f'ncrops_{nm}':len(crops),
                    f'nguard_{nm}':len(grown),f'rounds_{nm}':rounds,f'conv_{nm}':conv,
                    f'retro_calls_{nm}':r.calls if r else 0,f'retro_attempted_{nm}':r.attempted if r else 0,
                    f'retro_success_{nm}':r.success if r else 0,
                    f'closure_corrections_{nm}':close.new_corrections if close else 0})
                last[nm]=(crops,grown,sc,fl)
                push(nm,sc)
                return tpx[0]+rpx
''' +s[b:]
    a=s.index('            # ================================================================================================= parts')
    b=s.index('            imgs = (imgs + [img])[-2:]',a)
    s=s[:a]+'''            # Fixed sandbox rows. No test split and no adaptive configuration sweep.
            budget=loop('v6S',2.0,960,dec=True)
            for nm in ('v6S_retro','v6S_close','v6S_recheck'):
                loop(nm,2.0,960,dec=True)
            for cfg in ('tsp8','unc2'):
                heur(cfg,'v6budget',budget,dec=True,c2=False)
            for L in (960,NWS[ds]):
                md,ms=wf_mode(ds,L); nm=f'wf{L}'
                d=keep(det.detect(img,md)[0])
                tc1=lift_c(nm,'C1',ms)(d)
                scwf=scene_dict(d,tc1)
                score(nm+'+C1',scwf,('none','v3a'),'base',c2=False,dec=True)
                scclose=RefineClosure(refine,v,PP0)(scwf)
                score(nm+'+C1_close',scclose,('closed',),'base',c2=False,dec=True)
                rec[f'px_{nm}+C1']=md.net_h*md.net_w/NATIVE
                rec[f'px_{nm}+C1_close']=md.net_h*md.net_w/NATIVE
                push(nm,scene_dict(d,None))
''' +s[b:]
    s=replace_once(s,'        with open(fo, "wb") as fh:\n            pickle.dump(rows, fh)', '''        old=pickle.load((reference_dir/f'{unit}.pkl').open('rb'))
        oldmap={int(r['frame']):r for r in old}
        parity={'unit':unit,'dataset':ds,'frames':len(rows),'differences':{}}
        assert {int(r['frame']) for r in rows} == set(oldmap), ('frames',unit)
        for col in ('act_gt','a_gt','a_cheap','J_cheap','a_v6S|none','a_v6S|v3a','J_v6S|v3a',
                    'a_wf960+C1|v3a',f'a_wf{NWS[ds]}+C1|v3a'):
            delta=[abs(float(r[col])-float(oldmap[int(r['frame'])][col])) for r in rows]
            parity['differences'][col]=sum(x>1e-8 for x in delta)
        (od/f'{unit}.parity.json').write_text(json.dumps(parity,indent=2))
        # Preserve the failed-unit data for review, but fail closed before main launch.
        tmp=fo.with_suffix('.tmp')
        with tmp.open('wb') as fh: pickle.dump(rows,fh)
        tmp.replace(fo)
        assert not any(parity['differences'].values()), parity
''')
    s=s.replace('print(f"K30 {part.upper()} {ds} DONE", flush=True)',
                'print(f"W1 SANDBOX {ds} DONE", flush=True)')
    s=s.replace(s[s.index('"""'):s.index('"""',s.index('"""')+3)+3],
                '"""Generated isolated W1 development harness. See protocol.md and fork_manifest.json."""',1)
    (HERE/'runner.py').write_text(s)
    t=loop.read_text().split('\ndef run_bounded(',1)[0]
    t=replace_once(t,'          max_calls=None):','          max_calls=None, post=None, post_rounds=1, recheck=None):')
    t=replace_once(t,'    scene.setdefault("from_full", np.zeros(len(scene["conf"]), bool))\n    return scene, crops, grown, rounds, converged', '''    scene.setdefault("from_full", np.zeros(len(scene["conf"]), bool))
    if recheck is not None:
        ex=recheck(crops,grown,scene)
        if ex:
            scene=detect_merge(ex);rounds+=1
            converged=not new_crops(scene)
    if post is not None:
        scene=post(scene)
        for _ in range(post_rounds):
            new=new_crops(scene)
            if not new: break
            scene=post(detect_merge(new));rounds+=1
        converged=not new_crops(scene)
    return scene, crops, grown, rounds, converged''')
    (HERE/'sandbox_loop.py').write_text(t)
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [origin,loop,ROOT/'src/vtf/speculative.py',ROOT/'src/vtf/dsf.py',ROOT/'remote/A_method/k28_p3.py']}
    (HERE/'fork_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': main()
