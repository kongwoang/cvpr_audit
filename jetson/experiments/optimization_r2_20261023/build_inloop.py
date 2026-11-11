"""Isolated, assertive source transformation of the completed round-1 harness."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'experiments/w1_bounded_20261023'

def once(s,old,new):
    assert s.count(old)==1,(old[:100],s.count(old))
    return s.replace(old,new)

def main():
    (HERE/'sandbox_loop.py').write_bytes((OLD/'sandbox_loop.py').read_bytes())
    h=(OLD/'helpers.py').read_text()
    h=once(h,'        self.used = False','        self.events = []\n        self.used = False')
    h=once(h,"                    j = unique_match(box,full['coarse'][i],det)",
        "                    j = unique_match(box,full['coarse'][i],det)\n"
        "                    self.events.append(dict(lag=lag,key=key,target=box.copy(),coarse=str(full['coarse'][i]),\n"
        "                        detections={k:np.asarray(v).copy() for k,v in det.items()},match=j))")
    (HERE/'helpers.py').write_text(h)
    b=(OLD/'bounded_variant.py').read_text()
    b=once(b,'probes=None, field=True, ctx=SP.CTX, max_grow=2):','probes=None, field=True, ctx=SP.CTX, max_grow=2, seed=None):')
    b=once(b,'        for k in SP.support(scene["mono"], v, pp):',
        '        if rounds == 0 and seed:\n'
        '            new += [c for c in seed if not SP.covered(c,crops+grown+new)]\n'
        '        for k in SP.support(scene["mono"], v, pp):')
    (HERE/'bounded_variant.py').write_text(b)
    s=(OLD/'runner.py').read_text().replace('w1_bounded_20261023/results','optimization_r2_20261023/results')
    s=once(s,'from helpers import Retro, RefineClosure, obligations, draft_recheck',
        'from helpers import Retro, RefineClosure, obligations, draft_recheck\n'
        'from proactive import Pending\nfrom estimators import motion_floors, add_floor, scale_refinement')
    s=once(s,"        reference_map = {int(r['frame']):r for r in pickle.load((reference_dir/f'{unit}.pkl').open('rb'))}",
        "        reference_map = {int(r['frame']):r for r in pickle.load((REAL/'experiments/w1_bounded_20261023/results/dev'/ds/f'{unit}.pkl').open('rb'))}\n"
        "        case_units={c['unit'] for c in json.loads(Path(__file__).with_name('case_manifest_with_severity.json').read_text())['cases']}\n"
        "        trace_case=unit in case_units")
    s=once(s,'            rec = dict(reference_map[int(fr)])  # immutable generic-control rows; baseline overwritten below',
        '            rec = {}  # recompute controls; do not import outcomes into policies')
    marker='''                def f(full):
                    h = dict(original_h)'''
    s=once(s,marker,'''                def finish(full,mono):
                    if '_floor' not in name:return mono
                    h=original_h
                    h1=(h['s1'],h['c1']) if h.get('s1') is not None else None
                    h2=(h['s2'],h['c2']) if h.get('s2') is not None else None
                    _,rates,_=motion_floors(full['xyxy'],full['coarse'],h1,h2,sc,dt)
                    return add_floor(mono,rates,np.ones(len(full['xyxy']),bool))[0]

                def f(full):
                    h = dict(original_h)''')
    s=once(s,'                    if not chn:\n                        return z, lo, hi, ttc',
        '                    if not chn:\n                        return finish(full,(z,lo,hi,ttc))')
    s=once(s,'                    return z, lo, hi, ttc\n                return f\n\n            def push',
        '                    return finish(full,(z,lo,hi,ttc))\n                return f\n\n            def push')
    s=once(s,'                h["s2"] = h.get("s1")',
        '                h["c2"] = h.get("c1")\n'
        '                h["c1"] = np.asarray(scene["coarse"])[ff].copy()\n'
        '                h["s2"] = h.get("s1")')
    s=once(s,'                if name.startswith("v3a_"):',
        '                if name == "half":\n'
        '                    rf=refine(scene,"v3a",sup_fn)\n'
        '                    factor=np.ones(len(tt[0]));valid=np.isfinite(tt[0])&(tt[0]>0)&np.isfinite(rf[0])\n'
        '                    factor[valid]=np.sqrt(np.asarray(rf[0])[valid]/tt[0][valid])\n'
        '                    return scale_refinement(tt,factor)\n'
        '                if name.startswith("v3a_"):')
    s=once(s,"                if nm == 'v6S':\n                    rec['trace_v6S'] = dict(xyxy=sc['xyxy'],coarse=sc['coarse'],from_full=sc['from_full'],\n                        raw=raw,refined=tr,crops=list(crops)+list(grown),chain_ids=sorted(observed_ids))",
        "                if nm == 'v6S' or trace_case:\n"
        "                    rec['trace_'+nm] = dict(xyxy=sc['xyxy'],coarse=sc['coarse'],conf=sc['conf'],from_full=sc['from_full'],\n"
        "                        raw=raw,refined=tr,crops=list(crops)+list(grown),chain_ids=sorted(observed_ids),\n"
        "                        retro_events=RETRO[nm].events if nm in RETRO else [])\n"
        "                return sorted(observed_ids)")
    s=once(s,"                lf = lift_pol(key, parse(TPOL), scale) if t4 else lift_c(key, \"C1\", scale)",
        "                pending=st.setdefault(nm,{}).setdefault('pending',Pending()) if '_pending' in nm else None\n"
        "                seed=pending.seed(base,W,H,KL.box_crop_ctx) if pending else []\n"
        "                lf = lift_pol(key, parse(TPOL), scale) if t4 else lift_c(key, \"C1\", scale)")
    s=once(s,"make_detect(tpx,scale,log,None,cnt),lf,PP0,2,'A',ctx=ctx)",
        "make_detect(tpx,scale,log,None,cnt),lf,PP0,2,'A',ctx=ctx,seed=seed)")
    s=once(s,'                audit(nm,sc,crops,grown,tr)',
        "                observed=audit(nm,sc,crops,grown,tr)\n"
        "                score(nm,sc,('half',),'base',c2=False)\n"
        "                if pending:\n"
        "                    pending.update(sc,base,observed,v,PP0)\n"
        "                    rec.update({f'pending_requested_{nm}':pending.requested,f'pending_armed_{nm}':pending.armed,\n"
        "                                f'pending_completed_{nm}':pending.completed,f'pending_expired_{nm}':pending.expired})")
    a=s.index('            # Fixed 2x2 screen;')
    b=s.index('            imgs = (imgs + [img])[-2:]',a)
    s=s[:a]+'''            # Every fixed acquisition row, independent history; all three refiner controls.
            if trace_case:rec['trace_cheap']={k:np.asarray(v).copy() if k!='mono' else tuple(np.asarray(q).copy() for q in v) for k,v in base.items()}
            for nm in ('v6S','v6S_b2A_retro','v6S_b2A_pending','v6S_b2A_pending_retro','v6S_b2A_floor_retro'):
                loop(nm,2.0,960,dec=True)
''' +s[b:]
    a=s.index("        old=pickle.load((reference_dir/")
    b=s.index("        (od/f'{unit}.parity.json').write_text",a)
    s=s[:a]+'''        oldmap=reference_map
        parity={'unit':unit,'dataset':ds,'frames':len(rows),'differences':{},'recomputed':['cheap','v6S','v6S_b2A_retro']}
        assert {int(r['frame']) for r in rows} == set(oldmap), ('frames',unit)
        for col in ('act_gt','a_gt','a_cheap','J_cheap','a_v6S|none','a_v6S|v3a','J_v6S|v3a','px_v6S',
                    'a_v6S_b2A_retro|none','a_v6S_b2A_retro|v3a','J_v6S_b2A_retro|v3a','px_v6S_b2A_retro'):
            delta=[abs(float(r[col])-float(oldmap[int(r['frame'])][col])) for r in rows]
            parity['differences'][col]=sum(x>1e-8 for x in delta)
''' +s[b:]
    s=once(s,'        if fo.exists():\n            continue',
        "        if fo.exists():\n"
        "            prior=json.loads(fo.with_suffix('.parity.json').read_text())\n"
        "            assert not any(prior['differences'].values()),prior\n"
        "            continue")
    (HERE/'runner.py').write_text(s)
    j=(OLD/'job.py').read_text().replace('w1_bounded_20261023','optimization_r2_20261023')
    j=once(j,"        names=sorted(p.stem for p in (ROOT/'results/k30/dev_devref'/a.dataset).glob('*.pkl'))[:4]",
        "        names=sorted({c['unit'] for c in json.loads((HERE/'case_manifest_with_severity.json').read_text())['cases'] if c['dataset']==a.dataset})")
    j=once(j,"for p in ('runner.py','sandbox_loop.py','bounded_variant.py','helpers.py','protocol.md')",
        "for p in ('runner.py','sandbox_loop.py','helpers.py','protocol.md','inloop_protocol.md','proactive.py','estimators.py','bounded_variant.py')")
    (HERE/'job.py').write_text(j)
    (HERE/'group.py').write_text((OLD/'group.py').read_text().replace('w1_bounded_20261023','optimization_r2_20261023'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/'remote/A_method/k31_eval.py',ROOT/'remote/A_method/k_loop.py',ROOT/'src/vtf/speculative.py',ROOT/'src/vtf/dsf.py']}
    (HERE/'fork_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('Generated isolated runner, pending seed extension, instrumented recovery and guarded jobs')

if __name__=='__main__':main()
