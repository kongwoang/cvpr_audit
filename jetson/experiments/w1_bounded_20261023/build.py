"""Generate an isolated fixed factorial harness, preserving frozen sources."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'experiments/w1_dev_20261023'


def once(s, old, new):
    assert s.count(old) == 1, (old[:90], s.count(old))
    return s.replace(old, new)


def main():
    for name in ('helpers.py', 'sandbox_loop.py'):
        (HERE/name).write_bytes((OLD/name).read_bytes())
    source = (ROOT/'remote/A_method/k31_eval.py').read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'run_bounded_keep')
    function = '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])
    (HERE/'bounded_variant.py').write_text(
        '"""Verbatim frozen K31 B2K-A function; KL is bound by the sandbox harness."""\n'
        'import numpy as np\nfrom vtf import speculative as SP\nKL = None\n\n'+function+'\n')
    s = (OLD/'runner.py').read_text().replace('w1_dev_20261023/results', 'w1_bounded_20261023/results')
    s = once(s, 'KL = importlib.util.module_from_spec(_kl); _kl.loader.exec_module(KL)',
        'KL = importlib.util.module_from_spec(_kl); _kl.loader.exec_module(KL)\n'
        'import bounded_variant\nbounded_variant.KL = KL\nfrom bounded_variant import run_bounded_keep')
    s = once(s, '        prev_a3 = None\n        for fr, path, v, g, cached in gen():',
        "        prev_a3 = None\n        reference_map = {int(r['frame']):r for r in pickle.load((reference_dir/f'{unit}.pkl').open('rb'))}\n"
        '        for fr, path, v, g, cached in gen():')
    s = once(s, '            rec = {"unit": unit, "frame": fr, "v": v, "dt":dt, "gt": not isinstance(cached, str)}',
        "            rec = dict(reference_map[int(fr)])  # immutable generic-control rows; baseline overwritten below\n"
        '            rec.update({"unit": unit, "frame": fr, "v": v, "dt":dt, "gt": not isinstance(cached, str)})')
    old = '''                sc,crops,grown,rounds,conv = KL.run_g(base,v,cam,W,H,make_detect(tpx,scale,log,None,cnt),lf,
                    PP0,v_closes=(0.0,),ctx=ctx,max_calls=None,post=close,post_rounds=1,recheck=recheck)'''
    new = '''                if 'b2A' in nm:
                    sc,crops,grown,rounds,conv = run_bounded_keep(base,v,cam,W,H,
                        make_detect(tpx,scale,log,None,cnt),lf,PP0,2,'A',ctx=ctx)
                    assert rounds <= 2
                else:
                    sc,crops,grown,rounds,conv = KL.run_g(base,v,cam,W,H,make_detect(tpx,scale,log,None,cnt),lf,
                        PP0,v_closes=(0.0,),ctx=ctx,max_calls=None,post=close,post_rounds=1,recheck=recheck)'''
    s = once(s, old, new)
    s = once(s, '                rpx=r.pixels if r else 0',
        "                rpx=r.pixels if r else 0\n                assert r is None or r.calls <= 2")
    a = s.index('            # Fixed sandbox rows.')
    b = s.index('            imgs = (imgs + [img])[-2:]', a)
    s = s[:a]+'''            # Fixed 2x2 screen; full-loop recovery/whole-frame controls reused in analysis.
            loop('v6S',2.0,960,dec=True)
            loop('v6S_b2A',2.0,960,dec=True)
            combined_budget=loop('v6S_b2A_retro',2.0,960,dec=True)
            for cfg in ('tsp8','unc2'):
                heur(cfg,'combinedbudget',combined_budget,dec=True,c2=False)
                key=cfg+'@combinedbudget+C1'
                rec['requested_crop_px_'+key]=combined_budget/NATIVE
                rec['guard_budget_overrun_px_'+key]=max(0.,rec['px_'+key]-(cheap_px+combined_budget)/NATIVE)
''' + s[b:]
    s = s.replace("parity={'unit':unit,'dataset':ds,'frames':len(rows),'differences':{}}",
        "parity={'unit':unit,'dataset':ds,'frames':len(rows),'differences':{},"
        "'recomputed':['cheap','v6S'],'imported_controls':['u960+C1','native+C1']}")
    (HERE/'runner.py').write_text(s)
    for name in ('job.py','group.py'):
        (HERE/name).write_text((OLD/name).read_text().replace('w1_dev_20261023','w1_bounded_20261023'))
    s = (OLD/'analyze.py').read_text().replace('w1_dev_20261023','w1_bounded_20261023')
    s = once(s, "            rows=pickle.load(file.open('rb'))", """            rows=pickle.load(file.open('rb'))
            previous=ROOT/'experiments/w1_dev_20261023/results'/args.stage/ds/file.name
            assert previous.exists(), ('original screen still incomplete',previous)
            prior={int(r['frame']):r for r in pickle.load(previous.open('rb'))}
            rows=[dict(prior[int(r['frame'])],**r) for r in rows]
""")
    a=s.index("        pipes={'cheap'")
    b=s.index('        dist={};points={}',a)
    s=s[:a]+'''        pipes={'cheap':'a_cheap','v6-S':'a_v6S|v3a','v6-S no refiner':'a_v6S|none',
               'temporal evidence':'a_v6S_retro|v3a','temporal evidence no refiner':'a_v6S_retro|none',
               'B2K-A':'a_v6S_b2A|v3a','B2K-A no refiner':'a_v6S_b2A|none',
               'B2K-A + temporal evidence':'a_v6S_b2A_retro|v3a',
               'B2K-A + temporal evidence no refiner':'a_v6S_b2A_retro|none',
               'hybrid at combined requested budget':'a_tsp8@combinedbudget+C1|v3a',
               'unc2 at combined requested budget':'a_unc2@combinedbudget+C1|v3a'}
        for length,label in [(960,'u960'),(1600 if ds=='nusc' else 2048,'native')]:
            pipes[label+' + C1 + v3a']=f'a_wf{length}+C1|v3a'
            pipes[label+' + C1 no refiner']=f'a_wf{length}+C1|none'
''' +s[b:]
    marker="        for name in ('v6S','v6S_retro','v6S_close','v6S_recheck'):"
    s=once(s,marker,"""        for comparator in ('B2K-A','temporal evidence'):
            name='B2K-A + temporal evidence'
            for metric in points[name]:
                contrasts.append(dict(dataset=ds,pipeline=name,comparator=comparator,metric=metric,
                    candidate_minus_baseline=points[name][metric]-points[comparator][metric],
                    **ci(dist[name][metric]-dist[comparator][metric])))
        for metric in points['v6-S']:
            effect=points['B2K-A + temporal evidence'][metric]-points['B2K-A'][metric]-points['temporal evidence'][metric]+points['v6-S'][metric]
            sample=dist['B2K-A + temporal evidence'][metric]-dist['B2K-A'][metric]-dist['temporal evidence'][metric]+dist['v6-S'][metric]
            contrasts.append(dict(dataset=ds,pipeline='factorial interaction',comparator='additive component effects',
                metric=metric,candidate_minus_baseline=effect,**ci(sample)))
        for name in ('v6S','v6S_retro','v6S_b2A','v6S_b2A_retro'):
""")
    s=s.replace("'hybrid at v6 crop budget'", "'hybrid at combined requested budget'")
    s=s.replace('baseline v6-S crop-pixel allowance','combined candidate crop-pixel allowance including recovery')
    s=s.replace('# W1 isolated development experiments', '# Bounded verification x temporal evidence')
    (HERE/'analyze.py').write_text(s)
    (HERE/'check_semantics.py').write_bytes((OLD/'check_semantics.py').read_bytes())
    sources=[ROOT/'remote/A_method/k31_eval.py',ROOT/'remote/A_method/k_loop.py',
             ROOT/'src/vtf/speculative.py',ROOT/'src/vtf/dsf.py',ROOT/'remote/A_method/k28_p3.py']
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (HERE/'fork_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
