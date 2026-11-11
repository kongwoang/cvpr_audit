"""Generate a sandbox-only Jetson factorial port, importing frozen components."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def once(s,old,new):
    assert s.count(old)==1,(old[:70],s.count(old))
    return s.replace(old,new)


def main():
    (HERE/'helpers.py').write_bytes((ROOT/'experiments/w1_dev_20261023/helpers.py').read_bytes())
    s=(ROOT/'scripts/77_k34_latency.py').read_text()
    s=once(s,'import cv2\nfrom k34_bounded import bounded_keep',
        f'import sys\nsys.path.insert(0,{str(ROOT/"scripts")!r})\nsys.path.insert(0,{str(HERE)!r})\n'
        'import cv2\nfrom helpers import Retro\nfrom recovery_gpu import TracedCrops,Historical\nfrom k34_bounded import bounded_keep')
    s=once(s,'ROOT = Path(__file__).resolve().parents[1]',f'ROOT = Path({str(ROOT)!r})')
    s=once(s,'    ap.add_argument("--rep", type=int, default=1)',
        '    ap.add_argument("--recovery_disabled",action="store_true")\n'
        '    ap.add_argument("--rep", type=int, default=1)')
    s=once(s,'f"k34_{mode}_{args.dataset}_{args.config}"','f"temporal_jetson_{mode}_{args.dataset}_{args.config}"')
    s=once(s,'    fid = 0\n    for unit, fr, path, v, g, cam, (R, t) in gen:',
        '    fid = 0\n    gpu_history={}\n    for unit, fr, path, v, g, cam, (R, t) in gen:')
    s=once(s,'            hist, cur, prev_act, imgs = {}, unit, None, []',
        '            hist, cur, prev_act, imgs = {}, unit, None, []\n            gpu_history.clear()')
    s=once(s,'        rec = {"unit": unit, "frame": fr, "v": v, "act_gt": P.discrete_action(a_gt, pp)}',
        '        rec = {"unit": unit, "frame": fr, "v": v, "a_gt":float(a_gt),"dt":dt,"act_gt": P.discrete_action(a_gt, pp)}\n        RETROS={}')
    s=once(s,'        def lift_c1_fast(name, p):\n            h = hist.get(name, {})',
        '        def lift_c1_fast(name, p):\n            original_h = hist.get(name, {})')
    # Only the fast C1 closure gets explicit recovered chain entries.
    start=s.index('        def lift_c1_fast(name, p):')
    end=s.index('\n        def lift_c(name, p):',start)
    chunk=s[start:end]
    chunk=once(chunk,'            def f(full):\n                X =',
        '            def f(full):\n                h=dict(original_h)\n                X =')
    chunk=once(chunk,'                if not chn:\n                    return z, lo, hi, ttc',
        "                if name in RETROS:\n                    h,chn=RETROS[name].augment(full,(z,lo,hi,ttc),h)\n"
        '                if not chn:\n                    return z, lo, hi, ttc')
    s=s[:start]+chunk+s[end:]
    s=once(s,'            variants = [("ours", None), ("ours_b2", 2), ("ours_b2A", 2)]\n            rotation = (args.rep - 1) % 3',
        '            variants = [("ours",None),("ours_retro",None),("ours_b2A",2),("ours_b2A_retro",2)]\n'
        '            rotation = (args.rep - 1) % 4')
    s=once(s,'                lf = lift_c(nm, p) if LIFT == "C1" else lift_t4(nm, scale)',
        '''                assert p=='trt32' and not args.parity
                is_recovery=nm0.endswith('_retro')
                trace={}
                traced_many=TracedCrops(many_p,G,scale,trace)
                traced_one=TracedCrops(one_p,G,scale,trace,common=True)
                if is_recovery and not args.recovery_disabled:
                    history=gpu_history.get(nm,[])
                    assert len(history)==len(imgs)
                    historical=Historical(imgs,history,many.eng,Mode,THR,STATS)
                    RETROS[nm]=Retro(dd,imgs,trace,historical,chains,v,pp)
                lf = lift_c(nm, p) if LIFT == "C1" else lift_t4(nm, scale)''')
    s=once(s,'                mu = SpecMany(many_p, spec_hook(nm, p)) if (N is None and LIFT == "C1" and OV.get(p)) else many_p',
        '                mu=traced_many if is_recovery else many_p')
    s=once(s,'                    if nm0 == "ours_b2A":','                    if nm0.startswith("ours_b2A"):')
    s=once(s,'scene, calls, hit = bounded_keep(dd, v, cam, W, H, one_p, lf, pp, args.v_close, N)',
        'scene, calls, hit = bounded_keep(dd,v,cam,W,H,traced_one if is_recovery else one_p,lf,pp,args.v_close,N)')
    s=once(s,'                a_noref = act(scene["mono"])',
        "                recovery=RETROS.get(nm)\n                assert recovery is None or recovery.calls<=2\n"
        "                if recovery is not None:\n                    assert historical.provenance in ([],[1],[1,0])\n"
        "                    assert len(historical.provenance)==recovery.calls\n"
        '                a_noref = act(scene["mono"])')
    s=once(s,'                px = (CH.net_h * CH.net_w + (STATS["tpx"] if N is None else STATS["tpx_b"])) / NATIVE',
        '                px = (CH.net_h*CH.net_w+(STATS["tpx"] if N is None else STATS["tpx_b"])+(recovery.pixels if recovery else 0))/NATIVE\n'
        "                rec.update({f'retro_calls_{nm}':recovery.calls if recovery else 0,\n"
        "                    f'retro_attempted_{nm}':recovery.attempted if recovery else 0,\n"
        "                    f'retro_success_{nm}':recovery.success if recovery else 0,\n"
        "                    f'retro_pixels_{nm}':recovery.pixels if recovery else 0})\n"
        "                if is_recovery:\n                    gpu_history[nm]=(gpu_history.get(nm,[])+[G.x])[-2:]\n"
        "                rec['gpu_history_bytes']=sum(x.numel()*x.element_size() for hh in gpu_history.values() for x in hh)")
    s=once(s,'    (run / "meta.json").write_text(json.dumps(meta, indent=1))',
        '    meta["peak_gpu_allocated_bytes"]=torch.cuda.max_memory_allocated()\n'
        '    meta["peak_gpu_reserved_bytes"]=torch.cuda.max_memory_reserved()\n'
        '    (run / "meta.json").write_text(json.dumps(meta, indent=1))')
    # The new four-way main loop still records base-only crop allowance correctly.
    out=HERE/'scripts/81_temporal_latency.py';out.parent.mkdir(exist_ok=True);out.write_text(s)
    files=['scripts/77_k34_latency.py','scripts/k34_bounded.py','src/vtf/c1fast.py','src/vtf/speculative.py',
        'src/vtf/gpuprep.py','src/vtf/trtdyn.py','src/vtf/gpupost.py']
    (HERE/'source_manifest.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},indent=2)+'\n')


if __name__=='__main__':main()
