"""Port fixed point-fallback policy to isolated existing-engine Jetson harness."""
from pathlib import Path
import json
import hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'experiments/verified_jetson_20261023';R3=ROOT/'experiments/optimization_r3_20261023'
def once(s,a,b):
    assert s.count(a)==1,(s.count(a),a[:100]);return s.replace(a,b)
def main():
    for f in ('helpers.py','fast.py','recovery_gpu.py','compare.py','verified_log_manifest.json','source_manifest.json'):
        (HERE/f).write_bytes((OLD/f).read_bytes())
    for f in ('learned.py','bridge.py','common.py'):(HERE/f).write_bytes((R3/f).read_bytes())
    runtime=(R3/'inference.py').read_text()
    runtime=once(runtime,'e=np.zeros((len(valid),3,512),np.float32)','e=torch.zeros((len(valid),3,512),dtype=torch.float32,device="cuda")')
    runtime=once(runtime,"torch.from_numpy(e).cuda(),torch.from_numpy(fe).cuda()","e,torch.from_numpy(fe).cuda()")
    (HERE/'inference.py').write_text(runtime)
    s=(OLD/'scripts/83_verified_av2.py').read_text().replace(str(OLD),str(HERE))
    s=once(s,'from fast import FastRetro','from fast import FastRetro\nfrom runtime_fast import MissingRuntime')
    s=once(s,'    ap.add_argument("--helper",choices=["original","fast"],required=True)',
        '    ap.add_argument("--helper",choices=["original","fast"],required=True)\n'
        '    ap.add_argument("--adapter_encoder",choices=["trt32","t32"],default="trt32")')
    s=once(s,'f"verified_cfg_{args.helper}_{mode}_{args.dataset}_{args.config}"',
        'f"r3_point_{args.adapter_encoder}_{mode}_{args.dataset}_{args.config}"')
    s=once(s,'    HEAD = CF.Head(ckh)',
        '''    HEAD = CF.Head(ckh)
    ADAPTER_ENCODER=ENC['trt32'] if args.adapter_encoder=='trt32' else CF.Encoder('t32',ENG,ckh)
    adapters={}
    def adapter(name):
        if name not in adapters:adapters[name]=MissingRuntime(DS,head,ADAPTER_ENCODER)
        return adapters[name]''')
    s=once(s,'            gpu_history.clear()',
        '            gpu_history.clear()\n            for ar in adapters.values():ar.bank={}')
    s=once(s,'        RETROS={}',
        '        RETROS={}\n        for ar in adapters.values():ar.frame(fid)')
    s=once(s,'            c1 = C1F[p]',
        '''            c1 = C1F[p]
            def apply_missing(full,h,chn,mono):
                if '_missing' not in name:return mono
                missing=set(range(len(full['xyxy'])))-{i for i,_,_ in chn}
                if not missing:return mono
                ar=adapter(name);ar.fr=fid;ar.shared_bank=bank
                zz=K.predicted_geometry({'xyxy':full['xyxy'],'coarse':full['coarse']},None,cam.calib,cam.cam_h,dt,cam_to_ego=(R,t))
                pred,masks=ar.infer('geometry',full,mono,zz,h,chn,hist.get('cheap_history',{}),imgs,img,W,H,v,dt,only_indices=missing)
                out=[np.asarray(q).copy() for q in mono]
                for i in missing:
                    if np.isfinite(pred[i,0]):
                        rate=float(pred[i,0]);out[3][i]=min(1/rate,1e3) if rate>1e-3/dt else np.inf
                return tuple(out)''')
    a=s.index('        def lift_c1_fast(');b=s.index('        def lift_c(name',a)
    block=s[a:b].replace('return z, lo, hi, ttc','return apply_missing(full,h,chn,(z,lo,hi,ttc))')
    s=s[:a]+block+s[b:]
    s=once(s,'variants = [("ours",None),("ours_retro",None),("ours_b2A",2),("ours_b2A_retro",2)]',
        'variants = [("ours",None),("ours_b2A_retro",2),("ours_b2A_missing_retro",2)]')
    s=once(s,'rotation = (args.rep - 1) % 4','rotation = (args.rep - 1) % len(variants)')
    s=once(s,'                recovery=RETROS.get(nm)',
        '''                ar=adapters.get(nm)
                rec['adapter_encoded_'+nm]=ar.encoded if ar else 0
                rec['adapter_items_'+nm]=ar.head_items if ar else 0
                recovery=RETROS.get(nm)''')
    s=once(s,'        hist["cheap_prev"] = {"xyxy": d["xyxy"], "coarse": d["coarse"]}',
        '''        chist=hist.setdefault('cheap_history',{})
        chist['s2']=chist.get('s1');chist['c2']=chist.get('c1')
        chist['s1']=np.asarray(d['xyxy']).copy();chist['c1']=np.asarray(d['coarse']).copy()
        hist["cheap_prev"] = {"xyxy": d["xyxy"], "coarse": d["coarse"]}''')
    folder=HERE/'scripts';folder.mkdir(exist_ok=True);(folder/'84_masked_point.py').write_text(s)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    sources.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.py')})
    (HERE/'generated_manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
    print('Built isolated Jetson point adapter')
if __name__=='__main__':main()
