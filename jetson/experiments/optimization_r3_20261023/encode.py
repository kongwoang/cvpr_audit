"""Cache frozen C1 embeddings for completed training units; no fitting or dev use."""
import argparse
import importlib.util
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,REMOTE,save,sha
from learned import extras

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--shard',default='0/1')
    a=ap.parse_args();si,sn=map(int,a.shard.split('/'));torch.set_num_threads(4)
    spec=importlib.util.spec_from_file_location('frozen_c1',REMOTE/'scripts/A_method/k28_ttrain.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    cf=REMOTE/'out/k28_temporal/temporal_head.pt';ck=torch.load(cf,map_location='cpu',weights_only=False)
    net=m.TempHead(16);net.load_state_dict(ck['state']);net=net.cuda().eval()
    data=HERE/'training_data'/a.dataset;out=HERE/'embeddings'/a.dataset;out.mkdir(parents=True,exist_ok=True)
    files=sorted(data.glob('*.npz'));assert files
    summaries=[]
    for i,f in enumerate(files):
        if i%sn!=si:continue
        target=out/f.name
        if target.exists():continue
        with np.load(f,allow_pickle=False) as z:
            if len(z['frame'])==0:
                np.savez_compressed(target,frame=z['frame']);continue
            d={k:z[k] for k in z.files if k!='crops'};crops=z['crops'];n=len(crops);enc=[]
            for start in range(0,n,128):
                x=torch.from_numpy(crops[start:start+128]).permute(0,1,4,2,3).reshape(-1,3,64,64).cuda().float()/255
                x=F.interpolate(x.flip(1),size=128,mode='bilinear',align_corners=False)
                with torch.no_grad():e=net.cnn((x-net.mu)/net.sd).reshape(-1,3,512)
                enc.append(e.cpu().numpy().astype(np.float16))
            d['embedding']=np.concatenate(enc);d['extra']=extras(d['feats'],d['mask'],d['kind'],d['v'],d['raw_mono'],d['boxes'])
            d['fe']=((d['feats']-ck['fmu'])/ck['fsd']).astype(np.float32)
            assert np.isfinite(d['fe']).all()
            tmp=target.with_suffix('.tmp.npz');np.savez_compressed(tmp,**d);tmp.replace(target)
            summaries.append(dict(unit=f.stem,samples=n,source_sha256=sha(f),embedding_sha256=sha(target)))
            print(a.dataset,f.stem,n,flush=True)
    save(out/('DONE_'+a.shard.replace('/','of')+'.json'),dict(status='complete',checkpoint_sha256=sha(cf),units=summaries))
if __name__=='__main__':main()
