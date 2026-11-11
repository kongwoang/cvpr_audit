"""Frozen C1/v3a reference predictions on the SAME regenerated holdout objects."""
import argparse
import importlib.util
import json
import numpy as np
import pandas as pd
import torch
from common import HERE,REMOTE,save,sha
from learned import demand

def loadmod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--shard',default='0/1');a=ap.parse_args()
    torch.set_num_threads(4)
    KTT=loadmod('c1_original',REMOTE/'scripts/A_method/k28_ttrain.py')
    KT=loadmod('range_original',REMOTE/'scripts/A_method/k28_train.py')
    ck=torch.load(REMOTE/'out/k28_temporal/temporal_head.pt',map_location='cpu',weights_only=False)
    head=KTT.TempHead(16);head.load_state_dict(ck['state']);head=head.cuda().eval()
    cf=REMOTE/'out/resnet18_v3a/image_refiner.pt';rc=torch.load(cf,map_location='cpu',weights_only=False)
    ref=KT.Res18x3(len(rc['fmu']));ref.load_state_dict(rc['state']);ref=ref.cuda().eval()
    rows=[]
    for file in sorted((HERE/'embeddings'/a.dataset).glob('*.npz')):
        with np.load(file,allow_pickle=False) as z:
            if not len(z['frame']):continue
            ok=(z['split']=='val')&np.isfinite(z['target'])&np.isfinite(z['gt_mono'][:,0])&(z['gt_mono'][:,0]>0)&np.isfinite(z['raw_mono'][:,:3]).all(1)
            ix=np.flatnonzero(ok)
            if not len(ix):continue
            with np.load(HERE/'training_data'/a.dataset/file.name,allow_pickle=False) as t:
                crops=t['crops'][ix,0];feats=t['feats'][ix]
            rfe=np.c_[feats[:,:12],np.ones(len(ix))];rfe=((rfe-rc['fmu'])/rc['fsd']).astype(np.float32)
            emb=z['embedding'][ix].astype(np.float32);cfe=z['fe'][ix];pred=[];factors=[]
            for start in range(0,len(ix),256):
                s=slice(start,start+256);xx=torch.from_numpy(np.c_[emb[s].reshape(-1,1536),cfe[s]]).cuda()
                img=torch.from_numpy(crops[s]).permute(0,3,1,2).cuda().float()/255
                with torch.no_grad():
                    pred.append(head.head(xx)[:,0].clamp(0,2).cpu().numpy())
                    factors.append(ref(img,torch.from_numpy(rfe[s]).cuda())[:,0].exp().cpu().numpy())
            rate=np.concatenate(pred);factor=np.concatenate(factors);mask=z['mask'][ix]
            rate[mask.sum(1)<3]=0 # frozen C1 has no rate without its three real observations
            raw=z['raw_mono'][ix];v=z['v'][ix];gt=z['gt_mono'][ix];y=np.clip(z['target'][ix],0,2)
            args=[torch.tensor(raw),torch.tensor(v),torch.tensor(rate)]
            # Apply v3a to every object for geometry audit; frame-level support is NOT simulated.
            pdem=demand(*args,torch.tensor(factor)).numpy();rawdem=demand(*args,torch.ones(len(ix))).numpy()
            gt_dem=demand(torch.tensor(gt),torch.tensor(v),torch.tensor(y),torch.ones(len(ix))).numpy()
            for n,i in enumerate(ix):
                rows.append(dict(dataset=a.dataset,unit=file.stem,index=int(i),frame=int(z['frame'][i]),kind=str(z['kind'][i]),
                    observations=int(mask[n].sum()),known_switch=bool(np.any(z['identity'][i,1:]==0)),target=float(y[n]),C1=float(rate[n]),
                    v3a_factor=float(factor[n]),true_factor=float(gt[n,0]/raw[n,0]),raw_range=float(raw[n,0]),gt_range=float(gt[n,0]),
                    raw_demand=float(rawdem[n]),v3a_demand=float(pdem[n]),gt_demand=float(gt_dem[n])))
    out=HERE/'reports/model_audit';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/(a.dataset+'_baseline.csv'),index=False)
    save(out/(a.dataset+'_STATUS.json'),dict(status='complete',samples=len(rows),v3a_sha256=sha(cf),
        caveats=['C1 embeddings cached fp16; readout fp32.','v3a on all matched objects for geometry audit, not scene support.','Original holdouts used for earlier model selection.']))
if __name__=='__main__':main()
