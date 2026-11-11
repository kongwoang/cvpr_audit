"""Train compact fallback only on TRAIN bridge examples, select holdout MAE."""
import argparse
import copy
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from common import HERE,REMOTE,save,sha
from compact import Compact

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--shard',default='0/1');a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);parts=[];device=torch.device('cpu')
    keys=['fe','extra','mask','target','split','unit','frame','kind']
    for p in sorted((HERE/'embeddings'/a.dataset).glob('*.npz')):
        with np.load(p,allow_pickle=False) as z:
            if not len(z['frame']):continue
            ok=(z['kind']=='bridge')&np.isfinite(z['target']);parts.append({k:z[k][ok] for k in keys})
    d={k:np.concatenate([x[k] for x in parts]) for k in keys};tr=np.flatnonzero(d['split']=='train');va=np.flatnonzero(d['split']=='val')
    X=np.c_[d['fe'],d['extra']];mu=X[tr].mean(0);sd=X[tr].std(0)+1e-5
    net=Compact(mu,sd).to(device);opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,20)
    tensors={k:torch.from_numpy(d[k].astype(np.float32)) for k in ('fe','extra','mask','target')};tensors['target'].clamp_(0,2)
    history=[];best=np.inf
    def forward(ids):
        b={k:v[ids].to(device) for k,v in tensors.items()};return net(None,b['fe'],b['mask'],b['extra'])[:,0],b['target']
    def predict():
        net.eval();out=[]
        with torch.no_grad():
            for start in range(0,len(va),1024):out.append(forward(va[start:start+1024])[0].clamp(0,2).cpu().numpy())
        return np.concatenate(out)
    for epoch in range(20):
        net.train();rng=np.random.default_rng(epoch);losses=[]
        for ids in np.array_split(rng.permutation(tr),max(1,len(tr)//512)):
            p,y=forward(ids);loss=F.smooth_l1_loss(p,y,beta=.02);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss))
        sched.step();p=predict();mae=float(np.mean(abs(p-np.clip(d['target'][va],0,2))))
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),val_mae=mae);history.append(row);print(a.dataset,row,flush=True)
        if mae<best:best=mae;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};best_epoch=epoch
    net.load_state_dict(state);p=predict();ck=torch.load(REMOTE/'out/k28_temporal/temporal_head.pt',map_location='cpu',weights_only=False)
    out=HERE/'models'/a.dataset/'compact_s0';out.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=state,mu=mu,sd=sd,fmu=ck['fmu'],fsd=ck['fsd']),out/'compact.pt')
    dval=pd.DataFrame({k:d[k][va] for k in ('unit','frame','kind')});dval['target']=np.clip(d['target'][va],0,2);dval['rate']=p;dval['observations']=d['mask'][va].sum(1)
    dval.to_csv(out/'validation.csv',index=False)
    save(out/'REPORT.json',dict(status='complete',dataset=a.dataset,train_examples=len(tr),validation_examples=len(va),epochs=20,best_epoch=best_epoch,
        best_val_mae=best,history=history,device=str(device),parameters=sum(x.numel() for x in net.parameters()),sha256=sha(out/'compact.pt')))
if __name__=='__main__':main()
