"""Train point/geometric and decision-aware adapters using training holdouts only."""
import argparse
import copy
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from common import HERE,REMOTE,save,sha
from learned import Adapter,decode,demand

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--shard',default='0/1')
    ap.add_argument('--objective',choices=['geometry','decision'],required=True);ap.add_argument('--seed',type=int,default=0)
    a=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(a.seed)
    files=sorted((HERE/'embeddings'/a.dataset).glob('*.npz'));parts=[];manifest=[]
    keys=['embedding','fe','extra','mask','identity','target','raw_mono','gt_mono','v','split','unit','frame','kind']
    for f in files:
        with np.load(f,allow_pickle=False) as z:
            if not len(z['frame']):continue
            good=np.isfinite(z['target'])&np.isfinite(z['gt_mono'][:,0])&(z['gt_mono'][:,0]>0)&np.isfinite(z['raw_mono'][:,:3]).all(1)
            parts.append({k:z[k][good] for k in keys});manifest.append(dict(name=f.name,sha256=sha(f),samples=int(good.sum())))
    d={k:np.concatenate([x[k] for x in parts]) for k in keys};del parts
    tr=np.flatnonzero(d['split']=='train');va=np.flatnonzero(d['split']=='val');assert len(tr)>0 and len(va)>0
    assert not set(d['unit'][tr])&set(d['unit'][va])
    ck=torch.load(REMOTE/'out/k28_temporal/temporal_head.pt',map_location='cpu',weights_only=False)
    net=Adapter(d['extra'].shape[1],ck['state']).cuda();opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4)
    epochs=20;sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    tensor={k:torch.from_numpy(d[k].astype(np.float32)) for k in ('embedding','fe','extra','mask','identity','target','raw_mono','gt_mono','v')}
    tensor['target']=tensor['target'].clamp(0,2)
    target_log=torch.log(tensor['gt_mono'][:,0]/tensor['raw_mono'][:,0]).clamp(-1.1,1.1)
    gt_rate=tensor['target'];gt_dem=demand(tensor['gt_mono'],tensor['v'],gt_rate,torch.ones(len(gt_rate)))
    od=HERE/'models'/a.dataset/(a.objective+'_s'+str(a.seed));od.mkdir(parents=True,exist_ok=True)
    history=[];best=float('inf');best_state=None
    def batch(ids):return {k:v[ids].cuda() for k,v in tensor.items()}
    def predict(ids):
        net.eval();out=[]
        with torch.no_grad():
            for j in range(0,len(ids),1024):
                b=batch(ids[j:j+1024]);out.append(net(b['embedding'],b['fe'],b['mask'],b['extra']).cpu())
        return torch.cat(out)
    for ep in range(epochs):
        net.train();rng=np.random.default_rng(a.seed*1000+ep);losses=[]
        for ids in np.array_split(rng.permutation(tr),max(1,len(tr)//512)):
            b=batch(ids);o=net(b['embedding'],b['fe'],b['mask'],b['extra']);rate,factor,q,valid=decode(o)
            rate_loss=F.smooth_l1_loss(o[:,0],b['target'],beta=.02)
            range_loss=F.smooth_l1_loss(o[:,1],target_log[ids].cuda(),beta=.05)
            err=b['target'][:,None]-o[:,2:5];quantiles=torch.tensor([.1,.5,.9],device='cuda')
            pinball=torch.maximum(quantiles*err,(quantiles-1)*err).mean()
            ident=b['identity'][:,1:];known=torch.isfinite(ident)&(b['mask'][:,1:]>0)
            iloss=F.binary_cross_entropy_with_logits(o[:,5:7][known],ident[known]) if known.any() else o.sum()*0
            loss=rate_loss+.2*range_loss+.2*pinball+.02*iloss
            if a.objective=='decision':
                pdem=demand(b['raw_mono'],b['v'],rate,factor,soft=True)
                loss=loss+.2*F.smooth_l1_loss(torch.log1p(pdem),torch.log1p(gt_dem[ids].cuda()),beta=.1)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append(float(loss))
        sched.step();p=predict(va);rate,factor,q,valid=decode(p)
        dem=demand(tensor['raw_mono'][va],tensor['v'][va],rate,factor)
        rm=float((rate-tensor['target'][va]).abs().mean());zm=float((factor.log()-target_log[va]).abs().mean())
        dm=float((torch.log1p(dem)-torch.log1p(gt_dem[va])).abs().mean())
        score=rm+.2*zm+(.2*dm if a.objective=='decision' else 0)
        row=dict(epoch=ep,train_loss=float(np.mean(losses)),val_rate_mae=rm,val_range_log_mae=zm,val_log_demand_mae=dm,selection_score=score)
        history.append(row);print(a.dataset,a.objective,row,flush=True)
        if score<best:best=score;best_state=copy.deepcopy(net.cpu().state_dict());net.cuda();best_epoch=ep
    net.load_state_dict(best_state);p=predict(va);rate,factor,q,valid=decode(p)
    # Retain all validation predictions, including missing and wrong-ID categories.
    df=pd.DataFrame({k:d[k][va] for k in ('unit','frame','kind')})
    df['target']=tensor['target'][va].numpy();df['rate']=rate.numpy();df['factor']=factor.numpy();df['true_factor']=target_log[va].exp().numpy()
    df['true_factor_unclipped']=d['gt_mono'][va,0]/d['raw_mono'][va,0]
    df['raw_range']=d['raw_mono'][va,0];df['gt_range']=d['gt_mono'][va,0]
    for i,name in enumerate(('q10','q50','q90')):df[name]=q[:,i].numpy()
    df['observations']=d['mask'][va].sum(1);df['known_switch']=np.any(d['identity'][va,1:]==0,axis=1)
    df['identity1']=d['identity'][va,1];df['identity2']=d['identity'][va,2]
    df['pred_identity1']=valid[:,0].numpy();df['pred_identity2']=valid[:,1].numpy()
    df['pred_demand']=demand(tensor['raw_mono'][va],tensor['v'][va],rate,factor).numpy();df['gt_demand']=gt_dem[va].numpy()
    df.to_csv(od/'validation.csv',index=False)
    torch.save(dict(state=best_state,n_extra=d['extra'].shape[1],fmu=ck['fmu'],fsd=ck['fsd'],objective=a.objective,seed=a.seed),od/'adapter.pt')
    save(od/'REPORT.json',dict(status='complete',dataset=a.dataset,objective=a.objective,seed=a.seed,epochs=epochs,best_epoch=best_epoch,
        train_examples=len(tr),val_examples=len(va),train_units=len(set(d['unit'][tr])),val_units=len(set(d['unit'][va])),history=history,
        embedding_manifest=manifest,checkpoint_sha256=sha(od/'adapter.pt')))
if __name__=='__main__':main()
