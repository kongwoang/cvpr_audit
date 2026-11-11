"""Separate balanced identity classifier feasibility test; no pipeline mutation."""
import argparse
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,save,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--shard',default='0/1');a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);parts=[];rows=[]
    kinds=['crop','u960','cheap','bridge']
    for file in sorted((HERE/'embeddings'/a.dataset).glob('*.npz')):
        with np.load(file,allow_pickle=False) as z:
            if not len(z['frame']):continue
            e=z['embedding'].astype(np.float32);e/=np.maximum(np.linalg.norm(e,axis=2,keepdims=True),1e-6)
            boxes=z['boxes'];sizes=np.maximum(boxes[:,:,2:]-boxes[:,:,:2],1)
            centers=(boxes[:,:,:2]+boxes[:,:,2:])/2;identity=z['identity'];mask=z['mask'];kind=z['kind'];split=z['split'];fr=z['frame'];dt=z['feats'][:,14]
            for lag in (1,2):
                ix=np.flatnonzero(np.isfinite(identity[:,lag])&(mask[:,lag]>0))
                if not len(ix):continue
                cur=e[ix,0];past=e[ix,lag];onehot=np.stack([kind[ix]==k for k in kinds],1)
                geom=np.c_[(cur*past).sum(1),(centers[ix,lag]-centers[ix,0])/sizes[ix,0],np.log(sizes[ix,lag]/sizes[ix,0]),dt[ix]*lag,onehot]
                x=np.c_[cur,past,abs(cur-past),geom].astype(np.float32)
                parts.append(x)
                rows.extend(dict(unit=file.stem,frame=int(fr[i]),kind=str(kind[i]),lag=lag,split=str(split[i]),bad=int(identity[i,lag]==0)) for i in ix)
    d=pd.DataFrame(rows);X=np.concatenate(parts);del parts;tr=np.flatnonzero(d.split=='train');va=np.flatnonzero(d.split=='val')
    assert not set(d.unit.iloc[tr])&set(d.unit.iloc[va])
    mu=X[tr].mean(0);sd=X[tr].std(0)+1e-5;X=torch.from_numpy((X-mu)/sd);Y=torch.from_numpy(d.bad.to_numpy(np.float32))
    net=nn.Sequential(nn.Linear(X.shape[1],128),nn.ReLU(),nn.Linear(128,32),nn.ReLU(),nn.Linear(32,1)).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4);pos_weight=torch.tensor(float((Y[tr]==0).sum()/(Y[tr]==1).sum()),device='cuda')
    best=-1;history=[]
    def predict():
        net.eval();out=[]
        with torch.no_grad():
            for start in range(0,len(va),2048):out.append(net(X[va[start:start+2048]].cuda())[:,0].sigmoid().cpu().numpy())
        return np.concatenate(out)
    for epoch in range(15):
        net.train();rng=np.random.default_rng(epoch)
        for ids in np.array_split(rng.permutation(tr),max(1,len(tr)//1024)):
            score=net(X[ids].cuda())[:,0];loss=F.binary_cross_entropy_with_logits(score,Y[ids].cuda(),pos_weight=pos_weight)
            opt.zero_grad();loss.backward();opt.step()
        p=predict();auc=roc_auc_score(Y[va],p);ap=average_precision_score(Y[va],p)
        history.append(dict(epoch=epoch,auc=auc,average_precision=ap));print(a.dataset,history[-1],flush=True)
        if auc>best:best=auc;state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};best_epoch=epoch
    net.load_state_dict(state);p=predict();y=d.bad.to_numpy()[va];threshold=float(np.quantile(p[y==0],.99,method='higher'))
    pred=p>threshold;false=float(pred[y==0].mean());recall=float(pred[y==1].mean());auc=float(roc_auc_score(y,p));ap=float(average_precision_score(y,p))
    out=HERE/'models'/a.dataset/'identity_pair_s0';out.mkdir(parents=True,exist_ok=True)
    val=d.iloc[va].copy();val['score']=p;val['reject']=pred;val.to_csv(out/'validation.csv',index=False)
    torch.save(dict(state=state,mu=mu,sd=sd,threshold=threshold,n_in=X.shape[1]),out/'pair.pt')
    save(out/'REPORT.json',dict(status='complete',dataset=a.dataset,train_pairs=len(tr),val_pairs=len(va),epochs=15,best_epoch=best_epoch,
        auc=auc,average_precision=ap,valid_rejection=false,switch_recall=recall,threshold=threshold,history=history,
        feasibility_gate=bool(auc>=.8 and recall>=.2 and false<=.01),pipeline_changed=False,checkpoint_sha256=sha(out/'pair.pt')))
    print('IDENTITY FINAL',auc,recall,false,flush=True)
if __name__=='__main__':main()
