"""Training-only ordinal evidence, including annotation-unmatched negatives."""
import argparse
import pickle
import time
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, confusion_matrix
from common import HERE, REMOTE, save, sha
from features import features, demands, action, FEATURES, probability_action

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);a=ap.parse_args()
    torch.set_num_threads(1);started=time.time()
    ck=torch.load(REMOTE/'out/k28_temporal/temporal_head.pt',map_location='cpu',weights_only=False)
    head=torch.nn.Sequential(torch.nn.Linear(1552,256),torch.nn.ReLU(),torch.nn.Linear(256,1))
    head.load_state_dict({k[5:]:v for k,v in ck['state'].items() if k.startswith('head.')});head.eval()
    folder=REMOTE/'experiments/optimization_r3_20261023/embeddings'/a.dataset
    parts=[];provenance=[]
    dev_units={p.stem for p in (REMOTE/'results/k30/dev_devref'/a.dataset).glob('*.pkl')}
    for f in sorted(folder.glob('*.npz')):
        assert f.stem not in dev_units,f
        with np.load(f,allow_pickle=False) as d:
            if not len(d['frame']):continue
            keep=d['kind']!='bridge';idx=np.flatnonzero(keep)
            raw=d['raw_mono'][idx].copy();fe=d['feats'][idx];full=d['kind'][idx]!='cheap'
            history=(d['mask'][idx].sum(1)==3);ci=np.flatnonzero(full&history)
            raw[full,3]=np.inf
            emb=d['embedding'][idx]
            for s in range(0,len(ci),1024):
                j=ci[s:s+1024]
                x=np.c_[emb[j].reshape(len(j),1536).astype(np.float32),d['fe'][idx[j]]]
                with torch.no_grad():r=head(torch.from_numpy(x))[:,0].numpy().clip(0,2)
                raw[j,3]=np.where(r>1e-3/fe[j,14],1/np.maximum(r,1e-10),np.inf)
            gt=d['gt_mono'][idx];matched=d['ids'][idx,0]>=0
            valid=(~matched)|(np.isfinite(gt[:,:3]).all(1)&(gt[:,0]>0)&~np.isnan(gt[:,3]))
            demand=np.zeros(len(idx));demand[matched&valid]=demands(gt[matched&valid],d['v'][idx][matched&valid])
            X=features(raw,d['v'][idx],fe[:,6],fe[:,7].astype(int),fe[:,:4],full,history)
            part=dict(X=X[valid],y=action(demand)[valid],demand=demand[valid],matched=matched[valid],
                      unit=d['unit'][idx][valid],frame=d['frame'][idx][valid],split=d['split'][idx][valid],
                      kind=d['kind'][idx][valid],history=history[valid],raw_demand=demands(raw,d['v'][idx])[valid])
            parts.append(part)
            provenance.append(dict(unit=f.stem,split=str(d['split'][0]),source_sha256=sha(f),
                                   source_samples=len(d['frame']),used_samples=int(valid.sum()),
                                   unmatched=int((~matched).sum()),invalid_matched=int((matched&~valid).sum())))
        if len(parts)%50==0:print(a.dataset,'prepared units',len(parts),flush=True)
    d={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    tr=d['split']=='train';va=d['split']=='val'
    assert tr.any() and va.any() and np.all(tr|va)
    assert not (set(d['unit'][tr])&set(d['unit'][va]))
    out=HERE/'models'/a.dataset;out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'data.npz',**d)
    pd.DataFrame(provenance).to_csv(out/'provenance.csv',index=False)
    report=[];cal=[];predictions=[]
    for name in ['all','matched']:
        use=tr if name=='all' else tr&d['matched']
        model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,min_samples_leaf=50,
            learning_rate=.05,l2_regularization=1.,early_stopping=False,random_state=0)
        model.fit(d['X'][use],d['y'][use]);assert np.array_equal(model.classes_,[0,1,2])
        with (out/(name+'.pkl')).open('wb') as h:pickle.dump(model,h)
        prob=model.predict_proba(d['X'][va]);ix=np.flatnonzero(va)
        tab=pd.DataFrame({k:d[k][va] for k in ['unit','frame','kind','history','matched','y','demand','raw_demand']})
        tab[['p_keep','p_decel','p_hard']]=prob;tab['model']=name;predictions.append(tab)
        report.append(dict(model=name,train_examples=int(use.sum()),train_units=len(set(d['unit'][use])),
            train_class_counts=np.bincount(d['y'][use],minlength=3).tolist(),
            val_examples=int(va.sum()),val_units=len(set(d['unit'][va])),val_logloss=float(log_loss(d['y'][va],prob)),
            confusion=confusion_matrix(d['y'][va],prob.argmax(1),labels=[0,1,2]).tolist(),
            sha256=sha(out/(name+'.pkl'))))
        for scope,sel in [('all',np.ones(len(ix),bool)),('verified',d['kind'][va]!='cheap'),
                          ('verified_missing',(d['kind'][va]!='cheap')&~d['history'][va]),
                          ('unmatched',~d['matched'][va])]:
            for q in [.5,.8,.95]:
                act=probability_action(prob,q)
                for level in [1,2]:
                    pred=(act>=level)&sel;truth=(d['y'][va]>=level)&sel;tp=int((pred&truth).sum())
                    cal.append(dict(dataset=a.dataset,model=name,scope=scope,q=q,level=level,
                        samples=int(sel.sum()),predicted=int(pred.sum()),required=int(truth.sum()),correct=tp,
                        precision=tp/pred.sum() if pred.any() else np.nan,
                        recall=tp/truth.sum() if truth.any() else np.nan))
        print(a.dataset,name,report[-1],flush=True)
    pd.concat(predictions).to_csv(out/'holdout_predictions.csv',index=False)
    pd.DataFrame(cal).to_csv(out/'holdout_thresholds.csv',index=False)
    save(out/'REPORT.json',dict(status='complete',dataset=a.dataset,models=report,features=FEATURES,
        training_units=len(provenance),dev_overlap=0,invalid_matched=sum(p['invalid_matched'] for p in provenance),
        unmatched=sum(p['unmatched'] for p in provenance),protocol_sha256=sha(HERE/'protocol.md'),
        frozen_c1_sha256=sha(REMOTE/'out/k28_temporal/temporal_head.pt'),seconds=time.time()-started,
        caveat='Annotations define unmatched negatives; not manually certified false detections. C1 embeddings are cached float16.'))

if __name__=='__main__':main()
