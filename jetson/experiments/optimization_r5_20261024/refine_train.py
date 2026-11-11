"""Frozen v3a on original TRAIN/REF-VAL crops; fit pooled/conditional gates."""
import argparse
import importlib.util
import pickle
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from common import HERE,REMOTE,save,sha
from features import action,demands,features,probability_action
from conditional import augment_features

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);a=ap.parse_args();ds=a.dataset
    torch.set_num_threads(4);started=time.time();out=HERE/'conditional_models'/ds;out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('frozen_v3a_r5',REMOTE/'scripts/A_method/k23_train_refiner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    ck=torch.load(REMOTE/'out/resnet18_v3a/image_refiner.pt',map_location='cpu',weights_only=False)
    net=m.Res18x3(len(ck['fmu']));net.load_state_dict(ck['state']);net=net.cuda().eval()
    base=np.load(HERE/'models'/ds/'data.npz');parts=[];manifest=[]
    src=REMOTE/'experiments/optimization_r3_20261023/training_data'/ds
    for f in sorted(src.glob('*.npz')):
        ix=np.flatnonzero(base['unit']==f.stem)
        if not len(ix):continue
        with np.load(f,allow_pickle=False) as z:
            if not len(z['frame']):continue
            idx=np.flatnonzero(z['kind']!='bridge');gt=z['gt_mono'][idx];matched=z['ids'][idx,0]>=0
            valid=(~matched)|(np.isfinite(gt[:,:3]).all(1)&(gt[:,0]>0)&~np.isnan(gt[:,3]))
            idx=idx[valid];assert len(idx)==len(ix)
            assert np.array_equal(z['frame'][idx],base['frame'][ix]);assert np.array_equal(z['kind'][idx],base['kind'][ix])
            assert np.allclose(z['feats'][idx,:4],base['X'][ix,11:15],rtol=0,atol=0)
            full=z['kind'][idx]!='cheap';ids=np.flatnonzero(full);X=base['X'][ix];raw=np.c_[X[:,:3],np.where(X[:,3]>0,1/np.maximum(X[:,3],1e-10),np.inf)]
            # Base float32 features can round TTC. Use exact raw for cheap and C1
            # reconstruction values for full; this affects training only.
            factors=np.ones(len(idx));cr=z['crops'][idx[ids],0];fe=z['feats'][idx[ids]]
            ff=np.c_[fe[:,:12],np.ones(len(fe),np.float32)].astype(np.float32);ff=((ff-ck['fmu'])/ck['fsd']).astype(np.float32)
            for start in range(0,len(ids),128):
                sub=ids[start:start+128];im=torch.from_numpy(cr[start:start+128]).permute(0,3,1,2).cuda().float()/255
                with torch.no_grad():o=net(im,torch.from_numpy(ff[start:start+128]).cuda()).float().cpu().numpy()
                factors[sub]=np.exp(o[:,0])
            rd=demands(raw,base['X'][ix,4]);apply=np.zeros(len(ix),bool)
            for kind in ['crop','u960']:
                for fr in np.unique(base['frame'][ix][base['kind'][ix]==kind]):
                    group=(base['frame'][ix]==fr)&(base['kind'][ix]==kind);maximum=rd[group].max()
                    threshold=3.5 if maximum>=3.5 else 1.
                    apply|=group&(rd>=threshold)
            refined=raw.copy();refined[apply,:3]*=factors[apply,None]
            newx=augment_features(X,raw,refined,X[:,4]);drop=action(rd)>action(demands(refined,X[:,4]))
            parts.append(dict(X=newx[full],y=base['y'][ix][full],split=base['split'][ix][full],unit=base['unit'][ix][full],
                frame=base['frame'][ix][full],kind=base['kind'][ix][full],matched=base['matched'][ix][full],drop=drop[full],
                raw_demand=rd[full],ref_demand=demands(refined,X[:,4])[full]))
            manifest.append(dict(unit=f.stem,sha256=sha(f),full_examples=int(full.sum()),disagreements=int(drop.sum())))
        if len(parts)%50==0:print(ds,'v3a units',len(parts),flush=True)
    d={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};np.savez_compressed(out/'data.npz',**d)
    pd.DataFrame(manifest).to_csv(out/'provenance.csv',index=False)
    tr=d['split']=='train';va=d['split']=='val';report=[];thresholds=[];preds=[]
    assert not (set(d['unit'][tr])&set(d['unit'][va]))
    for name in ['pooled','conditional']:
        use=tr if name=='pooled' else tr&d['drop'];counts=np.bincount(d['y'][use],minlength=3)
        if (counts==0).any():
            save(out/(name+'_INFEASIBLE.json'),dict(status='infeasible',class_counts=counts.tolist()));continue
        model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,min_samples_leaf=50,learning_rate=.05,l2_regularization=1.,early_stopping=False,random_state=0)
        model.fit(d['X'][use],d['y'][use]);assert np.array_equal(model.classes_,[0,1,2]);pickle.dump(model,(out/(name+'.pkl')).open('wb'))
        prob=model.predict_proba(d['X'][va]);tab=pd.DataFrame({k:d[k][va] for k in ['unit','frame','kind','matched','drop','y','raw_demand','ref_demand']});tab[['p_keep','p_decel','p_hard']]=prob;tab['model']=name;preds.append(tab)
        for scope,sel in [('full',np.ones(int(va.sum()),bool)),('disagreement',d['drop'][va])]:
            for q in [.5,.8,.95]:
                act=probability_action(prob,q)
                for level in [1,2]:
                    pred=sel&(act>=level);truth=sel&(d['y'][va]>=level);tp=int((pred&truth).sum())
                    thresholds.append(dict(dataset=ds,model=name,scope=scope,q=q,level=level,samples=int(sel.sum()),predicted=int(pred.sum()),required=int(truth.sum()),correct=tp,precision=tp/pred.sum() if pred.any() else np.nan,recall=tp/truth.sum() if truth.any() else np.nan))
        report.append(dict(model=name,train_examples=int(use.sum()),class_counts=counts.tolist(),train_units=len(set(d['unit'][use])),
            val_disagreements=int((va&d['drop']).sum()),val_disagreement_logloss=float(log_loss(d['y'][va&d['drop']],prob[d['drop'][va]],labels=[0,1,2])),sha256=sha(out/(name+'.pkl'))))
        print(ds,report[-1],flush=True)
    pd.DataFrame(thresholds).to_csv(out/'holdout_thresholds.csv',index=False)
    if preds:pd.concat(preds).to_csv(out/'holdout_predictions.csv',index=False)
    save(out/'REPORT.json',dict(status='complete',dataset=ds,models=report,units=len(manifest),seconds=time.time()-started,
        protocol_sha256=sha(HERE/'protocol.md'),test_or_dev_training=False,source_refiner_sha256=sha(REMOTE/'out/resnet18_v3a/image_refiner.pt')))

if __name__=='__main__':main()
