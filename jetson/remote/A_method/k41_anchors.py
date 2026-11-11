"""Fit fixed K20 four-second trajectory anchors from allowed training poses, CPU only."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans

B=Path(os.environ['B']);OUT=B/'results/k41/anchors';TIMES=np.arange(1,9)*.5


def windows(t,xyz,rot,start_times):
    out=[]
    for start in start_times:
        if start+4>t[-1]:continue
        origin=np.array([np.interp(start,t,xyz[:,i]) for i in range(3)])
        j=int(np.argmin(np.abs(t-start)))
        future=np.stack([np.interp(start+TIMES,t,xyz[:,i]) for i in range(3)],axis=1)
        out.append(((future-origin)@rot[j])[:,:2])
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True);meta=[];sources=[]
    def source(p):sources.append(dict(path=str(p),md5=hashlib.md5(p.read_bytes()).hexdigest()))
    split=B/'results/nusc_full/split.csv';source(split);sp=pd.read_csv(split)
    from rap.nusc import NuScenesDB
    from rap.paths import NUSCENES_TRAINVAL
    db=NuScenesDB(NUSCENES_TRAINVAL,'v1.0-trainval');by={db.scene_name(s):s for s in db.scenes};allrows=[]
    for name in sorted(sp[(sp['set']=='new')&(sp.split=='train')].seq):
        tokens=db.samples(by[name]);t=np.array([db._t['sample'][k]['timestamp']/1e6 for k in tokens]);rt=[db._global_to_ego(k) for k in tokens]
        xyz=np.array([x[1] for x in rt]);rot=np.array([x[0] for x in rt]);rows=windows(t,xyz,rot,t)
        meta.append(dict(dataset='nusc',unit=name,windows=len(rows),split='train'));allrows.extend(rows)
    for nm in ('scene','sample','sample_data','ego_pose'):source(Path(NUSCENES_TRAINVAL)/'v1.0-trainval'/f'{nm}.json')
    fit('nusc',allrows)
    split=B/'results/k23/av2_split.csv';source(split);sp=pd.read_csv(split);train=sorted(sp[sp.role=='REF-TRAIN'].log)
    manifest=Path(__file__).parent/'k40r_verified_av2.json';source(manifest);dev=set(json.loads(manifest.read_text())['logs']);assert not set(train)&dev
    allrows=[]
    for log in train:
        p=B/'datasets/av2/sensor/train'/log/'city_SE3_egovehicle.feather';source(p);d=pd.read_feather(p).sort_values('timestamp_ns')
        t=d.timestamp_ns.to_numpy(np.float64)*1e-9;xyz=d[['tx_m','ty_m','tz_m']].to_numpy(float)
        rot=Rotation.from_quat(d[['qx','qy','qz','qw']].to_numpy(float)).as_matrix()
        rows=windows(t,xyz,rot,np.arange(t[0],t[-1]-4+.000001,.5));allrows.extend(rows);meta.append(dict(dataset='av2',unit=log,windows=len(rows),split='REF-TRAIN'))
    fit('av2',allrows)
    pd.DataFrame(meta).to_csv(OUT/'training_units.csv',index=False);pd.DataFrame(sources).to_csv(OUT/'sources.csv',index=False)
    (OUT/'run.json').write_text(json.dumps(dict(script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),K=20,seed=0,n_init=10,horizon_s=4,waypoint_dt_s=.5,ahd_transfer='av2',test_data_used=False),indent=2)+'\n')


def fit(ds,rows):
    x=np.asarray(rows,float);assert len(x)>=20 and x.shape[1:]==(8,2) and np.isfinite(x).all()
    km=KMeans(n_clusters=20,random_state=0,n_init=10).fit(x.reshape(len(x),-1))
    np.savez(OUT/f'{ds}.npz',anchors=km.cluster_centers_.reshape(20,8,2),counts=np.bincount(km.labels_,minlength=20),times=TIMES)
    print(ds,len(x),'training windows; K20 anchors saved',flush=True)


if __name__=='__main__':main()
