"""P01: official NAVSIM v1.1 PDMS for the fixed K31 selector's saved choices.

Original metric caches are absent. Rebuild each with the installed official
processor and original benchmark scene, score all requested trajectories, then
remove only that newly generated temporary file. No detector or policy fitting.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import dataclasses
import fcntl
import hashlib
import json
import lzma
import os
from pathlib import Path
import pickle
import shutil
import socket
import time
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
import k31_nav_p2 as K
from navsim.common.dataclasses import SceneFilter,SensorConfig,Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

B=Path(os.environ['B']);X=B/'results/navsim_exp';OLD=B/'results/k31/p2'
BASE={'cheap':'pi_cheap','full':'pi_full960','field':'pi_real','oracle':'pi_oracle','hybrid':None}
ALL=tuple(BASE)+('all','selected','selected_support','geometric')


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(path,data):
    tmp=path.with_suffix('.writing.pkl')
    with tmp.open('wb') as f:pickle.dump(data,f);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def json_save(path,data):
    tmp=path.with_suffix('.writing.json');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def trajectory(selector,index):
    assert 0<=index<15
    poses=selector.P[index,5:41:5,:].copy()
    poses[:,:2]-=K.EGO_RA*np.stack([np.cos(poses[:,2]),np.sin(poses[:,2])],1)
    assert poses.shape==(8,3)
    return Trajectory(poses.astype(np.float32),TrajectorySampling(time_horizon=4,interval_length=.5))


def compute_log(job):
    log,tokens,rows,out_s,input_s,deadline,parity_tokens=job
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    out=Path(out_s);source=OLD/(log+'.pkl');saved={r['token']:r for r in pickle.load(source.open('rb'))}
    assert not (out/'FAILED.json').exists(),'another worker failed; stop before new work'
    unit=out/(log+'.pkl');existing=pickle.load(unit.open('rb')) if unit.exists() else {}
    done={(tok,row) for tok,rec in existing.items() for row in rec.get('scores',{})}
    if all((t,r) in done for t in tokens for r in rows):return log,len(tokens),'checkpoint'
    cp0=time.process_time();wall0=time.monotonic()
    cent_p=X/'k30_center'/(log+'.pkl');det_p=X/'dets_k30b'/(log+'.pkl');raw_p=K.DATA/'navsim_logs/test'/(log+'.pkl')
    centers=pickle.load(cent_p.open('rb'));dets=pickle.load(det_p.open('rb'))
    raw={r['token']:r for r in pickle.load(raw_p.open('rb'))}
    extras={}
    if any(r not in BASE for r in rows):
        records=pickle.load((Path(input_s)/(log+'.pkl')).open('rb'))
        extras={r['token']:r for r in records}
    loader=SceneLoader(K.DATA/'navsim_logs/test',K.DATA/'sensor_blobs/test',
        SceneFilter(num_history_frames=4,num_future_frames=10,frame_interval=1,has_route=True,log_names=[log],tokens=tokens),SensorConfig.build_no_sensors())
    assert set(loader.tokens)==set(tokens),(log,'scene manifest mismatch',len(loader.tokens),len(tokens))
    cfg=Path(os.environ['NAVSIM_DEVKIT_ROOT'])/'navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml'
    params=OmegaConf.load(cfg);simulator=instantiate(params.simulator);scorer=instantiate(params.scorer)
    temp=out/'temporary_metric_cache'/str(os.getpid());temp.mkdir(parents=True,exist_ok=True)
    proc=MetricCacheProcessor(cache_path=str(temp),force_feature_computation=False)
    gz=float((X/'dets/ground_z.txt').read_text());priors=pickle.load((X/'dets/size_priors.pkl').open('rb'))
    base_path=B/'results/k20'/('base__'+log+'.pkl')
    old_plans={r['token']:r['poses'] for r in pickle.load(base_path.open('rb'))} if parity_tokens else {}
    sources={str(p.relative_to(B)):digest(p) for p in (source,cent_p,det_p,raw_p,cfg)}
    complete=0
    for tok in tokens:
        assert not (out/'FAILED.json').exists(),'another worker failed; stop at token boundary'
        if all((tok,r) in done for r in rows):continue
        assert time.time()<deadline,'P01 aggregate CPU ceiling (conservative wall bound)'
        assert shutil.disk_usage(B).free>=200*1024**3
        rec=saved[tok];fr=raw[tok];ann=fr['anns'];cam=K.NP.Cam(*dets[tok]['cam']);cl=centers.get(tok)
        cl=cl if isinstance(cl,np.ndarray) and len(cl)>=2 else None
        speed=float(np.hypot(*np.asarray(fr['ego_dynamic_state'],float)[:2]));assert speed==rec['v0']
        sel=K.Selector(cl,speed)
        for br,col in (('cheap','pi_cheap'),('full960','pi_full960')):
            base,objects=K.branch_objects(ann,dets[tok][br],cam,gz,priors)
            scene_objects=K.Scene(ann,base,zip(objects,np.asarray(dets[tok][br]['xyxy'],float).reshape(-1,4)))
            assert sel.pi(scene_objects.o)==rec[col],(log,tok,br,'selector parity')
        hybrid_index=None
        if 'hybrid' in rows:
            hd=dets[tok]['hybrid_pdm']
            base,objects=K.branch_objects(ann,hd,cam,gz,priors)
            hs=K.Scene(ann,base,zip(objects,np.asarray(hd['xyxy'],float).reshape(-1,4)))
            hybrid_index=sel.pi(hs.o)
        scene=loader.get_scene_from_token(tok)
        if complete<parity_tokens:
            idx=scene.scene_metadata.num_history_frames-1
            cheap=K.NP.branch_annotations(scene.frames[idx].annotations,dets[tok]['cheap'],cam,gz,priors,'cheap')
            frames=list(scene.frames);frames[idx]=dataclasses.replace(frames[idx],annotations=cheap)
            probe=K.NP.plan(MetricCacheProcessor(None,False),dataclasses.replace(scene,frames=frames))
            assert np.array_equal(probe.poses,old_plans[tok]),(log,tok,'original PDM cheap plan parity')
        scenario=NavSimScenario(scene,map_root=os.environ['NUPLAN_MAPS_ROOT'],map_version='nuplan-maps-v1.0')
        t0=time.monotonic();proc.compute_metric_cache(scenario)
        cache_path=temp/scenario.log_name/scenario.scenario_type/scenario.token/'metric_cache.pkl'
        with lzma.open(cache_path,'rb') as f:cache=pickle.load(f)
        cache_meta=dict(sha256=digest(cache_path),bytes=cache_path.stat().st_size,build_wall_seconds=time.monotonic()-t0)
        item=existing.setdefault(tok,dict(log=log,token=tok,scores={},trajectories={},selected={},cache=cache_meta))
        reuse={}
        for name in rows:
            if (tok,name) in done:continue
            if name in BASE:
                index=int(hybrid_index if name=='hybrid' else rec[BASE[name]]);tr=trajectory(sel,index)
            else:
                index=int(extras[tok]['selected'][name]);tr=trajectory(sel,index)
                assert np.array_equal(tr.poses,np.asarray(extras[tok]['trajectories'][name],np.float32))
            key=tr.poses.tobytes();score_start=time.monotonic();hit=key in reuse
            if not hit:
                result=pdm_score(cache,tr,simulator.proposal_sampling,simulator,scorer)
                reuse[key]={k:float(v) for k,v in dataclasses.asdict(result).items()}
                assert all(np.isfinite(v) for v in reuse[key].values())
            item['scores'][name]=dict(reuse[key]);item['trajectories'][name]=tr.poses;item['selected'][name]=index
            item.setdefault('scoring',{})[name]=dict(reused_identical_trajectory=hit,wall_seconds=time.monotonic()-score_start)
        item['selector_parity']=True;item['pdm_cheap_parity']=complete<parity_tokens
        dump(unit,existing)
        # Only this process's freshly reconstructed cache; original records remain untouched.
        assert temp in cache_path.parents;cache_path.unlink()
        complete+=1
        if complete%10==0:print(log,complete,'tokens scored',flush=True)
    meta=dict(log=log,tokens=len(tokens),rows=rows,cpu_seconds=time.process_time()-cp0,wall_seconds=time.monotonic()-wall0,
              host=socket.gethostname(),source_sha256=sources,script_sha256=digest(Path(__file__)))
    meta_path=out/(log+'.done.json')
    if meta_path.exists():
        previous=json.loads(meta_path.read_text());meta['cpu_seconds']+=previous['cpu_seconds'];meta['previous']=previous
    json_save(meta_path,meta)
    return log,len(tokens),'complete'


def worker(job):
    out=Path(job[3]);usage=out/'usage';usage.mkdir(exist_ok=True);start=time.process_time();wall=time.time()
    try:return compute_log(job)
    except BaseException as e:
        json_save(out/'FAILED.json',dict(log=job[0],error=repr(e),unix=time.time()))
        raise
    finally:
        json_save(usage/(job[0]+'_'+str(time.time_ns())+'.json'),dict(cpu_seconds=time.process_time()-start,
                  elapsed_seconds=time.time()-wall,log=job[0],pid=os.getpid()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['navtest'],default='navtest')
    ap.add_argument('--rows',default='cheap,full,field,oracle');ap.add_argument('--inputs',default='results/k41/pip_nav')
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--out',default='results/k41/selector_metrics')
    ap.add_argument('--smoke',type=int,default=0);ap.add_argument('--parity_tokens',type=int,default=0)
    args=ap.parse_args();assert 1<=args.workers<=4 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    names=args.rows.split(',');assert len(set(names))==len(names) and set(names)<=set(ALL)
    out=B/args.out;out.mkdir(parents=True,exist_ok=True);assert B in out.resolve().parents
    assert not (out/'FAILED.json').exists(),'retain failed attempt and select a fresh output directory'
    if (out/'state.json').exists():
        assert json.loads((out/'state.json').read_text())['script_sha256']==digest(Path(__file__)),'code changed since checkpoint'
    lock=(B/'logs/k41/cpu_planner.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        todo=[]
        for p in sorted(OLD.glob('*.pkl')):
            records=pickle.load(p.open('rb'));tokens=sorted(r['token'] for r in records)
            assert len(set(tokens))==len(tokens)
            todo.append((p.stem,tokens))
        assert len(todo)==136 and sum(len(t) for _,t in todo)==12146
        if args.smoke:todo=[(todo[0][0],todo[0][1][:args.smoke])]
        all_prior=list((B/'results/k41').glob('selector_metrics*/usage/*.json'))
        spent=sum(json.loads(p.read_text())['cpu_seconds'] for p in all_prior)
        remaining=100*3600-spent;assert remaining>0
        # At most workers single-thread processes: this wall bound overestimates CPU use.
        deadline=time.time()+remaining/(args.workers+1)
        state=dict(status='running',pid=os.getpid(),host=socket.gethostname(),rows=names,workers=args.workers,
                   tokens=sum(len(t) for _,t in todo),started_unix=time.time(),completed_logs=[],cpu_seconds_before=spent,
                   metric='official NAVSIM v1.1 PDMS; not EPDMS',script_sha256=digest(Path(__file__)))
        json_save(out/'state.json',state)
        jobs=[(log,t,names,str(out),str(B/args.inputs),deadline,args.parity_tokens) for log,t in todo]
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures=[pool.submit(worker,j) for j in jobs]
                for future in as_completed(futures):
                    state['completed_logs'].append(future.result());json_save(out/'state.json',state)
            state['status']='complete';state['finished_unix']=time.time();json_save(out/'state.json',state)
        except BaseException as e:
            state.update(status='failed',error=repr(e),finished_unix=time.time());json_save(out/'state.json',state);raise


if __name__=='__main__':main()
