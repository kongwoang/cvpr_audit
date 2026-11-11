"""P01 validation of all 15 output candidates and historical official PDMS."""
import dataclasses
import inspect
import json
import lzma
import os
from pathlib import Path
import pickle
import time
import numpy as np
import k41_selector_metrics as M


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    b=M.B;out=b/'results/k41/selector_metrics_checks';out.mkdir(parents=True,exist_ok=True)
    start=time.process_time();wall=time.monotonic()
    one=b/'results/k41/selector_metrics_smoke01';two=b/'results/k41/selector_metrics_smoke02'
    p1=next(one.glob('*.pkl'));p2=two/p1.name
    a=pickle.load(p1.open('rb'));c=pickle.load(p2.open('rb'))
    for token in a:
        for name in ('cheap','full','field','oracle'):
            assert a[token]['scores'][name]==c[token]['scores'][name]
            assert np.array_equal(a[token]['trajectories'][name],c[token]['trajectories'][name])
    log=p1.stem;tok=sorted(a)[0]
    old={r['token']:r for r in pickle.load((M.OLD/p1.name).open('rb'))}[tok]
    center=pickle.load((M.X/'k30_center'/p1.name).open('rb')).get(tok)
    selector=M.K.Selector(center if isinstance(center,np.ndarray) and len(center)>=2 else None,old['v0'])
    loader=M.SceneLoader(M.K.DATA/'navsim_logs/test',M.K.DATA/'sensor_blobs/test',
        M.SceneFilter(num_history_frames=4,num_future_frames=10,frame_interval=1,has_route=True,log_names=[log],tokens=[tok]),M.SensorConfig.build_no_sensors())
    scene=loader.get_scene_from_token(tok);scenario=M.NavSimScenario(scene,map_root=os.environ['NUPLAN_MAPS_ROOT'],map_version='nuplan-maps-v1.0')
    proc=M.MetricCacheProcessor(str(out/'temporary_metric_cache'),False);proc.compute_metric_cache(scenario)
    cache_path=out/'temporary_metric_cache'/scenario.log_name/scenario.scenario_type/scenario.token/'metric_cache.pkl'
    with lzma.open(cache_path,'rb') as f:cache=pickle.load(f)
    cfg=Path(os.environ['NAVSIM_DEVKIT_ROOT'])/'navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml'
    params=M.OmegaConf.load(cfg);sim=M.instantiate(params.simulator);scorer=M.instantiate(params.scorer)
    scores=[]
    for index in range(15):
        tr=M.trajectory(selector,index)
        restored=tr.poses[:,:2]+M.K.EGO_RA*np.stack([np.cos(tr.poses[:,2]),np.sin(tr.poses[:,2])],1)
        assert np.allclose(restored,selector.P[index,5:41:5,:2],rtol=0,atol=1e-5)
        first=dataclasses.asdict(M.pdm_score(cache,tr,sim.proposal_sampling,sim,scorer))
        second=dataclasses.asdict(M.pdm_score(cache,tr,sim.proposal_sampling,sim,scorer))
        assert first==second and all(np.isfinite(v) for v in first.values())
        scores.append(dict(candidate=index,**{k:float(v) for k,v in first.items()}))
    old_plan={r['token']:r['poses'] for r in pickle.load((b/'results/k20'/('base__'+log+'.pkl')).open('rb'))}[tok]
    old_scores={r['token']:r for r in pickle.load((M.X/'pdms_final'/p1.name).open('rb'))}[tok]
    tr=M.Trajectory(np.asarray(old_plan,np.float32),M.TrajectorySampling(time_horizon=4,interval_length=.5))
    current=dataclasses.asdict(M.pdm_score(cache,tr,sim.proposal_sampling,sim,scorer))
    for key,val in current.items():assert np.isclose(val,old_scores[key+'_cheap'],rtol=0,atol=1e-9),(key,val,old_scores[key+'_cheap'])
    paths=[Path(__file__),Path(M.__file__),Path(M.K.__file__),Path(M.K.NP.__file__),cfg]
    for obj in (M.pdm_score,M.MetricCacheProcessor,M.NavSimScenario,M.Trajectory,type(sim),type(scorer)):
        paths.append(Path(inspect.getsourcefile(obj)))
    provenance={str(p.relative_to(b)):M.digest(p) for p in paths}
    cache_path.unlink()
    report=dict(tokens_compared=2,rows_compared=4,candidate_score_repeat_checks=15,historical_PDM_score_metrics=len(current),
                original_cheap_plan_exact_smoke_tokens=2,source_sha256=provenance,network_inference=False,
                measured_cpu_seconds=time.process_time()-start,measured_wall_seconds=time.monotonic()-wall,
                interpretation='API/implementation validation only; not a population performance estimate')
    (out/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'candidate_fixture_scores.json').write_text(json.dumps(scores,indent=2)+'\n')
    usage=out/'usage';usage.mkdir(exist_ok=True)
    (usage/'checks.json').write_text(json.dumps(dict(cpu_seconds=report['measured_cpu_seconds']))+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
