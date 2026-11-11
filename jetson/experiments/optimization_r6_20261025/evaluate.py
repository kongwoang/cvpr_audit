"""Exact frozen-query readout and all-frame evidence audit."""
import argparse,importlib.util,pickle,sys,time
import numpy as np
import pandas as pd
from common import HERE,REMOTE,save
from policies import apply,action,demand,mx,NAMES,PP
from rap import planner as P
PREFIX='v6S_b2A_missing_extent_retro'
CONTROL=['cheap','v6S','v6S_b2A_retro',PREFIX]+['v6S_r5_point_residual_'+str(q) for q in (50,80,95)]
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--shard',default='0/1');a=p.parse_args();ds=a.dataset
    spec=importlib.util.spec_from_file_location('k27_r6eval',REMOTE/'scripts/A_method/k27_dev.py');K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K)
    src=REMOTE/'experiments/optimization_r3_20261023/results/low'/ds;r5=REMOTE/'experiments/optimization_r5_20261024/results/residual'/ds
    out=HERE/'results/dev'/ds;out.mkdir(parents=True,exist_ok=True);cp=P.CostParams();audit=[];changed=[];timings=[]
    files=sorted(src.glob('*.pkl'));assert len(files)==dict(nusc=152,av2=300)[ds]
    for file in files:
        old=pickle.load(file.open('rb'));prev5=pickle.load((r5/file.name).open('rb'));props=pickle.load((HERE/'proposals'/ds/file.name).open('rb'))
        assert len(old)==len(prev5)==len(props);hist=[];result=[];prev=None
        for r,s,p in zip(old,prev5,props):
            assert r['frame']==s['frame']==p['frame'];assert r['a_gt']==s['a_gt']
            t=r['trace_'+PREFIX];c=r['trace_cheap'];v=r['v'];ref=demand(t['refined'],v);base=int(action(mx(ref)))
            assert base==r['a_'+PREFIX+'|v3a']
            assert abs(P.decision_cost(base,r['a_gt'],prev,PP,cp)['J']-r['J_'+PREFIX+'|v3a'])<1e-8
            # Only selected scalar controls are passed to the all-row reporter.
            row={k:r[k] for k in ('unit','frame','v','dt','gt','a_gt','act_gt','dec')}
            for pre in CONTROL:
                suffix=pre if pre=='cheap' else pre+'|v3a'
                for k in ['a_','J_']:row[k+suffix]=s[k+suffix]
                for k in ['px_','ncalls_']:
                    if k+pre in s:row[k+pre]=s[k+pre]
                for k in ['dec_found_','dec_dem_']:
                    if k+pre in s:row[k+pre]=s[k+pre]
            started=time.perf_counter();choices,meta=apply(c,t,p,hist,v,r['dt'],.6 if ds=='nusc' else 960/2048)
            timings.append(dict(dataset=ds,unit=r['unit'],frame=r['frame'],all_20_rules_ms=(time.perf_counter()-started)*1000))
            # GT accessed only after all policy outputs have been computed.
            dec=int(r['dec'])
            def matched(d):
                return int(K.IE.matched(r['gt_boxes'][dec:dec+1],r['gt_classes'][dec:dec+1],d['xyxy'],d['coarse'])[0]) if dec>=0 else -1
            j,cj,lj=matched(t),matched(c),matched(p)
            for name,(cd,ld) in choices.items():
                prefix='v6S_r6_'+name;parent='v6S_r5_point_residual_50' if name.endswith('_r5') else PREFIX
                pa=int(s['a_'+parent+'|v3a']);act=max(pa,int(action(max(mx(cd),mx(ld)))))
                row['a_'+prefix+'|v3a']=act;row['J_'+prefix+'|v3a']=P.decision_cost(act,r['a_gt'],prev,PP,cp)['J']
                for k in ['px_','ncalls_']:row[k+prefix]=r[k+PREFIX]
                dd=max(ref[j] if j>=0 else 0,cd[cj] if cj>=0 else 0,ld[lj] if lj>=0 else 0)
                found=j>=0 or (cj>=0 and cd[cj]>0) or (lj>=0 and ld[lj]>0)
                if name.endswith('_r5') and s.get('dec_found_'+parent,False):
                    found=True;dd=max(dd,float(s['dec_dem_'+parent]))
                row['dec_found_'+prefix]=found;row['dec_dem_'+prefix]=dd if found else np.nan
                assert act>=pa
                if act!=base:
                    changed.append(dict(dataset=ds,unit=r['unit'],frame=r['frame'],pipeline=name,gt_action=r['act_gt'],gt_demand=r['a_gt'],base_action=base,r5_action=s['a_v6S_r5_point_residual_50|v3a'],action=act,
                        decisive_found=found,decisive_demand=dd,cheap_added=max(mx(cd),0),low_added=max(mx(ld),0),current_cheap_decisive=cj,current_low_decisive=lj,
                        decisive_added_demand=max(cd[cj] if cj>=0 else 0,ld[lj] if lj>=0 else 0)))
            audit.append(dict(dataset=ds,unit=r['unit'],frame=r['frame'],**meta));result.append(row)
            hist=(hist+[dict(cheap=c,scene=t,v=v)])[-2:];prev=r['a_cheap']
        target=out/file.name;tmp=target.with_suffix('.tmp');pickle.dump(result,tmp.open('wb'));tmp.replace(target)
        save(target.with_suffix('.parity.json'),dict(dataset=ds,unit=file.stem,differences=dict(parent_actions=0,parent_costs=0,reference=0)))
    report=HERE/'reports'/ds;report.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(audit).to_csv(report/'eligibility.csv',index=False);pd.DataFrame(changed).to_csv(report/'changed_frames.csv',index=False);pd.DataFrame(timings).to_csv(report/'server_all_rules_runtime.csv',index=False)
    save(report/'evaluate_DONE.json',dict(status='complete',dataset=ds,units=len(files),frames=len(audit),new_policies=len(NAMES),control_rows=len(CONTROL),timing_scope='all 20 rules together, server CPU, excludes proposal recovery; not deployed Jetson latency'))
if __name__=='__main__':main()
