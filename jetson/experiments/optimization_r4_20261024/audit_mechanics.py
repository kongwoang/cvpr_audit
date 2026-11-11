"""Population-wide mechanism audit; all labels used only for evaluation."""
import importlib.util
import pickle
import sys
import numpy as np
import pandas as pd
from common import HERE, REMOTE, save, source_hashes
from rap import planner as P
from vtf import speculative as SP
import simple

def main():
    before=source_hashes();pp=P.PlannerParams();cp=P.CostParams()
    out=HERE/'reports/mechanisms';out.mkdir(parents=True,exist_ok=True)
    records=[];units=[];transitions=[];timeline=[]
    sys.path.insert(0,str(REMOTE/'scripts/A_method'))
    spec=importlib.util.spec_from_file_location('k27_mechanics',REMOTE/'scripts/A_method/k27_dev.py')
    K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K)
    for ds in ['nusc','av2']:
        changes=pd.read_csv(HERE/'reports/readout/changed_episode_attribution.csv')
        changes=changes[(changes.dataset==ds)&(changes.pipeline=='objectwise median')]
        for file in sorted((REMOTE/'experiments/optimization_r3_20261023/results/dev'/ds).glob('*.pkl')):
            rows=pickle.load(file.open('rb'));prev=None;crosslinks=links_total=0
            local=[];changes_unit=changes[changes.unit==file.stem]
            for r in rows:
                sc=r['trace_v6S_b2A_retro'];ch=r['trace_cheap'];v=float(r['v'])
                rd=P.required_decel(*sc['raw'],v,pp)[1];fd=P.required_decel(*sc['refined'],v,pp)[1];cd=P.required_decel(*ch['mono'],v,pp)[1]
                md,hd,skip,match=simple.readout_demands(ch,sc,rd,fd,cd)
                sup=np.array([i for i in SP.support(sc['raw'],v,pp) if sc['from_full'][i]],int)
                agreed_hard=bool(np.any(skip&(rd>=pp.thr_hard)))
                # Reconstruct exactly what the candidate would read if skipped objects
                # kept raw geometry; the deployed refiner touches no other object.
                part=[np.asarray(x).copy() for x in sc['refined']]
                for a,b in zip(part,sc['raw']):a[skip]=np.asarray(b)[skip]
                pdem=P.required_decel(*part,v,pp)[1]
                partial=simple.readout_demands(ch,sc,rd,pdem,cd)[0]
                action=lambda x:P.discrete_action(float(np.max(x)) if len(x) else 0,pp)
                a=action(md);skipped_a=2 if agreed_hard else action(partial)
                assert a==skipped_a,(ds,r['unit'],r['frame'])
                f=np.asarray(sc['from_full'],bool);x=sc['xyxy'][f];c=sc['coarse'][f]
                cross=0;nlink=0
                if prev is not None:
                    px,pc=prev
                    m=simple.links(x,np.full(len(x),'vehicle'),px,np.full(len(px),'vehicle'))
                    ok=m>=0;nlink=int(ok.sum());cross=int(((c[ok]=='vehicle')!=(pc[m[ok]]=='vehicle')).sum())
                prev=(x,c);crosslinks+=cross;links_total+=nlink
                saved=len(sup) if agreed_hard else int(skip[sup].sum())
                rec=dict(dataset=ds,unit=r['unit'],frame=r['frame'],refiner_objects=len(sup),skippable_objects=saved,
                    active_refiner_frame=int(len(sup)>0),all_skipped_active=int(len(sup)>0 and saved==len(sup)),
                    crossgroup_history_links=cross,history_links=nlink,abstention_action_differences=int(a!=skipped_a))
                local.append(rec);records.append(rec)
                raw_a=action(rd);ref_a=action(fd)
                short=lambda aa:max(0,r['a_gt']-P.commanded_decel(aa,pp))**2
                excess=lambda aa:cp.lam_brake*max(0,P.commanded_decel(aa,pp)-r['a_gt'])**2
                transitions.append(dict(dataset=ds,unit=r['unit'],frame=r['frame'],gt_action=r['act_gt'],raw_action=raw_a,refined_action=ref_a,
                    insufficient_delta=short(ref_a)-short(raw_a),overbraking_delta=excess(ref_a)-excess(raw_a),median_action=a,hardvote_action=action(hd)))
                for ep in changes_unit.itertuples():
                    if not ep.start_frame<=r['frame']<=ep.end_frame:continue
                    j=int(K.IE.matched(r['gt_boxes'][int(r['dec']):int(r['dec'])+1],r['gt_classes'][int(r['dec']):int(r['dec'])+1],sc['xyxy'],sc['coarse'])[0]) if r['dec']>=0 else -1
                    k=int(match[j]) if j>=0 else -1
                    timeline.append(dict(dataset=ds,unit=r['unit'],frame=r['frame'],episode_start=ep.start_frame,episode_end=ep.end_frame,
                        gt_demand=r['a_gt'],gt_action=r['act_gt'],cheap_action=r['a_cheap'],raw_action=raw_a,refined_action=ref_a,median_action=a,
                        decisive_found=j>=0,matched_cheap=k>=0,
                        decisive_raw_demand=float(rd[j]) if j>=0 else np.nan,decisive_refined_demand=float(fd[j]) if j>=0 else np.nan,
                        decisive_median_demand=float(md[j]) if j>=0 else np.nan,matched_cheap_demand=float(cd[k]) if k>=0 else np.nan,
                        decisive_raw_z=float(sc['raw'][0][j]) if j>=0 else np.nan,decisive_refined_z=float(sc['refined'][0][j]) if j>=0 else np.nan,
                        decisive_raw_lmin=float(sc['raw'][1][j]) if j>=0 else np.nan,decisive_raw_lmax=float(sc['raw'][2][j]) if j>=0 else np.nan,
                        decisive_refined_lmin=float(sc['refined'][1][j]) if j>=0 else np.nan,decisive_refined_lmax=float(sc['refined'][2][j]) if j>=0 else np.nan,
                        decisive_ttc=float(sc['raw'][3][j]) if j>=0 else np.nan))
            u=pd.DataFrame(local).select_dtypes('number').sum().to_dict();u.update(dataset=ds,unit=file.stem,frames=len(local));units.append(u)
        print(ds,'mechanics audit complete',flush=True)
    u=pd.DataFrame(units);summary=[]
    for ds,g in u.groupby('dataset'):
        n=len(g);draw=np.random.default_rng(0).integers(0,n,(2000,n));counts=np.array([np.bincount(a,minlength=n) for a in draw])
        for metric,num,den in [('skippable_refiner_object_fraction','skippable_objects','refiner_objects'),('all_skipped_active_frame_fraction','all_skipped_active','active_refiner_frame'),('crossgroup_history_link_fraction','crossgroup_history_links','history_links')]:
            num=g[num].to_numpy();den=g[den].to_numpy();bs=(counts@num)/(counts@den)
            summary.append(dict(dataset=ds,metric=metric,estimate=num.sum()/den.sum(),ci_lo=np.percentile(bs,2.5),ci_hi=np.percentile(bs,97.5),numerator=num.sum(),denominator=den.sum()))
    for name,rows in [('abstention_frames',records),('units',units),('summary',summary),('refiner_transitions',transitions),('median_rescue_timelines',timeline)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    assert source_hashes()==before
    save(out/'audit_DONE.json',dict(status='complete',frames=len(records),abstention_action_differences=0,main_sources_unchanged=True,
        note='Stored-scene action parity only; altered inference batch sizes still require actual backend parity. Cross-group links are semantic compatibility diagnostics, not identity ground truth.'))

if __name__=='__main__':main()
