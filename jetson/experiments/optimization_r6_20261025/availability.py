"""Post-policy annotated evidence availability, including unresolved HARD cases."""
import argparse,importlib.util,pickle
import numpy as np
import pandas as pd
from common import HERE,REMOTE,save
from policies import demand,PP
from rap import planner as P
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--shard',default='0/1');a=ap.parse_args();ds=a.dataset
    spec=importlib.util.spec_from_file_location('k27_r6avail',REMOTE/'scripts/A_method/k27_dev.py');K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K)
    parent='v6S_b2A_missing_extent_retro';rows=[]
    for f in sorted((REMOTE/'experiments/optimization_r3_20261023/results/low'/ds).glob('*.pkl')):
        old=pickle.load(f.open('rb'));pr=pickle.load((HERE/'proposals'/ds/f.name).open('rb'));new=pickle.load((HERE/'results/dev'/ds/f.name).open('rb'))
        for r,p,s in zip(old,pr,new):
            if not r['gt']:continue
            t=r['trace_'+parent];c=r['trace_cheap'];dec=int(r['dec']);v=r['v']
            def match(o):return int(K.IE.matched(r['gt_boxes'][dec:dec+1],r['gt_classes'][dec:dec+1],o['xyxy'],o['coarse'])[0]) if dec>=0 else -1
            j,cj,pj=match(t),match(c),match(p);cd=demand(c['mono'],v);fd=demand(t['refined'],v);pd_=demand(p['mono'],v)
            act=int(s['a_'+parent+'|v3a']);gt=int(r['act_gt']);gd=float(r['a_gt'])
            row=dict(dataset=ds,unit=r['unit'],frame=r['frame'],gt_action=gt,gt_demand=gd,parent_action=act,r5_action=s['a_v6S_r5_point_residual_50|v3a'],cheap_action=r['a_cheap'],
                parent_decisive_match=j>=0,cheap_decisive_match=cj>=0,proposal_decisive_match=pj>=0,proposal_low=pj>=0 and p['conf'][pj]<p['threshold'],
                cheap_decisive_demand=cd[cj] if cj>=0 else 0,parent_decisive_demand=fd[j] if j>=0 else 0,proposal_ego_demand=pd_[pj] if pj>=0 else 0,
                proposal_confidence=float(p['conf'][pj]) if pj>=0 else np.nan,
                parent_risk=max(0,gd-P.commanded_decel(act,PP))**2,reference_action_risk=max(0,gd-P.commanded_decel(gt,PP))**2)
            for name in ['res_low','track_agree1','track_agree2','combo1_r5','combo2_r5']:
                row['action_'+name]=s['a_v6S_r6_'+name+'|v3a'];row['decisive_'+name]=s['dec_dem_v6S_r6_'+name]
            rows.append(row)
    d=pd.DataFrame(rows);out=HERE/'reports'/ds;d.to_csv(out/'evidence_availability_frames.csv',index=False)
    summary=[]
    for scope,sel in [('required',d.gt_action>=1),('hard_required',d.gt_action>=2),('underacted_required',(d.gt_action>=1)&(d.parent_action<d.gt_action)),('unbraked_required',(d.gt_action>=1)&(d.parent_action==0))]:
        g=d[sel]
        for name,mask in [('all',np.ones(len(g),bool)),('parent_unmatched',~g.parent_decisive_match),('cheap_matched',g.cheap_decisive_match),('cheap_brake_supported',g.cheap_decisive_demand>=1),('current_proposal_matched',g.proposal_decisive_match),('current_low_proposal_matched',g.proposal_low),('no_matched_current_evidence',~(g.parent_decisive_match|g.cheap_decisive_match|g.proposal_decisive_match))]:
            h=g[mask];summary.append(dict(dataset=ds,scope=scope,signature=name,frames=len(h),parent_risk=h.parent_risk.sum(),reference_action_residual=h.reference_action_risk.sum(),denominator_frames=len(g),flags_overlap=True))
    pd.DataFrame(summary).to_csv(out/'evidence_availability.csv',index=False)
    save(out/'availability_DONE.json',dict(status='complete',dataset=ds,warning='Post-hoc IoU-match availability, not a visibility certificate, GT exclusion, deployable oracle or method.'))
if __name__=='__main__':main()
