"""Attribute every changed required episode to its decisive-object evidence."""
import argparse
import pickle
import numpy as np
import pandas as pd
from common import HERE,save

LABELS={'v6S':'v6-S','v6S_b2A_retro':'round1 combined',
 'v6S_b2A_bridge':'mixed-history C1','v6S_b2A_bridge_retro':'mixed-history + recovery',
 'v6S_b2A_transfer_retro':'cheap TTC transfer','v6S_b2A_extent_retro':'critical extent retention',
 'v6S_b2A_motion_retro':'motion-conditioned field','v6S_b2A_missing_retro':'learned missing history',
 'v6S_b2A_allgeo_retro':'learned geometric','v6S_b2A_alldec_retro':'learned decision',
 'v6S_b2A_upper_retro':'upper-quantile missing history','v6S_b2A_query_retro':'uncertainty query',
 'v6S_b2A_liftgeo_retro':'joint geometric lift','v6S_b2A_liftdec_retro':'joint decision lift',
 'v6S_b2A_upper_extent_retro':'upper quantile + extent retention',
 'v6S_b2A_upper_identity_retro':'identity gate + uncertain fallback',
 'v6S_b2A_missing_extent_retro':'point fallback + extent retention',
 'v6S_b2A_missing_observed_retro':'observed-history point fallback',
 'v6S_b2A_missing_corroborated_retro':'corroborated upper fallback',
 'v6S_b2A_missing_compact_retro':'compact point fallback',
 'v6S_b2A_missing_compact_observed_retro':'compact observed-history fallback'}

LABELS['v6S_b2A_median']='objectwise median'
for mode in ['rollback','missing','both','median_gate']:
    for q in [50,80,95]:
        LABELS['v6S_r5_'+mode+'_'+str(q)]='R5 '+mode+' q'+str(q)
LABELS['v6S_r5_matched_both_80']='R5 matched-only both q80'

for mode in ['domain_both','residual','residual_both','extent_both','point_residual','extent_residual']:
    for q in [50,80,95]:
        LABELS['v6S_r5_'+mode+'_'+str(q)]='R5 '+mode+' q'+str(q)
LABELS['v6S_r5_residual_ungated']='R5 residual ungated'

for parent in ['round1','point']:
    for model in ['pooled','conditional']:
        for q in [50,80,95]:
            LABELS['v6S_r5_cond_'+parent+'_'+model+'_'+str(q)]='R5 cond '+parent+' '+model+' q'+str(q)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);ap.add_argument('--report',required=True);ap.add_argument('--comparator',default='round1 combined',choices=['round1 combined','point fallback + extent retention','critical extent retention'])
    a=ap.parse_args();out=HERE/'reports'/a.report;e=pd.read_csv(out/'episodes.csv');rows=[];summary=[]
    for ds,d in e.groupby('dataset'):
        base=d[d.pipeline==a.comparator].set_index(['unit','start_frame']);cache={}
        for prefix,name in LABELS.items():
            if name in ('v6-S','round1 combined'):continue
            # dec_info is recorded for the last evaluated refinement; do not attribute
            # its evidence to an earlier v3a/no-refiner row of the same acquisition.
            selected=name+' / adapt' if name+' / adapt' in set(d.pipeline) else name
            candidate=d[d.pipeline==selected].set_index(['unit','start_frame'])
            if candidate.empty:continue
            x=candidate.join(base[['missed']],rsuffix='_baseline');changed=x[x.missed!=x.missed_baseline]
            own=[]
            for (unit,start),ep in changed.iterrows():
                if unit not in cache:cache[unit]={int(r['frame']):r for r in pickle.load((HERE/'results'/a.stage/ds/(unit+'.pkl')).open('rb'))}
                refiner='adapt' if selected.endswith(' / adapt') else ('none' if '_lift' in prefix else 'v3a')
                hit=[];supported=[];hard_supported=[];detail=[]
                for fr in range(int(start),int(ep.end_frame)+1):
                    r=cache[unit].get(fr)
                    if r is None:continue
                    action=r['a_'+prefix+'|'+refiner]
                    if action<1:continue
                    hit.append(fr);demand=r.get('dec_dem_'+prefix,np.nan)
                    if r['act_gt']>=1 and r.get('dec_found_'+prefix,False) and demand>=1:supported.append(fr)
                    if r['act_gt']>=2 and r.get('dec_found_'+prefix,False) and demand>=3.5 and action>=2:hard_supported.append(fr)
                    detail.append(dict(frame=fr,gt_action=r['act_gt'],action=action,dec_found=r.get('dec_found_'+prefix),dec_demand=demand))
                row=dict(dataset=ds,pipeline=selected,comparator=a.comparator,unit=unit,start_frame=int(start),end_frame=int(ep.end_frame),peak_demand=ep.peak_demand,
                    transition='newly_missed' if ep.missed else 'rescued',first_action_frame=hit[0] if hit else np.nan,
                    any_decisive_supported_brake=bool(supported),any_decisive_supported_hard_brake=bool(hard_supported),
                    first_decisive_supported_frame=supported[0] if supported else np.nan,
                    only_gap_frame_brake=bool(hit and all(cache[unit][fr]['act_gt']==0 for fr in hit)),detail=repr(detail))
                rows.append(row);own.append(row)
            rescued=[r for r in own if r['transition']=='rescued'];lost=[r for r in own if r['transition']=='newly_missed']
            summary.append(dict(dataset=ds,pipeline=selected,comparator=a.comparator,rescued=len(rescued),newly_missed=len(lost),
                rescued_with_decisive_evidence=sum(r['any_decisive_supported_brake'] for r in rescued),
                rescued_without_decisive_evidence=sum(not r['any_decisive_supported_brake'] for r in rescued),
                hard_peak_rescued=sum(r['peak_demand']>=3.5 for r in rescued),
                hard_peak_rescued_with_hard_evidence=sum(r['peak_demand']>=3.5 and r['any_decisive_supported_hard_brake'] for r in rescued)))
    suffix='' if a.comparator=='round1 combined' else ('_vs_point' if a.comparator.startswith('point') else '_vs_extent')
    pd.DataFrame(rows).to_csv(out/('changed_episode_attribution'+suffix+'.csv'),index=False);pd.DataFrame(summary).to_csv(out/('rescue_attribution_summary'+suffix+'.csv'),index=False)
    save(out/('attribution_STATUS'+suffix+'.json'),dict(status='complete',stage=a.stage,report=a.report,
        caution='A positive decisive match/demand is evidence of attribution, not proof of correct identity or calibrated geometry. Missing evidence need not imply no other legitimately relevant object.'))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
if __name__=='__main__':main()
