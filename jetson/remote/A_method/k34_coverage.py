"""K34-2: decisive-object coverage by all actually verified crops by final round."""
import argparse
import numpy as np
import pandas as pd
from vtf import speculative as SP
from k34_common import *

def class_group(raw):
    s=raw.lower()
    if s=='vehicle.car' or s in ('regular_vehicle','car'):return 'car'
    if s=='vehicle.truck' or s in ('box_truck','truck','truck_cab','vehicular_trailer'):return 'truck'
    if s.startswith('vehicle.bus') or s in ('bus','articulated_bus','school_bus'):return 'bus'
    if s.startswith('human.pedestrian') or s in ('pedestrian','person'):return 'pedestrian'
    if s in ('vehicle.bicycle','vehicle.motorcycle','bicycle','bicyclist','motorcycle','motorcyclist','cyclist','wheeled_rider'):return 'cyclist'
    return 'other_or_unspecified'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);args=ap.parse_args();ds=args.dataset
    assert (OUT/'GATE_K34_1_COMPLETE.json').exists(), 'K34-1 gate missing'
    od=OUT/'coverage'/ds;od.mkdir(parents=True,exist_ok=True);manifest=[];d=load_base(ds,manifest)
    gt=pd.read_pickle(B/'results/k30/gt'/(ds+'.pkl')).set_index(['unit','frame'])
    match=read_rows(B/'results/k30/match'/ds,manifest).set_index(['unit','frame'])
    ann=[]
    for f in sorted((OUT/'replay'/ds).glob('*.pkl')):
        ann+=pickle.load(f.open('rb'))['annotations'];manifest.append({'path':str(f),'sha256':sha(f)})
    ann=pd.DataFrame(ann).set_index(['unit','frame']);assert ann.index.is_unique
    req=d[d.act_gt>=1].copy();idx=pd.MultiIndex.from_frame(req[['unit','frame']]);assert set(idx)==set(ann.index)
    rows=[]
    for rr in req.to_dict('records'):
        key=(rr['unit'],rr['frame']);g=gt.loc[key];a=ann.loc[key];m=match.loc[key]
        assert np.allclose(g.dec_box,a.box) and np.isclose(g.a_gt,a.demand_gt)
        crops=m['crops_v6S'];covered=bool(SP.centres_in(np.array([g.dec_box]),crops)[0])
        assert covered==bool(m['dec_in_crop_v6S'])
        rows.append({'dataset':ds,'unit':key[0],'frame':key[1],'covered':covered,'raw_class':a.raw_class,
                     'class':class_group(a.raw_class),'range_m':g.dec_range,'abs_lateral_m':abs(g.dec_lat),
                     'closing_speed_mps':a.closing_speed_mps,'has_rate':a.has_rate,'cheap_detected':a.cheap_detected,
                     'decision_loss':rr['J_v6S|v3a'],'act_gt':rr['act_gt'],'a_v6S':rr['a_v6S|v3a'],
                     'match_action_differs_K28':m['a_v6S|v3a']!=rr['a_v6S|v3a'],
                     'verified_crop_count':len(crops),'region_semantics':'union_of_actual_field_support_and_guard_crops_by_final_round'})
    q=pd.DataFrame(rows);x=q.closing_speed_mps.to_numpy();finite=np.isfinite(x)
    cuts=np.quantile(x[finite],[1/3,2/3]) if finite.any() else [np.nan,np.nan]
    q['closing_speed_tercile']='missing'
    q.loc[finite,'closing_speed_tercile']=np.array(['low','middle','high'])[np.searchsorted(cuts,x[finite],side='left')]
    q['range_bin']=np.select([q.range_m<20,q.range_m<40,q.range_m<=60],['<20','20-40','40-60'],default='>60')
    q['lateral_bin']=np.select([q.abs_lateral_m<1.2,q.abs_lateral_m<=3],['<1.2','1.2-3'],default='>3')
    q.to_csv(od/'coverage_frames.csv',index=False)
    units,C=bootstrap(d.unit);L=d.groupby('unit').J_cheap.sum().reindex(units).to_numpy();summ=[];ints=[]
    strata=[('all','all',np.ones(len(q),bool))]
    for col,levels in [('class',['car','truck','bus','pedestrian','cyclist','other_or_unspecified']),
                       ('range_bin',['<20','20-40','40-60','>60']),('lateral_bin',['<1.2','1.2-3','>3']),
                       ('closing_speed_tercile',['low','middle','high','missing']),('cheap_detected',[True,False])]:
        strata.extend((col,str(v),q[col]==v) for v in levels)
    for col,label,mask in strata:
        s=q[mask].copy();s['uncovered']=~s.covered;s['uncovered_loss']=s.decision_loss*s.uncovered
        a=s.groupby('unit')[['covered','uncovered','uncovered_loss']].sum().reindex(units,fill_value=0)
        n=s.groupby('unit').size().reindex(units,fill_value=0).to_numpy();nu=a.uncovered.to_numpy();loss=a.uncovered_loss.to_numpy()
        metrics={'coverage':(a.covered.to_numpy(),n),'uncovered_conditional_mean_loss':(loss,nu),'uncovered_loss_share_cheap_total':(loss,L)}
        point={}
        for metric,(num,den) in metrics.items():
            point[metric]=float(ratio(num.sum(),den.sum()))
            ints.append({'dataset':ds,'stratification':col,'stratum':label,'metric':metric,'estimate':point[metric],**interval(ratio(C@num,C@den))})
        summ.append({'dataset':ds,'stratification':col,'stratum':label,'objects':len(s),'frames':len(s),'contributing_units':s.unit.nunique(),
                     'all_units':len(units),'covered':int(a.covered.sum()),'uncovered':int(nu.sum()),'uncovered_loss_sum':loss.sum(),
                     'cheap_total_decision_loss':L.sum(),**point})
    for name,vals in [('coverage_summary',summ),('coverage_intervals',ints),('source_manifest',manifest)]:pd.DataFrame(vals).to_csv(od/(name+'.csv'),index=False)
    q[['raw_class','class']].drop_duplicates().sort_values('raw_class').to_csv(od/'class_mapping.csv',index=False)
    pd.DataFrame([{'dataset':ds,'stratum':'closing_speed_tercile','cut_1_mps':cuts[0],'cut_2_mps':cuts[1],
                   'speed_source':'negative annotated longitudinal range derivative; missing when has_rate=False',
                   'ties':'equal cut assigned to lower bin; empty bins retained'}]).to_csv(od/'stratum_definitions.csv',index=False)
    done(od/'DONE.json',dataset=ds,frames=len(q),code_sha256=sha(__file__))
    print('K34 COVERAGE DONE',ds,len(q),flush=True)

if __name__=='__main__':main()
