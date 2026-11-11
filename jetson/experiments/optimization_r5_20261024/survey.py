"""Secondary diagnostics: fragment/edge caveats and all-row gain concentration."""
import argparse
import importlib.util
import pickle
import sys
import subprocess
import numpy as np
import pandas as pd
from common import HERE,REMOTE,save
from rap.mono import box_iou
from features import PP,action
from rap import planner as P
sys.path.insert(0,str(REMOTE/'scripts/A_method'))
s=importlib.util.spec_from_file_location('k27_r5survey',REMOTE/'scripts/A_method/k27_dev.py');K=importlib.util.module_from_spec(s);s.loader.exec_module(K)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);a=ap.parse_args();ds=a.dataset
    out=HERE/'reports/survey'/ds;out.mkdir(parents=True,exist_ok=True);flags=[];unitrows=[]
    w,h=(1600,900) if ds=='nusc' else (1550,2048)
    for f in sorted((HERE/'results/residual'/ds).glob('*.pkl')):
        records=pickle.load(f.open('rb'));d=pd.DataFrame(records);d=d[d['gt'].astype(bool)]
        gt=d.a_gt.to_numpy();cmd=np.array([0.,2.5,6.]);hard=gt>=3.5
        for col in d:
            if not col.startswith('a_v6S'):continue
            act=d[col].to_numpy(int);risk=np.maximum(0,gt-cmd[act])**2
            unitrows.append(dict(dataset=ds,unit=f.stem,pipeline_column=col,frames=len(d),risk=risk.sum(),hard_frames=int(hard.sum()),
                hard_loss=risk[hard].sum(),hard_no_brake=int((hard&(act==0)).sum()),hard_under_action=int((hard&(act<2)).sum()),
                false_frames=int(((gt<1)&(act>=1)).sum()),false_hard=int(((gt<1)&(act>=2)).sum())))
        originals=pickle.load((REMOTE/'experiments/optimization_r4_20261024/results/dev'/ds/f.name).open('rb'))
        for r in originals:
            if not r['gt'] or r['act_gt']<1:continue
            t=r['trace_v6S_b2A_retro'];j=int(r['dec']);b=r['gt_boxes'][j];gtc=str(r['gt_classes'][j]);n=len(t['xyxy'])
            compat=np.array([str(c) in K.IE.COMPAT.get(gtc,{gtc}) for c in t['coarse']],bool)
            iou=box_iou(b[None,:],t['xyxy'])[0] if n else np.zeros(0)
            inter=np.maximum(0,np.minimum(t['xyxy'][:,2:],b[2:])-np.maximum(t['xyxy'][:,:2],b[:2])).prod(1) if n else np.zeros(0)
            area=np.maximum(1e-6,(t['xyxy'][:,2:]-t['xyxy'][:,:2]).prod(1)) if n else np.zeros(0)
            contained=compat&(inter/area>=.5)&(iou>=.05)
            best=float(iou[compat].max()) if compat.any() else 0.
            ba=max(1e-6,float(np.maximum(0,b[2:]-b[:2]).prod()))
            masked=max(0,min(float(b[3]),h)-max(float(b[1]),1830))*max(0,min(float(b[2]),w)-max(float(b[0]),0))/ba if ds=='av2' else 0.
            edge=bool(b[0]<=2 or b[1]<=2 or b[2]>=w-2 or b[3]>=h-2)
            aa=r['a_v6S_b2A_retro|v3a'];loss=max(0,r['a_gt']-cmd[aa])**2
            flags.append(dict(dataset=ds,unit=f.stem,frame=r['frame'],gt_demand=r['a_gt'],action=aa,loss=loss,
                best_compatible_iou=best,frozen_decisive_match=best>=.3,possible_contained_fragment=bool(contained.any()),
                gt_box_area_fraction=ba/(w*h),gt_box_touches_image_edge=edge,gt_box_mask_overlap_fraction=masked,
                gt_box_x1=b[0],gt_box_y1=b[1],gt_box_x2=b[2],gt_box_y2=b[3]))
    pd.DataFrame(flags).to_csv(out/'annotation_geometry_flags.csv',index=False)
    u=pd.DataFrame(unitrows);u.to_csv(out/'unit_metrics.csv',index=False)
    summaries=[];contrasts=[];concentration=[];units=sorted(u.unit.unique());n=len(units)
    C=np.array([np.bincount(x,minlength=n) for x in np.random.default_rng(0).integers(0,n,(2000,n))]);dist={};points={}
    for name,g in u.groupby('pipeline_column'):
        g=g.set_index('unit').loc[units];est={};bs={}
        for num,den,label in [('hard_loss','frames','hard_loss_per_all_frame'),('hard_loss','hard_frames','hard_loss_per_hard_frame'),
                              ('hard_no_brake','hard_frames','hard_no_brake_rate'),('hard_under_action','hard_frames','hard_under_action_rate')]:
            estimate=g[num].sum()/g[den].sum();draw=(C@g[num].to_numpy())/(C@g[den].to_numpy());lo,hi=np.percentile(draw,[2.5,97.5])
            summaries.append(dict(dataset=ds,pipeline_column=name,metric=label,estimate=estimate,ci_lo=lo,ci_hi=hi));est[label]=estimate;bs[label]=draw
        dist[name]=bs;points[name]=est
    for base in ['a_v6S_b2A_retro|v3a','a_v6S_b2A_missing_extent_retro|v3a']:
        reference=u[u.pipeline_column==base].set_index('unit').loc[units]
        for name,g in u.groupby('pipeline_column'):
            if name==base:continue
            g=g.set_index('unit').loc[units];gain=reference.risk-g.risk;pos=gain.clip(lower=0);total=pos.sum();rank=pos.sort_values(ascending=False)
            concentration.append(dict(dataset=ds,pipeline_column=name,comparator=base,net_loss_reduction_sum=gain.sum(),
                improved_units=int((gain>1e-10).sum()),harmed_units=int((gain< -1e-10).sum()),
                largest_gain_unit=str(rank.index[0]),largest_gain=rank.iloc[0],largest_gain_share=rank.iloc[0]/total if total else np.nan,
                top5_gain_share=rank.head(5).sum()/total if total else np.nan,
                reduction_per_frame_without_largest=(gain.sum()-gain.loc[rank.index[0]])/(g.frames.sum()-g.loc[rank.index[0],'frames'])))
            for metric,draw in dist[name].items():
                effect=draw-dist[base][metric];lo,hi=np.percentile(effect,[2.5,97.5]);contrasts.append(dict(dataset=ds,pipeline_column=name,comparator=base,metric=metric,delta=points[name][metric]-points[base][metric],ci_lo=lo,ci_hi=hi))
    pd.DataFrame(summaries).to_csv(out/'severity_intervals.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'severity_contrasts.csv',index=False)
    pd.DataFrame(concentration).to_csv(out/'gain_concentration.csv',index=False)
    ff=pd.DataFrame(flags);summary=[]
    for scope,sel in [('required',np.ones(len(ff),bool)),('unbraked_required',ff.action==0),('unbraked_unmatched',(ff.action==0)&~ff.frozen_decisive_match)]:
        q=ff[sel]
        for flag,mask in [('possible_contained_fragment',q.possible_contained_fragment),('image_edge',q.gt_box_touches_image_edge),
                          ('mask_overlap_positive',q.gt_box_mask_overlap_fraction>0),('mask_overlap_ge_half',q.gt_box_mask_overlap_fraction>=.5),
                          ('box_area_ge_10pct',q.gt_box_area_fraction>=.1)]:
            summary.append(dict(dataset=ds,scope=scope,flag=flag,frames=int(mask.sum()),of_frames=len(q),loss=q[mask].loss.sum(),of_loss=q.loss.sum()))
    pd.DataFrame(summary).to_csv(out/'geometry_flag_summary.csv',index=False)
    for comparator in ['point fallback + extent retention','critical extent retention']:
        subprocess.run([sys.executable,str(HERE/'attribution.py'),'--stage','residual','--report','final_'+ds,'--comparator',comparator],check=True)
    save(out/'DONE.json',dict(status='complete',units=n,secondary_flags='Overlapping annotation diagnostics only; do not prove visibility or change any primary metric.',bootstrap=2000,seed=0))

if __name__=='__main__':main()
