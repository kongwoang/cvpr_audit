"""Representative current/previous observations; selection is explicitly diagnostic."""
import argparse,importlib.util,pickle
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import HERE,REMOTE,save
from policies import demand,apply
PARENT='v6S_b2A_missing_extent_retro'
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--shard',default='0/1');a=p.parse_args();ds=a.dataset
    spec=importlib.util.spec_from_file_location('k27_r6cases',REMOTE/'scripts/A_method/k27_dev.py');K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K)
    changed=pd.read_csv(HERE/'reports'/ds/'changed_frames.csv');avail=pd.read_csv(HERE/'reports'/ds/'evidence_availability_frames.csv');chosen=[]
    q=changed[(changed.pipeline=='track_agree1')&(changed.action>changed.r5_action)&(changed.decisive_added_demand>=1)&(changed.gt_action>=1)]
    if len(q):r=q.sort_values(['gt_demand','unit','frame'],ascending=[False,True,True]).iloc[0];chosen.append(dict(kind='new_supported_action',unit=r.unit,frame=int(r.frame)))
    q=changed[(changed.pipeline=='track_agree1')&(changed.action>changed.r5_action)&(changed.gt_action==0)]
    if len(q):r=q.sort_values(['action','unit','frame'],ascending=[False,True,True]).iloc[0];chosen.append(dict(kind='new_false_action',unit=r.unit,frame=int(r.frame)))
    q=avail[(avail.gt_action>=2)&(avail.action_combo1_r5<2)]
    if len(q):r=q.sort_values(['parent_risk','unit','frame'],ascending=[False,True,True]).iloc[0];chosen.append(dict(kind='unresolved_hard_case',unit=r.unit,frame=int(r.frame)))
    ep=pd.read_csv(HERE/'reports'/ds/'rescued_episodes.csv');q=ep[(ep.pipeline=='res_low')&(ep.comparator=='v6S_r5_point_residual_50')&ep.rescued]
    if len(q):
        e=q.sort_values(['peak_demand','unit','start_frame'],ascending=[False,True,True]).iloc[0]
        c=changed[(changed.pipeline=='res_low')&(changed.unit==e.unit)&changed.frame.between(e.start_frame,e.end_frame)&(changed.action>changed.r5_action)]
        if len(c):chosen.append(dict(kind='low_score_episode_rescue',unit=e.unit,frame=int(c.frame.min())))
    lookup={};wanted={x['unit'] for x in chosen}
    for unit,cam,gen in K.units(ds):
        if unit not in wanted:continue
        lookup[unit]={fr:path for fr,path,v,g,c in gen()}
    out=HERE/'reports/cases'/ds
    import shutil
    archive=HERE/'reports/case_caption_v1'/ds
    if out.exists() and not archive.exists():shutil.copytree(out,archive)
    out.mkdir(parents=True,exist_ok=True);details=[]
    for ci,c in enumerate(chosen):
        unit=c['unit'];fr=c['frame'];old=pickle.load((REMOTE/'experiments/optimization_r3_20261023/results/low'/ds/(unit+'.pkl')).open('rb'));props=pickle.load((HERE/'proposals'/ds/(unit+'.pkl')).open('rb'));new=pickle.load((HERE/'results/followup'/ds/(unit+'.pkl')).open('rb'))
        idx=next(i for i,r in enumerate(old) if r['frame']==fr);r=old[idx];n=new[idx];pr=props[idx]
        candidate_prefix='v6S_r6_res_low' if c['kind']=='low_score_episode_rescue' else 'v6S_r6_track_agree1_r5'
        candidate_label='R6 residual low-score corroboration' if c['kind']=='low_score_episode_rescue' else 'R5 + bounded observed1'
        hist=[dict(cheap=x['trace_cheap'],scene=x['trace_'+PARENT],v=x['v']) for x in old[max(0,idx-2):idx]]
        rules,_=apply(r['trace_cheap'],r['trace_'+PARENT],pr,hist,r['v'],r['dt'],.6 if ds=='nusc' else 960/2048)
        ld=rules['track_agree1'][1];fig,axes=plt.subplots(1,3,figsize=(15,6))
        for ax,k in zip(axes,[max(0,idx-2),max(0,idx-1),idx]):
            rr=old[k];path=lookup[unit][rr['frame']];im=cv2.imread(path);assert im is not None;ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
            sc=rr['trace_'+PARENT];ch=rr['trace_cheap'];dd=demand(sc['refined'],rr['v']);cc=demand(ch['mono'],rr['v'])
            for trace,dem,color in [(ch,cc,'dodgerblue'),(sc,dd,'cyan')]:
                for i,(b,de) in enumerate(zip(trace['xyxy'],dem)):
                    if de<1:continue
                    x1,y1,x2,y2=b;ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color=color,lw=1.2));ax.text(x1,y1,f'{de:.1f}',fontsize=6,color=color,bbox=dict(facecolor='black',alpha=.6,pad=1))
            if k==idx:
                for i,b in enumerate(pr['xyxy']):
                    if pr['conf'][i]>=pr['threshold']:continue
                    x1,y1,x2,y2=b;ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color='magenta',lw=2 if ld[i]>0 else .5,alpha=1 if ld[i]>0 else .45))
                    if ld[i]>0:ax.text(x1,y2,f'conf {pr["conf"][i]:.2f}; add {ld[i]:g}',fontsize=6,color='magenta',bbox=dict(facecolor='black',alpha=.65,pad=1))
            dec=int(rr['dec'])
            if dec>=0:
                x1,y1,x2,y2=rr['gt_boxes'][dec];ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color='red',lw=2))
            if ds=='av2':ax.axhline(1830,color='orange',ls='--',lw=.6)
            ax.set_title(f'frame {rr["frame"]} | GT {rr["a_gt"]:.2f} | parent {rr["a_"+PARENT+"|v3a"]}',fontsize=9);ax.set_xticks([]);ax.set_yticks([])
        candidate_action=n['a_'+candidate_prefix+'|v3a']
        fig.suptitle(f'{ds} {unit} — {c["kind"]}\nBlue cheap / cyan final / magenta current low-score / red decisive annotation\nR5 {n["a_v6S_r5_point_residual_50|v3a"]} → {candidate_label} {candidate_action}',fontsize=10)
        fig.tight_layout();stem=f'{ci:02d}_{c["kind"]}'
        for ext in ['png','pdf']:fig.savefig(out/(stem+'.'+ext),dpi=150)
        plt.close(fig);details.append(dict(dataset=ds,**c,gt_demand=r['a_gt'],r5_action=n['a_v6S_r5_point_residual_50|v3a'],candidate=n['a_'+candidate_prefix+'|v3a'],candidate_prefix=candidate_prefix,image_path=lookup[unit][fr],panel=stem+'.png'))
    pd.DataFrame(details).to_csv(out/'manifest.csv',index=False);save(out/'DONE.json',dict(status='complete',panels=len(details),selection='Fixed categories, deterministic severity maxima/first rescue; diagnostic selection, not an unbiased sample. Orange line is original geometry exclusion boundary; pixels are unmasked.'))
if __name__=='__main__':main()
