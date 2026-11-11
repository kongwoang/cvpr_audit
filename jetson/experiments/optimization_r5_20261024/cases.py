"""Deterministic severe-failure, rescue and false-braking frame panels."""
import argparse
import importlib.util
import pickle
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import HERE,REMOTE,save
from features import PP
from rap import planner as P
sys.path.insert(0,str(REMOTE/'scripts/A_method'))
s=importlib.util.spec_from_file_location('k27_r5cases',REMOTE/'scripts/A_method/k27_dev.py');K=importlib.util.module_from_spec(s);s.loader.exec_module(K)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);a=ap.parse_args();ds=a.dataset
    audit=pd.read_csv(HERE/'reports/audit'/(ds+'_frames.csv'));chosen=[]
    miss=audit[(audit.signature=='decisive_detection_absent')&(audit.action==0)]
    for label,q in [('severe_absent_both',miss[~miss.cheap_found]),('severe_lost_cheap',miss[miss.cheap_dec_demand>=1])]:
        if len(q):r=q.nlargest(1,'insufficient').iloc[0];chosen.append(dict(kind=label,unit=r.unit,frame=int(r.frame)))
    ep=pd.read_csv(HERE/'reports'/('evaluation_'+ds)/'changed_episode_attribution.csv')
    for supported in [True,False]:
        q=ep[(ep.pipeline=='R5 both q50')&(ep.transition=='rescued')&(ep.any_decisive_supported_brake==supported)]
        if len(q):
            r=q.nlargest(1,'peak_demand').iloc[0]
            frame=int(r.first_decisive_supported_frame if supported else r.first_action_frame)
            chosen.append(dict(kind='supported_rescue' if supported else 'unsupported_rescue',unit=r.unit,frame=frame))
    newfalse=[]
    for f in (HERE/'results/readout'/ds).glob('*.pkl'):
        for r in pickle.load(f.open('rb')):
            if r['act_gt']==0 and r['a_v6S_r5_both_50|v3a']>r['a_v6S_b2A_retro|v3a']:
                newfalse.append(dict(kind='new_false_braking',unit=r['unit'],frame=r['frame'],penalty=r['J_v6S_r5_both_50|v3a']-r['J_v6S_b2A_retro|v3a']))
    if newfalse:chosen.append(sorted(newfalse,key=lambda r:(-r['penalty'],r['unit'],r['frame']))[0])
    units={r['unit'] for r in chosen};lookup={}
    for unit,cam,gen in K.units(ds,0):
        if unit not in units:continue
        lookup[unit]={fr:(path,v,g) for fr,path,v,g,cached in gen()}
    out=HERE/'reports/cases'/ds;out.mkdir(parents=True,exist_ok=True);details=[]
    for ci,c in enumerate(chosen):
        unit,fr=c['unit'],c['frame'];rows=pickle.load((REMOTE/'experiments/optimization_r4_20261024/results/dev'/ds/(unit+'.pkl')).open('rb'))
        r=next(r for r in rows if r['frame']==fr);new=next(r for r in pickle.load((HERE/'results/readout'/ds/(unit+'.pkl')).open('rb')) if r['frame']==fr)
        path,v,g=lookup[unit][fr];img=cv2.imread(path);assert img is not None
        if ds=='av2':img[1830:]=0
        t=r['trace_v6S_b2A_retro'];ch=r['trace_cheap'];dec=r['dec'];box=r['gt_boxes'][dec] if dec>=0 else None
        _,rd=P.required_decel(*t['raw'],v,PP);_,fd=P.required_decel(*t['refined'],v,PP);_,cd=P.required_decel(*ch['mono'],v,PP)
        fig,axes=plt.subplots(1,2,figsize=(13,6));H,W=img.shape[:2]
        for ax in axes:
            ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            for crop in t['crops']:
                x1,y1,x2,y2=crop;ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='orange',linewidth=.7,alpha=.55))
            for trace,dem,color in [(ch,cd,'dodgerblue'),(t,fd,'cyan')]:
                for i,(b,d) in enumerate(zip(trace['xyxy'],dem)):
                    if box is not None:
                        centre=(b[:2]+b[2:])/2;bc=(box[:2]+box[2:])/2
                        if np.linalg.norm(centre-bc)>max(250,2*np.linalg.norm(box[2:]-box[:2])):continue
                    x1,y1,x2,y2=b;ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,linewidth=1.1))
                    ax.text(x1,y1,f'{i}:{d:.2f}',fontsize=6,color=color,bbox=dict(facecolor='black',alpha=.45,pad=1))
            if box is not None:
                x1,y1,x2,y2=box;ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='red',linewidth=2))
            ax.set_xticks([]);ax.set_yticks([])
        if box is not None:
            center=(box[:2]+box[2:])/2;span=np.maximum((box[2:]-box[:2])*2.8,[240,180])
            axes[1].set_xlim(max(0,center[0]-span[0]),min(W,center[0]+span[0]))
            axes[1].set_ylim(min(H,center[1]+span[1]),max(0,center[1]-span[1]))
        axes[0].set_title('Red: decisive annotation; blue: cheap; cyan: final; orange: crops',fontsize=9)
        axes[1].set_title('Numbers: object index and required demand (m/s²)',fontsize=9)
        title=f"{ds} {unit[:18]} frame {fr} | {c['kind']}\nGT {r['a_gt']:.3f} | ego {v:.2f} m/s | baseline action {r['a_v6S_b2A_retro|v3a']} → R5 both q50 {new['a_v6S_r5_both_50|v3a']}"
        fig.suptitle(title);fig.tight_layout();stem=f'{ci:02d}_{c["kind"]}'
        fig.savefig(out/(stem+'.png'),dpi=130);fig.savefig(out/(stem+'.pdf'));plt.close(fig)
        d=dict(c,dataset=ds,gt_demand=r['a_gt'],v=v,baseline=r['a_v6S_b2A_retro|v3a'],candidate=new['a_v6S_r5_both_50|v3a'],
            gt_class=str(g['coarse'][dec]) if dec>=0 else '',gt_z=float(g['long_near'][dec]) if dec>=0 else np.nan,
            gt_ttc=float(g['ttc'][dec]) if dec>=0 else np.nan,image_path=path,panel=stem+'.png')
        details.append(d)
    pd.DataFrame(details).to_csv(out/'manifest.csv',index=False)
    save(out/'DONE.json',dict(status='complete',dataset=ds,panels=len(details),selection='Deterministic maxima by category, not a random sample; full population audit and every changed episode reported separately.'))

if __name__=='__main__':main()
