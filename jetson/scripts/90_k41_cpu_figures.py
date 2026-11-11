#!/usr/bin/env python3
"""Render executed CPU summaries only; no model or dataset imports."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
H=ROOT/'docs/paper/handoff';K=H/'K41';R=H/'K40R';L=H/'LATENCY_OPT_20261030'
DS=['nuscenes','av2','argoverse_hd'];LABEL=['nuScenes','AV2','Argoverse-HD']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})


def save(fig,folder,name):
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(folder/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    d=pd.read_csv(R/'k40_1c_taxonomy.csv')
    fig,axs=plt.subplots(1,3,figsize=(10,3))
    for ds,label,ax in zip(DS,LABEL,axs):
        q=d[(d.dataset==ds)&(d.setting=='no_refiner')&(d.metric=='contribution_delta_G')].set_index('category').loc[['i','ii','iii','iv']]
        ax.errorbar(np.arange(4),q.value,yerr=[q.value-q.lo,q.hi-q.value],fmt='o',capsize=3)
        ax.axhline(0,color='grey',lw=.7);ax.set_xticks(np.arange(4));ax.set_xticklabels(['outside','state','DESP-only','other'],rotation=20)
        ax.set_title(label);ax.set_ylabel('Contribution to ΔG')
    save(fig,R,'k40_1c_contributions')
    d=pd.read_csv(K/'k41_3_strata.csv');rows=['DESP','u960 + comp.','hybrid','uncertainty','AnA iterated']
    for feature in ['target_count','corridor_count','support_size','decisive_height','decisive_closing','decisive_range']:
        fig,axs=plt.subplots(1,3,figsize=(11,3.2),sharey=False)
        for ds,label,ax in zip(DS,LABEL,axs):
            for j,row in enumerate(rows):
                q=d[(d.dataset==ds)&(d.feature==feature)&(d.row==row)&(d.refiner=='v3a')&(d.metric=='G')].set_index('stratum').reindex(['low','middle','high','missing'])
                x=np.arange(4)+(j-2)*.12
                ax.errorbar(x,q.value,yerr=[q.value-q.lo,q.hi-q.value],fmt='.',capsize=2,label=row)
            ax.axhline(0,color='grey',lw=.7);ax.set_xticks(np.arange(4));ax.set_xticklabels(['low','mid','high','missing']);ax.set_ylabel('G');ax.set_title(label)
        axs[-1].legend(fontsize=7,loc='best');fig.suptitle(feature+' — descriptive test strata; unit bootstrap 95% CI')
        save(fig,K,'k41_3_'+feature)
    d=pd.read_csv(K/'k41_4_locality.csv');fig,axs=plt.subplots(1,3,figsize=(10,3.2),sharey=True)
    for ds,label,ax in zip(DS,LABEL,axs):
        for j,th in enumerate((.5,.7)):
            q=d[(d.dataset==ds)&(d.feature=='height_bin')&(d.metric=='match_rate')&(d.iou_threshold==th)].set_index('stratum').reindex(['low','middle','high'])
            ax.errorbar(np.arange(3)+j*.07,q.value,yerr=[q.value-q.lo,q.hi-q.value],fmt='o-',capsize=3,label=f'IoU ≥ {th}')
        ax.set_xticks(np.arange(3));ax.set_xticklabels(['small','middle','large']);ax.set_title(label);ax.set_ylim(0,1);ax.set_ylabel('Crop-origin match rate');ax.legend()
    save(fig,K,'k41_4_locality_height')
    d=pd.read_csv(L/'BENCHMARK.csv');fig,axs=plt.subplots(1,2,figsize=(10,3.5),sharey=True)
    names=['u512+c1comp@trt32','u640+c1comp@trt32','u960+c1comp@trt32','ours@trt32','ours_b2A@trt32','ours_b1A@trt32']
    for ds,label,ax in zip(['nusc_test20','av2t40'],LABEL,axs):
        q=d[d.dataset==ds].set_index('row').reindex(names)
        ax.errorbar(np.arange(len(q)),q.p95_ms,yerr=[q.p95_ms-q.p95_lo,q.p95_hi-q.p95_ms],fmt='o',capsize=3)
        ax.set_xticks(np.arange(len(q)));ax.set_xticklabels(['u512','u640','u960','DESP','B2K-A','B1K-A'],rotation=20)
        ax.set_title(label+' — historical K39');ax.set_ylabel('p95 latency (ms)');ax.axhline(50,color='grey',ls=':',lw=.7)
    save(fig,L,'historical_p95_frontier')


if __name__=='__main__':main()
