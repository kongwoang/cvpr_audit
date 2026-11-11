"""Case timelines expose why an L1 rescue is not adequate severe braking."""
import pickle
import numpy as np
import pandas as pd
from common import HERE,save
from rap import planner as P

def main():
    out=HERE/'reports/cases';out.mkdir(parents=True,exist_ok=True);selected=[]
    # Deterministic illustrative choices, not an unbiased case sample.
    for stage,report,ds,pipe,rule in [('compact','compact','nusc','compact point fallback','unsupported'),
                                    ('low','low','nusc','corroborated upper fallback','severe'),
                                    ('learned','learned','av2','learned missing history','unsupported')]:
        d=pd.read_csv(HERE/'reports'/report/'changed_episode_attribution.csv');q=d[(d.dataset==ds)&(d.pipeline==pipe)&(d.transition=='rescued')]
        if rule=='unsupported':q=q[~q.any_decisive_supported_brake]
        if len(q):
            r=q.sort_values(['peak_demand','unit','start_frame'],ascending=[False,True,True]).iloc[0]
            selected.append(dict(stage=stage,dataset=ds,pipeline=pipe,unit=r.unit,start=int(r.start_frame),end=int(r.end_frame),rule=rule))
    d=pd.read_csv(HERE/'reports/v3a_attribution/learned_av2_episodes.csv');q=d[(d.pipeline=='uncertainty query')&(d.transition=='rescued')]
    r=q.sort_values(['peak','unit','start'],ascending=[False,True,True]).iloc[0]
    selected.append(dict(stage='learned',dataset='av2',pipeline='uncertainty query',unit=r.unit,start=int(r.start),end=int(r.end),rule='highest peak among rescued episodes'))
    prefixes={'compact point fallback':'v6S_b2A_missing_compact_retro','corroborated upper fallback':'v6S_b2A_missing_corroborated_retro','learned missing history':'v6S_b2A_missing_retro','uncertainty query':'v6S_b2A_query_retro'}
    pp=P.PlannerParams();commands=np.array([P.commanded_decel(i,pp) for i in range(3)]);frames=[]
    v3a=pd.read_csv(HERE/'reports/v3a_attribution/learned_av2_frames.csv')
    for c in selected:
        rows=pickle.load((HERE/'results'/c['stage']/c['dataset']/(c['unit']+'.pkl')).open('rb'));prefix=prefixes[c['pipeline']]
        for r in rows:
            fr=int(r['frame'])
            if not c['start']-3<=fr<=c['end']+3:continue
            dec_demand=r.get('dec_dem_'+prefix,np.nan);found=r.get('dec_found_'+prefix,False)
            if c['pipeline']=='uncertainty query':
                q=v3a[(v3a.unit==c['unit'])&(v3a.frame==fr)&(v3a.prefix==prefix)]
                dec_demand=float(q.dec_demand.iloc[0]) if len(q) else np.nan;found=bool(q.dec_found.iloc[0]) if len(q) else False
            frames.append(dict(**c,frame=fr,seconds=(fr-c['start'])*r['dt'],gt_demand=r['a_gt'],gt_action=r['act_gt'],
                baseline_action=r['a_v6S_b2A_retro|v3a'],candidate_action=r['a_'+prefix+'|v3a'],decisive_found=found,decisive_predicted_demand=dec_demand))
    d=pd.DataFrame(frames);d.to_csv(out/'timelines.csv',index=False);save(out/'selection.json',dict(cases=selected,rule='Exploratory illustrative cases; severe and unsupported rescues, not representative sampling.'))
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(len(selected),1,figsize=(11,3*len(selected)),squeeze=False)
    for ax,c in zip(axs[:,0],selected):
        g=d[(d.unit==c['unit'])&(d.pipeline==c['pipeline'])].sort_values('frame')
        ax.plot(g.seconds,g.gt_demand,'k.-',label='Required demand (GT)')
        ax.step(g.seconds,commands[g.baseline_action.to_numpy(int)],where='mid',color='#c76a00',label='Round1 command')
        ax.step(g.seconds,commands[g.candidate_action.to_numpy(int)],where='mid',color='#005aa7',label='Candidate command')
        ax.plot(g.seconds,g.decisive_predicted_demand,':',color='#25905b',label='Matched decisive demand estimate')
        ax.axhline(3.5,color='gray',ls='--',lw=.7);ax.set_ylim(-.3,9.4);ax.set_ylabel('m/s²');ax.grid(alpha=.2)
        ax.set_title(c['dataset']+' / '+c['unit']+' / '+c['pipeline']+' ('+c['rule']+')',fontsize=9);ax.set_xlabel('Seconds relative to episode onset');ax.legend(fontsize=7,ncol=2)
    fig.tight_layout();fig.savefig(out/'cases.pdf');fig.savefig(out/'cases.png',dpi=160);plt.close(fig)
    (out/'REPORT.md').write_text('# Changed-episode case timelines\n\nIllustrative severe and unsupported rescues, chosen by deterministic rules in selection.json. All changed episodes remain available in stage attribution CSVs. Commands are0/2.5/6 m/s²; required demand is the frozen benchmark consumer. A binary DECELERATE-episode rescue may still miss the peak frame or under-brake severely. A missing decisive-object match cannot establish that the correct hazard was recovered. Query v3a decisive estimates are independently replayed and action-checked on episode frames; outside that replay range the estimate is omitted. These are offline records, not closed-loop stopping trajectories.\n')
    save(out/'DONE.json',dict(status='complete',cases=len(selected),frames=len(d)))
if __name__=='__main__':main()
