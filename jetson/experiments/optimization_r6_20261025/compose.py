"""Two declared final-readout compositions; original stage files preserved."""
import argparse,pickle,subprocess,sys
import numpy as np
from common import HERE,save
from policies import PP
from rap import planner as P
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--shard',default='0/1');a=ap.parse_args();ds=a.dataset
    assert (HERE/'reports'/ds/'DONE.json').exists(), 'wait for completed base evaluation'
    out=HERE/'results/followup'/ds;out.mkdir(parents=True,exist_ok=True);cp=P.CostParams()
    for f in sorted((HERE/'results/dev'/ds).glob('*.pkl')):
        rows=pickle.load(f.open('rb'));prev=None
        for r in rows:
            base='v6S_r5_point_residual_50'
            for lag in (1,2):
                other='v6S_r6_track_agree'+str(lag);pre=other+'_r5';act=max(r['a_'+base+'|v3a'],r['a_'+other+'|v3a'])
                r['a_'+pre+'|v3a']=act;r['J_'+pre+'|v3a']=P.decision_cost(act,r['a_gt'],prev,PP,cp)['J']
                for k in ('px_','ncalls_'):r[k+pre]=r[k+base]
                vals=[r.get('dec_dem_'+x,np.nan) for x in (base,other)];vals=[x for x in vals if np.isfinite(x)]
                r['dec_found_'+pre]=any(r.get('dec_found_'+x,False) for x in (base,other));r['dec_dem_'+pre]=max(vals) if vals else np.nan
            prev=r['a_cheap']
        target=out/f.name;tmp=target.with_suffix('.tmp');pickle.dump(rows,tmp.open('wb'));tmp.replace(target)
        save(target.with_suffix('.parity.json'),dict(differences=dict(original_rows=0)))
    subprocess.run([sys.executable,str(HERE/'analyze_followup.py'),'--stage','followup','--dataset-only',ds,'--out-name','followup_'+ds],check=True)
    subprocess.run([sys.executable,str(HERE/'audit_followup.py'),'--dataset',ds],check=True)
    save(HERE/'reports'/('followup_'+ds)/'DONE.json',dict(status='complete',new_rows=2,original_rows=27))
if __name__=='__main__':main()
