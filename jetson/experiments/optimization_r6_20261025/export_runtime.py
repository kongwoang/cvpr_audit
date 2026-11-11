"""No-image/no-GT runtime inputs; reference parity on every development frame."""
import argparse,pickle
from common import HERE,REMOTE,save
from bounded_readout import readout
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--shard',default='0/1');a=p.parse_args();ds=a.dataset;out=HERE/'runtime_inputs';out.mkdir(exist_ok=True)
    units=[];mismatches=[];total=0
    for f in sorted((REMOTE/'experiments/optimization_r3_20261023/results/low'/ds).glob('*.pkl')):
        old=pickle.load(f.open('rb'));props=pickle.load((HERE/'proposals'/ds/f.name).open('rb'));new=pickle.load((HERE/'results/dev'/ds/f.name).open('rb'));hist=[];rows=[]
        for r,p,s in zip(old,props,new):
            scene=r['trace_v6S_b2A_missing_extent_retro'];scene={k:scene[k] for k in ('xyxy','coarse','from_full','refined')}
            proposal={k:p[k] for k in ('xyxy','coarse','conf','mono','threshold')};base=r['a_v6S_b2A_missing_extent_retro|v3a'];scale=.6 if ds=='nusc' else 960/2048
            actual=[]
            for lag in (1,2):
                aa=readout(scene,proposal,hist,r['v'],r['dt'],scale,lag);expected=s['a_v6S_r6_track_agree'+str(lag)+'|v3a']
                if max(aa,base)!=expected:mismatches.append(dict(unit=f.stem,frame=r['frame'],lag=lag,expected=expected,actual=max(aa,base)))
                actual.append(aa)
            rows.append(dict(frame=r['frame'],scene=scene,proposal=proposal,v=r['v'],dt=r['dt'],scale=scale,expected_extra=actual,
                r5_action=s['a_v6S_r5_point_residual_50|v3a'],expected_final=[max(s['a_v6S_r5_point_residual_50|v3a'],s['a_v6S_r6_track_agree'+str(lag)+'|v3a']) for lag in (1,2)]))
            hist=(hist+[dict(scene=scene,v=r['v'])])[-2:];total+=1
        units.append(dict(unit=f.stem,rows=rows))
    save(out/(ds+'_parity.json'),dict(status='complete' if not mismatches else 'failed',frames=total,mismatches=mismatches,scope='optimized extra-action function versus reference after max with original parent; no detector/TRT parity claim'))
    assert not mismatches
    f=out/(ds+'.pkl');tmp=f.with_suffix('.tmp');pickle.dump(units,tmp.open('wb'),protocol=4);tmp.replace(f)
if __name__=='__main__':main()
