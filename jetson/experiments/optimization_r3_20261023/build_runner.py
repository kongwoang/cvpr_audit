"""Assertive local fork of completed R2 harness. Never edits frozen source files."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'optimization_r2_20261023'
def once(s,a,b):
    assert s.count(a)==1,(s.count(a),a[:90]);return s.replace(a,b)
def main():
    for f in ('helpers.py','bounded_variant.py','sandbox_loop.py','proactive.py','estimators.py','case_manifest_with_severity.json'):
        (HERE/f).write_bytes((OLD/f).read_bytes())
    s=(OLD/'runner.py').read_text().replace('optimization_r2_20261023/results','optimization_r3_20261023/results')
    s=once(s,'from proactive import Pending','from proactive import Pending\nimport bridge')
    s=once(s,'    ap.add_argument("--tag", default="")','    ap.add_argument("--tag", default="")\n    ap.add_argument("--suite",default="bridge",choices=["bridge","learned"])')
    s=once(s,'        trace_case=unit in case_units','        trace_case=True  # all-row mechanism attribution, no outcome-based recording')
    s=once(s,'            GT = {}',
        "            rec['gt_boxes']=gtb;rec['gt_classes']=gtc;rec['gt_ids']=np.asarray(g['track_id']) if len(g) else np.zeros(0,int)\n            GT = {}")
    s=once(s,'                    if not chn:\n                        return finish(full,(z,lo,hi,ttc))',
        '''                    if '_bridge' in name:
                        h,chn,nbridge=bridge.augment(full,h,chn,st.get('cheap_history',{}))
                        rec['bridge_chains_'+name]=rec.get('bridge_chains_'+name,0)+nbridge
                    if '_transfer' in name:
                        (z,lo,hi,ttc),ntransfer=bridge.transfer(full,(z,lo,hi,ttc),chn,base)
                        rec['transferred_'+name]=rec.get('transferred_'+name,0)+ntransfer
                    if not chn:
                        return finish(full,(z,lo,hi,ttc))''')
    s=once(s,'                seed=pending.seed(base,W,H,KL.box_crop_ctx) if pending else []',
        '''                seed=pending.seed(base,W,H,KL.box_crop_ctx) if pending else []
                if '_extent' in nm:seed+=bridge.extent_seed(st.setdefault(nm,{}),base,W,H,KL.box_crop_ctx)
                vc=bridge.motion_probe(base,PP0) if '_motion' in nm else 0.
                rec['probe_vclose_'+nm]=vc''')
    s=once(s,"PP0,2,'A',ctx=ctx,seed=seed)","PP0,2,'A',ctx=ctx,seed=seed,v_closes=(0.,vc))")
    s=once(s,"                score(nm,sc,('half',),'base',c2=False)",
        "                if nm=='v6S_b2A_retro':score(nm,sc,('half','speed','speedhalf'),'base',c2=False)\n"
        "                if '_extent' in nm:bridge.extent_update(st[nm],sc,tr,v,PP0)")
    s=once(s,'                if name == "half":',
        '''                if name in ('speed','speedhalf'):
                    rf=[np.asarray(q).copy() for q in refine(scene,'v3a',sup_fn)]
                    good=np.isfinite(tt[0])&(tt[0]>0)&np.isfinite(rf[0])
                    power=1. if name=='speed' else .5
                    rf[3][good]*=np.power(rf[0][good]/tt[0][good],power)
                    return tuple(rf)
                if name == "half":''')
    s=once(s,"            for nm in ('v6S','v6S_b2A_retro','v6S_b2A_pending','v6S_b2A_pending_retro','v6S_b2A_floor_retro'):",
        "            for nm in ('v6S','v6S_b2A_retro','v6S_b2A_bridge','v6S_b2A_bridge_retro','v6S_b2A_transfer_retro','v6S_b2A_extent_retro','v6S_b2A_motion_retro'):")
    s=once(s,'            imgs = (imgs + [img])[-2:]',
        '''            chist=st.setdefault('cheap_history',{})
            chist['s2']=chist.get('s1');chist['c2']=chist.get('c1')
            chist['s1']=np.asarray(base['xyxy']).copy();chist['c1']=np.asarray(base['coarse']).copy()
            imgs = (imgs + [img])[-2:]''')
    (HERE/'runner.py').write_text(s)
    print('Built isolated bridge runner')
if __name__=='__main__':main()
