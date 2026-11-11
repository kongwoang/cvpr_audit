"""Focused low-tradeoff compositions, no changes to existing in-flight policies."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
def once(s,a,b):
    assert s.count(a)==1,(s.count(a),a[:80]);return s.replace(a,b)
def main():
    s=(HERE/'runner_learned.py').read_text()
    s=once(s,'                    out=[np.asarray(q).copy() for q in mono];known={i for i,_,_ in chn}',
        '''                    out=[np.asarray(q).copy() for q in mono];known={i for i,_,_ in chn}
                    corroborated=set()
                    if '_corroborated' in name:
                        tt,_=bridge.transfer(full,mono,chn,base)
                        _,dc=P.required_decel(*tt,v,PP0)
                        links=bridge.match(full['xyxy'],full['coarse'],base['xyxy'],base['coarse'])
                        corroborated={i for i,j in enumerate(links) if j>=0 and np.isfinite(base['mono'][3][j]) and base['mono'][3][j]>0 and dc[i]>=PP0.thr_decel}
                    rec.setdefault('fallback_evidence_'+name,[])''')
    s=once(s,"                        rate=float(pred[i,4] if '_upper' in name else pred[i,0])",
        '''                        if '_observed' in name and mask[i].sum()<2:continue
                        use_upper=('_upper' in name) or ('_corroborated' in name and i in corroborated and mask[i].sum()>=2)
                        rate=float(pred[i,4] if use_upper else pred[i,0])
                        rec['fallback_evidence_'+name].append(dict(box=full['xyxy'][i].copy(),observations=int(mask[i].sum()),rate=rate,use_upper=use_upper))''')
    s=once(s,"('v6S','v6S_b2A_retro','v6S_b2A_missing_retro','v6S_b2A_allgeo_retro','v6S_b2A_alldec_retro','v6S_b2A_upper_retro','v6S_b2A_query_retro')",
        "('v6S','v6S_b2A_retro','v6S_b2A_missing_retro','v6S_b2A_upper_retro','v6S_b2A_missing_extent_retro','v6S_b2A_missing_observed_retro','v6S_b2A_missing_corroborated_retro')")
    (HERE/'runner_low.py').write_text(s)
    print('Built focused low-tradeoff suite')
if __name__=='__main__':main()
