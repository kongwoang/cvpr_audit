"""CPU-only D01 routes from stored scores and locked dev choices, including history frames."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
from k40r_mean_policy import ada_side,canvas,eligible_history
from k40r_matched import policy


def digest(p):return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only routing preparation'
    b=Path(os.environ['B']);out=b/'results/k40r/routes';out.mkdir(parents=True,exist_ok=True)
    choice_path=b/'results/k40r/matched/dev_choices.json';choices=json.loads(choice_path.read_text())
    manifest=dict(choices_md5=digest(choice_path),source_hashes={},datasets={},history_rule='own policy, consecutive frames at the same scale; no skipped-arm inference',
                  script_md5=digest(Path(__file__)))
    totals=dict(nusc=(228,9181,9181),av2=(150,11818,11818),ahd=(24,15062,15006))
    for ds in ('nusc','av2','ahd'):
        src=b/'results/k33'/f'{ds}_pub_test';dst=out/ds;dst.mkdir(exist_ok=True)
        files=sorted(src.glob('*.pkl'));rows=[]
        for p in files:
            manifest['source_hashes'][str(p.relative_to(b))]=digest(p)
            rows.extend(pickle.load(p.open('rb')))
        tab=pd.DataFrame(rows);dev='av2' if ds=='ahd' else ds;choice=choices[dev]
        assert (len(files),len(rows),int(tab['gt'].astype(bool).sum()))==totals[ds]
        assert not tab.duplicated(['unit','frame']).any()
        score,sizes=policy(tab,dev);native=dict(nusc=1600,av2=2048,ahd=1920)[ds]
        result={};counts={name:{} for name in ('AdaScale','BAR','Router-G','Chanakya')}
        for i,r in enumerate(rows):
            routes={'AdaScale':dict(side=ada_side(int(r['ada_pred']),choice['AdaScale']['row'],native),prediction=int(r['ada_pred']),policy=choice['AdaScale']['row'])}
            for label in ('BAR','Router-G'):
                c=choice[label];s=float(r[c['score']]);assert np.isfinite(s)
                routes[label]=dict(side=1280 if s>=c['threshold'] else 0,score=s,threshold=c['threshold'])
            routes['Chanakya']=dict(side=int(sizes[i]) if score[i]>=choice['Chanakya']['threshold'] else 0,
                                     score=float(score[i]),threshold=choice['Chanakya']['threshold'],strong_side=int(sizes[i]))
            for label,route in routes.items():
                sd=route['side'];counts[label][str(sd)]=counts[label].get(str(sd),0)+1
            unit=str(r['unit']);frame=str(int(r['frame']))
            result.setdefault(unit,{})[frame]=dict(gt=bool(r['gt']),routes=routes)
        for unit,data in result.items():
            p=dst/(unit+'.json');encoded=json.dumps(data,separators=(',',':'))+'\n'
            if p.exists():assert p.read_text()==encoded,('immutable routing changed',p)
            else:p.write_text(encoded)
        manifest['datasets'][ds]=dict(units=len(result),frames=len(rows),evaluated_frames=int(tab['gt'].astype(bool).sum()),dev_dataset=dev,route_counts=counts)
        ck=b/'results/k33/rc'/f'chanakya_ch128B_csc_{dev}.pt';manifest['source_hashes'][str(ck.relative_to(b))]=digest(ck)
    # K33 AdaScale grid rule and own-history eligibility fixtures.
    for ds,native in (('nusc',1600),('av2',2048),('ahd',1920)):
        for p in (320,480,640,800,960):assert ada_side(p,'ada-1t1280',native)=={320:320,480:320,640:480,800:640,960:800}[p]
    old=dict(s1=np.ones((1,4)),s2=np.ones((2,4)),scales=(640,960))
    assert eligible_history(old,640)['s1'] is old['s1'] and eligible_history(old,640)['s2'] is None
    assert eligible_history(old,960)['s1'] is None and eligible_history(old,960)['s2'] is old['s2']
    assert eligible_history(old,1280)['s1'] is None and eligible_history(old,1280)['s2'] is None
    assert old['s1'].shape==(1,4) and old['s2'].shape==(2,4)
    manifest['cpu_checks']=dict(ada_ladder_cases=15,history_cases=3,detector_invoked=False)
    manifest['routing_hashes']={str(p.relative_to(out)):digest(p) for ds in ('nusc','av2','ahd') for p in sorted((out/ds).glob('*.json'))}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['datasets'],indent=2),flush=True)


if __name__=='__main__':main()
