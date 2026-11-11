"""Read-only history-association audit on the fixed earlier dev snapshot."""
import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR')
sys.path.insert(0,str(ROOT/'scripts/A_method'))
from k27_dev import assoc_idx
from k28_tdata import chains
from rap.mono import box_iou
from rap import planner as P


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    units=json.loads(args.manifest.read_text());pp=P.PlannerParams()
    totals=[];unitrows=[]
    for ds,names in units.items():
        total=Counter()
        for name in names:
            rows=pickle.load((ROOT/'experiments/w1_dev_20261023/results/dev'/ds/(name+'.pkl')).open('rb'))
            previous=[];c=Counter()
            for row in rows:
                t=row['trace_v6S'];full=np.flatnonzero(t['from_full'])
                x=t['xyxy'][full];coarse=t['coarse'][full]
                c['frames']+=1;c['full_objects']+=len(x)
                relevant=P.in_corridor(t['raw'][1][full],t['raw'][2][full],pp.corridor_half_w)&(t['raw'][0][full]<=pp.max_range)
                c['corridor_range_full_objects']+=int(relevant.sum())
                if previous:
                    c['nonconsecutive_frame_transitions']+=int(row['frame']-previous[-1]['frame']!=1)
                    prev=previous[-1]
                    j1=assoc_idx(x,prev['x'])
                    if len(prev['x']) and len(x):
                        overlaps=box_iou(x,prev['x'])>=.3
                        amb=overlaps.sum(1)>1
                        compatible=overlaps&(coarse[:,None]==prev['coarse'][None,:])
                        c['ambiguous_current_links']+=int((amb&(j1>=0)).sum())
                        c['ambiguous_relevant_current_links']+=int((amb&(j1>=0)&relevant).sum())
                        c['multiple_compatible_relevant_options']+=int(((compatible.sum(1)>1)&relevant).sum())
                    for i,j in enumerate(j1):
                        if j>=0:
                            c['current_links']+=1
                            c['class_switch_current_links']+=int(coarse[i]!=prev['coarse'][j])
                            c['class_switch_relevant_current_links']+=int(relevant[i] and coarse[i]!=prev['coarse'][j])
                if len(previous)==2:
                    p1,p2=previous[-1],previous[-2]
                    ch=chains(x,p1['x'],p2['x'])
                    assert {int(full[i]) for i,_,_ in ch}==set(t['chain_ids']),(ds,name,row['frame'])
                    for i,j,k in ch:
                        c['chains']+=1
                        bad=coarse[i]!=p1['coarse'][j] or p1['coarse'][j]!=p2['coarse'][k]
                        c['class_switch_chains']+=int(bad)
                        if relevant[i]:
                            c['relevant_chains']+=1;c['class_switch_relevant_chains']+=int(bad)
                previous=(previous+[dict(x=x,coarse=coarse,frame=row['frame'])])[-2:]
            unitrows.append(dict(dataset=ds,unit=name,**c));total.update(c)
        totals.append(dict(dataset=ds,units=len(names),**total))
    args.out.mkdir(parents=True,exist_ok=True)
    for filename,rows in [('summary',totals),('units',unitrows)]:
        columns=list(dict.fromkeys(k for row in rows for k in row))
        with (args.out/(filename+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    (args.out/'DONE.json').write_text(json.dumps(dict(status='complete',
        scope='Fixed 152-scene/116-log snapshot; coarse-class changes and box ambiguity only, no GT track-ID validation',
        frozen_chain_ids_reproduced=True),indent=2)+'\n')
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
