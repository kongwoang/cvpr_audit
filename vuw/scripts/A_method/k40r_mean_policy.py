"""D01 fixed routing and causal, fidelity-consistent history helpers; no fitting."""
import math
import re
import numpy as np

SCALES=(320,480,640,800,960)


def ada_side(prediction, policy, native):
    match=re.fullmatch(r'ada([+-]\d+)t(960|1280|nat)',policy)
    assert match and prediction in SCALES,(prediction,policy)
    shift=int(match.group(1));top=native if match.group(2)=='nat' else int(match.group(2))
    ladder=sorted(set(list(SCALES)+list(range(1120,native,160))+[native]))
    top_idx=ladder.index(top) if top in ladder else len(ladder)-1
    return ladder[int(np.clip(SCALES.index(prediction)+shift,0,top_idx))]


def canvas(ds,side,ada=False):
    if ada and side in SCALES and ds!='ahd':
        return (int(round(side*9/16/32))*32,side) if ds=='nusc' else (side,int(round(.8*side/32))*32)
    if ds=='ahd':return int(math.ceil(side*1200/1920/32))*32,side
    if ds=='nusc':
        return {640:(384,640),960:(576,960),1280:(736,1280),1536:(864,1536),1600:(928,1600)}.get(side,(int(math.ceil(side*900/1600/32))*32,side))
    assert ds=='av2'
    return {640:(640,512),960:(960,768),1280:(1280,992),1536:(1536,1184),2048:(2048,1568)}.get(side,(side,int(math.ceil(side*1550/2048/32))*32))


def eligible_history(history, side):
    """No stale-frame carry or unexecuted-arm history. C1 needs both adjacent frames."""
    result=dict(history);scales=history.get('scales',(None,None))
    if scales[0]!=side:result['s1']=None
    if scales[1]!=side:result['s2']=None
    return result
