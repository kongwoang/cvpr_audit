"""Exact K31 bottleneck insertion scores; no planner or parameter change."""
import numpy as np

def field_support(sel,scene,probes):
    scores, H=sel.score(scene.o)
    action=int(np.argmax(scores))
    if probes:
        hp=sel.h(np.asarray([row for row,_ in probes]))
        inserted=np.minimum(scores[:,None],hp)
        changed=np.argmax(inserted,axis=0)!=action
        field=[box for (_,box),keep in zip(probes,changed) if keep]
    else:field=[]
    others=np.arange(len(scores))!=action
    support=[scene.dbox[k] for k in range(len(scene.o)) if scene.dbox[k] is not None and (H[others,k]<=scores[action]).any()]
    return action,field,support
