"""Sandbox-only mechanics. Imported by a generated copy of the frozen dev harness."""
import numpy as np
from rap import planner as P
from rap.mono import box_iou
from vtf import speculative as SP


def unique_match(box, coarse, det):
    if not len(det['xyxy']):
        return -1
    scores = box_iou(np.asarray(box).reshape(1, 4), det['xyxy'])[0]
    ids = np.flatnonzero((scores >= .3) & (np.asarray(det['coarse']) == coarse))
    return int(ids[0]) if len(ids) == 1 else -1


def obligations(scene, crops, grown, field_fn, W, H, v, pp):
    field = field_fn(scene['mono'])
    fc = int(field is not None and not SP.covered(field, crops))
    support = [SP.box_crop(scene['xyxy'][i], W, H) for i in SP.support(scene['mono'], v, pp)]
    ns = sum(not SP.covered(c, list(crops) + list(grown)) for c in support)
    return fc, int(ns)


class RefineClosure:
    """Keep corrections for stable detections; compute all factors from raw geometry."""
    def __init__(self, refine, v, pp):
        self.refine, self.v, self.pp = refine, v, pp
        self.factors = {}
        self.new_corrections = 0

    def __call__(self, scene):
        raw = scene.get('_raw_mono', scene['mono'])
        keys = [(tuple(b), str(c), float(conf), float(z)) for b,c,conf,z in
                zip(scene['xyxy'],scene['coarse'],scene['conf'],raw[0])]
        out = dict(scene)
        out['_raw_mono'] = tuple(np.asarray(a).copy() for a in raw)
        # Each iteration permanently adds >=1 corrected detection, bounded by object count.
        for _ in range(len(keys) + 1):
            tt = [np.asarray(a, float).copy() for a in raw]
            for i,key in enumerate(keys):
                if key in self.factors:
                    for a in tt[:3]: a[i] *= self.factors[key]
            out['mono'] = tuple(tt)
            sup = [int(i) for i in SP.support(out['mono'],self.v,self.pp)
                   if scene['from_full'][i] and keys[i] not in self.factors]
            if not sup: return out
            original = dict(scene, mono=tuple(raw))
            refined = self.refine(original,'v3a',lambda _: sup)
            for i in sup:
                assert np.isfinite(raw[0][i]) and raw[0][i] > 0
                self.factors[keys[i]] = float(refined[0][i] / raw[0][i])
                self.new_corrections += 1
        raise AssertionError('refinement failed finite-object termination')


def draft_recheck(base, scene, crops, grown, W, H, v, pp, box_crop):
    """One extra evidence acquisition; no unconditional preservation of cheap actions."""
    inside = SP.centres_in(base['xyxy'],list(crops) + list(grown))
    _,dem = P.required_decel(*base['mono'],v,pp)
    support = SP.support(base['mono'],v,pp)
    for i in sorted(support,key=lambda i: (-dem[i],int(i))):
        if inside[i] and unique_match(base['xyxy'][i],base['coarse'][i],scene) < 0:
            cr = box_crop(base['xyxy'][i],W,H,3.0)
            if not SP.covered(cr,list(crops) + list(grown)):
                return [cr]
    return []


class Retro:
    """One canvas / two genuine past observations; acquired boxes never become stale state."""
    def __init__(self, base, imgs, log, run_det, chains, v, pp):
        self.base,self.imgs,self.log,self.run_det,self.chains = base,imgs,log,run_det,chains
        self.v,self.pp = v,pp
        self.matcher = unique_match
        self.events = []
        self.used = False
        self.recovered = {}  # current exact box -> actual boxes in t-1 and t-2
        self.calls,self.pixels,self.attempted,self.success = 0,0,0,0

    def augment(self, full, raw, history):
        X = np.asarray(full['xyxy']).reshape(-1,4)
        chn = self.chains(X,history.get('s1'),history.get('s2')) if len(self.imgs)==2 else []
        missing = set(range(len(X))) - {i for i,_,_ in chn}
        if not self.used and len(self.imgs)==2:
            _,dc = P.required_decel(*self.base['mono'],self.v,self.pp)
            _,df = P.required_decel(*raw,self.v,self.pp)
            candidates = []
            for i in sorted(missing):
                j = unique_match(X[i],full['coarse'][i],self.base)
                if (j >= 0 and dc[j] >= self.pp.thr_decel and df[i] < self.pp.thr_decel
                    and np.isfinite(raw[0][i]) and raw[0][i] <= self.pp.max_range
                    and raw[1][i] < self.pp.corridor_half_w and raw[2][i] > -self.pp.corridor_half_w
                    and tuple(X[i]) in self.log):
                    candidates.append((-float(dc[j]),i))
            if candidates:
                _,i = min(candidates)
                self.used = True; self.attempted += 1
                key = self.log[tuple(X[i])]
                past=[]; box=X[i]
                for lag in (1,2):
                    det = self.run_det(self.imgs[-lag],key)
                    self.calls += 1; self.pixels += key[-2]*key[-1]
                    j = self.matcher(box,full['coarse'][i],det)
                    self.events.append(dict(lag=lag,key=key,target=box.copy(),coarse=str(full['coarse'][i]),
                        detections={k:np.asarray(v).copy() for k,v in det.items()},match=j))
                    if j < 0: break
                    box = np.asarray(det['xyxy'][j]).copy(); past.append(box)
                if len(past)==2:
                    self.recovered[tuple(X[i])] = past; self.success += 1
        # Construct explicit chain entries. Do not re-associate the entire population and
        # accidentally steal a valid original chain when adding a historical observation.
        h = dict(history)
        p1 = list(np.asarray(h.get('s1',np.zeros((0,4)))) if h.get('s1') is not None else np.zeros((0,4)))
        p2 = list(np.asarray(h.get('s2',np.zeros((0,4)))) if h.get('s2') is not None else np.zeros((0,4)))
        for i in sorted(missing):
            past = self.recovered.get(tuple(X[i]))
            if past is not None:
                chn.append((i,len(p1),len(p2))); p1.append(past[0]);p2.append(past[1])
        h['s1'] = np.asarray(p1).reshape(-1,4);h['s2'] = np.asarray(p2).reshape(-1,4)
        return h,chn
