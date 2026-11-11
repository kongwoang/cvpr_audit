"""K35-5 (post-hoc, GPU): zero-value-of-computation test on the stored v6-S scenes of the three test splits.
On top of v6-S's final scene of each frame (K30 apdump 'v6S': boxes and POST-refiner geometry; verified that the action
from it equals the stored a_v6S|v3a), add random crops strictly outside v6-S's verified crops (field, support and guard
crops, K30 match crops_v6S) and outside the field footprint (field_v6S), at +25% and +50% of v6-S's crop px of the frame
(5 seeds each).
- Candidates: K28's random-crop heuristic (rnd): up to 40 boxes with sizes from the cheap draft's boxes at 2x context
  (200 x 120 px without boxes), uniform positions in the lower 65% of the image, seeded per (seed, unit, frame). A
  candidate that overlaps any v6-S crop or field box is rejected; the rest are taken while they fit the extra budget
  (K28 crop canvas px at v6-S's verification scale 960 / native long side).
- Detection: YOLOv8s (conf >= 0.25) on each crop at that scale, with the truncation guard. K28's merge_once puts them
  into the v6-S scene.
- New detections have no fine history (TTC infinite); their spatial geometry comes from the mono lift (AV2 ego mask as
  K10). The frozen refiner v3a runs on the new from-crop objects in the merged scene's braking support. Existing objects
  keep their stored refined geometry.
- Action and cost: deployed thresholds; K28 decision cost (jerk term on cheap's previous action).
Per changed frame (action differs from v6-S's), the change is attributed by the decisive object of the side that brakes
more:
- history: a scene object that v6-S had was replaced by a new detection without TTC;
- truncation: the decisive new detection touches its crop's border;
- new_object_gt / new_object_spurious: a new detection outside the field that matches / matches no annotated object;
- other.
usage: k35_k5.py --dataset nusc|av2|ahd --shard i/n   -> results/k35/k5/<ds>/<unit>.pkl
"""
import argparse
import math
import pickle
import socket
import time
import zlib

import cv2
import numpy as np
import torch
from rap import planner as P
from rap.detect import Mode, TwoFidelityDetector
from rap.mono import box_iou, predicted_geometry
from vtf import speculative as SP
import k24_test as K24
from instr_eval import merge_once
from nusc_refiner_crop import crop64
from k34_common import B, NWS, Path, sha

CLS = {'vehicle': 0, 'person': 1, 'cyclist': 2}
THR = 0.25
BUDGETS = (0.25, 0.5)
SEEDS = range(5)


def ckey(cr, scale, W, H):
    x1, y1 = int(cr[0]), int(cr[1])
    x2, y2 = max(int(math.ceil(cr[2])), x1 + 8), max(int(math.ceil(cr[3])), y1 + 8)
    xe, ye = min(x2, W), min(y2, H)
    return (x1, y1, x2, y2, max(32, int(math.ceil((ye - y1) * scale / 32)) * 32), max(32, int(math.ceil((xe - x1) * scale / 32)) * 32))


def overlaps(c, boxes):
    return any(min(c[2], b[2]) > max(c[0], b[0]) and min(c[3], b[3]) > max(c[1], b[1]) for b in boxes)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--max_units', type=int, default=0); ap.add_argument('--tag', default='')
    args = ap.parse_args(); ds = args.dataset; si, sn = map(int, args.shard.split('/'))
    torch.set_num_threads(2); cv2.setNumThreads(1)
    od = B / ('results/k35/k5' + ('_' + args.tag if args.tag else '')) / ds; od.mkdir(parents=True, exist_ok=True); nd = 0
    det = TwoFidelityDetector(str(B / 'models' / 'yolov8s.pt'), device='cuda:0', backend='torch')
    checkpoint = B / 'out/resnet18_v3a/image_refiner.pt'; ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    net = K24.KT.Res18x3(len(ck['fmu'])); net.load_state_dict(ck['state']); net = net.eval()
    pp, cp = P.PlannerParams(), P.CostParams()
    H, W = {'nusc': (900, 1600), 'av2': (2048, 1550), 'ahd': (1200, 1920)}[ds]
    scale = 960 / NWS[ds]; NATIVE = W * H
    for ui, (unit, cam, gen) in enumerate(K24.units(ds)):
        if ui % sn != si:
            continue
        target = od / (unit + '.pkl')
        if target.exists():
            continue
        if args.max_units and nd >= args.max_units:
            break
        nd += 1
        tic = time.perf_counter()
        dump = pickle.load(open(B / 'results/k30/apdump' / ds / (unit + '.pkl'), 'rb')); dets = dump['dets']
        ar = {int(r['frame']): r for r in dump['rows']}
        mt = {int(r['frame']): r for r in pickle.load(open(B / 'results/k30/match' / ds / (unit + '.pkl'), 'rb'))}
        R, t = cam.transform('ego'); dt = float(cam.dt); rows = []
        for fr, path, v, g, cached in gen():
            if fr not in ar:
                continue
            r0 = ar[fr]
            if not r0['gt']:
                continue
            sc = dets[(fr, 'v6S')]['sc']; xy0 = np.asarray(sc['xyxy'], float).reshape(-1, 4)
            base = {'xyxy': xy0, 'conf': np.asarray(sc['conf'], float), 'coarse': np.asarray(sc['coarse']),
                    'mono': tuple(np.asarray(x, float) for x in sc['geo'])}
            a_v6 = P.discrete_action(P.required_decel(*base['mono'], v, pp)[0], pp)
            assert a_v6 == int(r0['a_v6S|v3a']), (unit, fr, 'stored v6-S scene does not reproduce its action')
            crops_v6 = [np.asarray(c, float) for c in mt[fr]['crops_v6S']]
            field = [np.asarray(c, float) for c in (mt[fr].get('field_v6S') or [])]
            crop_px = max(float(r0['px_v6S']) - float(r0['px_cheap']), 0.0) * NATIVE
            ch = dets[(fr, 'cheap')]['sc']; cxy = np.asarray(ch['xyxy'], float).reshape(-1, 4)
            gt_xy = np.stack([g[k] for k in ('x1', 'y1', 'x2', 'y2')], 1).astype(float) if len(g) else np.zeros((0, 4))
            img = None; cache = {}
            for bud in BUDGETS:
                extra = bud * crop_px
                for seed in SEEDS:
                    rr = np.random.default_rng(zlib.crc32(f'{seed}|{unit}|{fr}'.encode()))
                    sel, used = [], 0
                    if extra >= 32 * 32:
                        for _ in range(40):
                            if len(cxy):
                                b = cxy[rr.integers(len(cxy))]
                                w_, h_ = min((b[2] - b[0]) * 2, W), min((b[3] - b[1]) * 2, H)
                            else:
                                w_, h_ = 200.0, 120.0
                            h_ = min(h_, H * 0.65)
                            x0, y0 = rr.uniform(0, max(W - w_, 1)), rr.uniform(H * 0.35, max(H - h_, H * 0.35 + 1))
                            c = np.array([x0, y0, x0 + w_, y0 + h_])
                            if overlaps(c, crops_v6) or overlaps(c, field) or overlaps(c, sel):
                                continue
                            k = ckey(c, scale, W, H)
                            if used + k[4] * k[5] <= extra:
                                sel.append(c); used += k[4] * k[5]
                    rec = {'unit': unit, 'frame': int(fr), 'budget': bud, 'seed': seed, 'n_crops': len(sel), 'extra_px': used / NATIVE,
                           'a_v6S': a_v6, 'act_gt': int(r0['act_gt'])}
                    if not sel:
                        rec.update({'a_new': a_v6, 'changed': False, 'attribution': 'no_crop'})
                        rows.append(rec); continue
                    if img is None:
                        img = cv2.imread(path); assert img is not None and img.shape[:2] == (H, W), (unit, fr, path)

                    def detect_crop(cr):
                        k = ckey(cr, scale, W, H)
                        if k not in cache:
                            x1, y1, x2, y2, nh, nw = k
                            d = det.detect(img[y1:y2, x1:x2], Mode(f'c{nh}x{nw}', nh, nw))[0]
                            kp = d['conf'] >= THR
                            cache[k] = {'xyxy': d['xyxy'][kp].astype(float) + np.array([x1, y1, x1, y1], float),
                                        'conf': d['conf'][kp].astype(float), 'coarse': d['coarse'][kp]}
                        return {kk: x.copy() for kk, x in cache[k].items()}

                    def lift(full):
                        xy = np.asarray(full['xyxy'], float).reshape(-1, 4)
                        geom = predicted_geometry({'xyxy': xy, 'coarse': full['coarse']}, None, cam.calib, cam.cam_h, dt, cam_to_ego=(R, t))
                        z, lo, hi = (np.asarray(geom[k], float).copy() for k in ('z', 'lat_min', 'lat_max'))
                        if ds == 'av2' and len(xy):
                            m = np.maximum(xy[:, 3] - np.maximum(xy[:, 1], 1830), 0) / np.maximum(xy[:, 3] - xy[:, 1], 1e-6) >= .5
                            z[m] = np.inf
                        return z, lo, hi, np.full(len(xy), np.inf)
                    scn, allc = merge_once(base, sel, SP.guarded(detect_crop, W, H), lift)
                    ff = np.asarray(scn['from_full'], bool)
                    mono = [np.asarray(x, float).copy() for x in scn['mono']]
                    sup = [p for p in SP.support(tuple(mono), v, pp) if ff[p]]
                    if sup:
                        xy = np.asarray(scn['xyxy'], float)
                        zz = predicted_geometry({'xyxy': xy[sup], 'coarse': np.asarray(scn['coarse'])[sup]}, None, cam.calib, cam.cam_h, dt, cam_to_ego=(R, t))
                        F = []
                        for n, i in enumerate(sup):
                            x1, y1, x2, y2 = xy[i]; zg, zh = float(zz['z_ground'][n]), float(zz['z_height'][n])
                            F.append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(scn['conf'][i]),
                                      float(CLS[str(scn['coarse'][i])]), float(mono[0][i]), zg, zh, float(np.log(max(zg, .5) / max(zh, .5))), 1.])
                        fe = ((np.array(F, np.float32) - ck['fmu']) / ck['fsd']).astype(np.float32)
                        x = torch.from_numpy(np.stack([crop64(img, xy[i]) for i in sup])).permute(0, 3, 1, 2).float() / 255.
                        with torch.no_grad():
                            pr = net(x, torch.from_numpy(fe)).float().numpy().reshape(len(sup), -1)
                        for j, i in enumerate(sup):
                            for q in mono[:3]:
                                q[i] *= float(np.exp(pr[j, 0]))
                    dem, each = P.required_decel(*mono, v, pp)
                    a_new = P.discrete_action(dem, pp)
                    rec.update({'a_new': int(a_new), 'changed': bool(a_new != a_v6), 'n_new_dets': int(ff.sum()),
                                'n_removed_base': int(len(xy0) - (~ff).sum())})
                    att = 'unchanged'
                    if a_new != a_v6:
                        if a_new > a_v6:                          # the new scene brakes more: its decisive object
                            j = int(np.argmax(each))
                            if not ff[j]:
                                att = 'other'
                            else:
                                b = np.asarray(scn['xyxy'], float)[j]
                                cut = any(((b[0] - c[0] < 2 and c[0] > 1) or (c[2] - b[2] < 2 and c[2] < W - 1) or
                                           (b[1] - c[1] < 2 and c[1] > 1) or (c[3] - b[3] < 2 and c[3] < H - 1)) for c in allc
                                          if c[0] <= b[0] + 1 and c[1] <= b[1] + 1 and c[2] >= b[2] - 1 and c[3] >= b[3] - 1)
                                mgt = len(gt_xy) and box_iou(b[None], gt_xy).max() >= 0.3
                                att = 'truncation' if cut else ('new_object_gt' if mgt else 'new_object_spurious')
                        else:                                     # v6-S braked more: was its decisive object replaced?
                            _, each0 = P.required_decel(*base['mono'], v, pp)
                            j0 = int(np.argmax(each0))
                            kept = SP.centres_in(xy0[j0:j0 + 1], sel)[0] if sel else False
                            att = 'history' if kept else 'other'
                    rec['attribution'] = att
                    rows.append(rec)
        payload = {'rows': rows, 'code_sha256': sha(__file__), 'elapsed_s': time.perf_counter() - tic, 'host': socket.gethostname()}
        tmp = Path(str(target) + '.tmp'); tmp.write_bytes(pickle.dumps(payload)); tmp.replace(target)
        print(ds, unit, 'frames', len(rows) // 10, 'changed', sum(r['changed'] for r in rows), 's', round(payload['elapsed_s'], 1), flush=True)
    print('K35-5 SHARD DONE', ds, args.shard, flush=True)


if __name__ == '__main__':
    main()
