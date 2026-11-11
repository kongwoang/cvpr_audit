"""K35-3 replay (post-hoc, CPU): u960 + C1 + v3a on frames WITHOUT a fine history, for the temporal-escalation
baselines. Under "C1 history as in deployment" an unescalated frame produces no fine observation, so on an escalated
frame whose two previous frames were not both escalated, C1 has no 3-observation chain and its TTC is infinite.
The rest is K34's replay (k34_replay.py) unchanged: the stored whole-frame u960 boxes (K30 apdump), the pre-refiner
spatial geometry from boxes / calibration (AV2 ego mask as K10), the frozen refiner v3a on the deployed braking support,
the deployed thresholds. No detector or C1 inference.
Parity: on frames where the stored continuous-history C1 TTC is already infinite for every box, the replayed action
must equal the stored a_wf960+C1|v3a (recorded per frame).
usage: k35_k3_replay.py --dataset nusc|av2|ahd --shard i/n   -> results/k35/k3_replay/<ds>/<unit>.pkl
"""
import argparse
import pickle
import socket
import time

import cv2
import numpy as np
import torch
from rap import planner as P
from rap.mono import predicted_geometry
from vtf import speculative as SP
import k24_test as K24
from nusc_refiner_crop import crop64
from k34_common import B, Path, sha

CLS = {'vehicle': 0, 'person': 1, 'cyclist': 2}
KEY = 'wf960+C1'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True); ap.add_argument('--shard', default='0/1')
    args = ap.parse_args(); ds = args.dataset; si, sn = map(int, args.shard.split('/'))
    torch.set_num_threads(1); cv2.setNumThreads(1)
    od = B / 'results/k35/k3_replay' / ds; od.mkdir(parents=True, exist_ok=True)
    checkpoint = B / 'out/resnet18_v3a/image_refiner.pt'; ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    net = K24.KT.Res18x3(len(ck['fmu'])); net.load_state_dict(ck['state']); net = net.eval()
    pp = P.PlannerParams()
    H, W = {'nusc': (900, 1600), 'av2': (2048, 1550), 'ahd': (1200, 1920)}[ds]
    for ui, (unit, cam, gen) in enumerate(K24.units(ds)):
        if ui % sn != si:
            continue
        target = od / (unit + '.pkl')
        if target.exists():
            continue
        tic = time.perf_counter(); src = B / 'results/k30/apdump' / ds / (unit + '.pkl')
        dump = pickle.load(src.open('rb')); ar = {int(r['frame']): r for r in dump['rows']}; dets = dump['dets']
        R, t = cam.transform('ego'); dt = float(cam.dt); rows = []
        expected = {int(r['frame']) for r in dump['rows'] if r['gt']}
        for fr, path, v, g, cached in gen():
            if fr not in expected:
                continue
            sc = dets[(fr, KEY)]['sc']; xy = np.asarray(sc['xyxy'], float).reshape(-1, 4)
            geom = predicted_geometry({'xyxy': xy, 'coarse': sc['coarse']}, None, cam.calib, cam.cam_h, dt, cam_to_ego=(R, t))
            if ds == 'av2' and len(xy):
                mask = np.maximum(xy[:, 3] - np.maximum(xy[:, 1], 1830), 0) / np.maximum(xy[:, 3] - xy[:, 1], 1e-6) >= .5
                for k in ('z', 'z_ground', 'z_height', 'ttc'):
                    geom[k] = np.asarray(geom[k], float).copy(); geom[k][mask] = np.inf
            ttc_c1 = np.asarray(sc['geo'][3], float)
            mono = tuple(np.asarray(geom[k], float) for k in ('z', 'lat_min', 'lat_max')) + (np.full(len(xy), np.inf),)
            a_pre = P.discrete_action(P.required_decel(*mono, v, pp)[0], pp)
            sup = list(SP.support(mono, v, pp))
            tt = [x.copy() for x in mono]
            if sup:
                img = cv2.imread(path); assert img is not None and img.shape[:2] == (H, W), (unit, fr, path)
                F = []
                for i in sup:
                    x1, y1, x2, y2 = xy[i]; zg = float(geom['z_ground'][i]); zh = float(geom['z_height'][i])
                    F.append([x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H, float(sc['conf'][i]),
                              float(CLS[str(sc['coarse'][i])]), float(mono[0][i]), zg, zh, float(np.log(max(zg, .5) / max(zh, .5))), 1.])
                feats = ((np.array(F, np.float32) - ck['fmu']) / ck['fsd']).astype(np.float32)
                crops = torch.from_numpy(np.stack([crop64(img, xy[i]) for i in sup])).permute(0, 3, 1, 2).float() / 255.
                with torch.no_grad():
                    pred = net(crops, torch.from_numpy(feats)).float().numpy().reshape(len(sup), -1)
                for j, i in enumerate(sup):
                    for x in tt[:3]:
                        x[i] *= float(np.exp(pred[j, 0]))
            a = P.discrete_action(P.required_decel(*tt, v, pp)[0], pp)
            no_c1 = bool(len(ttc_c1) == 0 or not np.isfinite(ttc_c1).any())
            rows.append({'unit': unit, 'frame': int(fr), 'a_u960_nohist|none': int(a_pre), 'a_u960_nohist|v3a': int(a),
                         'stored_c1_all_inf': no_c1, 'stored_a': int(ar[fr]['a_' + KEY + '|v3a']),
                         'parity': (int(a) == int(ar[fr]['a_' + KEY + '|v3a'])) if no_c1 else None, 'n_support': len(sup)})
        assert {r['frame'] for r in rows} == expected, (unit, 'missing frames')
        payload = {'rows': rows, 'source': str(src), 'source_sha256': sha(src), 'checkpoint_sha256': sha(checkpoint),
                   'code_sha256': sha(__file__), 'elapsed_s': time.perf_counter() - tic, 'host': socket.gethostname()}
        tmp = Path(str(target) + '.tmp'); tmp.write_bytes(pickle.dumps(payload)); tmp.replace(target)
        par = [r['parity'] for r in rows if r['parity'] is not None]
        print(ds, unit, 'frames', len(rows), 's', round(payload['elapsed_s'], 1), 'parity', f'{sum(par)}/{len(par)}', flush=True)
    print('K35-3 REPLAY SHARD DONE', ds, args.shard, flush=True)


if __name__ == '__main__':
    main()
