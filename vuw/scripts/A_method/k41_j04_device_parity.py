"""J04 device action parity (reported, not a gate): Jetson K41 rows (D = inf, refiner v3a) vs the VUW K33 test records
(results/k33/{nusc,av2}_{de,pub}_test/<unit>.pkl, results/k33/rc/test_<ds>.pkl) on the shared frames."""
import glob, pickle, sys
import pandas as pd
D = 'results/k41/j04_device_actions'
ROWS = ['ana05hinfo', 'ana05hinfi', 'cf40q20o', 'cf40q20i', 'ada+1t1280', 'fovs1280b64a1', 'fovi1280b64a2', 'dvr1280_0.5', 'bar1280_0.5', 'ch128B_csc', 'v6S']
out = []
for ds, vds in (('nusc_test20', 'nusc'), ('av2t40', 'av2')):
    dev = {t: pd.read_csv(f'{D}/dev_{ds}_{t}.csv', dtype={'unit': str}) for t in ('r1', 'r2', 'r3')}
    units = set(dev['r1'].unit)
    recs = []
    for part in ('de', 'pub'):
        for u in units:
            for p in glob.glob(f'results/k33/{vds}_{part}_test/{u}.pkl'):
                recs += [{k: v for k, v in r.items() if k in ('unit', 'frame') or (k.startswith('a_') and k.endswith('|v3a'))} for r in pickle.load(open(p, 'rb'))]
    R = pd.DataFrame(recs).groupby(['unit', 'frame'], as_index=False).first() if recs else pd.DataFrame()
    rc = pd.read_pickle(f'results/k33/rc/test_{vds}.pkl'); rc = rc[rc.unit.astype(str).isin(units)]
    print(ds, 'rc columns sample', [c for c in rc.columns if c.startswith('a_')][:12], file=sys.stderr)
    for row in ROWS:
        dc = 'a_ours@trt32' if row == 'v6S' else f'a_{row}@trt32'
        src = R if f'a_{row}|v3a' in R else rc
        vc = next((c for c in (f'a_{row}|v3a', f'a_{row}') if c in src), None)
        if vc is None or dc not in dev['r1']:
            out.append(dict(dataset=ds, row=row, status='no VUW record column' if vc is None else 'no device column')); continue
        v = src[['unit', 'frame', vc]].assign(unit=lambda x: x.unit.astype(str))
        for t, d in dev.items():
            m = d[['unit', 'frame', dc]].merge(v, on=['unit', 'frame'])
            out.append(dict(dataset=ds, row=row, rep=t, vuw_column=vc, shared_frames=len(m), agree=int((m[dc] == m[vc]).sum()),
                            agreement=float((m[dc] == m[vc]).mean()) if len(m) else float('nan'), status='ok'))
o = pd.DataFrame(out); o.to_csv(sys.argv[1], index=False); print(o.to_string(index=False))
