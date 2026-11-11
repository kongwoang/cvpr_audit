"""Reproduce Appendix C historical reference from allowed release timing records."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

B = Path(os.environ['B'])
sys.path.insert(0, str(B / 'results/k40r/vendor'))
from analysis.jetson import per_rep, rep_of


def main():
    source = B / 'tmp/desp_k40/artifacts/records/jetson_latency.parquet'
    out = B / 'results/latency_opt/tab6_cached'
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(source)
    summaries, rows = [], []
    for ds, count, units in [('nusc_test20', 805, 20), ('av2', 586, 8)]:
        for row in ['cheap', 'ours@trt32', 'ours_b2@trt32', 'u512+c1comp@trt32',
                    'u640+c1comp@trt32', 'u960+c1comp@trt32', 'u1280+c1comp@trt32', 'native+c1comp@trt32']:
            reference = per_rep(d, ds, row, 'v6S')
            q = d[(d.dataset == ds) & (d.pipeline == row) & (d.config == 'v6S') & ~d.warm]
            q = q[q.tag.map(rep_of).notna() & ~q.tag.str.contains('_g')]
            selected = []
            seen = set()
            for run, g in q.groupby('run', sort=True):
                rep = rep_of(g.tag.iloc[0])
                if rep in seen:
                    continue
                seen.add(rep)
                assert len(g) == count and g.unit.nunique() == units
                r = dict(dataset=ds, row=row, rep=rep, run=run, frames=len(g), units=units,
                         mean_ms=g.ms.mean(), p50_ms=g.ms.quantile(.5), p90_ms=g.ms.quantile(.9),
                         p95_ms=g.ms.quantile(.95), p99_ms=g.ms.quantile(.99), max_ms=g.ms.max(),
                         G=reference[rep]['G'], px=reference[rep]['px'])
                assert abs(r['p95_ms'] - reference[rep]['p95']) < 1e-10
                rows.append(r); selected.append(r)
            assert len(selected) == 3
            summaries.append(dict(dataset=ds, row=row, frames=count, units=units, repetitions=3,
                                  aggregation='median of per-repetition statistics; original Tab6 convention',
                                  **{k: float(np.median([r[k] for r in selected])) for k in
                                     ['mean_ms', 'p50_ms', 'p90_ms', 'p95_ms', 'p99_ms', 'max_ms', 'G', 'px']}))
    pd.DataFrame(rows).to_csv(out / 'repetitions.csv', index=False)
    pd.DataFrame(summaries).to_csv(out / 'summary.csv', index=False)
    (out / 'run.json').write_text(json.dumps(dict(
        source=str(source), source_md5=hashlib.md5(source.read_bytes()).hexdigest(),
        script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
        new_device_execution=False, protocol='K29 fast TensorRT; 20 nuScenes scenes / 8 AV2 logs'), indent=2) + '\n')
    print(pd.DataFrame(summaries).query("row in ['ours@trt32','u960+c1comp@trt32']").to_string(index=False))


if __name__ == '__main__':
    main()
