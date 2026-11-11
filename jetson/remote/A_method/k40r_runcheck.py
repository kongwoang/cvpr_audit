"""Per-unit scientific parity/checkpoint safeguards for isolated detector jobs."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import time
import numpy as np


def room(root):
    assert shutil.disk_usage(root).free >= 200*1024**3, 'B has less than200GiB free'


def check_and_save(rows, output, root, dataset, script):
    room(root)
    reference_path = root/'results/k30/match'/dataset/output.name
    reference = {(str(r['unit']),int(r['frame'])):r for r in pickle.load(reference_path.open('rb'))}
    cols=['a_cheap','J_cheap','a_v6S|none','J_v6S|none','a_v6S|v3a','J_v6S|v3a','px_v6S','ncrops_v6S','nguard_v6S','rounds_v6S']
    failures=[];comparisons=0
    for r in rows:
        ref=reference[(str(r['unit']),int(r['frame']))]
        for col in cols:
            assert col in r and col in ref,col
            comparisons+=1
            if not np.isclose(r[col],ref[col],rtol=0,atol=1e-9,equal_nan=True):
                failures.append(dict(unit=r['unit'],frame=int(r['frame']),endpoint=col,actual=float(r[col]),reference=float(ref[col])))
        for col in [c for c in r if c.startswith('px_') and ('@k39_strict' in c or '@strict_observed' in c)]:
            assert r[col] <= r['px_v6S']+1e-9,(col,r[col],r['px_v6S'])
    report=dict(rows=len(rows),evaluated_frames=sum(bool(r['gt']) for r in rows),comparisons=comparisons,
                failures=failures,reference=str(reference_path),reference_md5=hashlib.md5(reference_path.read_bytes()).hexdigest(),
                script_md5=hashlib.md5(Path(script).read_bytes()).hexdigest(),completed_unix=time.time())
    if failures:
        bad=output.with_suffix('.failed.pkl')
        with bad.open('wb') as f:pickle.dump(rows,f)
        output.with_suffix('.parity_failed.json').write_text(json.dumps(report,indent=2)+'\n')
        raise AssertionError(f'{len(failures)} frozen parity mismatches; retained {bad}')
    tmp=output.with_suffix('.partial.pkl')
    with tmp.open('wb') as f:
        pickle.dump(rows,f);f.flush();os.fsync(f.fileno())
    tmp.replace(output)
    output.with_suffix('.done.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
