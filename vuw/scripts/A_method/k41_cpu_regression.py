"""Read-only copy of existing release CPU tests. CUDA regression explicitly skipped."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

B=Path(os.environ['B']);SOURCE=B/'tmp/desp_k40';DEST=B/'results/latency_opt/release_cpu_tests'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA must be hidden before importing tests'
    DEST.mkdir(parents=True,exist_ok=True)
    shutil.copytree(SOURCE/'desp',DEST/'desp',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (DEST/'tests/regression').mkdir(parents=True,exist_ok=True)
    paths=['tests/test_core.py','tests/test_regression.py','tests/regression/expected.pkl']
    for p in paths:shutil.copy2(SOURCE/p,DEST/p)
    manifest={str(p.relative_to(DEST)):hashlib.md5(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file() and p.suffix in ('.py','.pkl')}
    (DEST/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(DEST)+os.pathsep+os.environ.get('PYTHONPATH',''))
    cmd=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_core.py','tests/test_regression.py','--junitxml',str(DEST/'junit.xml')]
    r=subprocess.run(cmd,cwd=DEST,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (DEST/'stdout.txt').write_text(r.stdout);print(r.stdout)
    (DEST/'run.json').write_text(json.dumps(dict(command=cmd,returncode=r.returncode,CUDA_VISIBLE_DEVICES='',source=str(SOURCE),destination=str(DEST)),indent=2)+'\n')
    raise SystemExit(r.returncode)


if __name__=='__main__':main()
