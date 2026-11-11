"""Exactly two modest GPU workers per unchanged scheduling wrapper."""
import json
import subprocess
import sys
from pathlib import Path
here=Path(__file__).resolve().parent
jobs=json.loads(sys.argv[1]);assert 1<=len(jobs)<=2
ps=[subprocess.Popen([sys.executable,str(here/'job.py')]+args) for args in jobs]
codes=[p.wait() for p in ps]
print('GROUP EXIT',codes,flush=True);sys.exit(int(any(codes)))
