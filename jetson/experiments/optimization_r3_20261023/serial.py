"""Run preregistered follow-up suites sequentially within one scheduled GPU slot."""
import argparse
import subprocess
import sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--shard',required=True);a=ap.parse_args()
here=Path(__file__).resolve().parent
stages=['calibrated','lift','persist']+(['iddev'] if a.dataset=='nusc' else [])
for stage in stages:
    r=subprocess.run([sys.executable,str(here/'job.py'),stage,a.dataset,a.shard])
    if r.returncode:sys.exit(r.returncode)
print('SERIAL FOLLOWUPS COMPLETE',a.dataset,a.shard,stages,flush=True)
