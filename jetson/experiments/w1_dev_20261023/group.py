"""Group two independent jobs per wrapper launch; wait for and report every exit."""
import argparse
import subprocess
import sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','dev']);ap.add_argument('--dataset',choices=['nusc','av2'])
a=ap.parse_args();here=Path(__file__).resolve().parent
jobs=[('nusc','0/1'),('av2','0/1')] if a.stage=='smoke' else [(a.dataset,'0/2'),(a.dataset,'1/2')]
ps=[subprocess.Popen([sys.executable,str(here/'job.py'),'--stage',a.stage,'--dataset',ds,'--shard',sh]) for ds,sh in jobs]
codes=[p.wait() for p in ps]
print('GROUP EXIT',jobs,codes,flush=True)
sys.exit(int(any(codes)))
