"""Two workers per GPU wrapper; remaining AV2 logs partitioned into four shards."""
import argparse
import subprocess
import sys
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--pair',type=int,choices=[0,1],required=True)
a=ap.parse_args();here=Path(__file__).resolve().parent
jobs=[f'{2*a.pair+i}/4' for i in (0,1)]
ps=[subprocess.Popen([sys.executable,str(here/'job.py'),'--stage','dev','--dataset','av2','--shard',s]) for s in jobs]
codes=[p.wait() for p in ps]
print('GROUP4 EXIT',jobs,codes,flush=True)
sys.exit(int(any(codes)))
