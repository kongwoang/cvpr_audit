"""Run one CPU job with a persistent status and a duplicate-launch lock."""
import fcntl
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def main():
    name=sys.argv[1];cmd=sys.argv[2:];root=Path(os.environ['B'])/'results/k34/jobs';root.mkdir(parents=True,exist_ok=True)
    lock=(root/(name+'.lock')).open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    path=root/(name+'.json')
    if path.exists() and json.loads(path.read_text()).get('status')=='complete':return
    state={'name':name,'host':socket.gethostname(),'pid':os.getpid(),'command':cmd,
           'started_at':datetime.now().astimezone().isoformat(),'status':'running'}
    def write():
        t=path.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(path)
    write()
    rc=subprocess.call(cmd)
    state.update(status='complete' if rc==0 else 'failed',returncode=rc,finished_at=datetime.now().astimezone().isoformat());write()
    sys.exit(rc)

if __name__=='__main__':main()
