"""Bounded child launcher. GPU invocations must come from the host-local wrapper."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from k40r_runcheck import room


def save(path, data):
    tmp=path.with_suffix('.writing.json')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--status',required=True)
    ap.add_argument('--seconds',type=float,required=True)
    ap.add_argument('command',nargs=argparse.REMAINDER)
    args=ap.parse_args();cmd=args.command
    if cmd and cmd[0]=='--':cmd=cmd[1:]
    root=Path(os.environ['B']).resolve();out=(root/args.status).resolve()
    assert root in out.parents and cmd and args.seconds>0
    assert os.environ.get('CUDA_VISIBLE_DEVICES'), 'Only launch through run_when_gpu_free.sh'
    assert os.environ.get('SHARE','0')=='0'
    assert not out.exists(), 'Execution status already exists; retain it and use a new attempt ID'
    room(root);out.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    info=dict(status='running',command=cmd,host=socket.gethostname(),cuda=os.environ['CUDA_VISIBLE_DEVICES'],
              started_unix=time.time(),elapsed_seconds=0,limit_seconds=args.seconds)
    proc=subprocess.Popen(cmd,cwd=root,start_new_session=True)
    info['pid']=proc.pid;save(out,info)
    def stop():
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    def interrupted(signum,frame):
        raise InterruptedError('received signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        while proc.poll() is None:
            elapsed=time.monotonic()-start
            if elapsed>args.seconds:raise TimeoutError('registered GPU occupancy ceiling reached')
            room(root);info['elapsed_seconds']=elapsed;save(out,info)
            try:proc.wait(timeout=min(30,max(.1,args.seconds-elapsed)))
            except subprocess.TimeoutExpired:pass
        info.update(status='complete' if proc.returncode==0 else 'failed',returncode=proc.returncode)
    except BaseException as e:
        stop();info.update(status='failed',error=repr(e),returncode=proc.returncode)
        raise
    finally:
        info.update(elapsed_seconds=time.monotonic()-start,finished_unix=time.time());save(out,info)
    if proc.returncode:raise SystemExit(proc.returncode)


if __name__=='__main__':main()
