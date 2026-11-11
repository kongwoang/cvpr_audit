"""Exclusive Jetson window; always restore owned hardware controls and pause."""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
LOG=HERE/'logs';LOG.mkdir(exist_ok=True)
PAUSE=Path.home()/'.jetson_pause';BUSY=ROOT.parent/'risk-aware-perception/scripts/busy.sh'
state=dict(status='preflight',pid=os.getpid(),runs=[],started=dt.datetime.now().astimezone().isoformat())
child=telemetry=None;owned=clocks=False;fan=None
def save(**kw):
    state.update(kw,updated=dt.datetime.now().astimezone().isoformat());tmp=LOG/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(LOG/'state.json')
def api(payload=None):
    if payload is None:return json.load(urllib.request.urlopen('http://localhost:8765/api/jetson/status',timeout=10))
    req=urllib.request.Request('http://localhost:8765/api/jetson/fan',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req,timeout=10))
def runcommand(command,log):
    global child
    with (LOG/log).open('w') as output:
        child=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
        save(child_pid=child.pid,current_log=log);code=child.wait()
    child=None
    if code:raise RuntimeError((log,code))
def stop(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
def main():
    global owned,clocks,fan,telemetry
    def handle(sig,frame):raise InterruptedError(sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,handle)
    lock=(LOG/'window.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources=json.loads((HERE/'local_sources_before.json').read_text())
    sources.update({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'bounded_readout.py',HERE/'scripts/89_r6_readout.py']})
    for p,h in sources.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    (LOG/'source_hashes_before.json').write_text(json.dumps(sources,indent=2))
    try:
        while PAUSE.exists():save(status='waiting_existing_window',owner=PAUSE.read_text());time.sleep(20)
        fd=os.open(str(PAUSE),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
        with os.fdopen(fd,'w') as f:f.write(f'CVPR R6 readout timing pid {os.getpid()} {state["started"]}\n')
        owned=True
        while True:
            busy=subprocess.check_output(['bash',str(BUSY)],text=True).strip()
            if busy=='paused' and not (Path.home()/'.r3_profiling').exists():break
            save(status='waiting_exclusive',busy=busy);time.sleep(20)
        hardware=api();fan=hardware['fan'];(LOG/'hardware_before.json').write_text(json.dumps(hardware,indent=2));assert hardware['powerMode']=='MAXN'
        subprocess.run(['sudo','-n','/usr/bin/jetson_clocks','--store',str(LOG/'clock_before.cfg')],check=True);clocks=True
        subprocess.run(['sudo','-n','/usr/bin/jetson_clocks'],check=True);api({'mode':'manual','pwm':255})
        with (LOG/'tegrastats.log').open('w') as f:telemetry=subprocess.Popen(['tegrastats','--interval','500'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        save(status='timing')
        runcommand([str(ROOT/'scripts/py'),str(HERE/'scripts/89_r6_readout.py')],'readout.log')
        for p,h in sources.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
        save(status='complete',source_hashes_unchanged=True,engines_used=False)
    except BaseException as e:save(status='failed',error=repr(e));raise
    finally:
        stop(child);stop(telemetry);errors=[]
        if clocks:
            try:subprocess.run(['sudo','-n','/usr/bin/jetson_clocks','--restore',str(LOG/'clock_before.cfg')],check=True)
            except Exception as e:errors.append(str(e))
        if fan is not None:
            try:api({'mode':'manual','pwm':fan['pwm']} if fan['mode']=='manual' else {'mode':'auto','profile':fan.get('profile') or 'quiet'})
            except Exception as e:errors.append(str(e))
        if owned and PAUSE.exists() and f'pid {os.getpid()} ' in PAUSE.read_text():PAUSE.unlink()
        save(window_released=True,restore_errors=errors)
if __name__=='__main__':main()
