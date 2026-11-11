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
def run(ds,encoder,rep,smoke=False,old=False):
    tag=('smoke' if smoke else 'r'+str(rep))+'_'+('old' if old else encoder)
    script=ROOT/'experiments/verified_jetson_20261023/scripts/83_verified_av2.py' if old else HERE/'scripts/84_masked_point.py'
    cmd=[str(ROOT/'scripts/py'),str(script),'--dataset',ds,'--helper','fast','--config','v6S','--wf','none','--paths','trt32','--bounds','2',
         '--rep',str(rep),'--tag',tag,'--max_frames','80' if smoke else '712','--warmup','0' if smoke else '200']
    if not old:cmd+=['--adapter_encoder',encoder]
    if ds=='av2_dev':cmd+=['--mask_y0','1830']
    log=tag+'_'+ds+'.log';save(status='smoke' if smoke else 'timing',dataset=ds,encoder=encoder,rep=rep)
    runcommand(cmd,log)
    folder=Path(next(x.split('run: ',1)[1].strip() for x in (LOG/log).read_text().splitlines() if 'run: ' in x));assert (folder/'DONE.json').exists()
    state['runs'].append(dict(dataset=ds,encoder=encoder,rep=rep,smoke=smoke,old=old,run=str(folder)));save();return folder
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
    sources=json.loads((HERE/'source_manifest.json').read_text())
    sources.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(HERE.glob('*.py'))+list((HERE/'scripts').glob('*.py'))+list((HERE/'models').rglob('*.pt'))})
    for p,h in sources.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    engines={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'engines').glob('*') if p.is_file() and p.suffix in ('.engine','.pt','.npz')}
    (LOG/'source_hashes_before.json').write_text(json.dumps(sources,indent=2));(LOG/'engine_hashes_before.json').write_text(json.dumps(engines,indent=2))
    try:
        while PAUSE.exists():save(status='waiting_existing_window',owner=PAUSE.read_text());time.sleep(20)
        fd=os.open(str(PAUSE),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
        with os.fdopen(fd,'w') as f:f.write(f'CVPR R3 point timing pid {os.getpid()} {state["started"]}\n')
        owned=True
        while True:
            busy=subprocess.check_output(['bash',str(BUSY)],text=True).strip()
            if busy=='paused' and not (Path.home()/'.r3_profiling').exists():break
            save(status='waiting_exclusive',busy=busy);time.sleep(20)
        hardware=api();fan=hardware['fan'];(LOG/'hardware_before.json').write_text(json.dumps(hardware,indent=2));assert hardware['powerMode']=='MAXN'
        subprocess.run(['sudo','-n','/usr/bin/jetson_clocks','--store',str(LOG/'clock_before.cfg')],check=True);clocks=True
        subprocess.run(['sudo','-n','/usr/bin/jetson_clocks'],check=True);api({'mode':'manual','pwm':255})
        with (LOG/'tegrastats.log').open('w') as f:telemetry=subprocess.Popen(['tegrastats','--interval','500'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        for ds in ('nusc_val','av2_dev'):
            old=run(ds,'trt32',1,True,True);fast=run(ds,'trt32',1,True);reference=run(ds,'t32',1,True)
            runcommand([str(ROOT/'scripts/py'),str(HERE/'check.py'),'--dataset',ds,'--old',str(old),'--new',str(fast),'--torch',str(reference)],'check_'+ds+'.log')
        for rep in (1,2,3):
            for ds in ('nusc_val','av2_dev'):run(ds,'trt32',rep)
        for p,h in sources.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
        for p,h in engines.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
        save(status='complete',source_engine_hashes_unchanged=True)
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
