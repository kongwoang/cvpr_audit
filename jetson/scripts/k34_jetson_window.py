"""Exclusive K34-3 Jetson window; release owned controls on success or failure."""
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

R=Path(__file__).resolve().parents[1]
L=R/'logs/k34j';L.mkdir(parents=True,exist_ok=True)
PAUSE=Path.home()/'.jetson_pause'
BUSY=R.parent/'risk-aware-perception/scripts/busy.sh'
state={'started_local':dt.datetime.now().astimezone().isoformat(),'pid':os.getpid(),'status':'preflight','runs':[]}
child=None
owned=False
clock_saved=False
fan_before=None
telemetry=None

def stamp():return dt.datetime.now().astimezone().isoformat()
def save():
    state['updated_local']=stamp();p=L/'state.json';q=L/'state.tmp';q.write_text(json.dumps(state,indent=2)+'\n');q.replace(p)
def log(msg):print(stamp(),msg,flush=True)
def api(payload=None):
    if payload is None:return json.load(urllib.request.urlopen('http://localhost:8765/api/jetson/status',timeout=10))
    req=urllib.request.Request('http://localhost:8765/api/jetson/fan',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req,timeout=10))
def command(args):subprocess.run(args,cwd=str(R),check=True)
def stop_group(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
def interrupted(sig,frame):raise InterruptedError('signal '+str(sig))
for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)

def run(ds,rep,smoke=False):
    global child
    tag=('smoke_'+ds if smoke else 'r'+str(rep))
    args=[str(R/'scripts/py'),str(R/'scripts/77_k34_latency.py'),'--dataset',ds,'--config','v6S','--wf','none','--bounds','2','--paths','ref,trt32' if smoke else 'trt32','--rep',str(rep),'--tag',tag]
    if ds=='av2':args+=['--mask_y0','1830']
    if smoke:args+=['--parity','--check_bounded','--max_frames','80']
    lf=L/(tag+'_'+ds+'.log')
    state.update(status='smoke' if smoke else 'timing',dataset=ds,rep=rep,current_log=str(lf));save();log('start '+tag+' '+ds)
    with lf.open('w') as out:
        child=subprocess.Popen(args,cwd=str(R),stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        state['child_pid']=child.pid;save()
        rc=child.wait()
    child=None
    if rc:raise RuntimeError('run failed rc='+str(rc)+'; '+str(lf))
    lines=lf.read_text().splitlines();run_path=Path(next(x.split('run: ',1)[1].strip() for x in lines if 'run: ' in x))
    assert (run_path/'DONE.json').exists(),'missing run completion marker'
    if smoke:
        import csv
        rows=list(csv.DictReader((run_path/'parity.csv').open()))
        wanted={r['pipeline']:r for r in rows if r['pipeline'] in ['ours','ours_b2','ours_b2A']}
        assert len(wanted)==3 and all(int(r['differing_frames'])==0 for r in wanted.values()),'GPU smoke parity differs; inspect before timing'
    state['runs'].append({'dataset':ds,'rep':rep,'smoke':smoke,'run':str(run_path),'log':str(lf)});save();log('done '+tag+' '+ds)

lock=(L/'window.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
    # Never adopt a pre-existing pause owned by another task.
    fd=os.open(str(PAUSE),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
    with os.fdopen(fd,'w') as f:f.write('CVPR K34-3 pid '+str(os.getpid())+' '+stamp()+'\n')
    owned=True;state['pause_owned']=True;save()
    while True:
        busy=subprocess.check_output(['bash',str(BUSY)],text=True).strip()
        if busy=='paused' and not (Path.home()/'.r3_profiling').exists():break
        state['status']='waiting_exclusive';state['busy']=busy;save();log('waiting: '+busy);time.sleep(20)
    snapshot=api();fan_before=snapshot['fan'];(L/'hardware_before.json').write_text(json.dumps(snapshot,indent=2))
    assert snapshot['powerMode']=='MAXN',('expected existing MAXN',snapshot['powerMode'])
    command(['sudo','-n','/usr/bin/jetson_clocks','--store',str(L/'clock_before.cfg')]);clock_saved=True
    command(['sudo','-n','/usr/bin/jetson_clocks'])
    api({'mode':'manual','pwm':255})
    source_files=['scripts/77_k34_latency.py','scripts/k34_bounded.py','scripts/64_k29_latency.py','scripts/59_bounded_latency.py','remote/A_method/k31_eval.py','src/vtf/speculative.py','src/vtf/c1fast.py','src/vtf/gpuprep.py','src/vtf/gpupost.py']
    sources={f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in source_files}
    (L/'source_hashes.json').write_text(json.dumps(sources,indent=2))
    with (L/'tegrastats.log').open('w') as out:
        telemetry=subprocess.Popen(['tegrastats','--interval','500'],stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    for ds in ['nusc_test20','av2']:run(ds,1,True)
    for rep in (1,2,3):
        for ds in ['nusc_test20','av2']:run(ds,rep)
    for f,h in sources.items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==h,('source changed during window',f)
    state['status']='timing_complete';state['timing_finished_local']=stamp();save()
except BaseException as exc:
    state.update(status='failed',error=repr(exc));save();log('FAILED '+repr(exc));raise
finally:
    stop_group(child);stop_group(telemetry)
    restore_errors=[]
    if clock_saved:
        try:command(['sudo','-n','/usr/bin/jetson_clocks','--restore',str(L/'clock_before.cfg')])
        except Exception as ex:restore_errors.append('clocks: '+str(ex))
    if fan_before is not None:
        try:
            payload={'mode':'manual','pwm':fan_before['pwm']} if fan_before['mode']=='manual' else {'mode':'auto','profile':fan_before.get('profile') or 'quiet'}
            api(payload)
        except Exception as ex:restore_errors.append('fan: '+str(ex))
    if owned and PAUSE.exists() and ('pid '+str(os.getpid())+' ') in PAUSE.read_text():PAUSE.unlink()
    state['pause_owned']=False;state['restore_errors']=restore_errors;state['window_released_local']=stamp();save();log('window released')

# CPU reporting follows after releasing the GPU window.
state['status']='reporting';save()
try:
    command([str(R/'scripts/py'),str(R/'scripts/78_k34_report.py')])
    state['status']='complete';save();log('K34-3 COMPLETE')
except Exception as ex:
    state.update(status='report_failed',error=repr(ex));save();raise
