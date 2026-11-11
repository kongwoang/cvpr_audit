"""Exclusive Jetson factorial timing after runtime and VUW smoke gates."""
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LOG=HERE/'logs';LOG.mkdir(exist_ok=True)
PAUSE=Path.home()/'.jetson_pause'
BUSY=ROOT.parent/'risk-aware-perception/scripts/busy.sh'
state=dict(status='preflight',pid=os.getpid(),runs=[],started=dt.datetime.now().astimezone().isoformat())
child=telemetry=None
owned=clocks=False
fan=None


def save(**kw):
    state.update(kw,updated=dt.datetime.now().astimezone().isoformat())
    tmp=LOG/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(LOG/'state.json')


def api(payload=None):
    if payload is None:
        return json.load(urllib.request.urlopen('http://localhost:8765/api/jetson/status',timeout=10))
    req=urllib.request.Request('http://localhost:8765/api/jetson/fan',data=json.dumps(payload).encode(),
                               headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req,timeout=10))


def run_command(command,log):
    global child
    with (LOG/log).open('w') as output:
        child=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
        save(child_pid=child.pid,current_log=log)
        code=child.wait()
    child=None
    if code:
        raise RuntimeError((log,code))


def run(ds,runtime,rep,smoke=False):
    tag=('smoke' if smoke else 'r'+str(rep))+'_'+runtime
    logfile=tag+'_'+ds+'.log'
    command=[str(ROOT/'scripts/py'),str(HERE/'scripts/81_temporal_latency.py'),
             '--dataset',ds,'--config','v6S','--wf','none','--paths','trt32',
             '--bounds','2','--rep',str(rep),'--tag',tag,'--max_frames',('80' if runtime=='disabled' else '160') if smoke else '712',
             '--warmup','0' if smoke else '200']
    if runtime=='disabled':command+=['--recovery_disabled']
    if ds=='av2_dev':
        command+=['--mask_y0','1830']
    save(status='smoke' if smoke else 'timing',dataset=ds,runtime=runtime,rep=rep)
    run_command(command,logfile)
    lines=(LOG/logfile).read_text().splitlines()
    folder=Path(next(x.split('run: ',1)[1].strip() for x in lines if 'run: ' in x))
    assert (folder/'DONE.json').exists()
    state['runs'].append(dict(dataset=ds,runtime=runtime,rep=rep,smoke=smoke,run=str(folder)))
    save()
    return folder


def stop(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def signal_stop(sig,frame):
    raise InterruptedError(sig)


def main():
    global owned,clocks,fan,telemetry
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,signal_stop)
    lock=(LOG/'window.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not all((ROOT/p).exists() for p in ('docs/paper/handoff/RUNTIME_20261023/DONE.json',
            'docs/paper/handoff/NAVSIM_REUSE_20261023/DONE.json','docs/paper/handoff/W1_BOUNDED_20261023/smoke/REPORT.md')):
        save(status='waiting_runtime_and_smoke_gates');time.sleep(20)
    sources=json.loads((HERE/'source_manifest.json').read_text())
    for p in [HERE/'helpers.py',HERE/'recovery_gpu.py',HERE/'scripts/81_temporal_latency.py',HERE/'protocol.md']:
        sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (LOG/'source_hashes_before.json').write_text(json.dumps(sources,indent=2)+'\n')
    for p,h in sources.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    engines={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((ROOT/'engines').glob('*')) if p.is_file() and p.suffix in ('.engine','.pt','.npz')}
    (LOG/'engine_hashes_before.json').write_text(json.dumps(engines,indent=2)+'\n')
    try:
        while PAUSE.exists():
            save(status='waiting_existing_window',owner=PAUSE.read_text());time.sleep(20)
        fd=os.open(str(PAUSE),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
        with os.fdopen(fd,'w') as f:f.write(f'CVPR runtime experiment pid {os.getpid()} {state["started"]}\n')
        owned=True
        while True:
            busy=subprocess.check_output(['bash',str(BUSY)],text=True).strip()
            if busy=='paused' and not (Path.home()/'.r3_profiling').exists():break
            save(status='waiting_exclusive',busy=busy);time.sleep(20)
        hardware=api();fan=hardware['fan']
        (LOG/'hardware_before.json').write_text(json.dumps(hardware,indent=2)+'\n')
        assert hardware['powerMode']=='MAXN'
        subprocess.run(['sudo','-n','/usr/bin/jetson_clocks','--store',str(LOG/'clock_before.cfg')],check=True)
        clocks=True
        subprocess.run(['sudo','-n','/usr/bin/jetson_clocks'],check=True)
        api({'mode':'manual','pwm':255})
        with (LOG/'tegrastats.log').open('w') as f:
            telemetry=subprocess.Popen(['tegrastats','--interval','500'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        previous=json.loads((ROOT/'experiments/runtime_20261023/logs/state.json').read_text())
        for ds in ('nusc_val','av2_dev'):
            base=next(Path(r['run']) for r in previous['runs'] if r['dataset']==ds and r['runtime']=='reference' and r['smoke'])
            port=run(ds,'disabled',1,True)
            run_command([str(ROOT/'scripts/py'),str(HERE/'check.py'),'--run',str(port),
                '--disabled','--reference',str(base),'--out',str(LOG/(ds+'_disabled_parity.json'))],ds+'_disabled_check.log')
            enabled=run(ds,'enabled',1,True)
            run_command([str(ROOT/'scripts/py'),str(HERE/'check.py'),'--run',str(enabled),
                '--out',str(LOG/(ds+'_enabled_budget.json'))],ds+'_enabled_check.log')
        for rep in (1,2,3):
            for ds in ('nusc_val','av2_dev'):
                folder=run(ds,'enabled',rep)
                run_command([str(ROOT/'scripts/py'),str(HERE/'check.py'),'--run',str(folder),
                    '--out',str(LOG/f'{ds}_r{rep}_budget.json')],f'{ds}_r{rep}_check.log')
        for p,h in sources.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
        for p,h in engines.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
        save(status='timing_complete',source_engine_hashes_unchanged=True)
    except BaseException as exc:
        save(status='failed',error=repr(exc));raise
    finally:
        stop(child);stop(telemetry)
        errors=[]
        if clocks:
            try:subprocess.run(['sudo','-n','/usr/bin/jetson_clocks','--restore',str(LOG/'clock_before.cfg')],check=True)
            except Exception as exc:errors.append(str(exc))
        if fan is not None:
            try:api({'mode':'manual','pwm':fan['pwm']} if fan['mode']=='manual' else {'mode':'auto','profile':fan.get('profile') or 'quiet'})
            except Exception as exc:errors.append(str(exc))
        if owned and PAUSE.exists() and f'pid {os.getpid()} ' in PAUSE.read_text():PAUSE.unlink()
        save(window_released=True,restore_errors=errors)


if __name__=='__main__':
    main()
