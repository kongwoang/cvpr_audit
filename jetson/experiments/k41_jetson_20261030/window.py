"""J04 (+ J02 breakdown, J03 rows) exclusive Jetson window (K38-1 protocol, copied from experiments/k39_jetson_20261027/
window.py with new paths and the K41 job list); always restores owned hardware controls and the pause file.
  1. dev calibration, no stopping (nuScenes val / AV2 CFG-DEV, first 712 frames, 200 warm-up): round medians of v6-S /
     B2K-A / B1K-A (K39 medians.py) and of the iterated K41 rows (medians41.py), sha256 recorded before any test run
  2. 80-frame smoke of every row on both test sets
  3. three repetitions x (nusc_test20, av2t40), every row in one process, rows rotated by --rep, dataset order alternated
  4. J02: one --breakdown repetition per test set (v6-S / bounded rows only; profiler overhead kept out of step 3)
usage: scripts/py experiments/k41_jetson_20261030/window.py"""
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

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
K39 = HERE.parent / 'k39_jetson_20261027'; K38 = HERE.parent / 'k38_deadline_20261026'
LOG = HERE / 'logs'; LOG.mkdir(exist_ok=True)
PAUSE = Path.home() / '.jetson_pause'; BUSY = ROOT.parent / 'risk-aware-perception/scripts/busy.sh'
state = dict(status='preflight', pid=os.getpid(), runs=[], started=dt.datetime.now().astimezone().isoformat())
child = telemetry = None; owned = clocks = False; fan = None


def save(**kw):
    state.update(kw, updated=dt.datetime.now().astimezone().isoformat()); tmp = LOG / 'state.tmp'
    tmp.write_text(json.dumps(state, indent=2) + '\n'); tmp.replace(LOG / 'state.json')


def api(payload=None):
    if payload is None: return json.load(urllib.request.urlopen('http://localhost:8765/api/jetson/status', timeout=10))
    req = urllib.request.Request('http://localhost:8765/api/jetson/fan', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=10))


def runcommand(command, log):
    global child
    with (LOG / log).open('w') as output:
        child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                                 env=dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))
        save(child_pid=child.pid, current_log=log); code = child.wait()
    child = None
    if code: raise RuntimeError((log, code))


def run(ds, rep, smoke=False, extra=(), tag=None, frames=None):
    tag = tag or (('smoke' if smoke else 'r' + str(rep)))
    cmd = [str(ROOT / 'scripts/py'), str(HERE / 'scripts/92_k41.py'), '--graphs', 'static', '--dataset', ds, '--runtime', 'staged',
           '--helper', 'fast', '--config', 'v6S', '--paths', 'trt32', '--bounds', '2', '--adapter_encoder', 'trt32', '--rep', str(rep),
           '--tag', tag] + list(extra)
    if smoke: cmd += ['--max_frames', '80', '--warmup', '0']
    elif frames: cmd += ['--max_frames', str(frames), '--warmup', '200']
    if ds.startswith('av2'): cmd += ['--mask_y0', '1830']
    log = tag + '_' + ds + '.log'; save(status='smoke' if smoke else 'timing', dataset=ds, rep=rep, tag=tag)
    runcommand(cmd, log)
    folder = Path(next(x.split('run: ', 1)[1].strip() for x in (LOG / log).read_text().splitlines() if 'run: ' in x))
    assert (folder / 'DONE.json').exists()
    state['runs'].append(dict(dataset=ds, rep=rep, smoke=smoke, tag=tag, run=str(folder))); save(); return folder


def stop(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()


def main():
    global owned, clocks, fan, telemetry
    def handle(sig, frame): raise InterruptedError(sig)
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, handle)
    lock = (LOG / 'window.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = json.loads((K38 / 'source_manifest.json').read_text())
    files = (list(HERE.glob('*.py')) + list((HERE / 'scripts').glob('*.py')) + list(K39.glob('*.py')) + list((K39 / 'scripts').glob('*.py'))
             + list(K38.glob('*.py')) + list((K38 / 'models').rglob('*.pt')) + [ROOT / 'src/vtf/k33sel.py', ROOT / 'src/vtf/k33pub.py']
             + list((ROOT / 'engines/k33').rglob('*.*')))
    sources.update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    for p, h in sources.items(): assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h, p
    engines = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'engines').glob('*')
               if p.is_file() and p.suffix in ('.engine', '.pt', '.npz')}
    (LOG / 'source_hashes_before.json').write_text(json.dumps(sources, indent=2)); (LOG / 'engine_hashes_before.json').write_text(json.dumps(engines, indent=2))
    try:
        while PAUSE.exists(): save(status='waiting_existing_window', owner=PAUSE.read_text()); time.sleep(20)
        fd = os.open(str(PAUSE), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, 'w') as f: f.write(f'CVPR K41 J04 adaptive-deadline window pid {os.getpid()} {state["started"]}\n')
        owned = True
        while True:
            busy = subprocess.check_output(['bash', str(BUSY)], text=True).strip()
            if busy == 'paused' and not (Path.home() / '.r3_profiling').exists(): break
            save(status='waiting_exclusive', busy=busy); time.sleep(20)
        hardware = api(); fan = hardware['fan']; (LOG / 'hardware_before.json').write_text(json.dumps(hardware, indent=2)); assert hardware['powerMode'] == 'MAXN'
        subprocess.run(['sudo', '-n', '/usr/bin/jetson_clocks', '--store', str(LOG / 'clock_before.cfg')], check=True); clocks = True
        subprocess.run(['sudo', '-n', '/usr/bin/jetson_clocks'], check=True); api({'mode': 'manual', 'pwm': 255})
        with (LOG / 'tegrastats.log').open('w') as f:
            telemetry = subprocess.Popen(['tegrastats', '--interval', '500'], stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        # 1. dev calibration (no stopping)
        CAL = ['--wf', 'none', '--calibrate', '--k41', 'de', '--k41_rows', 'ana05hinfi,cf40q20i']
        cal = {ds: run(ds, 1, extra=CAL, tag='cal', frames=712) for ds in ('nusc_val', 'av2_dev')}
        runcommand([str(ROOT / 'scripts/py'), str(K39 / 'medians.py'), str(cal['nusc_val']), str(cal['av2_dev']), str(HERE / 'medians_k41.json')], 'medians.log')
        runcommand([str(ROOT / 'scripts/py'), str(HERE / 'medians41.py'), str(cal['nusc_val']), str(cal['av2_dev']), str(HERE / 'medians41.json')], 'medians41.log')
        state['medians_sha256'] = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ('medians_k41.json', 'medians41.json')}; save()
        ROWS = ['--wf', 'u512,u640,u960,u1280,native', '--wf_c1only', '--heur', '--heur_only', 'tsp8S+C1,unc2S+C1',
                '--deadlines', '33.3,50,66.7,100', '--medians', str(HERE / 'medians_k41.json'),
                '--k41', 'de,pub,rc', '--medians41', str(HERE / 'medians41.json')]
        DS = ('nusc_test20', 'av2t40')
        # 2. smoke (80 frames, every row)
        for ds in DS: run(ds, 1, smoke=True, extra=ROWS)
        # 3. three repetitions, every row, dataset order alternated
        for rep in (1, 2, 3):
            for ds in (DS if rep % 2 else DS[::-1]): run(ds, rep, extra=ROWS)
        # 4. J02: breakdown repetition (v6-S and bounded rows, per-stage syncs)
        for ds in DS: run(ds, 1, extra=['--breakdown', '--wf', 'none', '--medians', str(HERE / 'medians_k41.json')], tag='bd')
        for p, h in sources.items(): assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h, p
        for p, h in engines.items(): assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h, p
        save(status='complete', source_engine_hashes_unchanged=True)
    except BaseException as e: save(status='failed', error=repr(e)); raise
    finally:
        stop(child); stop(telemetry); errors = []
        if clocks:
            try: subprocess.run(['sudo', '-n', '/usr/bin/jetson_clocks', '--restore', str(LOG / 'clock_before.cfg')], check=True)
            except Exception as e: errors.append(str(e))
        if fan is not None:
            try: api({'mode': 'manual', 'pwm': fan['pwm']} if fan['mode'] == 'manual' else {'mode': 'auto', 'profile': fan.get('profile') or 'quiet'})
            except Exception as e: errors.append(str(e))
        if owned and PAUSE.exists() and f'pid {os.getpid()} ' in PAUSE.read_text(): PAUSE.unlink()
        save(window_released=True, restore_errors=errors)


if __name__ == '__main__': main()
