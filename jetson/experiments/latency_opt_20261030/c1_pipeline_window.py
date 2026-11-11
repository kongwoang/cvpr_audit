"""Authorized JP03 dev pilot. Own the window; restore controls even on failure."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAMP = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
LOG = HERE / 'logs' / ('JP03_' + STAMP)
PAUSE = Path.home() / '.jetson_pause'
BUSY = ROOT.parent / 'risk-aware-perception/scripts/busy.sh'
state = dict(status='preflight', pid=os.getpid(), runs=[], started=dt.datetime.now().astimezone().isoformat())
child = telemetry = None
owned = clocks = False
fan = None


def save(**kw):
    state.update(kw, updated=dt.datetime.now().astimezone().isoformat())
    p = LOG / 'state.tmp'; p.write_text(json.dumps(state, indent=2) + '\n'); p.replace(LOG / 'state.json')
    print(state['status'], kw.get('current_log', ''), flush=True)


def api(payload=None):
    url = 'http://localhost:8765/api/jetson/' + ('status' if payload is None else 'fan')
    req = url if payload is None else urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=10))


def stop_owned(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()


def command(cmd, name):
    global child
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('CUDA_VISIBLE_DEVICES', None)
    remaining = 10800 - (time.monotonic() - state['monotonic_start'])
    if remaining <= 0:
        raise TimeoutError('JP03 three-hour ceiling reached')
    with (LOG / name).open('w') as out:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        save(child_pid=child.pid, current_log=name, command=cmd)
        rc = child.wait(timeout=remaining)
    child = None
    if rc:
        raise RuntimeError((name, rc))


def run(ds, mode, rep, stage, frames=712):
    tag = f'JP03_{STAMP}_{stage}_{mode}_r{rep}'
    cmd = [str(ROOT / 'scripts/py'), str(HERE / 'scripts/94_latency_c1_graph.py'),
           '--dataset', ds, '--config', 'v6S', '--runtime', 'staged', '--helper', 'fast',
           '--adapter_encoder', 'trt32', '--paths', 'trt32', '--graphs', 'static',
           '--columns_mode', 'baseline', '--c1_graph_mode', mode, '--wf', 'u512,u640,u960', '--wf_c1only',
           '--bounds', '2', '--warmup', '200' if frames == 712 else '0',
           '--rep', str(rep), '--max_frames', str(frames), '--tag', tag]
    if ds.startswith('av2'):
        cmd += ['--mask_y0', '1830']
    if stage in ('smoke', 'parity'):
        cmd += ['--record_history']
    if stage == 'profile':
        cmd += ['--breakdown']
    name = f'{stage}_{ds}_{mode}_r{rep}.log'
    save(status=stage, dataset=ds, mode=mode, rep=rep)
    command(cmd, name)
    folder = Path(next(x.split('run: ', 1)[1].strip() for x in (LOG / name).read_text().splitlines() if 'run: ' in x))
    done = json.loads((folder / 'DONE.json').read_text())
    assert done['frames'] == frames, done
    assert done['rep'] == rep
    state['runs'].append(dict(dataset=ds, mode=mode, stage=stage, run=str(folder), **done))
    save()
    return folder


def main():
    global owned, clocks, fan, telemetry
    LOG.mkdir(parents=True)
    print('pilot log:', LOG, flush=True)
    def interrupt(sig, frame):
        raise InterruptedError(sig)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupt)
    lock = (HERE / 'pilot.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protected = json.loads((ROOT / 'docs/paper/handoff/K40R/protected_files_before.json').read_text())
    for p, h in protected.items():
        assert hashlib.md5((ROOT / p).read_bytes()).hexdigest() == h, p
    source = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(HERE.rglob('*.py'))}
    engines = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((ROOT / 'engines').glob('*')) if p.suffix in ('.engine', '.pt', '.npz')}
    (LOG / 'source_hashes_before.json').write_text(json.dumps(source, indent=2))
    (LOG / 'engine_hashes_before.json').write_text(json.dumps(engines, indent=2))
    try:
        if PAUSE.exists():
            raise RuntimeError('Another window owns .jetson_pause: ' + PAUSE.read_text())
        fd = os.open(str(PAUSE), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, 'w') as f:
            f.write(f'CVPR LATENCY_OPT JP03 pid {os.getpid()} {state["started"]}\n')
        owned = True
        while True:
            busy = subprocess.check_output(['bash', str(BUSY)], text=True).strip()
            if busy == 'paused' and not (Path.home() / '.r3_profiling').exists():
                break
            save(status='waiting_exclusive', busy=busy); time.sleep(20)
        hardware = api(); fan = hardware['fan']
        (LOG / 'hardware_before.json').write_text(json.dumps(hardware, indent=2))
        assert hardware['powerMode'] == 'MAXN'
        subprocess.run(['sudo', '-n', '/usr/bin/jetson_clocks', '--store', str(LOG / 'clock_before.cfg')], check=True)
        clocks = True
        subprocess.run(['sudo', '-n', '/usr/bin/jetson_clocks'], check=True)
        api({'mode': 'manual', 'pwm': 255})
        with (LOG / 'tegrastats.log').open('w') as out:
            telemetry = subprocess.Popen(['tegrastats', '--interval', '500'], stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        state['monotonic_start'] = time.monotonic()
        command([str(ROOT / 'scripts/py'), '-c', 'import cv2; import torch; import tensorrt; print(cv2.__version__, torch.__version__, tensorrt.__version__); print(torch.cuda.get_device_name()); x=torch.ones(4,device="cuda"); torch.cuda.synchronize(); print(x.cpu().tolist())'], 'device_preflight.log')
        for ds in ('nusc_val', 'av2_dev'):
            run(ds, 'audit', 1, 'smoke', 80)
            reference = run(ds, 'baseline', 1, 'parity')
            candidate = run(ds, 'audit', 1, 'parity')
            command([str(ROOT / 'scripts/py'), str(HERE / 'compare.py'), str(reference), str(candidate), str(LOG / f'parity_{ds}.csv')], f'compare_{ds}.log')
        for rep in (1, 2, 3):
            modes = ('baseline', 'candidate') if rep % 2 else ('candidate', 'baseline')
            for ds in ('nusc_val', 'av2_dev'):
                for mode in modes:
                    run(ds, mode, rep, 'timing')
        for p, h in protected.items():
            assert hashlib.md5((ROOT / p).read_bytes()).hexdigest() == h, p
        for p, h in engines.items():
            assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h, p
        save(status='complete', protected_sources_and_engines_unchanged=True)
    except BaseException as e:
        save(status='failed', error=repr(e)); raise
    finally:
        stop_owned(child); stop_owned(telemetry)
        errors = []
        if clocks:
            try:
                subprocess.run(['sudo', '-n', '/usr/bin/jetson_clocks', '--restore', str(LOG / 'clock_before.cfg')], check=True)
            except Exception as e:
                errors.append(str(e))
        if fan is not None:
            try:
                api({'mode': 'manual', 'pwm': fan['pwm']} if fan['mode'] == 'manual' else {'mode': 'auto', 'profile': fan.get('profile') or 'quiet'})
            except Exception as e:
                errors.append(str(e))
        if owned and PAUSE.exists() and f'pid {os.getpid()} ' in PAUSE.read_text():
            PAUSE.unlink()
        save(window_released=True, restore_errors=errors)
        if errors:
            raise RuntimeError(errors)


if __name__ == '__main__':
    main()
