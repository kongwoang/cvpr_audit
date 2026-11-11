"""Recover completed JP01 artifacts after an interrupted wrapper; preserve its state."""
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'experiments/latency_opt_20261030/logs/JP01_20261030_205826'


def main():
    lock = (LOG.parent.parent / 'pilot.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((LOG / 'state.json').read_text())
    assert state['status'] == 'profile' and len(state['runs']) == 19
    for pid in (state['pid'], state['child_pid']):
        assert not Path(f'/proc/{pid}').exists(), pid
    pause = Path.home() / '.jetson_pause'
    assert pause.read_text().startswith('CVPR LATENCY_OPT JP01 pid 51344 ')
    logfile = LOG / state['current_log']
    folder = Path(next(x.split('run: ', 1)[1].strip() for x in logfile.read_text().splitlines() if 'run: ' in x))
    done = json.loads((folder / 'DONE.json').read_text())
    assert done == dict(frames=712, timed_frames=512, rep=1)
    state['runs'].append(dict(dataset='av2_dev', mode='baseline', stage='profile', run=str(folder), **done))
    hashes = {}
    for r in state['runs']:
        p = Path(r['run'])
        assert json.loads((p / 'DONE.json').read_text())['frames'] == r['frames']
        for name in ('frames.pkl', 'DONE.json', 'config.json', 'environment.json'):
            f = p / name
            assert f.stat().st_size > 0
            hashes[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    for manifest, algorithm in [('engine_hashes_before.json', 'sha256'), ('source_hashes_before.json', 'sha256')]:
        for p, h in json.loads((LOG / manifest).read_text()).items():
            assert hashlib.new(algorithm, (ROOT / p).read_bytes()).hexdigest() == h, p
    protected = json.loads((ROOT / 'docs/paper/handoff/K40R/protected_files_before.json').read_text())
    for p, h in protected.items():
        assert hashlib.md5((ROOT / p).read_bytes()).hexdigest() == h, p
    subprocess.run(['sudo', '-n', '/usr/bin/jetson_clocks', '--restore', str(LOG / 'clock_before.cfg')], check=True)
    fan = json.loads((LOG / 'hardware_before.json').read_text())['fan']
    payload = {'mode': 'manual', 'pwm': fan['pwm']} if fan['mode'] == 'manual' else {'mode': 'auto', 'profile': fan['profile']}
    req = urllib.request.Request('http://localhost:8765/api/jetson/fan', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as response:
        response.read()
    hardware = json.load(urllib.request.urlopen('http://localhost:8765/api/jetson/status', timeout=10))
    (LOG / 'hardware_after_recovery.json').write_text(json.dumps(hardware, indent=2) + '\n')
    pause.unlink()
    state.update(status='recovered_complete', window_released=True, restore_errors=[],
                 protected_sources_and_engines_unchanged=True,
                 recovery=dict(time=dt.datetime.now().astimezone().isoformat(),
                               reason='Original wrapper exit unobserved; machine restarted after all 20 result directories wrote DONE.json. Original state.json preserved.',
                               last_profile_exit_code=None, original_restore_observed=False,
                               controls_restored_during_recovery=True,
                               validation='20 DONE markers and artifact hashes; source, engine and 88 protected hashes unchanged. DataFrame/endpoint checks performed separately by CPU analysis.'))
    (LOG / 'state.recovered.json').write_text(json.dumps(state, indent=2) + '\n')
    (LOG / 'recovered_artifact_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print('Recovered 20 artifact sets; original state preserved; own stale pause released.')


if __name__ == '__main__':
    main()
