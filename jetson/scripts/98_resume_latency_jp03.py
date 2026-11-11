#!/usr/bin/env python3
"""Resume interrupted JP03 orchestration without discarding completed trials."""
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'experiments/latency_opt_20261030'
LOG = HERE / 'logs/JP03_20261031_014821'


def main():
    spec = importlib.util.spec_from_file_location('jp03', HERE/'c1_pipeline_window.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.LOG = LOG; m.STAMP = '20261031_014821'
    lock = (HERE/'pilot.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads((LOG/'state.json').read_text())
    assert state['status']=='timing' and len(state['runs'])==10
    assert state['pid']==18948 and state['child_pid']==21247
    assert not Path('/proc/18948').exists() and not Path('/proc/21247').exists()
    pause = Path.home()/'.jetson_pause'
    assert pause.read_text().startswith('CVPR LATENCY_OPT JP03 pid 18948 ')
    for manifest in ('source_hashes_before.json','engine_hashes_before.json'):
        for p,h in json.loads((LOG/manifest).read_text()).items():
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    protected=json.loads((ROOT/'docs/paper/handoff/K40R/protected_files_before.json').read_text())
    for p,h in protected.items(): assert hashlib.md5((ROOT/p).read_bytes()).hexdigest()==h,p
    before=m.api()
    assert before['fan']['pwm']==255 and before['powerMode']=='MAXN'
    (LOG/'hardware_at_resume.json').write_text(json.dumps(before,indent=2)+'\n')
    folder=Path(next(x.split('run: ',1)[1].strip() for x in (LOG/state['current_log']).read_text().splitlines() if 'run: ' in x))
    done=json.loads((folder/'DONE.json').read_text())
    assert done==dict(frames=712,timed_frames=512,rep=2)
    cfg=json.loads((folder/'config.json').read_text())
    assert cfg['dataset']=='nusc_val' and cfg['c1_graph_mode']=='candidate' and cfg['rep']==2
    assert not cfg['record_history'] and not cfg['breakdown']
    # Keep the original state verbatim and disclose the unobserved child exit.
    backup=LOG/'state.interrupted_before_resume.json'
    with backup.open('x') as f: f.write((LOG/'state.json').read_text())
    state['runs'].append(dict(dataset='nusc_val',mode='candidate',stage='timing',run=str(folder),**done))
    state['resume']=dict(time=dt.datetime.now().astimezone().isoformat(),
                         original_wrapper_pid=18948,original_child_pid=21247,
                         original_exit_code=None,recovered_run=str(folder),
                         reason='Tool execution session disappeared; wrapper and child absent. Completed child has DONE marker. Device did not reboot and telemetry/fan remained active.',
                         completed_clean_runs_retained=5,
                         original_state=str(backup),
                         boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    state['pid']=os.getpid();m.state=state
    def interrupted(sig, frame): raise InterruptedError(sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    m.fan=json.loads((LOG/'hardware_before.json').read_text())['fan']
    # This telemetry PID is ours: its output fd points exactly to this window.
    telemetry_pid=19050
    telemetry_fd=Path(f'/proc/{telemetry_pid}/fd/1')
    assert telemetry_fd.resolve()==LOG/'tegrastats.log'
    assert Path(f'/proc/{telemetry_pid}/cmdline').read_bytes()==b'tegrastats\x00--interval\x00500\x00'
    os.kill(telemetry_pid,signal.SIGTERM)
    pause.write_text(f'CVPR LATENCY_OPT JP03 pid {os.getpid()} resumed {state["resume"]["time"]}\n')
    try:
        subprocess.run(['sudo','-n','/usr/bin/jetson_clocks'],check=True)
        with (LOG/'tegrastats.log').open('a') as out:
            m.telemetry=subprocess.Popen(['tegrastats','--interval','500'],stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        m.save(status='resuming',telemetry_pid=m.telemetry.pid)
        completed={(r['dataset'],r['mode'],r['rep']) for r in state['runs'] if r['stage']=='timing'}
        for rep in (1,2,3):
            modes=('baseline','candidate') if rep%2 else ('candidate','baseline')
            for ds in ('nusc_val','av2_dev'):
                for mode in modes:
                    if (ds,mode,rep) not in completed:
                        name=f'timing_{ds}_{mode}_r{rep}.log'
                        assert not (LOG/name).exists(),name
                        m.run(ds,mode,rep,'timing')
        for p,h in protected.items():assert hashlib.md5((ROOT/p).read_bytes()).hexdigest()==h,p
        for manifest in ('source_hashes_before.json','engine_hashes_before.json'):
            for p,h in json.loads((LOG/manifest).read_text()).items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
        m.save(status='complete',protected_sources_and_engines_unchanged=True)
    except BaseException as e:
        m.save(status='failed',error=repr(e));raise
    finally:
        m.stop_owned(m.child);m.stop_owned(m.telemetry)
        errors=[]
        try:subprocess.run(['sudo','-n','/usr/bin/jetson_clocks','--restore',str(LOG/'clock_before.cfg')],check=True)
        except Exception as e:errors.append(str(e))
        try:m.api({'mode':'manual','pwm':m.fan['pwm']} if m.fan['mode']=='manual' else {'mode':'auto','profile':m.fan.get('profile') or 'quiet'})
        except Exception as e:errors.append(str(e))
        if pause.exists() and f'pid {os.getpid()} ' in pause.read_text():pause.unlink()
        m.save(window_released=True,restore_errors=errors)
        if errors:raise RuntimeError(errors)


if __name__=='__main__':main()
