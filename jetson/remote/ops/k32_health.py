#!/usr/bin/env python3
"""Hourly, read-only K32 health checks. Run locally; --probe runs via SSH stdin.

No evaluator is stopped or launched here. k32_auto.sh owns recovery and analysis.
Reports: docs/paper/handoff/K32/health/latest.md; history: logs/k32_health/.
"""
import concurrent.futures
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REMOTE = Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR')
ROOT = Path('/home/kongwoang/research/CVPR')


def tail(path, size=8192):
    try:
        with path.open('rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - size))
            return f.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def probe():
    now = time.time()
    processes = subprocess.check_output(['ps', '-eo', 'pid=,args='], universal_newlines=True).splitlines()
    active, evaluators, dispatchers = {}, {}, []
    for line in processes:
        pid, args = line.strip().split(None, 1)
        if 'bash ' in args and '/k32_dispatch.sh ' in args:
            dispatchers.append(int(pid))
        for name in re.findall(r'results/k32/((?:test|shadow)/[A-Za-z0-9_]+)\.json', args):
            active.setdefault(name, []).append(int(pid))
            if 'leaderboard_evaluator.py' in args and 'aeb_pilot.py' in args:
                evaluators.setdefault(name, []).append(int(pid))
    jobs = {}
    for filename in ('jobs_test_vuw.txt', 'jobs_test_vuw1.txt'):
        for line in (REMOTE / 'results/k32' / filename).read_text().splitlines():
            if not line.strip():
                continue
            name = line.split('|')[0]
            checkpoint = REMOTE / 'results/k32' / (name + '.json')
            log = REMOTE / 'logs/k32' / (name + '.log')
            owner = REMOTE / 'results/k32/claims' / name.replace('/', '_') / 'host'
            job = {'state': 'not_started', 'progress': [0, 0],
                   'owner': owner.read_text().strip() if owner.exists() else None}
            if checkpoint.exists():
                try:
                    p = json.loads(checkpoint.read_text())['_checkpoint']['progress']
                    job.update(progress=p, state='done' if p and p[0] >= p[1] else 'incomplete')
                except (ValueError, KeyError, TypeError) as exc:
                    job.update(state='unreadable', error=str(exc))
                job['checkpoint_age_min'] = (now - checkpoint.stat().st_mtime) / 60
            if job['state'] != 'done' and log.exists():
                job['log_age_min'] = (now - log.stat().st_mtime) / 60
                content = tail(log)
                job['tail'] = content.splitlines()[-3:]
                job['error_lines'] = [x[-300:] for x in content.splitlines()
                                      if re.search(r'Traceback|out of memory|segmentation fault|segfault|exit [1-9]', x, re.I)][-3:]
            jobs[name] = job
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                       '--format=csv,noheader'], universal_newlines=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        gpu = 'unavailable: ' + str(exc)
    return {'host': os.uname().nodename.split('.')[0], 'epoch': now,
            'dispatchers': dispatchers, 'active': active, 'evaluators': evaluators,
            'load': Path('/proc/loadavg').read_text().split()[:3], 'gpu': gpu, 'jobs': jobs,
            'final_analysis': (REMOTE / 'results/k32/ANALYSIS_DONE').exists()}


def fetch(host, source):
    try:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                 '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
                                 host, 'python3 - --probe'], input=source,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, timeout=120, check=True)
        return json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {'error': str(exc)}


def issues_for(hosts, previous):
    alerts = []
    good = {h: v for h, v in hosts.items() if 'jobs' in v}
    for host, data in hosts.items():
        if 'error' in data:
            alerts.append('%s: không kết nối/đọc được trạng thái: %s' % (host, data['error']))
    if not good:
        return alerts, {}, {}
    primary = next(iter(good.values()))
    jobs = primary['jobs']
    counts = {state: sum(j['state'] == state for j in jobs.values())
              for state in ('done', 'incomplete', 'not_started', 'unreadable')}
    for host, data in good.items():
        responsible = [name for name, j in jobs.items() if j['state'] != 'done' and j['owner'] == data['host']]
        if counts['done'] < len(jobs) and (responsible or counts['not_started']) and not data['dispatchers']:
            alerts.append('%s: thiếu dispatcher trong khi còn việc.' % host)
        if len(data['dispatchers']) > 1:
            alerts.append('%s: có %d dispatcher; cần kiểm tra chạy trùng.' % (host, len(data['dispatchers'])))
        for name, pids in data['evaluators'].items():
            if len(pids) > 1:
                alerts.append('%s: %s có nhiều evaluator %s.' % (host, name, pids))
    for name, job in jobs.items():
        if job['state'] == 'done':
            continue
        live_hosts = [h for h, d in good.items() if name in d['active']]
        ev_hosts = [h for h, d in good.items() if name in d['evaluators']]
        age = job.get('log_age_min', 0)
        if len(ev_hosts) > 1:
            alerts.append('%s: evaluator chạy trên cả hai host.' % name)
        if ev_hosts and age >= 45:
            alerts.append('%s (%s): evaluator còn sống, log im %.0f phút; checkpoint %s.' %
                          (name, ', '.join(ev_hosts), age, job['progress']))
        if len(good) == 2 and job['state'] == 'incomplete' and not live_hosts and age >= 30:
            alerts.append('%s: checkpoint dở %s, không có tiến trình, log cũ %.0f phút.' %
                          (name, job['progress'], age))
        if job['state'] == 'unreadable':
            alerts.append('%s: không đọc được checkpoint (có thể đang ghi); kiểm tra lần sau.' % name)
        if job.get('error_lines') and age < 75:
            alerts.append('%s: lỗi gần đây trong log: %s' % (name, ' | '.join(job['error_lines'])))
        old = previous.get('jobs', {}).get(name, {})
        # Log ticks may continue while the simulator never finishes a route.
        if (ev_hosts and old.get('progress') == job['progress'] and job['state'] == 'incomplete'
                and job.get('checkpoint_age_min', 0) >= 180):
            alerts.append('%s: checkpoint không đổi qua hai lần check, lần ghi cuối >3 giờ.' % name)
    return alerts, jobs, counts


def main():
    out = ROOT / 'docs/paper/handoff/K32/health'
    history = ROOT / 'logs/k32_health'
    out.mkdir(parents=True, exist_ok=True)
    history.mkdir(parents=True, exist_ok=True)
    lock = (history / 'check.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    if (out / 'COMPLETE').exists():
        return
    now = dt.datetime.now().astimezone()
    previous_path = out / 'latest.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    source = Path(__file__).read_text()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {h: pool.submit(fetch, h, source) for h in ('VUW', 'VUW1')}
        hosts = {h: f.result() for h, f in futures.items()}
    alerts, jobs, counts = issues_for(hosts, previous)
    auto_log = ROOT / 'logs/k32_auto.log'
    auto_tail = tail(auto_log, 4000)
    auto_done = (ROOT / 'docs/paper/handoff/K32/auto/FINAL_PULLED').exists()
    if jobs and counts['done'] < len(jobs) and auto_done:
        alerts.append('FINAL_PULLED đã có nhưng còn job chưa xong; không coi đây là kết quả hoàn tất.')
    if auto_log.exists() and not auto_done and time.time() - auto_log.stat().st_mtime > 5400:
        alerts.append('Cron k32_auto không ghi log hơn 90 phút.')
    last_failure = max(auto_tail.rfind('analysis FAILED'), auto_tail.rfind('Traceback'))
    last_success = max(auto_tail.rfind('analysis done'), auto_tail.rfind('final tables pulled'))
    if last_failure > last_success:
        alerts.append('Lần tổng hợp kết quả gần đây có lỗi, chưa thấy thông báo thành công sau đó trong cuối log.')
    stamp = now.strftime('%Y%m%d_%H%M%S')
    data = {'time': now.isoformat(), 'epoch': time.time(), 'hosts': hosts, 'jobs': jobs,
            'counts': counts, 'alerts': alerts, 'auto_final_pulled': auto_done}
    lines = ['# K32 — kiểm tra mỗi giờ', '', 'Cập nhật: ' + now.strftime('%d/%m/%Y %H:%M:%S %z'), '',
             '**%s**' % ('CẦN KIỂM TRA' if alerts else 'Không phát hiện bất thường theo các kiểm tra hiện tại.'), '']
    if jobs:
        lines += ['Hoàn tất **%d/%d**; đang dở %d; chưa bắt đầu %d; checkpoint không đọc được %d.' %
                  (counts['done'], len(jobs), counts['incomplete'], counts['not_started'], counts['unreadable']), '']
        for group in ('test', 'shadow'):
            subset = [j for n, j in jobs.items() if n.startswith(group + '/')]
            lines.append('- %s: %d/%d hoàn tất.' % (group, sum(j['state'] == 'done' for j in subset), len(subset)))
        lines += ['']
    for host, d in hosts.items():
        if 'error' not in d:
            lines.append('- %s: %d evaluator, %d dispatcher; load %s.' %
                         (host, sum(map(len, d['evaluators'].values())), len(d['dispatchers']), '/'.join(d['load'])))
    lines += ['', 'Bất thường:', ''] + (['- ' + x for x in alerts] or ['- Không.'])
    lines += ['', 'Bộ kiểm tra này chỉ đọc trạng thái; cơ chế khôi phục hiện có là k32_auto.sh.',
              'Báo cáo được lưu trên máy; không tự gửi tin nhắn vào cuộc trò chuyện.', '']
    text = '\n'.join(lines)
    for path, content in ((history / (stamp + '.json'), json.dumps(data, ensure_ascii=False, indent=2)),
                          (history / (stamp + '.md'), text),
                          (out / 'latest.json', json.dumps(data, ensure_ascii=False, indent=2)),
                          (out / 'latest.md', text)):
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(content)
        tmp.replace(path)
    with (history / 'events.log').open('a') as f:
        f.write('%s done=%s/%s alerts=%d\n' % (now.isoformat(), counts.get('done', '?'), len(jobs) or '?', len(alerts)))
        for alert in alerts:
            f.write('  ' + alert + '\n')
    if jobs and counts['done'] == len(jobs) and auto_done:
        (out / 'COMPLETE').write_text(now.isoformat() + '\n')
    print(text)


if __name__ == '__main__':
    if '--probe' in sys.argv:
        print(json.dumps(probe()))
    else:
        main()
