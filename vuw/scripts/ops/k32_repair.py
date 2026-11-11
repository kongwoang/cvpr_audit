#!/usr/bin/env python3
"""Run infrastructure-failed K32 route units separately; never select by outcome quality.

Raw attempts are immutable once finished. Canonical merging is a separate, reviewed step.
"""
import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

B = Path(os.environ.get('K32_ROOT', '/vol/dl-nguyenb5-solar/users/hoangpc/CVPR'))
K = B / 'results/k32'
VALID = {'Completed', 'Perfect', 'Failed', 'Failed - TickRuntime', 'Failed - Agent got blocked',
         'Failed - Agent timed out', 'Failed - Agent deviated from the route'}


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(d, indent=2) + '\n')
    os.replace(tmp, p)


def plan(batch):
    """Append only infrastructure-failed registered units; preserve prior repair attempts."""
    batch.mkdir(parents=True, exist_ok=True)
    old = json.loads((batch/'manifest.json').read_text()) if (batch/'manifest.json').exists() else {'created_utc':stamp(), 'units':[]}
    known = {u['id'] for u in old['units']}
    for jf in ('jobs_test_vuw.txt', 'jobs_test_vuw1.txt'):
        for row in (K/jf).read_text().splitlines():
            if not row.strip(): continue
            name, xml, env = row.split('|')
            ck = K/(name+'.json')
            if not ck.exists(): continue
            data = json.loads(ck.read_text())
            tree = ET.parse(xml)
            root = tree.getroot()
            route_by_id = {r.get('id'):r for r in root.findall('route')}
            env = dict(x.split('=',1) for x in env.split())
            for rec in data['_checkpoint']['records']:
                if rec['status'] in VALID: continue
                # Do not silently turn unknown evaluation outcomes into retries.
                if not any(x in rec['status'] for x in ("couldn't be set up", 'Simulation crashed', "Agent crashed")):
                    raise ValueError(('unreviewed status', name, rec['status']))
                rid = rec['route_id'].split('_')[1]
                uid = name.replace('/','_') + '_r' + rid
                if uid in known: continue
                ud = batch/'units'/uid
                ud.mkdir(parents=True, exist_ok=True)
                single = copy.deepcopy(root)
                for r in list(single.findall('route')): single.remove(r)
                single.append(copy.deepcopy(route_by_id[rid]))
                # Keep the original basename, which the pilot logs as its route-set metadata.
                xml_out = ud/Path(xml).name
                ET.ElementTree(single).write(xml_out, encoding='utf-8', xml_declaration=True)
                unit = {'id':uid, 'job':name, 'route':rid, 'route_id':rec['route_id'],
                        'original_index':rec['index'], 'original_status':rec['status'],
                        'original_xml':xml, 'xml':str(xml_out), 'xml_sha256':digest(xml_out),
                        'original_xml_sha256':digest(Path(xml)), 'env':env,
                        'source_record':rec, 'created_utc':stamp()}
                write_json(ud/'unit.json', unit)
                old['units'].append(unit); known.add(uid)
    old['updated_utc'] = stamp()
    write_json(batch/'manifest.json', old)
    print(json.dumps({'units':len(old['units']), 'ids':[u['id'] for u in old['units']]}, indent=2), flush=True)


def validate(unit, attempt):
    ck = attempt/'checkpoint.json'
    data = json.loads(ck.read_text())
    assert data['_checkpoint']['progress'] == [1,1], 'incomplete checkpoint'
    recs = data['_checkpoint']['records']
    assert len(recs) == 1 and recs[0]['route_id'] == unit['route_id'], 'wrong route'
    rec = recs[0]
    assert rec['status'] in VALID, rec['status']
    assert rec['meta']['duration_game'] > 0, 'zero simulation duration'
    e = unit['env']; br=e['AEB_BRANCH']; st=e['AEB_LAT']
    sv = attempt/'save'/('aeb_'+br+('_logall' if e.get('AEB_LOGALL')=='1' else '')+'_'+st+'.jsonl')
    lines = [json.loads(x) for x in sv.read_text().splitlines()]
    assert len(lines) == 1, 'attempt must contain exactly one route log'
    d = lines[0]
    expected = {'branch':br, 'setting':st, 'seed':int(e['AEB_SEED']),
                'logall':e.get('AEB_LOGALL','0')=='1', 'c1':e.get('AEB_C1','1'),
                'ref':e.get('AEB_REF','1'), 'front':e.get('AEB_FRONT','0'), 'routes':Path(unit['xml']).stem}
    assert all(d.get(k)==v for k,v in expected.items()), 'configuration mismatch'
    start = ET.parse(unit['xml']).getroot().find('route/waypoints/position')
    assert d.get('start') and math.hypot(float(start.get('x'))-d['start']['x'], float(start.get('y'))-d['start']['y']) < 1, 'start mismatch'
    ts = d['ticks']
    assert ts and all('a_gt' in t and not isinstance(t.get('err'),str) for t in ts), 'missing/error ticks'
    assert all(t['t']==i+1 for i,t in enumerate(ts)), 'nonconsecutive ticks'
    assert all(t[k] in (0,1,2) for t in ts for k in ('a_gt','a_frame','a_applied')), 'invalid action'
    assert abs(rec['meta']['duration_game']-len(ts)*.05) <= 2, 'duration mismatch'
    assert digest(Path(unit['xml'])) == unit['xml_sha256'], 'XML changed'
    return {'status':rec['status'], 'ticks':len(ts), 'checkpoint':str(ck), 'jsonl':str(sv),
            'checkpoint_sha256':digest(ck), 'jsonl_sha256':digest(sv), 'validated_utc':stamp()}


def processes():
    out = {}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            stat = (p/'stat').read_text(); end=stat.rfind(')'); fields=stat[end+2:].split()
            out[int(p.name)] = {'ppid':int(fields[1]), 'state':fields[0], 'start':fields[19],
                                'comm':stat[stat.find('(')+1:end], 'args':(p/'cmdline').read_bytes().split(b'\0')}
        except (OSError,ValueError,IndexError): pass
    return out


def descendants(root, ps):
    ids = {root}
    for _ in range(12):
        nxt = ids | {p for p,r in ps.items() if r['ppid'] in ids}
        if nxt == ids: break
        ids = nxt
    return ids


def stop_owned(root):
    ps = processes(); ids = descendants(root, ps)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        now = processes()
        for pid in sorted(ids, reverse=True):
            if pid in ps and pid in now and ps[pid]['start']==now[pid]['start']:
                try: os.kill(pid,sig)
                except ProcessLookupError: pass
        if sig == signal.SIGTERM: time.sleep(3)


def worker(batch):
    host = os.uname().nodename.split('.')[0]
    print(stamp(), 'worker',host,'pid',os.getpid(),flush=True)
    while True:
        units = json.loads((batch/'manifest.json').read_text())['units']
        pending = [u for u in units if not (batch/'units'/u['id']/'DONE.json').exists()]
        if not pending: print(stamp(),'all planned repair units done',flush=True); return
        picked = False
        for u in pending:
            ud = batch/'units'/u['id']; claim=ud/'claim'
            try: claim.mkdir()
            except FileExistsError: continue
            write_json(claim/'owner.json', {'host':host,'pid':os.getpid(),'time':stamp()})
            picked=True
            try:
                # The pending snapshot may predate another worker's completed claim.
                if (ud/'DONE.json').exists():
                    continue
                for _ in range(8):
                    attempts = sorted(ud.glob('attempt_*'))
                    ad = ud/('attempt_%03d' % (len(attempts)+1));ad.mkdir()
                    env = dict(os.environ, **u['env'], SHARE='1', MIN_FREE_MB='10000')
                    log=ad/'run.log'
                    cmd=[str(B/'run_when_gpu_free.sh'),str(log),'bash',str(B/'scripts/A_closedloop/k32_run.sh'),
                         u['xml'],str(ad/'checkpoint.json'),str(ad/'save')]
                    print(stamp(),'start',u['id'],ad.name,flush=True)
                    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
                    write_json(ud/'CURRENT.json',{'host':host,'worker_pid':os.getpid(),'wrapper_pid':proc.pid,'attempt':str(ad),'started_utc':stamp()})
                    eval_started=None; reason=None
                    while proc.poll() is None:
                        time.sleep(10)
                        ps=processes(); owned=descendants(proc.pid,ps)
                        ev=[p for p in owned if p in ps and ('--checkpoint='+str(ad/'checkpoint.json')).encode() in ps[p]['args']
                            and any(a.endswith(b'/leaderboard_evaluator.py') for a in ps[p]['args'])]
                        if ev:
                            eval_started=eval_started or time.monotonic()
                            child=[r for r in ps.values() if r['ppid'] in ev and r['comm'].startswith('CarlaUE4')]
                            if child and all(r['state']=='Z' for r in child): reason='simulator exited'
                            if log.exists() and time.time()-log.stat().st_mtime > 600: reason='no simulator progress for 10 minutes'
                            if time.monotonic()-eval_started > 2700: reason='single-route attempt exceeded 45 minutes'
                        if reason: stop_owned(proc.pid);break
                    proc.wait()
                    try:
                        result=validate(u,ad)
                    except Exception as ex:
                        write_json(ad/'REJECTED.json',{'time':stamp(),'reason':reason or str(ex),'returncode':proc.returncode})
                        print(stamp(),'retry',u['id'],reason or str(ex),flush=True)
                        time.sleep(15)
                    else:
                        result.update(unit=u['id'],host=host,attempt=str(ad))
                        write_json(ud/'DONE.json',result)
                        print(stamp(),'accepted',u['id'],result['status'],result['ticks'],'ticks',flush=True)
                        break
                else:
                    write_json(ud/'NEEDS_ATTENTION.json',{'time':stamp(),'reason':'8 attempts failed; inspect before continuing'})
                    raise RuntimeError('repair repeatedly failed: '+u['id'])
            finally:
                (claim/'owner.json').unlink(missing_ok=True);claim.rmdir()
            break
        if not picked: time.sleep(20)


def matched_lines(path, xml):
    """Same ordered matching and longer-retry rule as the existing K32 analyzer."""
    routes=ET.parse(xml).getroot().findall('route')
    ids=[r.get('id') for r in routes]
    xy=[tuple(float(r.find('waypoints/position').get(k)) for k in ('x','y')) for r in routes]
    out={};j=0
    for line in path.read_text().splitlines():
        d=json.loads(line);s=d.get('start')
        if not s: continue
        near=lambda k:math.hypot(xy[k][0]-s['x'],xy[k][1]-s['y'])<30
        k=next((k for k in range(j,len(ids)) if near(k)),None)
        if k is not None:out[ids[k]]=d;j=k+1
        elif j and near(j-1):
            if len(d['ticks'])>len(out[ids[j-1]]['ticks']):out[ids[j-1]]=d
        else:raise ValueError('unmatched nonempty original route line')
    return out


def preview(batch):
    """Prepare merged artifacts with official global statistics; do not touch canonical data."""
    from leaderboard.utils.statistics_manager import StatisticsManager
    manifest=json.loads((batch/'manifest.json').read_text())
    groups={}
    for u in manifest['units']:groups.setdefault(u['job'],[]).append(u)
    prepared=[]
    for name,units in groups.items():
        ck=K/(name+'.json');source=json.loads(ck.read_text());source_hash=digest(ck)
        ids=[r.get('id') for r in ET.parse(units[0]['original_xml']).getroot().findall('route')]
        assert source['_checkpoint']['progress']==[len(ids),len(ids)], ('original job still incomplete',name)
        records=copy.deepcopy(source['_checkpoint']['records'])
        assert [r['route_id'].split('_')[1] for r in records]==ids
        e=units[0]['env'];sv=K/(name+'_save')/('aeb_'+e['AEB_BRANCH']+('_logall' if e.get('AEB_LOGALL')=='1' else '')+'_'+e['AEB_LAT']+'.jsonl')
        sv_hash=digest(sv);lines=matched_lines(sv,units[0]['original_xml'])
        before_lines=copy.deepcopy(lines);replaced=[]
        for u in units:
            assert digest(Path(u['original_xml']))==u['original_xml_sha256']
            done=json.loads((batch/'units'/u['id']/'DONE.json').read_text())
            validated=validate(u,Path(done['attempt']))
            assert all(validated[k]==done[k] for k in ('checkpoint_sha256','jsonl_sha256'))
            i=u['original_index'];assert records[i]==u['source_record'], 'original failed record changed'
            assert records[i]['status'] not in VALID, 'refuse replacing a valid outcome'
            new=json.loads(Path(done['checkpoint']).read_text())['_checkpoint']['records'][0]
            for key in ('route_id','scenario_name','weather_id','town_name'):
                assert records[i][key]==new[key], (name,u['route'],key,'scenario metadata mismatch')
            new['index']=i;records[i]=new
            lines[u['route']]=json.loads(Path(done['jsonl']).read_text())
            replaced.append(u['route'])
        assert set(lines)==set(ids), 'not all original routes have logs after repair'
        for i,r in enumerate(records):
            rid=ids[i];assert r['status'] in VALID and lines[rid]['ticks']
            if rid not in replaced:
                assert r==source['_checkpoint']['records'][i], 'valid record changed'
                assert lines[rid]==before_lines[rid], 'valid tick data changed'
        staging=batch/'prepared'/name.replace('/','_');staging.mkdir(parents=True,exist_ok=True)
        outck=staging/'checkpoint.json';outsv=staging/'aeb.jsonl'
        merged=copy.deepcopy(source);merged['_checkpoint']['records']=records
        write_json(outck,merged)
        manager=StatisticsManager(str(outck),str(staging/'debug.txt'))
        manager.add_file_records(str(outck));manager.save_sensors(source['sensors']);manager.save_progress(len(ids),len(ids))
        manager.compute_global_statistics();manager.validate_and_write_statistics(True,False)
        check=json.loads(outck.read_text())
        assert check['_checkpoint']['records']==records and check['entry_status']=='Finished'
        outsv.write_text(''.join(json.dumps(lines[rid])+'\n' for rid in ids))
        assert set(matched_lines(outsv,units[0]['original_xml']))==set(ids)
        prepared.append({'job':name,'routes_replaced':replaced,'checkpoint':str(ck),'jsonl':str(sv),
                         'before_checkpoint_sha256':source_hash,'before_jsonl_sha256':sv_hash,
                         'prepared_checkpoint':str(outck),'prepared_jsonl':str(outsv),
                         'after_checkpoint_sha256':digest(outck),'after_jsonl_sha256':digest(outsv),
                         'preserved_valid_routes':len(ids)-len(replaced)})
    write_json(batch/'merge_preview.json',{'created_utc':stamp(),'jobs':prepared})
    print(json.dumps(prepared,indent=2),flush=True)


def commit(batch):
    """Commit only an unchanged preview; caller must exclude concurrent analyzers/evaluators."""
    assert not (batch/'MERGED.json').exists(), 'already committed'
    preview=json.loads((batch/'merge_preview.json').read_text())
    for item in preview['jobs']:
        for role in ('checkpoint','jsonl'):
            assert digest(Path(item[role]))==item['before_'+role+'_sha256'], 'canonical data changed; regenerate preview'
            assert digest(Path(item['prepared_'+role]))==item['after_'+role+'_sha256'], 'prepared data changed'
    # Back up all sources before replacing any of them.
    for item in preview['jobs']:
        backup=batch/'canonical_backups'/item['job'].replace('/','_');backup.mkdir(parents=True,exist_ok=False)
        for role in ('checkpoint','jsonl'):
            target=backup/Path(item[role]).name;shutil.copy2(item[role],target)
            assert digest(target)==item['before_'+role+'_sha256']
        write_json(backup/'provenance.json',item)
    write_json(batch/'COMMITTING.json',preview)
    for item in preview['jobs']:
        for role in ('jsonl','checkpoint'):
            dst=Path(item[role]);tmp=dst.with_name(dst.name+'.repair_tmp')
            shutil.copy2(item['prepared_'+role],tmp);os.replace(tmp,dst)
            assert digest(dst)==item['after_'+role+'_sha256']
    preview['committed_utc']=stamp();write_json(batch/'MERGED.json',preview)
    print('Merged',sum(len(j['routes_replaced']) for j in preview['jobs']),'repaired route units; originals preserved.',flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['plan','worker','preview','commit']);ap.add_argument('--batch',required=True)
    a=ap.parse_args();globals()[a.mode](Path(a.batch))
