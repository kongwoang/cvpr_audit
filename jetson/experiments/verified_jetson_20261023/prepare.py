"""Freeze a cached AV2 log list verified against available training-array IDs."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
H=ROOT/'docs/paper/handoff'


def main():
    audit=H/'OPTIMIZATION_20261023/train_audit'
    status=json.loads((audit/'DONE.json').read_text());assert status['status']=='PASS'
    splitfile=ROOT/'experiments/optimization_20261023/current_av2_split.csv'
    roles={r['log']:r['role'] for r in csv.DictReader(splitfile.open())}
    used=list(csv.DictReader((audit/'training_selection_units.csv').open()))
    blocked={r['unit'] for r in used if r['dataset']=='av2'}
    cached=[s.strip() for s in (Path.home()/'datasets/k29/k29_av2dev_logs.txt').read_text().splitlines() if s.strip()]
    assert len(cached)==128 and len(set(cached))==128
    logs=[s for s in cached if roles.get(s)=='CFG-DEV' and s not in blocked]
    assert len(logs)>=10
    data=Path.home()/'datasets/k29/av2/sensor/train'
    for unit in logs:
        assert (data/unit/'annotations.feather').exists()
        assert (data/unit/'calibration/intrinsics.feather').exists()
        assert any((data/unit/'sensors/cameras/ring_front_center').glob('*.jpg'))
    manifest=dict(logs=logs,cache_logs=128,verified_logs=len(logs),
        current_cache_roles=dict(Counter(roles.get(s,'UNRECOGNIZED') for s in cached)),
        rule='Preserve original K29 ordering; keep current CFG-DEV and exclude every available C1/refiner training or selection unit.',
        split_sha256=hashlib.sha256(splitfile.read_bytes()).hexdigest(),
        training_audit_sha256=hashlib.sha256((audit/'DONE.json').read_bytes()).hexdigest(),
        checkpoints=status['checkpoints'])
    (HERE/'verified_log_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    out=H/'OPTIMIZATION_20261023/dataset_audit';out.mkdir(exist_ok=True)
    (out/'verified_log_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rows=[];summary=[]
    for report in ('RUNTIME_20261023','TEMPORAL_JETSON_20261023','FAST_RECOVERY_20261023'):
        if report=='RUNTIME_20261023':
            source=H/report/'frame_manifest.csv';data_rows=list(csv.DictReader(source.open()))
        else:
            source=H/report/'frames.csv'
            data_rows=list({(r['dataset'],r['unit'],r['frame']):r for r in csv.DictReader(source.open())}.values())
        for r in data_rows:
            role=roles.get(r['unit'],'UNRECOGNIZED') if r['dataset']=='av2_dev' else 'nuScenes DEV'
            rows.append(dict(report=report,dataset=r['dataset'],unit=r['unit'],frame=r['frame'],current_role=role,
                in_available_training_or_selection=r['unit'] in blocked if r['dataset']=='av2_dev' else False))
        av=[r for r in rows if r['report']==report and r['dataset']=='av2_dev']
        summary.append(dict(report=report,unique_timed_frames=len(av),
            cfg_dev=sum(r['current_role']=='CFG-DEV' for r in av),
            other_roles=sum(r['current_role']!='CFG-DEV' for r in av),
            known_training_or_selection=sum(r['in_available_training_or_selection'] for r in av)))
    for name,data_rows in [('legacy_timed_roles',rows),('legacy_summary',summary)]:
        with (out/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data_rows[0]));w.writeheader();w.writerows(data_rows)
    correction=('> Dataset correction (2026-10-23): the legacy CLI label `av2_dev` below refers to the cached K29 runtime subset, '
        'which includes REF-TRAIN logs. It is not an independent CFG-DEV quality evaluation. All measurements are retained; '
        'use them only for runtime/output-preservation engineering. The full VUW quality screen was audited against actual '
        'available training/selection unit IDs and checkpoint feature norms, with zero overlap. Corrected AV2 timing on a '
        'verified CFG-DEV manifest is reported separately in VERIFIED_JETSON_20261023. See '
        '../OPTIMIZATION_20261023/dataset_audit/legacy_summary.csv.\n\n')
    for report in ('RUNTIME_20261023','TEMPORAL_JETSON_20261023','FAST_RECOVERY_20261023'):
        p=H/report/'REPORT.md';text=p.read_text()
        if 'Dataset correction (2026-10-23)' not in text:p.write_text(correction+text)
    for name in ('FINAL_REPORT.md','FEASIBILITY.md'):
        p=H/'OPTIMIZATION_20261023'/name;text=p.read_text()
        if 'Dataset correction (2026-10-23)' not in text:p.write_text(correction+text)
    (out/'DONE.json').write_text(json.dumps(dict(status='complete',verified_logs=len(logs),legacy=summary),indent=2)+'\n')
    print(json.dumps(dict(verified_logs=len(logs),cache_roles=manifest['current_cache_roles'],legacy=summary),indent=2))


if __name__=='__main__':main()
