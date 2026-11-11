"""Create the corrected-manifest harness without changing any deployed source."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'experiments/temporal_fast_20261023'


def once(s,old,new):
    assert s.count(old)==1,(old[:80],s.count(old))
    return s.replace(old,new)


def main():
    for name in ('helpers.py','fast.py','recovery_gpu.py','compare.py'):
        (HERE/name).write_bytes((OLD/name).read_bytes())
    s=(OLD/'scripts/82_temporal_helper.py').read_text().replace(str(OLD),str(HERE))
    s=once(s,'    logs = [ln.strip() for ln in open(K29D / "k29_av2dev_logs.txt") if ln.strip()]',
        f'    logs = json.loads(Path({str(HERE/"verified_log_manifest.json")!r}).read_text())["logs"]')
    s=once(s,'f"temporal_helper_{args.helper}_{mode}_{args.dataset}_{args.config}"',
        'f"verified_cfg_{args.helper}_{mode}_{args.dataset}_{args.config}"')
    s=once(s,'        RETROS={}',
        '        rec["image_filename"]=Path(path).name\n'
        '        rec["annotation_timestamp_ns"]=int(cam.frame_ts[fr]) if hasattr(cam,"frame_ts") else -1\n'
        '        RETROS={}')
    path=HERE/'scripts/83_verified_av2.py';path.parent.mkdir(exist_ok=True);path.write_text(s)
    (HERE/'source_manifest.json').write_bytes((OLD/'source_manifest.json').read_bytes())
    # New exclusive window; reuse the validated implementation, replace only the data manifest.
    s=(OLD/'window.py').read_text()
    s=s.replace("str(HERE/'scripts/82_temporal_helper.py')","str(HERE/'scripts/83_verified_av2.py')")
    s=s.replace("'80' if smoke else '456'","'80' if smoke else '712'")
    s=s.replace("while not (ROOT/'docs/paper/handoff/TEMPORAL_JETSON_20261023/DONE.json').exists():",
        "while not (ROOT/'docs/paper/handoff/FAST_RECOVERY_20261023/DONE.json').exists():")
    s=s.replace("HERE/'scripts/82_temporal_helper.py'","HERE/'scripts/83_verified_av2.py'")
    s=s.replace("HERE/'protocol.md']", "HERE/'protocol.md',HERE/'verified_log_manifest.json']")
    a=s.index("        save(status='cpu_semantics')")
    b=s.index('        for p,h in sources.items():assert',a)
    s=s[:a]+'''        original=run('av2_dev','original',1,True);fast=run('av2_dev','fast',1,True)
        run_command([str(ROOT/'scripts/py'),str(HERE/'compare.py'),'--reference',str(original),'--staged',str(fast),
            '--out',str(LOG/'verified_smoke_parity.json')],'verified_smoke_compare.log')
        for rep in (1,2,3):run('av2_dev','fast',rep)
''' +s[b:]
    (HERE/'window.py').write_text(s)
    s=(ROOT/'experiments/temporal_jetson_20261023/report.py').read_text()
    s=s.replace('TEMPORAL_JETSON_20261023','VERIFIED_JETSON_20261023')
    s=s.replace('Temporal evidence recovery — Jetson feasibility','AV2 verified CFG-DEV — Jetson timing')
    s=s.replace('Two dev prefixes, 512 timed frames/dataset/rep after 200 warm-up frames; three repetitions.',
        'Verified AV2 CFG-DEV only, 512 timed frames/rep after 200 warm-up frames; three repetitions.')
    s=s.replace('Disabled-recovery port parity and enabled provenance/budget checks passed before timing.',
        'Original/fast helper parity passed on the first 80 verified frames. Actual training/selection unit IDs were excluded.')
    s=s.replace('Historical execution and pixels are included.',
        'Historical execution and pixels are included. All four rows use the validated exact fast bookkeeping helper.')
    s=s.replace('experiments/temporal_jetson_20261023/logs/','experiments/verified_jetson_20261023/logs/')
    s=s.replace("    d=pd.concat(frames,ignore_index=True)",
        "    d=pd.concat(frames,ignore_index=True)\n"
        "    assert set(d.dataset)=={'av2_dev'}\n"
        "    actual=pd.concat([pd.read_pickle(Path(r['run'])/'frames.pkl')[['unit','frame','image_filename','annotation_timestamp_ns','warm']] for r in s['runs'] if not r['smoke']],ignore_index=True).drop_duplicates()\n"
        "    actual.to_csv(OUT/'frame_manifest.csv',index=False)\n"
        "    (OUT/'verified_log_manifest.json').write_text((HERE/'verified_log_manifest.json').read_text())")
    (HERE/'report.py').write_text(s)


if __name__=='__main__':main()
