"""Isolated original/fast helper comparison; frozen methods/engines untouched."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'experiments/temporal_jetson_20261023'


def once(s,old,new):
    assert s.count(old)==1,(old[:80],s.count(old))
    return s.replace(old,new)


def main():
    for name in ('helpers.py','recovery_gpu.py','check.py'):
        (HERE/name).write_bytes((OLD/name).read_bytes())
    s=(OLD/'scripts/81_temporal_latency.py').read_text()
    s=s.replace(str(OLD),str(HERE))
    s=once(s,'from helpers import Retro','from helpers import Retro\nfrom fast import FastRetro')
    s=once(s,'    ap.add_argument("--recovery_disabled",action="store_true")',
        '    ap.add_argument("--helper",choices=["original","fast"],required=True)\n'
        '    ap.add_argument("--recovery_disabled",action="store_true")')
    s=once(s,'f"temporal_jetson_{mode}_{args.dataset}_{args.config}"',
        'f"temporal_helper_{args.helper}_{mode}_{args.dataset}_{args.config}"')
    s=once(s,'                    h,chn=RETROS[name].augment(full,(z,lo,hi,ttc),h)',
        "                    h,chn=(RETROS[name].augment(full,(z,lo,hi,ttc),h,known_chains=chn)\n"
        "                           if args.helper=='fast' else RETROS[name].augment(full,(z,lo,hi,ttc),h))")
    s=once(s,'                if is_recovery and not args.recovery_disabled:',
        "                eligible_frame=is_recovery and not args.recovery_disabled and (args.helper=='original' or np.any(np.asarray(P.required_decel(*dd['mono'],v,pp)[1])>=pp.thr_decel))\n"
        '                if eligible_frame:')
    s=once(s,'                    RETROS[nm]=Retro(dd,imgs,trace,historical,chains,v,pp)',
        "                    cls=FastRetro if args.helper=='fast' else Retro\n                    RETROS[nm]=cls(dd,imgs,trace,historical,chains,v,pp)")
    s=once(s,'                mu=traced_many if is_recovery else many_p',
        "                mu=traced_many if is_recovery and (args.helper=='original' or eligible_frame) else many_p")
    s=once(s,'traced_one if is_recovery else one_p',
        "traced_one if is_recovery and (args.helper=='original' or eligible_frame) else one_p")
    out=HERE/'scripts/82_temporal_helper.py';out.parent.mkdir(exist_ok=True);out.write_text(s)
    (HERE/'source_manifest.json').write_bytes((OLD/'source_manifest.json').read_bytes())


if __name__=='__main__':main()
