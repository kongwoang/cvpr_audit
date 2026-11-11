"""Fixed-context C1 TRT32 graph feasibility only; run in an exclusive window."""
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from vtf.c1fast import _TRTStatic, BUCKETS

ROOT=Path(__file__).resolve().parents[2]


def measure(fn,n=100):
    for _ in range(10):fn()
    torch.cuda.synchronize();out=[]
    for _ in range(n):
        torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize()
        out.append((time.perf_counter()-t)*1000)
    return out


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0);rows=[];timings=[]
    for b in BUCKETS:
        runner=_TRTStatic(ROOT/'engines'/f'c1enc_fp32_b{b}.engine',b)
        # Each bucket already owns an independent fixed-shape context and buffers.
        runner.inp.copy_(torch.rand_like(runner.inp))
        for _ in range(10):runner.run()
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):runner.run()
        torch.cuda.synchronize()
        mismatches=0;maxdiff=0.
        for _ in range(20):
            runner.inp.copy_(torch.rand_like(runner.inp));runner.run();reference=runner.out.clone()
            graph.replay();torch.cuda.synchronize()
            mismatches+=int(not torch.equal(reference,runner.out))
            maxdiff=max(maxdiff,float((reference-runner.out).abs().max()))
        assert mismatches==0,(b,mismatches,maxdiff)
        for rep in (1,2,3):
            modes=('plain','graph') if rep%2 else ('graph','plain')
            result={}
            for mode in modes:
                ts=measure(runner.run if mode=='plain' else graph.replay)
                result[mode]=ts
                timings.extend(dict(bucket=b,rep=rep,mode=mode,iteration=i,ms=v) for i,v in enumerate(ts))
            x,y=[np.asarray(result[m]) for m in ('plain','graph')]
            rows.append(dict(bucket=b,rep=rep,plain_mean_ms=x.mean(),graph_mean_ms=y.mean(),plain_p95_ms=np.quantile(x,.95),graph_p95_ms=np.quantile(y,.95),mean_saved_ms=x.mean()-y.mean(),mean_reduction_percent=100*(x.mean()-y.mean())/x.mean(),output_mismatches=mismatches,max_abs_diff=maxdiff,scope='isolated encoder execution; no full-pipeline latency claim'))
        del graph,runner
        torch.cuda.empty_cache()
    for name,data in [('summary.csv',rows),('samples.csv',timings)]:
        with (args.output/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (args.output/'protocol.json').write_text(json.dumps(dict(seed=0,buckets=list(BUCKETS),parity_inputs_per_bucket=20,warmup_per_timing=10,samples_per_rep=100,repetitions=3,end_to_end=False,contexts='one independent fixed context per bucket; no dynamic crop graph'),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
