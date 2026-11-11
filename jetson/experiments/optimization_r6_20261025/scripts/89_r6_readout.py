"""Jetson component microbenchmark only; see runtime_protocol.md."""
import json,pickle,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from bounded_readout import readout
class Compat(pickle.Unpickler):
    def find_class(self,module,name):return super().find_class(module.replace('numpy._core','numpy.core'),name)
def main():
    out=HERE/'reports/jetson';out.mkdir(parents=True,exist_ok=True);records=[];mismatch=[];inputs={}
    for ds in ('nusc','av2'):
        with (HERE/'runtime_inputs'/(ds+'.pkl')).open('rb') as f:inputs[ds]=Compat(f).load()
        warmed=0
        for unit in inputs[ds]:
            hist=[]
            for r in unit['rows']:
                for lag in (1,2):readout(r['scene'],r['proposal'],hist,r['v'],r['dt'],r['scale'],lag)
                hist=(hist+[dict(scene=r['scene'],v=r['v'])])[-2:];warmed+=1
                if warmed>=200:break
            if warmed>=200:break
    for rep in (1,2,3):
        for ds in ('nusc','av2'):
            for unit in inputs[ds]:
                hist=[]
                for r in unit['rows']:
                    for lag in ((1,2) if (rep+r['frame'])%2 else (2,1)):
                        start=time.perf_counter_ns();extra=readout(r['scene'],r['proposal'],hist,r['v'],r['dt'],r['scale'],lag);ms=(time.perf_counter_ns()-start)/1e6
                        act=max(r['r5_action'],extra)
                        if extra!=r['expected_extra'][lag-1] or act!=r['expected_final'][lag-1]:mismatch.append(dict(dataset=ds,unit=unit['unit'],frame=r['frame'],rep=rep,lag=lag,actual=act,expected=r['expected_final'][lag-1]))
                        records.append(dict(dataset=ds,unit=unit['unit'],frame=r['frame'],rep=rep,lag=lag,ms=ms,extra_action=extra,raises_r5=act>r['r5_action']))
                    hist=(hist+[dict(scene=r['scene'],v=r['v'])])[-2:]
            print(ds,rep,'complete',flush=True)
    pd.DataFrame(records).to_csv(out/'frames.csv',index=False)
    (out/'parity.json').write_text(json.dumps(dict(status='complete' if not mismatch else 'failed',evaluations=len(records),mismatches=mismatch,scope='CPU readout on stored VUW observations, not TensorRT or full-pipeline parity'),indent=2)+'\n')
    assert not mismatch
if __name__=='__main__':main()
