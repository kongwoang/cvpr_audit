"""Wait for owned collection jobs, then evaluate and report on one CPU thread."""
import argparse,json,subprocess,sys,time
from common import HERE,save
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--shard',default='0/1');a=p.parse_args();ds=a.dataset
    while True:
        states=[]
        for i in (0,1):
            f=HERE/'jobs'/('collect_'+ds+'_'+str(i)+'of2.json')
            states.append(json.loads(f.read_text()).get('status') if f.exists() else 'waiting')
        save(HERE/'reports'/ds/'WAIT_STATUS.json',dict(collect=states,updated=time.time()))
        if 'failed' in states:raise RuntimeError(('collection failed',states))
        if states==['complete','complete']:break
        time.sleep(15)
    subprocess.run([sys.executable,str(HERE/'evaluate.py'),'--dataset',ds],check=True)
    subprocess.run([sys.executable,str(HERE/'analyze.py'),'--stage','dev','--dataset-only',ds,'--out-name',ds],check=True)
    subprocess.run([sys.executable,str(HERE/'audit.py'),'--dataset',ds],check=True)
    save(HERE/'reports'/ds/'DONE.json',dict(status='complete',dataset=ds))
if __name__=='__main__':main()
