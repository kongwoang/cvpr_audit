import pickle,glob,numpy as np,csv,sys,collections
S=[r for f in sorted(glob.glob('results/k41/d07/score/*.pkl')) for r in pickle.load(open(f,'rb'))]
O=[r for f in sorted(glob.glob('results/k41/superseded/d07_score_shared_cache_proc/*.pkl')) for r in pickle.load(open(f,'rb'))]
print('tokens',len(S),'logs',len({r['log'] for r in S}))
d=np.array([r['p02_base_dist'] for r in S]); print('P02 base dist >1e-6:',(d>1e-6).sum(),' >0.5m:',(d>0.5).sum(), 'tokens',[r['token'][:8]+' %.2f'%r['p02_base_dist'] for r in S if r['p02_base_dist']>1e-6])
for row in ('cheap','full960','singleton'):
    m=[(r['token'],r['scores'][row]['score'],r['stored'][row]) for r in S if row in r['stored']]
    bad=[x for x in m if abs(x[1]-x[2])>1e-9]
    print(row,'vs stored pdms_final: n',len(m),'differ',len(bad),' mean diff %.5f'%np.mean([x[1]-x[2] for x in m]),[ (t[:8],round(a,3),round(b,3)) for t,a,b in bad[:6]])
om={r['token']:r for r in O}
nd=sum(any(abs(r['scores'][k]['score']-om[r['token']]['scores'][k]['score'])>1e-9 for k in r['scores']) for r in S if r['token'] in om)
print('fresh vs shared-cache D07 (overlap',sum(r['token'] in om for r in S),'): tokens with any score diff',nd)
ROWS=('cheap','full960','singleton','k2','k2all','geometric')
logs=sorted({r['log'] for r in S}); L={l:i for i,l in enumerate(logs)}
X={k:np.array([r['scores'][k]['score'] for r in S]) for k in ROWS}; A={k:np.array([r['agree_full'][k] for r in S]) for k in ROWS}
g=np.array([L[r['log']] for r in S]); rng=np.random.default_rng(0)
def boot(v):
    s=np.zeros(len(logs)); c=np.zeros(len(logs)); np.add.at(s,g,v); np.add.at(c,g,1); bs=[]
    for _ in range(10000):
        i=rng.integers(0,len(logs),len(logs)); bs.append(s[i].sum()/c[i].sum())
    return np.percentile(bs,[2.5,97.5])
out=[]
for k in ROWS:
    lo,hi=boot(X[k]-X['cheap']); lo2,hi2=boot(X[k]-X['full960'])
    px=np.mean([r['px'].get(k,np.nan) for r in S]) if k in ('singleton','geometric','k2','k2all') else np.nan
    out.append(dict(row=k,pdms=X[k].mean(),d_vs_cheap=(X[k]-X['cheap']).mean(),ci_lo=lo,ci_hi=hi,d_vs_full=(X[k]-X['full960']).mean(),ci2_lo=lo2,ci2_hi=hi2,agree_full=A[k].mean(),px_mean=px))
    print('%-9s PDMS %.4f  d_cheap %+.4f [%+.4f,%+.4f]  d_full %+.4f [%+.4f,%+.4f]  agree_full %.4f  px %.0f'%(k,X[k].mean(),out[-1]['d_vs_cheap'],lo,hi,out[-1]['d_vs_full'],lo2,hi2,A[k].mean(),px))
print('k2==singleton all tokens:',all(r['scores']['k2']==r['scores']['singleton'] and r['scores']['k2all']==r['scores']['singleton'] for r in S))
w=csv.DictWriter(open(sys.argv[1],'w',newline=''),fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
w=csv.writer(open(sys.argv[2],'w',newline='')); w.writerow(['log','token','p02_base_dist']+[f'stored_{k}' for k in ('cheap','full960','singleton')]+[f'fresh_{k}' for k in ('cheap','full960','singleton')])
for r in S: w.writerow([r['log'],r['token'],r['p02_base_dist']]+[r['stored'].get(k) for k in ('cheap','full960','singleton')]+[r['scores'][k]['score'] for k in ('cheap','full960','singleton')])
