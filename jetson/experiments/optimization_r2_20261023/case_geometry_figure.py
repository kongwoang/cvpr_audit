"""Explain the measured case with the frozen consumer, not a fitted curve."""
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rap import planner as P

class Compat(pickle.Unpickler):
    def find_class(self,module,name):
        return super().find_class(module.replace('numpy._core','numpy.core'),name)

root=Path(__file__).resolve().parents[2]
out=root/'docs/paper/handoff/OPTIMIZATION_R2_20261023'
case=out/'cases/04_av2_refiner_turns_braking_to_keep'
with (case/'records.pkl').open('rb') as f:r=Compat(f).load()[-1]
rec=r['record'];tr=rec['trace_v6S'];idx=r['decisive_match'];g=r['ground_truth'][r['decisive_gt_index']]
v,ttc=rec['v'],tr['raw'][3][idx];pp=P.PlannerParams()
z=np.linspace(40,80,801)
ego=v*v/(2*np.maximum(z-pp.standoff-pp.reaction_t*v,.05))
closing=(z/ttc)**2/(2*np.maximum(z-pp.standoff,.05))
fig,ax=plt.subplots(figsize=(8.2,4.5))
ax.plot(z,ego,label='Ego stopping term',color='#4878b5')
ax.plot(z,closing,label='Closing term at stored C1 TTC',color='#d66c34')
ax.plot(z,np.maximum(ego,closing),color='#222222',lw=2.5,label='Required demand = max of both')
ax.axhline(pp.thr_decel,color='gray',ls='--',label='DECELERATE threshold')
for zz,label,col in [(tr['raw'][0][idx],'raw 52.15 m','#ab4499'),(tr['refined'][0][idx],'refined 64.19 m','#259b78'),(float(g['long_near']),'GT range 72.81 m','#347fc4')]:
    dd=float(P.required_decel(np.array([zz]),np.array([-.5]),np.array([.5]),np.array([ttc]),v,pp)[0])
    ax.scatter([zz],[dd],color=col,zorder=5)
    ax.annotate(label,(zz,dd),xytext=(0,16 if col!='#259b78' else -22),textcoords='offset points',ha='center',color=col)
ax.set(xlabel='Object range (m)',ylabel='Demand (m/s²)',ylim=(.65,1.55),
       title='Closer range accuracy can still cross the wrong action threshold')
ax.text(.02,.97,f'AV2 development case 04, frame 75; ego {v:.2f} m/s\nC1 TTC {ttc:.3f} s; GT TTC {float(g["ttc"]):.3f} s',
        transform=ax.transAxes,va='top',fontsize=9)
ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
fig.savefig(out/'range_decision_valley.png',dpi=180);fig.savefig(out/'range_decision_valley.pdf')
print('Saved range_decision_valley PNG/PDF; GT-range marker holds C1 TTC fixed, not full GT demand')
