"""Joint state inside the acquisition loop, isolated from support-only adapter runs."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
def once(s,a,b):
    assert s.count(a)==1,(s.count(a),a[:100]);return s.replace(a,b)
def main():
    s=(HERE/'runner_learned.py').read_text()
    s=once(s,"('_missing','_allgeo','_alldec','_upper','_query')","('_missing','_allgeo','_alldec','_upper','_query','_liftgeo','_liftdec')")
    s=once(s,"objective='decision' if '_alldec' in name else 'geometry'","objective='decision' if ('_alldec' in name or '_liftdec' in name) else 'geometry'")
    s=once(s,"                        out[3][i]=min(1/rate,1e3) if rate>1e-3/dt else np.inf",
        "                        out[3][i]=min(1/rate,1e3) if rate>1e-3/dt else np.inf\n"
        "                        if '_lift' in name:\n"
        "                            for geometry in out[:3]:geometry[i]*=float(pred[i,1])")
    s=once(s,"                adapted=any(tag in nm for tag in ('_allgeo','_alldec','_query'))",
        "                lifted='_lift' in nm\n                if lifted:refs=('none',)\n"
        "                adapted=any(tag in nm for tag in ('_allgeo','_alldec','_query'))")
    s=once(s,"                tr = sc['mono'] if close else refine(sc,'adapt' if adapted else 'v3a',brake_sup(PP0))",
        "                tr = sc['mono'] if (close or lifted) else refine(sc,'adapt' if adapted else 'v3a',brake_sup(PP0))")
    s=once(s,"('v6S','v6S_b2A_retro','v6S_b2A_missing_retro','v6S_b2A_allgeo_retro','v6S_b2A_alldec_retro','v6S_b2A_upper_retro','v6S_b2A_query_retro')",
        "('v6S','v6S_b2A_retro','v6S_b2A_liftgeo_retro','v6S_b2A_liftdec_retro','v6S_b2A_upper_extent_retro')")
    (HERE/'runner_lift.py').write_text(s)
    print('Built joint-lift runner')
if __name__=='__main__':main()
