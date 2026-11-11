"""Build isolated K39-derived baseline/candidate driver. Never execute inference."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE=ROOT/'experiments/k39_jetson_20261027/scripts/89_k39.py'


def once(text,old,new):
    assert text.count(old)==1,(old[:80],text.count(old))
    return text.replace(old,new)


def main():
    text=SOURCE.read_text()
    text=once(text,'    args = ap.parse_args()\n', '''    ap.add_argument('--columns_mode', choices=['baseline','candidate','audit'], default='baseline')
    ap.add_argument('--record_history', action='store_true', help='Debug outputs/lift/TTC/history; invalid for timing')
    args = ap.parse_args()
    assert args.graphs != 'all', 'Dynamic crop graphs remain rejected after K39 illegal memory access'
    sys.path.insert(0, str(ROOT / 'experiments/latency_opt_20261030'))
    import columns_only as CO
    reference_columns = GP.host_columns
    column_checks = [0]
    debug = None
    if args.record_history:
        from debug_audit import Audit
        debug = Audit()
        debug.install(GP, StagedCrops)
    if args.columns_mode == 'candidate':
        CO.install()
    elif args.columns_mode == 'audit':
        def checked_columns(raw, mode, min_conf=None):
            reference = reference_columns(raw, mode, min_conf)
            candidate = CO.host_columns(raw, mode, min_conf)
            assert len(reference) == len(candidate)
            for a, b in zip(reference, candidate):
                assert np.array_equal(a, b, equal_nan=True), 'selected-column mismatch'
            column_checks[0] += 1
            return candidate
        GP.host_columns = checked_columns
''')
    text=once(text,'f"k39_{args.graphs}', 'f"latency_opt_{args.columns_mode}_{args.graphs}')
    text=once(text,'            h["s1"] = np.asarray(xyxy, float).reshape(-1, 4)\n', '''            h["s1"] = np.asarray(xyxy, float).reshape(-1, 4)
            if args.record_history:
                rec[f'parity_history_{name}'] = [None if h.get(k) is None else np.asarray(h[k]).copy() for k in ('s1','s2')]
''')
    text=once(text,'        rows.append(rec)\n', '''        rec['debug_column_calls'] = column_checks[0]
        rec['production_timing_valid'] = not (args.columns_mode == 'audit' or args.record_history or args.parity or args.breakdown)
        rows.append(rec)
''')
    text=once(text,'            return lift_c1(name) if p == "ref" else lift_c1_fast(name, p)\n', '''            original = lift_c1(name) if p == "ref" else lift_c1_fast(name, p)
            if debug is None:
                return original
            def audited(full):
                result = original(full)
                debug.add('lift_C1_TTC', [name, full, result])
                return result
            return audited
''')
    text=once(text,'            return tuple(refined), len(sup)\n', '''            if debug is not None:
                debug.add('merged_scene_refinement', [scene, sup, corr, tuple(refined)])
            return tuple(refined), len(sup)
''')
    text=once(text,'        def start(name):\n', '''        def start(name):
            if debug is not None:
                debug.start()
''')
    text=once(text,'            rec[f"ms_{name}"] = ms\n', '''            rec[f"ms_{name}"] = ms
            if debug is not None:
                rec[f'parity_outputs_{name}'] = debug.finish()
''')
    ast.parse(text)
    dest=HERE/'scripts/91_latency_opt.py';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
    manifest={str(p.relative_to(ROOT)):hashlib.md5(p.read_bytes()).hexdigest() for p in [SOURCE,dest,HERE/'columns_only.py',HERE/'debug_audit.py',Path(__file__).resolve()]}
    (HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Built',dest,'(not run)')


if __name__=='__main__':main()
