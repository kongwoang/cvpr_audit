"""Create a runtime-only copy of K34 timing harness, leaving the original untouched."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def once(text, old, new):
    assert text.count(old) == 1, (old[:70], text.count(old))
    return text.replace(old, new)


def main():
    s = (ROOT/'scripts/77_k34_latency.py').read_text()
    s = once(s, 'import cv2\nfrom k34_bounded import bounded_keep',
        'import sys\n'
        f'sys.path.insert(0, {str(ROOT/"scripts")!r})\n'
        f'sys.path.insert(0, {str(HERE)!r})\n'
        'import cv2\nfrom k34_bounded import bounded_keep\nfrom staged import StagedCrops, StagedOneCall')
    s = once(s, 'ROOT = Path(__file__).resolve().parents[1]', f'ROOT = Path({str(ROOT)!r})')
    s = once(s, '    ap.add_argument("--rep", type=int, default=1)',
        '    ap.add_argument("--runtime", choices=["reference","staged"], required=True)\n'
        '    ap.add_argument("--rep", type=int, default=1)')
    s = once(s, 'f"k34_{mode}_{args.dataset}_{args.config}"',
        'f"runtime_{args.runtime}_{mode}_{args.dataset}_{args.config}"')
    s = once(s, '    many = KF.Crops(G, scale)',
        '    many = (KF.Crops(G, scale) if args.runtime == "reference" else\n'
        '            StagedCrops(G,scale,ENG/"yolov8s_dyn.engine",Mode,THR,STATS))')
    s = once(s, '    one = KB.OneCall(G, scale, many.eng)',
        '    one = KB.OneCall(G,scale,many.eng) if args.runtime == "reference" else StagedOneCall(many)')
    path = HERE/'scripts/79_runtime_latency.py'
    path.parent.mkdir(exist_ok=True)
    path.write_text(s)
    files = ['scripts/77_k34_latency.py','scripts/58_large_latency_fast.py','scripts/59_bounded_latency.py',
             'scripts/k34_bounded.py','src/vtf/gpuprep.py','src/vtf/trtdyn.py','src/vtf/c1fast.py',
             'src/vtf/gpupost.py','src/vtf/speculative.py']
    (HERE/'source_manifest.json').write_text(json.dumps(
        {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}, indent=2)+'\n')


if __name__ == '__main__':
    main()
