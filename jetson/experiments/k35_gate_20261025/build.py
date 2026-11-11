"""K35-4b gate harness: an isolated copy of the R3 tradeoff Jetson harness (FAST_RECOVERY helper, verified AV2 CFG-DEV
manifest) with (1) rows v6-S / B2K-A / v7a = B2K-A + recovery / v7b = v7a + image-point fallback + critical-extent
retention (= VUW R3 low 'v6S_b2A_missing_extent_retro'), (2) --runtime reference|staged: RUNTIME_20261023 crop-buffer
staging for every row. No deployed source, weight, threshold or engine changes."""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT/'experiments/optimization_r3_tradeoff_jetson_20261024'
STG = ROOT/'experiments/runtime_20261023/staged.py'


def once(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:80])
    return s.replace(a, b)


def main():
    for p in SRC.glob('*.py'):
        if p.name not in ('window.py', 'check.py'):
            shutil.copyfile(p, HERE/p.name)
    for n in ('MANIFEST.json', 'verified_log_manifest.json'):
        shutil.copyfile(SRC/n, HERE/n)
    shutil.copyfile(STG, HERE/'staged.py')
    if not (HERE/'models').exists():
        shutil.copytree(SRC/'models', HERE/'models')
    s = (SRC/'scripts/85_tradeoff.py').read_text()
    s = s.replace(str(SRC), str(HERE))
    assert str(SRC) not in s
    s = once(s, 'from compact import CompactRuntime\n', 'from compact import CompactRuntime\nfrom staged import StagedCrops, StagedOneCall\n')
    s = once(s, '    ap.add_argument("--rep", type=int, default=1)',
             '    ap.add_argument("--runtime", choices=["reference", "staged"], required=True)\n'
             '    ap.add_argument("--rep", type=int, default=1)')
    s = once(s, 'f"r3_tradeoff_{args.adapter_encoder}_{mode}_{args.dataset}_{args.config}"',
             'f"k35_gate_{args.runtime}_{args.helper}_{mode}_{args.dataset}_{args.config}"')
    s = once(s, '    many = KF.Crops(G, scale)',
             '    many = (KF.Crops(G, scale) if args.runtime == "reference" else\n'
             '            StagedCrops(G, scale, ENG / "yolov8s_dyn.engine", Mode, THR, STATS))')
    s = once(s, '    one = KB.OneCall(G, scale, many.eng)',
             '    one = KB.OneCall(G, scale, many.eng) if args.runtime == "reference" else StagedOneCall(many)')
    s = once(s, 'variants = [("ours",None),("ours_b2A_retro",2),("ours_b2A_missing_retro",2),("ours_b2A_missing_compact_retro",2),("ours_b2A_extent_retro",2),("ours_b2A_missing_corroborated_retro",2)]',
             'variants = [("ours",None),("ours_b2A",2),("ours_b2A_retro",2),("ours_b2A_missing_extent_retro",2)]')
    out = HERE/'scripts/86_k35_gate.py'
    out.parent.mkdir(exist_ok=True)
    out.write_text(s)
    src = json.loads((SRC/'source_manifest.json').read_text())
    src['experiments/runtime_20261023/staged.py'] = hashlib.sha256(STG.read_bytes()).hexdigest()
    src['experiments/optimization_r3_tradeoff_jetson_20261024/scripts/85_tradeoff.py'] = hashlib.sha256((SRC/'scripts/85_tradeoff.py').read_bytes()).hexdigest()
    (HERE/'source_manifest.json').write_text(json.dumps(src, indent=2)+'\n')
    print('built', out)


if __name__ == '__main__':
    main()
