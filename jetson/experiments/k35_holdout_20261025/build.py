"""K35-4 holdout harness (VUW): isolated copies of the frozen test-path runners. The only changes are:
- the Argoverse-HD unit source: k24_test.units('ahd') reads Argoverse-HD <split>.json and the AV1.1 tracking <split> roots;
  split = env K35_AHD_SPLIT, 'val' = adapter check (reproduces the stored test records), 'train' = the K35-0 holdout;
- the output folders: results/k35/k4/<split>/...;
- R3 runner_low: dataset 'ahd' allowed; the learned adapters use the AV2 models (the AV2 configuration transfers, as
  for Argoverse-HD); rows restricted to v6S / v6S_b2A_retro (v7a) / v6S_b2A_missing_extent_retro (v7b); the dev parity
  against the W1 records is replaced by a parity against k28_p3 on the same unit (done in check.py).
No source, weight, threshold, episode definition or planner change."""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR')
A = B/'scripts/A_method'
R3 = B/'experiments/optimization_r3_20261023'


def once(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:90])
    return s.replace(a, b)


def main():
    # 1. unit source
    s = (A/'k24_test.py').read_text()
    s = once(s, '            for sid, log, names, ids in HDm["seqs"]():\n                L = A1.AV1Log(HDm["AV1"] / "val", log, names)',
             '            for sid, log, names, ids in holdout_seqs(HDm):\n                L = A1.AV1Log(ROOTS[log], log, names)')
    s = once(s, '            yield f"seq{sid:02d}", L, frames',
             '            yield (f"seq{sid:02d}" if SPLIT == "val" else f"hold{sid:02d}"), L, frames')
    s = once(s, 'def units(ds, max_units=0):', '''SPLIT = os.environ["K35_AHD_SPLIT"]
assert SPLIT in ("val", "train"), SPLIT
_AV1 = B / "datasets" / "argoverse1" / "argoverse-tracking"
ROOTS = {p.name: p.parent for s_ in (("val",) if SPLIT == "val" else ("train1", "train2", "train3", "train4"))
         for p in (_AV1 / s_).glob("*") if p.is_dir()}


def holdout_seqs(HDm):
    """= argoverse_hd_eval.seqs() on Argoverse-HD <SPLIT>.json"""
    import json as _json
    d = _json.loads((HDm["HD"] / f"{SPLIT}.json").read_text())
    by = {}
    for im in d["images"]:
        by.setdefault(im["sid"], []).append(im)
    out = []
    for sid, ims in sorted(by.items()):
        ims = sorted(ims, key=lambda x: x["fid"])
        out.append((sid, d["seq_dirs"][sid].split("/")[1], [i["name"] for i in ims], [i["id"] for i in ims]))
    return out


def units(ds, max_units=0):''')
    (HERE/'k24_holdout.py').write_text(s)
    K24 = 'B / "scripts" / "A_method" / "k24_test.py"'
    K24H = f'Path({str(HERE/"k24_holdout.py")!r})'
    OUTK = 'REAL / "results" / "k35" / "k4" / os.environ["K35_AHD_SPLIT"]'
    # 2. k28_p3 (v6-S, cheap, u960 / native + C1 + v3a, tsp8@S, unc2@S; part instr as the stored test rows)
    s = (A/'k28_p3.py').read_text()
    s = once(s, K24, K24H)
    s = once(s, 'od = REAL / "results" / "k28" / "p3" / (', f'od = {OUTK} / "k28p3" / (')
    (HERE/'k28_p3_h.py').write_text(s)
    # 3. k33_eval (AnA iterated, ana05hinfi)
    s = (A/'k33_eval.py').read_text()
    s = once(s, K24, K24H)
    s = once(s, 'od = REAL / "results" / "k33" / (', f'od = {OUTK} / "k33" / (')
    (HERE/'k33_eval_h.py').write_text(s)
    # 4. R3 low runner (v7a, v7b; its own v6-S / cheap recomputation)
    r3 = HERE/'r3'
    r3.mkdir(exist_ok=True)
    for p in R3.glob('*.py'):
        shutil.copyfile(p, r3/p.name)
    shutil.copyfile(R3/'case_manifest_with_severity.json', r3/'case_manifest_with_severity.json')
    if not (r3/'models').exists():
        shutil.copytree(R3/'models', r3/'models')
    s = (R3/'runner_low.py').read_text()
    s = once(s, K24, K24H)
    s = once(s, 'ap.add_argument("--dataset", required=True, choices=["nusc", "av2"])',
             'ap.add_argument("--dataset", required=True, choices=["ahd"])')
    s = once(s, 'ap.add_argument("--split", default="dev", choices=["dev"])',
             'ap.add_argument("--split", default="test", choices=["test"])')
    s = once(s, 'runtime=Runtime(ds,head,crop64,K27.assoc_idx)', "runtime=Runtime('av2',head,crop64,K27.assoc_idx)")
    s = once(s, 'od = REAL / "experiments/optimization_r3_20261023/results" / (args.tag or "dev") / ds',
             f'od = {OUTK} / "r3low" / (args.tag or "main") / ds')
    i, j = s.index("    reference_dir = REAL / 'results/k30/dev_devref' / ds"), s.index('    SETTINGS = {"base": PP0}')
    s = s[:i]+'    import pandas as pd\n'+s[j:]
    s = once(s, 'enumerate(K27.units(ds, args.max_units))', 'enumerate(K24.units(ds, args.max_units))')
    s = once(s, "        assert unit in allowed, ('non-dev unit',unit)\n", '')
    s = once(s, '''            prior=json.loads(fo.with_suffix('.parity.json').read_text())
            assert not any(prior['differences'].values()),prior
''', '')
    s = once(s, "        reference_map = {int(r['frame']):r for r in pickle.load((REAL/'experiments/w1_bounded_20261023/results/dev'/ds/f'{unit}.pkl').open('rb'))}\n", '')
    s = once(s, "for nm in ('v6S','v6S_b2A_retro','v6S_b2A_missing_retro','v6S_b2A_upper_retro','v6S_b2A_missing_extent_retro','v6S_b2A_missing_observed_retro','v6S_b2A_missing_corroborated_retro'):",
             "for nm in ('v6S','v6S_b2A_retro','v6S_b2A_missing_extent_retro'):")
    i, j = s.index('        oldmap=reference_map'), s.index('        # Preserve the failed-unit data')
    s = s[:i]+s[j:]
    s = once(s, '        assert not any(parity[\'differences\'].values()), parity\n', '')
    (r3/'runner_h.py').write_text(s)
    files = {str(p.relative_to(B)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
             [A/'k24_test.py', A/'k28_p3.py', A/'k33_eval.py', R3/'runner_low.py', A/'argoverse_hd_eval.py']}
    files.update({str(p.relative_to(B)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [HERE/'k24_holdout.py', HERE/'k28_p3_h.py', HERE/'k33_eval_h.py', r3/'runner_h.py']})
    (HERE/'source_manifest.json').write_text(json.dumps(files, indent=2)+'\n')
    print('built', HERE)


if __name__ == '__main__':
    main()
