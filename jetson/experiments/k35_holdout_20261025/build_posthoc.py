"""K35-4 consequence of a pass: v7a once, POST-HOC, on the three frozen test splits (nuScenes new test 228 / AV2 val 150
with the K10 ego mask / Argoverse-HD val 24). Isolated copy of R3 runner_low with: the frozen test unit source
(k24_test.units, unchanged), dataset choices nusc / av2 / ahd, split 'test', the learned adapter models of the dataset
(av2 for ahd; not used by these rows), rows restricted to v6S and v6S_b2A_retro (v7a), output results/k35/k4/posthoc/;
the dev parity against W1 records is replaced by posthoc_check.py (cheap / v6-S vs the stored K28 instr test records)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = Path('/vol/dl-nguyenb5-solar/users/hoangpc/CVPR')
R3 = B/'experiments/optimization_r3_20261023'


def once(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:90])
    return s.replace(a, b)


def main():
    s = (R3/'runner_low.py').read_text()
    s = once(s, 'ap.add_argument("--dataset", required=True, choices=["nusc", "av2"])',
             'ap.add_argument("--dataset", required=True, choices=["nusc", "av2", "ahd"])')
    s = once(s, 'ap.add_argument("--split", default="dev", choices=["dev"])',
             'ap.add_argument("--split", default="test", choices=["test"])')
    s = once(s, 'runtime=Runtime(ds,head,crop64,K27.assoc_idx)', "runtime=Runtime('av2' if ds=='ahd' else ds,head,crop64,K27.assoc_idx)")
    s = once(s, 'od = REAL / "experiments/optimization_r3_20261023/results" / (args.tag or "dev") / ds',
             'od = REAL / "results" / "k35" / "k4" / "posthoc" / "r3low" / ds')
    i, j = s.index("    reference_dir = REAL / 'results/k30/dev_devref' / ds"), s.index('    SETTINGS = {"base": PP0}')
    s = s[:i]+"    import pandas as pd\n    if ds == 'av2':\n        assert os.environ.get('K10_MASK_Y0') == '1830', 'fixed AV2 mask required'\n"+s[j:]
    s = once(s, 'enumerate(K27.units(ds, args.max_units))', 'enumerate(K24.units(ds, args.max_units))')
    s = once(s, "        assert unit in allowed, ('non-dev unit',unit)\n", '')
    s = once(s, '''            prior=json.loads(fo.with_suffix('.parity.json').read_text())
            assert not any(prior['differences'].values()),prior
''', '')
    s = once(s, "        reference_map = {int(r['frame']):r for r in pickle.load((REAL/'experiments/w1_bounded_20261023/results/dev'/ds/f'{unit}.pkl').open('rb'))}\n", '')
    s = once(s, "for nm in ('v6S','v6S_b2A_retro','v6S_b2A_missing_retro','v6S_b2A_upper_retro','v6S_b2A_missing_extent_retro','v6S_b2A_missing_observed_retro','v6S_b2A_missing_corroborated_retro'):",
             "for nm in ('v6S','v6S_b2A_retro'):")
    i, j = s.index('        oldmap=reference_map'), s.index('        # Preserve the failed-unit data')
    s = s[:i]+s[j:]
    s = once(s, "        assert not any(parity['differences'].values()), parity\n", '')
    (HERE/'r3'/'runner_t.py').write_text(s)
    print('built runner_t.py')


if __name__ == '__main__':
    main()
