"""Read-only CPU audit of a fixed, already reported development snapshot.

No inference, tuning, or modification of running experiment code/data.
Canonical crop counts estimate repeated tensor inputs, not saved engine calls.
"""
import argparse
import csv
import hashlib
import json
import math
import pickle
from collections import Counter
from pathlib import Path


def canonical(crop, width, height):
    x1, y1 = int(crop[0]), int(crop[1])
    x2 = max(math.ceil(crop[2]), x1 + 8)
    y2 = max(math.ceil(crop[3]), y1 + 8)
    scale = 960 / max(width, height)
    h = max(0, min(y2, height) - y1)
    w = max(0, min(x2, width) - x1)
    nh = min(1024, max(32, math.ceil(h * scale / 32) * 32))
    nw = min(1024, max(32, math.ceil(w * scale / 32) * 32))
    # Keep requested bounds too: they enter postprocessing image shape/offset.
    return x1, y1, x2, y2, nh, nw


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    totals, unit_rows, concentration, sources = [], [], [], []
    for ds, units in manifest.items():
        width, height = (1600, 900) if ds == 'nusc' else (1550, 2048)
        counts = Counter()
        for unit in units:
            file = args.root / 'results/dev' / ds / (unit + '.pkl')
            raw = file.read_bytes()
            sources.append(dict(dataset=ds, unit=unit, sha256=hashlib.sha256(raw).hexdigest()))
            records = [r for r in pickle.loads(raw) if r['gt']]
            u = dict(dataset=ds, unit=unit, frames=len(records), cheap_loss=0.,
                     retro_delta_J=0., close_delta_J=0., recheck_delta_J=0.)
            for r in records:
                trace = r['trace_v6S']
                counts['frames'] += 1
                keys = [canonical(c, width, height) for c in trace['crops']]
                seen = set()
                duplicates = 0
                for key in keys:
                    counts['crop_requests'] += 1
                    counts['crop_tensor_pixels'] += key[-2] * key[-1]
                    if key in seen:
                        duplicates += 1
                        counts['duplicate_requests'] += 1
                        counts['duplicate_tensor_pixels'] += key[-2] * key[-1]
                    seen.add(key)
                counts['frames_with_duplicate_requests'] += bool(duplicates)
                dropped = bool(r['diag_action_drop_v6S'])
                obligation = bool(r['diag_post_field_v6S'] or r['diag_post_support_v6S'])
                required = r['act_gt'] >= 1
                under = required and r['a_v6S|v3a'] == 0
                counts['action_drop_frames'] += dropped
                counts['post_refine_unmet_frames'] += obligation
                counts['action_drop_and_unmet_frames'] += dropped and obligation
                counts['required_frames'] += required
                counts['required_but_keep_frames'] += under
                counts['required_but_keep_and_drop_and_unmet_frames'] += under and dropped and obligation
                u['cheap_loss'] += r['J_cheap']
                for variant in ('retro', 'close', 'recheck'):
                    ref = 'closed' if variant == 'close' else 'v3a'
                    # Positive means the candidate lowers total decision cost.
                    u[variant + '_delta_J'] += r['J_v6S|v3a'] - r[f'J_v6S_{variant}|{ref}']
            unit_rows.append(u)
        total = dict(dataset=ds, units=len(units), **counts)
        total['duplicate_request_share'] = counts['duplicate_requests'] / max(counts['crop_requests'], 1)
        total['duplicate_crop_pixel_share'] = counts['duplicate_tensor_pixels'] / max(counts['crop_tensor_pixels'], 1)
        totals.append(total)
        dd = [u for u in unit_rows if u['dataset'] == ds]
        denominator = sum(u['cheap_loss'] for u in dd)
        for variant in ('retro', 'close', 'recheck'):
            col = variant + '_delta_J'
            ordered = sorted(dd, key=lambda u: u[col], reverse=True)
            top = ordered[0]
            net = sum(u[col] for u in dd)
            positive = sum(max(u[col], 0) for u in dd)
            concentration.append(dict(dataset=ds, variant=variant, units=len(dd),
                delta_G=net / denominator, benefiting_units=sum(u[col] > 1e-9 for u in dd),
                harmed_units=sum(u[col] < -1e-9 for u in dd),
                largest_benefit_unit=top['unit'], largest_benefit_delta_J=top[col],
                largest_share_of_positive_gain=top[col] / positive if positive else 0,
                delta_G_without_largest_benefit_unit=(net-top[col]) / (denominator-top['cheap_loss'])))
    for name, rows in [('crop_and_closure', totals), ('unit_cost_deltas', unit_rows),
                       ('gain_concentration', concentration), ('source_hashes', sources)]:
        write_csv(args.out / (name + '.csv'), rows)
    (args.out / 'SNAPSHOT.json').write_text(json.dumps(dict(
        units=manifest, manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        note='Fixed prior snapshot; nuScenes complete, AV2 116/300. CPU diagnostics only. '
             'Crop counts use VUW baseline traces and Jetson canonicalization; no latency savings measured. '
             'Concentration is descriptive; removing a unit is not a new selection criterion.'), indent=2)+'\n')
    print(json.dumps(dict(crops=totals, concentration=concentration), indent=2))


if __name__ == '__main__':
    main()
