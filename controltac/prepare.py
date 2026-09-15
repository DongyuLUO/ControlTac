"""Reconstruct reproducible manifests from original FeelAnyForce CSVs."""
import argparse
import ast
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SUBSETS = ['cross7', 'cylinder7', 'cylinder92_1', 'cylinder92_2', 'cylinder142', 'sphere28', 'triple_cylinder7']
FORCE_QUOTAS = [3334, 3333, 1667, 1667, 3333, 3333, 3333]
POSE_QUOTAS = [1167, 1167, 583, 583, 1167, 1167, 1166]
FIELDS = ['object', 'subset', 'image', 'mask', 'pose', 'force', 'source_csv', 'source_row', 'sample_id']

def physical(subset):
    return 'cylinder92' if subset.startswith('cylinder92_') else subset

def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def canonical_pose(value):
    return json.dumps([round(float(v), 10) for v in ast.literal_eval(value)], separators=(',', ':'))

def read_source(root, path, subset):
    rows = []
    for i, row in enumerate(read_rows(root/path)):
        raw = row.get('tactile_nobg') or row.get('tactile')
        if not raw:
            raise ValueError(f'{path}: missing image column')
        image = f'data_all/{physical(subset)}/tactile_nobg/{Path(raw).name}'
        if not (root/image).is_file():
            raise FileNotFoundError(root/image)
        mask = row.get('depth_align', '').replace('\\', '/')
        if mask and not (root/mask).is_file():
            raise FileNotFoundError(root/mask)
        rows.append(dict(object=physical(subset), subset=subset, image=image, mask=mask,
                         pose=canonical_pose(row['indentation_init_pose']),
                         force=json.dumps(ast.literal_eval(row['FT'])[:3]),
                         source_csv=path, source_row=i+2, sample_id=''))
    return rows

def grouped(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['object'], row['pose'])].append(row)
    return groups

def unique(rows):
    return list({r['image']:r for r in reversed(rows)}.values())

def stratified(rows, count, rng, poses=None, allow_repeat=False):
    groups = grouped(unique(rows))
    keys = sorted(groups)
    rng.shuffle(keys)
    if poses is not None:
        if len(keys) < poses:
            raise ValueError(f'Need {poses} poses, found {len(keys)}')
        keys = keys[:poses]
    candidates = {k:list(groups[k]) for k in keys}
    for values in candidates.values():
        rng.shuffle(values)
    selected = []
    # One image per pose per round: preserve pose coverage without replacement.
    while len(selected) < count and any(candidates.values()):
        for key in keys:
            if candidates[key] and len(selected) < count:
                selected.append(candidates[key].pop())
    if len(selected) < count:
        if not allow_repeat:
            raise ValueError(f'Need {count} unique images but only {len(selected)} available')
        originals = list(selected)
        if not originals:
            raise ValueError('Empty candidate pool')
        while len(selected) < count:
            order = list(originals)
            rng.shuffle(order)
            selected.extend(order[:count-len(selected)])
    return [dict(r) for r in selected]

def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({**row, 'sample_id':f'{Path(path).stem}_{i:06d}'})

def build(root, output, seed=42, allow_repeat=False):
    rng = random.Random(seed)
    output.mkdir(parents=True, exist_ok=True)
    force, position, val, test = [], [], [], []
    report = {'seed':seed, 'reconstruction':True, 'allow_repeated_force_samples':allow_repeat,
              'rounding':'Nearest integer allocation; cylinder92 split equally.', 'subsets':{}, 'sources':{}}
    pools, pose_rows = {}, {}
    for subset in SUBSETS:
        obj = physical(subset)
        base = f'data_all/{obj}/'
        primary = base + (f'{obj}_train.csv' if obj != 'cylinder92' else f'{subset}.csv')
        # Full local source is needed to recover deficient training CSVs.
        full = base + (f'{subset}.csv' if obj == 'cylinder92' else f'{obj}.csv')
        posefile = base + f'{subset}_train_pos_{200 if subset == "cylinder92_1" else 300}.csv'
        sources = list(dict.fromkeys([primary, full, posefile]))
        loaded = {p:read_source(root,p,subset) for p in sources}
        for p in sources:
            with (root/p).open('rb') as f:
                report['sources'][p] = hashlib.file_digest(f, 'sha256').hexdigest()
        pools[subset] = unique(loaded[primary] + loaded[full])
        pose_rows[subset] = loaded[posefile]
    # Reserve every annotated training pose, including masks not selected this run.
    reserved = set().union(*(set(grouped(rows)) for rows in pose_rows.values()))
    all_groups = grouped([r for rows in pools.values() for r in rows])
    holdout_val, holdout_test = set(), set()
    for obj in dict.fromkeys(map(physical, SUBSETS)):
        available = sorted(k for k in all_groups if k[0] == obj and k not in reserved)
        rng.shuffle(available)
        n = min(30, len(available)//3)
        if n < 1:
            raise ValueError(f'No independent holdout poses for {obj}')
        holdout_val.update(available[:n])
        holdout_test.update(available[n:2*n])
    for subset, nf, np in zip(SUBSETS, FORCE_QUOTAS, POSE_QUOTAS):
        pool = pools[subset]
        train_pool = [r for r in pool if (r['object'],r['pose']) not in holdout_val|holdout_test]
        selected = stratified(train_pool, nf, rng, allow_repeat=allow_repeat)
        poses = 150 if subset.startswith('cylinder92_') else 300
        used_poses = set(grouped(position))
        annotated = [r for r in pose_rows[subset] if (r['object'],r['pose']) not in used_poses]
        pos = stratified(annotated, np, rng, poses=poses)
        force.extend(selected)
        position.extend(pos)
        val.extend(r for r in pool if (r['object'],r['pose']) in holdout_val)
        test.extend(r for r in pool if (r['object'],r['pose']) in holdout_test)
        report['subsets'][subset] = dict(force_samples=len(selected), force_unique_images=len(unique(selected)),
                                        force_repeated_samples=len(selected)-len(unique(selected)),
                                        pose_samples=len(pos), unique_contact_poses=len(grouped(pos)),
                                        available_force_images=len(train_pool))
    assert len(force) == 20000 and len(position) == 7000
    assert len(unique(position)) == 7000 and len(grouped(position)) == 1800
    train_keys = set(grouped(force+position))
    assert not (train_keys & holdout_val or train_keys & holdout_test or holdout_val & holdout_test)
    for name, rows in [('force_train',force),('force_pose_train',position),('validation',val),('test',test)]:
        write_csv(output/f'{name}.csv', rows)
        report[name] = dict(samples=len(rows),unique_images=len(unique(rows)),poses=len(grouped(rows)))
    report['pose_disjoint_train_validation_test'] = True
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report

def compute_normalization(root, output):
    # The maintainer confirmed one fixed profile is shared by all objects/stages.
    profile = Path(__file__).with_name('normalization.json')
    (output/'normalization.json').write_text(profile.read_text(encoding='utf-8'), encoding='utf-8')
    print('Shared normalization saved', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='Directory containing data_all/')
    parser.add_argument('--output', type=Path, default=Path('splits'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--allow-repeated-force-samples', action='store_true')
    parser.add_argument('--skip-normalization', action='store_true')
    args = parser.parse_args()
    report = build(args.data_root.resolve(), args.output, args.seed, args.allow_repeated_force_samples)
    print(json.dumps(report['subsets'], indent=2), flush=True)
    if not args.skip_normalization:
        compute_normalization(args.data_root.resolve(), args.output)

if __name__ == '__main__':
    main()
