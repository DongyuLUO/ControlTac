"""Inspect local recovery inputs without changing them (stdlib only)."""
import ast
import csv
import json
import argparse
from pathlib import Path
from collections import Counter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, required=True, help='Directory containing original data_all/ and data/')
args = parser.parse_args()
root = args.source_root.resolve()
result = {}
for obj in ['cross7', 'cylinder7', 'cylinder92', 'cylinder142', 'sphere28', 'triple_cylinder7']:
    for p in sorted((root / 'data_all' / obj).glob('*train*.csv')):
        if any(s in p.stem for s in ['aug', 'TacForce', 'supp', 'xyr', 'ref', 'only', 'con']):
            continue
        with p.open(newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        poses = Counter(r.get('indentation_init_pose') for r in rows)
        masks = Counter(r.get('depth_align') for r in rows)
        result[str(p.relative_to(root))] = dict(rows=len(rows), images=len(set(r.get('tactile_nobg') for r in rows)), poses=len(poses), masks=len(masks), min_per_pose=min(poses.values()), columns=list(rows[0]), example=rows[0])
for p, r in result.items():
    print(p, {k:r[k] for k in ['rows','images','poses','masks','min_per_pose']})
