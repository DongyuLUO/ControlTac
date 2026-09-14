"""Validate exact counts, file existence, duplicates, and pose leakage."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--splits',type=Path,default=Path('splits'))
    args = p.parse_args()
    all_rows = {}
    for name in ['force_train','force_pose_train','validation','test']:
        with (args.splits/f'{name}.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if not (args.data_root/row['image']).is_file():
                raise FileNotFoundError(row['image'])
            if name == 'force_pose_train' and not (args.data_root/row['mask']).is_file():
                raise FileNotFoundError(row['mask'])
        all_rows[name] = rows
    assert len(all_rows['force_train']) == 20000
    pos = all_rows['force_pose_train']
    assert len(pos) == len({r['image'] for r in pos}) == 7000
    poses = {(r['object'],r['pose']) for r in pos}
    assert set(Counter(o for o,p in poses).values()) == {300}
    train = {(r['object'],r['pose']) for r in all_rows['force_train']+pos}
    val = {(r['object'],r['pose']) for r in all_rows['validation']}
    test = {(r['object'],r['pose']) for r in all_rows['test']}
    assert not (train&val or train&test or val&test)
    print(json.dumps({'valid':True,'counts':{k:len(v) for k,v in all_rows.items()},
                      'pose_counts':dict(Counter(o for o,p in poses)),
                      'force_unique_images':len({r['image'] for r in all_rows['force_train']})},indent=2))

if __name__ == '__main__':
    main()
