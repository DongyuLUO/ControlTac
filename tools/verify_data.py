"""Validate training files, sample counts, force vectors and image separation."""
import argparse
import csv
import json
import math
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--splits',type=Path,default=Path('splits'))
    args=parser.parse_args()
    data={}
    for name in ['force_train','force_pose_train','validation','test']:
        with (args.splits/f'{name}.csv').open(newline='') as f:
            reader=csv.DictReader(f)
            expected={'object','image','force','reference_image' if name=='force_train' else 'mask'}
            assert set(reader.fieldnames)==expected,(name,reader.fieldnames)
            rows=list(reader)
        for row in rows:
            vector=json.loads(row['force'])
            assert len(vector)==3 and all(math.isfinite(x) for x in vector)
            for field in ['image','reference_image' if name=='force_train' else 'mask']:
                if field=='mask' and name!='force_pose_train' and not row[field]: continue
                assert row[field] and (args.data_root/row[field]).is_file(),row[field]
        data[name]=rows
    assert len(data['force_train'])==20000
    assert len(data['force_pose_train'])==len({r['image'] for r in data['force_pose_train']})==7000
    train={r['image'] for r in data['force_train']+data['force_pose_train']}
    val={r['image'] for r in data['validation']}
    test={r['image'] for r in data['test']}
    assert not (train&val or train&test or val&test)
    print(json.dumps({'valid':True,'counts':{k:len(v) for k,v in data.items()}},indent=2))

if __name__=='__main__': main()
