"""Create a portable data archive for the checked-in manifests."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--splits',type=Path,default=Path('splits'))
    p.add_argument('--output',type=Path,default=Path('release_assets/controltac_data.zip'))
    args = p.parse_args()
    files = set()
    for name in ['force_train','force_pose_train','validation','test']:
        with (args.splits/f'{name}.csv').open(newline='') as f:
            for row in csv.DictReader(f):
                files.add(row['image'])
                if row.get('mask'):
                    files.add(row['mask'])
    root = args.data_root.resolve()
    for name in files:
        target = (root/name).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f'Unsafe or missing data path: {name}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for name in sorted(files):
            path = root/name
            with path.open('rb') as f:
                hashes[name] = hashlib.file_digest(f,'sha256').hexdigest()
            archive.write(path,name)
        archive.writestr('data_manifest.json',json.dumps(hashes,indent=2))
        archive.writestr('DATA_ATTRIBUTION.txt','Tactile images: FeelAnyForce, amirsh1376, CC BY 4.0.\nhttps://huggingface.co/datasets/amirsh1376/FeelAnyForce\nLocal derivatives: background removal, contact masks, reconstructed selection.\n')
    with zipfile.ZipFile(args.output) as archive:
        bad = archive.testzip()
        if bad:
            raise IOError(f'Archive CRC failure: {bad}')
    print(json.dumps(dict(archive=str(args.output),files=len(files),bytes=args.output.stat().st_size)))

if __name__ == '__main__':
    main()
