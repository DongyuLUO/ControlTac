"""Connect extracted FeelAnyForce background-subtracted images to ControlTac splits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile
import cv2

def safe_members(z,root):
    for entry in z.infolist():
        path = (root/entry.filename).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f'Unsafe archive path: {entry.filename}')
        yield entry

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True,help='Extracted FeelAnyForce tree (contains object/tactile_nobg folders)')
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('data'))
    p.add_argument('--copy',action='store_true',help='Copy instead of linking; requires additional disk space')
    args = p.parse_args()
    source,output = args.source_root.resolve(),args.output.resolve()
    manifest_path = Path(__file__).resolve().parents[1]/'splits/annotation_manifest.json'
    if manifest_path.exists():
        expected_archive = json.loads(manifest_path.read_text())
        with args.annotations.open('rb') as f:
            digest = hashlib.file_digest(f,'sha256').hexdigest()
        if digest != expected_archive['sha256']:
            raise ValueError('Annotation archive checksum mismatch; use the release matching this checkout.')
    output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.annotations) as z:
        entries = list(safe_members(z,output))
        index = json.loads(z.read('image_index.json'))
        # Extract only our masks and metadata; never overwrite existing tactile images.
        if any(e.filename not in ['image_index.json','ATTRIBUTION.txt'] and not e.filename.endswith('.npy') for e in entries):
            raise ValueError('Expected an annotation-only archive')
        z.extractall(output,members=entries)
    folders = {}
    for name in index:
        parts = Path(name).parts
        obj = parts[1]
        if obj in folders:
            continue
        candidates = [source/'data_all'/obj/'tactile_nobg',source/obj/'tactile_nobg',source/'dataset'/obj/'tactile_nobg']
        found = [d for d in candidates if d.is_dir()]
        if not found:
            found = [d for d in source.rglob('tactile_nobg') if d.parent.name==obj]
        if len(found)!=1:
            raise ValueError(f'Expected exactly one {obj}/tactile_nobg folder; found {len(found)}. Point --source-root at the extracted dataset root.')
        folders[obj] = found[0]
    count = 0
    for name,expected in sorted(index.items()):
        relative = Path(name)
        src = folders[relative.parts[1]]/relative.name
        image = cv2.imread(str(src),cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(src)
        if list(image.shape)!=expected['shape'] or hashlib.sha256(image.tobytes()).hexdigest()!=expected['bgr_sha256']:
            raise ValueError(f'Pixel mismatch: {src}. Do not substitute raw tactile images or silently change preprocessing.')
        dest = output/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            old = cv2.imread(str(dest),cv2.IMREAD_COLOR)
            if old is None or list(old.shape)!=expected['shape'] or hashlib.sha256(old.tobytes()).hexdigest()!=expected['bgr_sha256']:
                raise ValueError(f'Existing destination differs: {dest}')
        elif args.copy:
            shutil.copyfile(src,dest)
        else:
            try:
                os.link(src,dest)
            except OSError:
                try:
                    dest.symlink_to(src)
                except OSError as error:
                    raise OSError('Cannot link across these disks. Use --copy or place --output on the same volume.') from error
        count += 1
    print(json.dumps({'verified_images':count,'data_root':str(output),'mode':'copy' if args.copy else 'link'}))

if __name__=='__main__':
    main()
