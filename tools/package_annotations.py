"""Package aligned masks and an image verification index, without tactile images."""
import csv
import hashlib
import json
from pathlib import Path
import argparse
import zipfile
import cv2

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--splits',type=Path,default=Path('splits'))
    p.add_argument('--output',type=Path,default=Path('release_assets/controltac_annotations.zip'))
    args = p.parse_args()
    images, masks = set(),set()
    for name in ['force_train','force_pose_train','validation','test']:
        with (args.splits/f'{name}.csv').open(newline='') as f:
            for row in csv.DictReader(f):
                images.add(row['image'])
                if row['mask']:
                    masks.add(row['mask'])
    index = {}
    for i,name in enumerate(sorted(images)):
        image = cv2.imread(str(args.data_root/name),cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(name)
        index[name] = {'shape':list(image.shape),'bgr_sha256':hashlib.sha256(image.tobytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(masks):
            z.write(args.data_root/name,name)
        z.writestr('image_index.json',json.dumps(index,sort_keys=True))
        z.writestr('ATTRIBUTION.txt','Source tactile images: FeelAnyForce (CC BY 4.0), https://huggingface.co/datasets/amirsh1376/FeelAnyForce\nImages are referenced, not redistributed in this archive.\nAligned contact masks and reconstructed manifests: ControlTac.\n')
    with zipfile.ZipFile(args.output) as z:
        assert z.testzip() is None
    with args.output.open('rb') as f:
        checksum = hashlib.file_digest(f,'sha256').hexdigest()
    (args.splits/'annotation_manifest.json').write_text(json.dumps({'file':args.output.name,
        'sha256':checksum,'bytes':args.output.stat().st_size,'images_referenced':len(images),'masks':len(masks)},indent=2))
    print(json.dumps({'images_referenced':len(images),'masks':len(masks),'bytes':args.output.stat().st_size,'archive':str(args.output)}))

if __name__=='__main__':
    main()
