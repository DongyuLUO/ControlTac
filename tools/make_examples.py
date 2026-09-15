"""Select measured examples with Fz in [-10, -1] N."""
import argparse
import csv
import json
import math
import shutil
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, required=True)
args = parser.parse_args()
root = args.source_root.resolve()
out = Path(__file__).resolve().parents[1]/'examples'
assets = out/'assets'
source_csv = 'data_all/cross7/cross7_train_pos_300.csv'
with (root/source_csv).open(newline='',encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
def force(row): return json.loads(row['FT'])[:3]
def pose(row): return json.loads(row['indentation_init_pose'])
valid = [r for r in rows if -10 <= force(r)[2] <= -1]
group = [r for r in valid if pose(r) == pose(valid[0])]
reference = min(group, key=lambda r: abs(force(r)[2]+2))
force_target = min(group, key=lambda r: abs(force(r)[2]+8))
reference_mask = np.load(root/reference['depth_align'], allow_pickle=False)
candidates = []
for row in valid:
    distance = math.dist(pose(reference)[:2], pose(row)[:2])
    if 0.002 <= distance <= 0.005 and pose(row)[2:] == pose(reference)[2:]:
        candidates.append((abs(force(row)[2]+8)+100*abs(distance-0.003), row))
candidates.sort(key=lambda item: item[0])
pose_target = next(row for _,row in candidates if not np.array_equal(reference_mask,np.load(root/row['depth_align'],allow_pickle=False)))
provenance = {'object':'Cross','force_units':'N','position_units':'m','fz_range':[-10,-1],'examples':{}}
def record(row):
    return {'source_csv':source_csv,'source_row':rows.index(row)+2,'image':row['tactile_nobg'],
            'mask':row['depth_align'],'force':force(row),'contact_pose':pose(row)}
for stage,target in [('force',force_target),('force_pose',pose_target)]:
    assert force(reference) != force(target)
    assert (pose(reference) == pose(target)) == (stage == 'force')
    folder = assets/stage
    folder.mkdir(parents=True,exist_ok=True)
    for source,name in [(reference['tactile_nobg'],'reference.png'),(target['tactile_nobg'],'target_residual.png'),
                        (reference['depth_align'],'reference_mask.npy'),(target['depth_align'],'target_mask.npy')]:
        shutil.copy2(root/source,folder/name)
    example = dict(stage=stage,checkpoint=f'../checkpoints/{stage}_control.pth',
                   reference=f'assets/{stage}/reference.png',background='assets/background.png',
                   initial_force=force(reference),target_force=force(target),normalization='normalization.json')
    if stage == 'force_pose': example['mask']=f'assets/{stage}/target_mask.npy'
    (out/f'{stage}.json').write_text(json.dumps(example,indent=2)+'\n',encoding='utf-8')
    provenance['examples'][stage] = {'reference':record(reference),'target':record(target),
        'pose_changed':stage=='force_pose','position_change_m':math.dist(pose(reference)[:2],pose(target)[:2])}
shutil.copy2(root/'data/cross7/tactile/background.png',assets/'background.png')
shutil.copy2(out.parent/'controltac/normalization.json',out/'normalization.json')
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
for name in ['reference.png','target_residual.png','contact_mask.npy']:
    obsolete=assets/name
    if obsolete.is_file(): obsolete.unlink()
print(json.dumps(provenance,indent=2))
