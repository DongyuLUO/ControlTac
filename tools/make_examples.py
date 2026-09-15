"""Recover two self-contained, documented inference examples from local sources."""
import csv
import json
import argparse
import shutil
from pathlib import Path
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, required=True, help='Directory containing original data_all/ and data/')
args = parser.parse_args()
root = args.source_root.resolve()
out = Path(__file__).resolve().parents[1]/'examples'
assets = out/'assets'
assets.mkdir(exist_ok=True)
rows = list(csv.DictReader((root/'data_all/cross7/cross7_train_pos_300.csv').open(newline='')))
group = [r for r in rows if r['indentation_init_pose'] == rows[0]['indentation_init_pose']]
group.sort(key=lambda r:abs(json.loads(r['FT'])[2]))
reference, target = group[0], group[-1]
for source, name in [(reference['tactile_nobg'],'reference.png'), (target['tactile_nobg'],'target_residual.png'),
                      (target['depth_align'],'contact_mask.npy'), ('data/cross7/tactile/background.png','background.png')]:
    shutil.copy2(root/source,assets/name)
stats = torch.load(root/'data_all/cross7/minmax_cross7_train.pt',map_location='cpu',weights_only=True)
(out/'normalization.json').write_text(json.dumps({'shared':{k:v.flatten().tolist() for k,v in stats.items()}},indent=2))
for stage, filename, checkpoint in [('force','force.json','force_control.pth'),('force_pose','force_pose.json','force_pose_control.pth')]:
    example = dict(stage=stage, checkpoint='../checkpoints/'+checkpoint,
                   reference='assets/reference.png', background='assets/background.png',
                   initial_force=json.loads(reference['FT'])[:3],target_force=json.loads(target['FT'])[:3],
                   normalization='normalization.json')
    if stage == 'force_pose':
        example['mask']='assets/contact_mask.npy'
    (out/filename).write_text(json.dumps(example,indent=2))
(out/'provenance.json').write_text(json.dumps(dict(reference=reference,target=target,
    normalization_source='data_all/cross7/minmax_cross7_train.pt',
    note='The maintainer confirmed these normalization values are shared by all objects and both stages.'),indent=2))
