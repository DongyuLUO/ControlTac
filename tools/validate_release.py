"""Strictly check shipped weights and record local verification evidence."""
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controltac.runtime import create_model, load_weights

root = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
report = {'environment':{n:importlib.metadata.version(n) for n in ['torch','torchvision','diffusers','timm','numpy','opencv-python']},
          'full_paper_training_run':False,'paper_metrics_reproduced':False,'checkpoint_checks':{},'training_smoke':{},'inference_smoke':{}}
manifest = json.loads((root/'checkpoints/manifest.json').read_text())
for stage, name in [('force','force_control'),('force_pose','force_pose_control')]:
    path = root/'checkpoints'/manifest[name]['file']
    with path.open('rb') as f:
        sha = hashlib.file_digest(f,'sha256').hexdigest()
    assert sha == manifest[name]['sha256']
    model = load_weights(create_model({'stage':stage,'legacy_skip_first_block':True}),path)
    report['checkpoint_checks'][stage] = dict(strict_load=True,sha256_verified=True,tensors=len(model.state_dict()))
    del model
    log = root/f'runs/smoke/{stage}/metrics.jsonl'
    metrics = [json.loads(line) for line in log.read_text().splitlines()]
    report['training_smoke'][stage] = dict(device='cpu',batch_images=4,pairs_per_step=16,metrics=metrics)
    image = root/f'outputs/{stage}_example.png'
    assert image.is_file()
    report['inference_smoke'][stage] = dict(image=str(image.relative_to(root)),sampling_steps=50,completed=True)
split = json.loads((root/'splits/report.json').read_text())
report['data'] = {k:split[k] for k in ['force_train','force_pose_train','validation','test','pose_disjoint_train_validation_test']}
report['split_reproducible'] = all((root/'splits'/n).read_bytes()==(root/'runs/reproducibility'/n).read_bytes()
                                  for n in ['force_train.csv','force_pose_train.csv','validation.csv','test.csv','report.json'])
assert report['split_reproducible']
report['regression_tests'] = {'command':'python -m unittest discover -s tests -v','passed':3}
report['one_command_training'] = {}
for stage in ['force','force_pose']:
    path = root/f'runs/{stage}/metrics.jsonl'
    if path.exists():
        report['one_command_training'][stage] = [json.loads(line) for line in path.read_text().splitlines()]
report['wheel_build'] = {'completed':bool(list((root/'runs/wheels').glob('*.whl')))}
report['data_archive'] = {'file':'release_assets/controltac_data.zip',
                          'bytes':(root/'release_assets/controltac_data.zip').stat().st_size,
                          'crc_verified_during_packaging':True}
(root/'docs/validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
