"""Exercise real model training for one optimizer step per stage on CPU."""
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controltac.prepare import read_rows, write_csv

repo = Path(__file__).resolve().parents[1]
out = repo/'runs/smoke'
out.mkdir(parents=True,exist_ok=True)
rows = [r for r in read_rows(repo/'splits/force_pose_train.csv') if r['subset']=='cross7']
write_csv(out/'data.csv',rows[:8])
for stage in ['force','force_pose']:
    config = json.loads((repo/f'configs/{stage}.json').read_text())
    config.update(steps=1,manifest=str(out/'data.csv'),normalization=str(repo/'examples/normalization.json'),output=str(out/stage))
    path = out/f'{stage}.json'
    path.write_text(json.dumps(config))
    command = [sys.executable,'-m','controltac.train','--config',str(path),'--data-root',str(repo.parent),
               '--device','cpu','--codec-device','cpu','--local-only','--save-training-state']
    if stage == 'force_pose':
        command += ['--initialize-from',str(repo/'checkpoints/Only_Force_00_B_phase_2_checkpoint_epoch_65.pth')]
    subprocess.run(command,cwd=repo,check=True)
