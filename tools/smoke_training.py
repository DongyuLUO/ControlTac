"""Exercise real model training for one optimizer step per stage on CPU."""
import json
import argparse
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controltac.prepare import read_rows, write_csv

repo = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-root', type=Path, required=True, help='Directory containing data_all/')
args = parser.parse_args()
out = repo/'runs/smoke'
out.mkdir(parents=True,exist_ok=True)
for stage in ['force','force_pose']:
    rows = [r for r in read_rows(repo/f'splits/{stage}_train.csv') if r['object']=='Cross']
    manifest = out/f'{stage}_data.csv'
    write_csv(manifest, rows[:8], force_pairing=stage != 'force_pose')
    config = json.loads((repo/f'configs/{stage}.json').read_text())
    config.update(steps=1,manifest=str(manifest),normalization=str(repo/'examples/normalization.json'),output=str(out/stage))
    path = out/f'{stage}.json'
    path.write_text(json.dumps(config))
    command = [sys.executable,'-m','controltac.train','--config',str(path),'--data-root',str(args.data_root.resolve()),
               '--device','cpu','--codec-device','cpu','--local-only','--save-training-state']
    if stage == 'force_pose':
        command += ['--initialize-from',str(repo/'checkpoints/force_control.pth')]
    subprocess.run(command,cwd=repo,check=True)
