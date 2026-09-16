"""Validate distributed weights and run regression tests without writing reports."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import torch
root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from controltac.runtime import create_model,load_weights
torch.set_num_threads(4)
manifest=json.loads((root/'checkpoints/manifest.json').read_text())
for stage in ['force','force_pose']:
    info=manifest[stage+'_control']
    path=root/'checkpoints'/info['file']
    with path.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==info['sha256']
    model=load_weights(create_model({'stage':stage,'legacy_skip_first_block':True}),path)
    print(stage+': SHA-256 and strict model loading passed',flush=True)
    del model
subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=root,check=True)
