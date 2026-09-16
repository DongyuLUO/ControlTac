"""Inspect tensor shapes in explicitly supplied checkpoint files."""
import json
import argparse
from pathlib import Path
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkpoints', type=Path, nargs='+')
parser.add_argument('--output', type=Path, help='Optional destination; prints results by default')
args = parser.parse_args()
out = {}
for path in args.checkpoints:
    name = path.name
    ckpt = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    state = ckpt.get('model_state_dict', ckpt)
    out[name] = {'keys': list(ckpt), 'tensors': len(state), 'shapes': {k:list(v.shape) for k,v in state.items()}, 'metadata': {k:str(v)[:500] for k,v in ckpt.items() if k not in ['model_state_dict','optimizer_state_dict','scheduler_state_dict','scaler_state_dict']}}
    print(name, list(ckpt), len(state), flush=True)
    print({k:list(v.shape) for k,v in state.items() if 'embedder' in k or 'controlnet.0.before' in k}, flush=True)
if args.output:
    args.output.write_text(json.dumps(out, indent=2))
else:
    print(json.dumps(out, indent=2))
