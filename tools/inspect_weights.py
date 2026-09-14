import json
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[2]
out = {}
for name in ['Only_Force_00_B_phase_2_checkpoint_epoch_65.pth', 'CN_300_00_phase_2_checkpoint_epoch_60.pth']:
    ckpt = torch.load(root / name, map_location='cpu', weights_only=True, mmap=True)
    state = ckpt.get('model_state_dict', ckpt)
    out[name] = {'keys': list(ckpt), 'tensors': len(state), 'shapes': {k:list(v.shape) for k,v in state.items()}, 'metadata': {k:str(v)[:500] for k,v in ckpt.items() if k not in ['model_state_dict','optimizer_state_dict','scheduler_state_dict','scaler_state_dict']}}
    print(name, list(ckpt), len(state), flush=True)
    print({k:list(v.shape) for k,v in state.items() if 'embedder' in k or 'controlnet.0.before' in k}, flush=True)
(root / 'ControlTac/docs/checkpoint_audit.json').write_text(json.dumps(out, indent=2))
