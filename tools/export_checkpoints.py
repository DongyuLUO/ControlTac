"""Export tensor-only weights and verify exact equality; never overwrite inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('checkpoints'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mapping = {'force_control': 'Only_Force_00_B_phase_2_checkpoint_epoch_65.pth',
               'force_pose_control': 'CN_300_00_phase_2_checkpoint_epoch_60.pth'}
    report = {}
    for name, original in mapping.items():
        source = args.source_root / original
        obj = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
        state = obj.get('model_state_dict', obj)
        if not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError('Expected a tensor-only state_dict')
        destination = args.output / original
        if destination.resolve() == source.resolve():
            raise ValueError('Refusing to overwrite original checkpoint')
        torch.save(state, destination)
        recovered = torch.load(destination, map_location='cpu', weights_only=True, mmap=True)
        assert state.keys() == recovered.keys()
        assert all(torch.equal(state[k], recovered[k]) for k in state)
        sha = hashlib.file_digest(destination.open('rb'), 'sha256').hexdigest()
        report[name] = dict(source=original, file=destination.name, source_bytes=source.stat().st_size,
                            output_bytes=destination.stat().st_size, tensors=len(state), sha256=sha,
                            exact_tensor_equality=True)
        print(json.dumps(report[name]), flush=True)
    (args.output/'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
