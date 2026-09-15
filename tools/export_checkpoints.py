"""Export tensor-only weights and verify exact equality; never overwrite inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force-checkpoint', type=Path, required=True)
    parser.add_argument('--force-pose-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('checkpoints'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mapping = {'force_control': args.force_checkpoint,
               'force_pose_control': args.force_pose_checkpoint}
    report = {}
    for name, original in mapping.items():
        source = original
        obj = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
        state = obj.get('model_state_dict', obj)
        if not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError('Expected a tensor-only state_dict')
        destination = args.output / (name + '.pth')
        if destination.resolve() == source.resolve():
            raise ValueError('Refusing to overwrite original checkpoint')
        torch.save(state, destination)
        recovered = torch.load(destination, map_location='cpu', weights_only=True, mmap=True)
        assert state.keys() == recovered.keys()
        assert all(torch.equal(state[k], recovered[k]) for k in state)
        sha = hashlib.file_digest(destination.open('rb'), 'sha256').hexdigest()
        report[name] = dict(source=source.name, file=destination.name, source_bytes=source.stat().st_size,
                            output_bytes=destination.stat().st_size, tensors=len(state), sha256=sha,
                            exact_tensor_equality=True)
        print(json.dumps(report[name]), flush=True)
    (args.output/'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
