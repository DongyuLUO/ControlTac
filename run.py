"""Single command entry point for the two-stage training pipeline or examples."""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    train = sub.add_parser('train',help='Train force, then force+pose with its learned weights')
    train.add_argument('--data-root',type=Path,required=True)
    train.add_argument('--device',default='cuda')
    train.add_argument('--codec-device',default='cpu')
    train.add_argument('--local-only',action='store_true')
    train.add_argument('--max-steps',type=int)
    infer = sub.add_parser('infer',help='Run one of two self-contained examples')
    infer.add_argument('--stage',choices=['force','force_pose'],default='force')
    infer.add_argument('--device',default='cuda')
    infer.add_argument('--codec-device',default='cpu')
    infer.add_argument('--local-only',action='store_true')
    args = parser.parse_args()
    common = ['--device',args.device,'--codec-device',args.codec_device]
    if args.local_only:
        common += ['--local-only']
    if args.command == 'infer':
        subprocess.run([sys.executable,'-m','controltac.infer','--example',f'examples/{args.stage}.json',
                        '--output',f'outputs/{args.stage}.png',*common],cwd=repo,check=True)
    else:
        root = args.data_root.resolve()
        if args.max_steps:
            common += ['--max-steps',str(args.max_steps)]
        for stage in ['force','force_pose']:
            command = [sys.executable,'-m','controltac.train','--config',f'configs/{stage}.json',
                       '--data-root',str(root),*common]
            if stage == 'force_pose':
                command += ['--initialize-from','runs/force/model.pth']
            subprocess.run(command,cwd=repo,check=True)

if __name__ == '__main__':
    main()
