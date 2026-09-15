"""Generate a tactile image from a reference image and target force."""
import argparse
import json
from pathlib import Path
import cv2
import torch
from .diffusion import Diffusion
from .runtime import (read_config, create_model, load_weights, load_autoencoder,
                      read_rgb, read_mask, normalize, denormalize, LATENT_SCALE, AE_ID)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--example', type=Path, help='Portable example JSON, paths relative to its parent')
    p.add_argument('--stage', choices=['force','force_pose'])
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--model-config', type=Path, help='Training config for newly trained weights')
    p.add_argument('--reference', type=Path)
    p.add_argument('--background', type=Path)
    p.add_argument('--mask', type=Path)
    p.add_argument('--initial-force', nargs=3, type=float)
    p.add_argument('--target-force', nargs=3, type=float)
    p.add_argument('--normalization', type=Path)
    p.add_argument('--subset', help='Key in the normalization JSON; selects preprocessing statistics, not a model class condition')
    p.add_argument('--output', type=Path, default=Path('outputs/generated.png'))
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--codec-device', default='cpu')
    p.add_argument('--ae', default=AE_ID)
    p.add_argument('--local-only', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.example:
        example = read_config(args.example)
        for k in ['stage','initial_force','target_force','subset']:
            if getattr(args,k) is None:
                setattr(args,k,example.get(k))
        for k in ['checkpoint','reference','background','mask','normalization','model_config']:
            if getattr(args,k) is None and example.get(k):
                setattr(args,k,args.example.parent/example[k])
    for k in ['stage','checkpoint','reference','background','initial_force','target_force','normalization','subset']:
        if getattr(args,k) is None:
            p.error(f'--{k.replace("_","-")} is required (or use --example)')
    if args.stage == 'force_pose' and args.mask is None:
        p.error('force_pose requires --mask')
    if args.stage == 'force' and args.mask is not None:
        p.error('--mask is only supported by the force_pose stage')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    # Shipped legacy weights require their historical backbone traversal.
    config = read_config(args.model_config) if args.model_config else {'stage':args.stage,'legacy_skip_first_block':True}
    if config['stage'] != args.stage:
        p.error('Model config stage mismatch')
    model = load_weights(create_model(config), args.checkpoint).to(args.device).eval()
    normalization = read_config(args.normalization)
    if args.subset not in normalization:
        p.error(f'Unknown normalization key {args.subset!r}; available: {", ".join(normalization)}')
    stats = normalization[args.subset]
    reference = normalize(read_rgb(args.reference), stats)[None]
    ae = load_autoencoder(args.codec_device, args.ae, args.local_only)
    with torch.inference_mode():
        latent = ae.encode(reference.to(args.codec_device)).latent.mul(LATENT_SCALE).to(args.device)
        mask = None
        if args.mask:
            mask = ae.encode(read_mask(args.mask)[None].to(args.codec_device)).latent.mul(LATENT_SCALE).to(args.device)
        force = torch.tensor(args.target_force, device=args.device) - torch.tensor(args.initial_force, device=args.device)
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        output = Diffusion(config.get('timesteps', 1000)).sample(model, latent, force[None], mask, args.steps, generator)
        decoded = ae.decode((output/LATENT_SCALE).to(args.codec_device)).sample.cpu()[0]
        rgb = (denormalize(decoded, stats) + read_rgb(args.background)).clamp(0,1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image = (rgb.permute(1,2,0).numpy()*255).round().astype('uint8')
    if not cv2.imwrite(str(args.output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f'Failed to write {args.output}')
    metadata = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    metadata['model_config_used'] = config
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(args.output.resolve())

if __name__ == '__main__':
    main()
