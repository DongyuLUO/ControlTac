"""Train either stage with paper defaults and optional full-state resumption."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
import torch
from .diffusion import Diffusion
from .prepare import read_rows
from .runtime import (read_config, create_model, load_weights, load_autoencoder,
                      read_rgb, read_mask, normalize, LATENT_SCALE, AE_ID)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--initialize-from', help='Stage-one tensor weights for stage two')
    p.add_argument('--resume', help='Full training state saved with --save-training-state')
    p.add_argument('--save-training-state', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--codec-device', default='cpu', help='CPU saves GPU memory; use cuda on >=16 GB GPUs')
    p.add_argument('--ae', default=AE_ID)
    p.add_argument('--local-only', action='store_true')
    p.add_argument('--max-steps', type=int, help='Override for a short smoke run')
    p.add_argument('--pair-microbatch', type=int, default=1, help='Accumulate all 16 pairs before optimizer.step')
    p.add_argument('--save-every', type=int, default=5000)
    p.add_argument('--output', type=Path)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    cfg = read_config(args.config)
    if args.max_steps is not None:
        if args.max_steps < 1:
            p.error('--max-steps must be positive')
        cfg['steps'] = args.max_steps
    if args.pair_microbatch < 1:
        p.error('--pair-microbatch must be positive')
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    rng = random.Random(cfg['seed'])
    rows = read_rows(cfg['manifest'])
    stats = read_config(cfg['normalization'])
    groups = defaultdict(list)
    for row in rows:
        key = (row['object'], row['reference_image']) if cfg['stage'] == 'force' else (row['object'],)
        groups[key].append(row)
    keys = sorted(groups)
    # Group weights preserve manifest sample proportions, including explicit repeats.
    weights = [len(groups[k]) for k in keys]
    output = args.output or Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=True)
    (output/'config.json').write_text(json.dumps(cfg, indent=2))
    (output/'normalization.json').write_text(json.dumps(stats, indent=2))
    model = create_model(cfg)
    if cfg['stage'] == 'force' and args.initialize_from:
        load_weights(model, args.initialize_from)
    if cfg['stage'] == 'force_pose':
        if not args.initialize_from and not args.resume:
            p.error('Stage two requires --initialize-from or --resume')
        if args.initialize_from:
            base = create_model({'stage':'force'})
            load_weights(base, args.initialize_from)
            result = model.load_state_dict(base.state_dict(), strict=False)
            if result.unexpected_keys or any(not k.startswith(('controlnet.','x_con_embedder.')) for k in result.missing_keys):
                raise ValueError(f'Incompatible stage-one weights: {result}')
            del base
        model.initialize_all()
        model.x_con_embedder.requires_grad_(True)
    model.to(args.device).train()
    parameters = [v for v in model.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['steps'], eta_min=cfg['min_learning_rate'])
    amp = args.device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        if state['config'] != cfg:
            raise ValueError('Resume configuration must match saved configuration')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        rng.setstate(state['random_state'])
        torch.set_rng_state(state['torch_rng'])
        if amp and state.get('cuda_rng'):
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        start = state['step']
    ae = load_autoencoder(args.codec_device, args.ae, args.local_only)
    # Codec construction can consume RNG; restore after all modules are created.
    if args.resume:
        rng.setstate(state['random_state'])
        torch.set_rng_state(state['torch_rng'])
        if amp and state.get('cuda_rng'):
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    diffusion = Diffusion(cfg['timesteps'])
    for step in range(start, cfg['steps']):
        key = rng.choices(keys, weights=weights, k=1)[0]
        candidates = groups[key]
        batch = rng.sample(candidates, cfg['batch_size']) if len(candidates) >= cfg['batch_size'] else rng.choices(candidates, k=cfg['batch_size'])
        images = torch.stack([normalize(read_rgb(args.data_root/r['image']), stats['shared']) for r in batch])
        with torch.no_grad():
            latent = ae.encode(images.to(args.codec_device)).latent.mul(LATENT_SCALE).to(args.device)
            masks = None
            if cfg['stage'] == 'force_pose':
                mask_images = torch.stack([read_mask(args.data_root/r['mask']) for r in batch])
                masks = ae.encode(mask_images.to(args.codec_device)).latent.mul(LATENT_SCALE).to(args.device)
        forces = torch.tensor([json.loads(r['force']) for r in batch], device=args.device)
        pairs = [(i,j) for i in range(len(batch)) for j in range(len(batch))]
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for offset in range(0, len(pairs), args.pair_microbatch):
            chunk = pairs[offset:offset+args.pair_microbatch]
            i, j = zip(*chunk)
            i, j = list(i), list(j)
            with torch.autocast(device_type='cuda' if amp else 'cpu', enabled=amp):
                loss = diffusion.loss(model, latent[i], latent[j], forces[j]-forces[i], None if masks is None else masks[j])
                weighted = loss * len(chunk)/len(pairs)
            if not torch.isfinite(weighted):
                raise FloatingPointError(f'Non-finite loss at step {step+1}')
            scaler.scale(weighted).backward()
            total_loss += weighted.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, cfg['max_grad_norm'], error_if_nonfinite=True)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        record = dict(step=step+1, loss=total_loss, learning_rate=scheduler.get_last_lr()[0])
        with (output/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        if (step+1) % args.save_every == 0 or step+1 == cfg['steps']:
            torch.save(model.state_dict(), output/'model.pth')
            if args.save_training_state:
                torch.save(dict(config=cfg, step=step+1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), random_state=rng.getstate(),
                                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if amp else []), output/'training_state.pt')

if __name__ == '__main__':
    main()
