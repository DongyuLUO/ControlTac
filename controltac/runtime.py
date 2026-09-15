"""Shared model, codec, and strict checkpoint loading."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch

AE_ID = 'mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers'
LATENT_SCALE = 0.1633485559287139

def read_config(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def create_model(config):
    from .models.force_control import ForceControlTransformer
    from .models.force_pose_control import ForcePoseControlTransformer
    kwargs = dict(input_size=(10, 8), patch_size=1, in_channels=64,
                  hidden_size=768, depth=12, num_heads=12, mlp_ratio=4,
                  ffn_type='glumbconv', attn_type='linear', use_pe=True, force_norm=False)
    if config['stage'] == 'force':
        return ForceControlTransformer(**kwargs)
    model = ForcePoseControlTransformer(**kwargs, copy_blocks_num=6)
    model.legacy_skip_first_block = config.get('legacy_skip_first_block', False)
    return model

def load_weights(model, path):
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(state.get('model_state_dict', state), strict=True)
    return model

def load_autoencoder(device, path=AE_ID, local_only=False):
    from diffusers import AutoencoderDC
    ae = AutoencoderDC.from_pretrained(path, local_files_only=local_only)
    return ae.eval().requires_grad_(False).to(device)

def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(cv2.resize(image, (320, 256)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255

def read_mask(path):
    if Path(path).suffix.lower() == '.npy':
        mask = np.load(path, allow_pickle=False).astype(np.float32).squeeze()
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        mask = mask.astype(np.float32) / 255
    if mask.ndim != 2 or not np.isfinite(mask).all():
        raise ValueError(f'Invalid contact mask: {path}')
    # Preserve historical masks and bilinear interpolation; no per-image renormalization.
    mask = cv2.resize(mask, (320, 256), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(mask.copy())[None].repeat(3, 1, 1)

def normalize(image, stats):
    low = image.new_tensor(stats['min']).view(3, 1, 1)
    high = image.new_tensor(stats['max']).view(3, 1, 1)
    return 2 * (image - 127/255 - low) / (high - low).clamp_min(1e-8) - 1

def denormalize(image, stats):
    low = image.new_tensor(stats['min']).view(3, 1, 1)
    high = image.new_tensor(stats['max']).view(3, 1, 1)
    return (image + 1) * 0.5 * (high - low) + low
