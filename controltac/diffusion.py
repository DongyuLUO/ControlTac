"""Epsilon-prediction diffusion with a corrected deterministic DDIM sampler."""
import torch
import torch.nn.functional as F

class Diffusion:
    def __init__(self, timesteps=1000):
        if timesteps < 20:
            raise ValueError('Training diffusion timesteps must be at least 20')
        self.timesteps = timesteps
        betas = torch.linspace(0.0001 * 1000/timesteps, 0.02 * 1000/timesteps, timesteps)
        self.alpha = (1 - betas).cumprod(0)

    def predict(self, model, noisy, reference, timestep, force, mask=None):
        inputs = torch.cat([noisy, reference], dim=1)
        return model(inputs, timestep, force) if mask is None else model(inputs, timestep, mask, force)

    def loss(self, model, reference, target, force, mask=None):
        t = torch.randint(self.timesteps, (target.shape[0],), device=target.device)
        alpha = self.alpha.to(target.device)[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(target)
        noisy = alpha.sqrt() * target + (1-alpha).sqrt() * noise
        pred = self.predict(model, noisy, reference, t, force, mask).float()
        return 0.5 * F.l1_loss(pred, noise.float()) + 0.5 * F.mse_loss(pred, noise.float())

    @torch.no_grad()
    def sample(self, model, reference, force, mask=None, steps=50, generator=None):
        if not 1 <= steps <= self.timesteps:
            raise ValueError('sampling steps must be in [1, training timesteps]')
        x = torch.randn(reference.shape, device=reference.device, dtype=reference.dtype, generator=generator)
        times = torch.linspace(self.timesteps-1, 0, steps).long().tolist()
        alphas = self.alpha.to(reference.device)
        for i, t in enumerate(times):
            time = torch.full((len(x),), t, dtype=torch.long, device=x.device)
            eps = self.predict(model, x, reference, time, force, mask).float()
            a = alphas[t]
            prev = alphas[times[i+1]] if i+1 < len(times) else x.new_tensor(1.0)
            x0 = (x - (1-a).sqrt() * eps) / a.sqrt()
            x = prev.sqrt() * x0 + (1-prev).sqrt() * eps
        return x
