# ported directly from LucidRains: 
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281

import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

import numpy as np
from tqdm import tqdm
from einops import rearrange


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)



class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, timesteps=1000, loss_type='l1', betas = None):
        super().__init__()
        self.denoise_fn = denoise_fn

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod.to('cuda'), t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod.to('cuda'), t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1.to('cuda'), t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2.to('cuda'), t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance.to('cuda'), t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped.to('cuda'), t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond, clip_denoised: bool):
        #print("x device:",x.device, "t device:", t.device, "cond device:", cond.device)
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t, cond))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        # print("x_recon device:",x_recon.device, "x device", x.device, "t device", t.device)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        assert x.dim() == 2, f"Expected x to have 2 dimensions (batch_size, num_features), but got {x.dim()} dimensions."

        b, num_feats, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, n_samples, input_dim, cond, unscale = True):
        device = next(self.denoise_fn.parameters()).device

        # Initialize sample with random noise
        #print(device)

        sample = torch.randn((n_samples, input_dim), device=device)

        #print("sample device:",sample.device, "cond device:", cond.device)
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((n_samples,), i, dtype=torch.long, device=device)
            sample = self.p_sample(sample, 
                                   t, 
                                   cond
                                   ).to(device)
        # updating the sample one-by-one

        # DIRC unscaling: 

        if unscale:
            assert input_dim >= 3

            x = unscale_inputs(sample[:,0].flatten(),STATS['x_max'],STATS['x_min'])
            y = unscale_inputs(sample[:,1].flatten(),STATS['y_max'],STATS['y_min'])
            t = unscale_inputs(sample[:,2].flatten(),STATS['time_max'],STATS['time_min'])

            x = _set_to_closest(x, ALLOWED_X, device)
            y = _set_to_closest(y, ALLOWED_Y, device)

            sample = torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1) 

        return sample
    
    @torch.no_grad()
    def sample(self, cond, n_samples, input_dim, unscale = True): 
        return self.p_sample_loop(n_samples=n_samples, input_dim=input_dim, cond=cond, unscale=unscale)
    
    # Resampling specific for FastDIRC
    @torch.no_grad()
    def resample(self, cond, n_samples, input_dim):
        # Resampling with the DIRC detector.
        initial = self.sample(
                            cond, 
                            n_samples, 
                            input_dim,
                            )     
        updated = _apply_mask(initial, STATS)
        n_resample = n_samples - len(updated)
        while n_resample != 0:
            resampled = self.sample(
                                cond, 
                                n_resample, 
                                input_dim,
                                )
            updated = torch.concat((updated, resampled), 0)
            updated = _apply_mask(updated, STATS)
            n_resample = n_samples - len(updated)

        x = _set_to_closest(updated[:,0],ALLOWED_X,cond.device)
        y = _set_to_closest(updated[:,1],ALLOWED_Y,cond.device)
        t = updated[:,2]

        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond, noise = None):
        b, s = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, cond, *args, **kwargs):
        b, *_, device = *x.shape, x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, cond, *args, **kwargs)


