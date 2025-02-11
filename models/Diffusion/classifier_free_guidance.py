import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.amp import autocast

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

import numpy as np

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

# classifier free guidance functions

def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# Deleted building block modules

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class CFGDiffusion(nn.Module):
    def __init__(
        self,
        model,
        stats,
        *,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 1.,
        offset_noise_strength = 0.,
        min_snr_loss_weight = False,
        min_snr_gamma = 5,
        use_cfg_plus_plus = False # https://arxiv.org/pdf/2406.08070
    ):
        super().__init__()

        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.input_dim = self.model.input_dim

        # ------------ hpDIRC constants ------------

        _allowed_x = torch.tensor(np.array([  3.65625   ,   6.96875   ,  10.28125   ,  13.59375   ,
                                                   16.90625   ,  20.21875   ,  23.53125   ,  26.84375   ,
                                                   30.15625   ,  33.46875   ,  36.78125   ,  40.09375   ,
                                                   43.40625   ,  46.71875   ,  50.03125   ,  53.34375   ,
                                                   62.54841111,  65.86091111,  69.17341111,  72.48591111,
                                                   75.79841111,  79.11091111,  82.42341111,  85.73591111,
                                                   89.04841111,  92.36091111,  95.67341111,  98.98591111,
                                                   102.29841111, 105.61091111, 108.92341111, 112.23591111,
                                                   121.44057223, 124.75307223, 128.06557223, 131.37807223,
                                                   134.69057223, 138.00307223, 141.31557223, 144.62807223,
                                                   147.94057223, 151.25307223, 154.56557223, 157.87807223,
                                                   161.19057223, 164.50307223, 167.81557223, 171.12807223,
                                                   180.33273334, 183.64523334, 186.95773334, 190.27023334,
                                                   193.58273334, 196.89523334, 200.20773334, 203.52023334,
                                                   206.83273334, 210.14523334, 213.45773334, 216.77023334,
                                                   220.08273334, 223.39523334, 226.70773334, 230.02023334,
                                                   239.22489446, 242.53739446, 245.84989446, 249.16239446,
                                                   252.47489446, 255.78739446, 259.09989446, 262.41239446,
                                                   265.72489446, 269.03739446, 272.34989446, 275.66239446,
                                                   278.97489446, 282.28739446, 285.59989446, 288.91239446,
                                                   298.11705557, 301.42955557, 304.74205557, 308.05455557,
                                                   311.36705557, 314.67955557, 317.99205557, 321.30455557,
                                                   324.61705557, 327.92955557, 331.24205557, 334.55455557,
                                                   337.86705557, 341.17955557, 344.49205557, 347.80455557]))
        _allowed_y = torch.tensor(np.array([  3.65625   ,   6.96875   ,  10.28125   ,  13.59375   ,
                                                   16.90625   ,  20.21875   ,  23.53125   ,  26.84375   ,
                                                   30.15625   ,  33.46875   ,  36.78125   ,  40.09375   ,
                                                   43.40625   ,  46.71875   ,  50.03125   ,  53.34375   ,
                                                   62.01339286,  65.32589286,  68.63839286,  71.95089286,
                                                   75.26339286,  78.57589286,  81.88839286,  85.20089286,
                                                   88.51339286,  91.82589286,  95.13839286,  98.45089286,
                                                   101.76339286, 105.07589286, 108.38839286, 111.70089286,
                                                   120.37053571, 123.68303571, 126.99553571, 130.30803571,
                                                   133.62053571, 136.93303571, 140.24553571, 143.55803571,
                                                   146.87053571, 150.18303571, 153.49553571, 156.80803571,
                                                   160.12053571, 163.43303571, 166.74553571, 170.05803571,
                                                   178.72767857, 182.04017857, 185.35267857, 188.66517857,
                                                   191.97767857, 195.29017857, 198.60267857, 201.91517857,
                                                   205.22767857, 208.54017857, 211.85267857, 215.16517857,
                                                   218.47767857, 221.79017857, 225.10267857, 228.41517857]))
        
        self.stats = stats

        self.photons_generated = 0
        self.photons_resampled = 0
        self.gapx =  1.89216111455965 + 4.
        self.gapy = 1.3571428571428572 + 4.
        self.pixel_width = 3.3125
        self.pixel_height = 3.3125

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # use cfg++ when ddim sampling

        self.use_cfg_plus_plus = use_cfg_plus_plus

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        
        register_buffer('_allowed_x', _allowed_x)
        register_buffer('_allowed_y', _allowed_y)

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - 0.1 was claimed ideal

        self.offset_noise_strength = offset_noise_strength

        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

    # --------- hpDIRC util functions --------- 
    
    def unscale(self,x,max_,min_):
        return x*0.5*(max_ - min_) + min_ + (max_-min_)/2

    def unscale_conditions(self,x,max_,min_):
        return x * (max_ - min_) + max_

    def set_to_closest(self, x, allowed):
        x = x.unsqueeze(1)  # Adding a dimension to x for broadcasting
        diffs = torch.abs(x - allowed.to(self.device).float())
        closest_indices = torch.argmin(diffs, dim=1)
        closest_values = allowed[closest_indices]
        return closest_values

    def set_to_closest_2d(self,hits):
        allowed_pairs = torch.cartesian_prod(self._allowed_x.to(self.device).float(), self._allowed_y.to(self.device).float()) 

        diffs = hits.unsqueeze(1) - allowed_pairs 
        distances = torch.norm(diffs, dim=2) 

        closest_indices = torch.argmin(distances, dim=1)
        closest_values = allowed_pairs[closest_indices]

        return closest_values[:,0].detach().cpu(),closest_values[:,1].detach().cpu()

    # -----------------------------------------

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance.to(self.device), t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped.to(self.device), t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, cond, cond_scale = 1.5, rescaled_phi = 0., clip_x_start = False):
        model_output, model_output_null = self.model.forward_with_cond_scale(x, t, cond, cond_scale = cond_scale, rescaled_phi = rescaled_phi)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output if not self.use_cfg_plus_plus else model_output_null

            x_start = self.predict_start_from_noise(x, t, model_output)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            x_start_for_pred_noise = x_start if not self.use_cfg_plus_plus else maybe_clip(model_output_null)

            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)

            x_start_for_pred_noise = x_start
            if self.use_cfg_plus_plus:
                x_start_for_pred_noise = self.predict_start_from_v(x, t, model_output_null)
                x_start_for_pred_noise = maybe_clip(x_start_for_pred_noise)

            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, cond, cond_scale, rescaled_phi, clip_denoised = True):
        preds = self.model_predictions(x, t, cond, cond_scale, rescaled_phi)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, cond, cond_scale = 1.5, rescaled_phi = 0., clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, cond = cond, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_sample = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_sample, x_start

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, cond_scale = 1.5, rescaled_phi = 0.):
        batch, device = shape[0], self.betas.device

        sample = torch.randn(shape, device=device) # of size (n_samples, input_dim)
        
        x_start = None

        for t in reversed(range(0, self.num_timesteps)):
            sample, x_start = self.p_sample(sample, t, cond, cond_scale, rescaled_phi)

        # Add hpDIRC resampling later
        return sample

    @torch.no_grad()
    def ddim_sample(self, cond, shape, cond_scale = 1.5, rescaled_phi = 0., clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        sample = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(sample, time_cond, cond, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_x_start = clip_denoised)

            if time_next < 0:
                sample = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(sample)

            sample = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise # DDPM noising

        # Add hpDIRC resampling later
        return sample

    @torch.no_grad()
    def sample(self, cond, n_samples, cond_scale = 1.5, rescaled_phi = 0.):
        batch_size, input_dim = cond.shape[0], self.input_dim
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(cond, (n_samples, input_dim), cond_scale, rescaled_phi)
        # our batch size here is n_samples per track for dirc sampling

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += self.offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond, noise = None):
        b, input_dim = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # predict and take gradient step

        model_out = self.model(x, t, cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        # loss = loss * extract(self.loss_weight, t, loss.shape)

        return loss.mean()

    def forward(self, x, cond, *args, **kwargs):
        b, *_, device = *x.shape, x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(x, t, cond, *args, **kwargs)
    
    # --------------- hpDIRC sampling ----------------

    def __get_track(self,n_samples, cond):
        batch_cond = cond.repeat(n_samples, 1)
        samples = self.sample(
                            batch_cond, 
                            n_samples, 
                            cond_scale=1.1,
                            rescaled_phi=0.
                            )
        x = self.unscale(samples[:,0].flatten(),self.stats['x_max'],self.stats['x_min'])#.round()
        y = self.unscale(samples[:,1].flatten(),self.stats['y_max'],self.stats['y_min'])#.round()
        t = self.unscale(samples[:,2].flatten(),self.stats['time_max'],self.stats['time_min'])

        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1)

    def _apply_mask(self, hits):
        # Time > 0 
        mask = torch.where((hits[:,2] > 0) & (hits[:,2] < self.stats['time_max']))
        hits = hits[mask]
        # Outter bounds
        mask = torch.where((hits[:, 0] > self.stats['x_min']) & (hits[:, 0] < self.stats['x_max']) & (hits[:, 1] > self.stats['y_min']) & (hits[:, 1] < self.stats['y_max']))[0] # Acceptance mask
        hits = hits[mask]

        return hits

    def create_tracks(self,num_samples,context,plotting=False): # resampling logic
        hits = self.__get_track(num_samples,context)
        updated_hits = self._apply_mask(hits)
        n_resample = int(num_samples - len(updated_hits))
        

        self.photons_generated += len(hits)
        self.photons_resampled += n_resample
        while n_resample != 0:
            resampled_hits = self.__get_track(n_resample,context)
            updated_hits = torch.concat((updated_hits,resampled_hits),0)
            updated_hits = self._apply_mask(updated_hits)
            n_resample = int(num_samples - len(updated_hits))
            self.photons_resampled += n_resample
            self.photons_generated += len(resampled_hits)
            

        # Use euclidean distance
        x,y = self.set_to_closest_2d(updated_hits[:,:-1])
        #x,y = updated_hits[:,0].detach().cpu(),updated_hits[:,1].detach().cpu()
        t = updated_hits[:,2].detach().cpu()

        pmtID = torch.div(x,torch.tensor(58,dtype=torch.int),rounding_mode='floor') + torch.div(y, torch.tensor(58,dtype=torch.int),rounding_mode='floor') * 6
        col = (1.0/self.pixel_width) * (x - 2 - self.pixel_width/2. - (pmtID%6)*self.gapx)
        row = (1.0/self.pixel_height) * (y - 2 - self.pixel_height/2. - self.gapy * torch.div(pmtID,torch.tensor(6,dtype=torch.int),rounding_mode='floor'))

        assert(len(row) == num_samples)
        assert(len(col) == num_samples)
        assert(len(pmtID) == num_samples)

        P = self.unscale_conditions(context[0][0].detach().cpu().numpy(),self.stats['P_max'],self.stats['P_min'])
        Theta = self.unscale_conditions(context[0][1].detach().cpu().numpy(),self.stats['theta_max'],self.stats['theta_min'])
        Phi = 0.0

        if not plotting:
            return {"NHits":num_samples,"P":P,"Theta":Theta,"Phi":Phi,"x":x.numpy(),"y":y.numpy(),"leadTime":t.numpy(),"pmtID":pmtID.numpy()}
        else:
            return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1)

