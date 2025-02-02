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
    def __init__(self, 
                 denoise_fn, 
                 stats,
                 timesteps=1000, 
                 loss_type='l1', 
                 betas = None):
        super().__init__()
        self.denoise_fn = denoise_fn

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

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

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('_allowed_x', _allowed_x)
        self.register_buffer('_allowed_y', _allowed_y)
        
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

        # DIRC unscaling: 

        if unscale:
            assert input_dim >= 3

            x = self.unscale(sample[:,0].flatten(),self.stats['x_max'],self.stats['x_min'])
            y = self.unscale(sample[:,1].flatten(),self.stats['y_max'],self.stats['y_min'])
            t = self.unscale(sample[:,2].flatten(),self.stats['time_max'],self.stats['time_min'])

            sample = torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1) 

        return sample
    
    @torch.no_grad()
    def sample(self, cond, n_samples, input_dim, unscale = True): 
        return self.p_sample_loop(n_samples=n_samples, input_dim=input_dim, cond=cond, unscale=unscale)
    
    # Resampling specific for FastDIRC
    @torch.no_grad()
    def resample(self, cond, n_samples, input_dim):
        # Resampling with the DIRC detector.
        samples = self.sample(
                            cond, 
                            n_samples, 
                            input_dim,
                            unscale=True
                            )     
        
        x = self.set_to_closest(samples[:,0],self._allowed_x,cond.device)
        y = self.set_to_closest(samples[:,1],self._allowed_y,cond.device)
        t = samples[:,2]

        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1).detach().cpu().numpy()

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

    # --------------- hpDIRC sampling ----------------

    def __get_track(self,n_samples, cond, input_dim=3):
        samples = self.sample(
                            cond, 
                            n_samples, 
                            input_dim,
                            unscale=False
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


