# ported directly from LucidRains: 
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281

import math
import numpy as np
import torch
from torch import sqrt
from torch import nn, einsum
import torch.nn.functional as F
from torch.amp import autocast
from torch.special import expm1

from tqdm import tqdm
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from utils.hpDIRC import ALLOWED_X,ALLOWED_Y

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# diffusion helpers

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# neural net helpers

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)

class MonotonicLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return F.linear(x, self.net.weight.abs(), self.net.bias.abs())

# continuous schedules

# equations are taken from https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material
# @crowsonkb Katherine's repository also helped here https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py

# log(snr) that approximates the original linear schedule

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def beta_linear_log_snr(t):
    return -log(expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, s = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5)

class learned_noise_schedule(nn.Module):
    """ described in section H and then I.2 of the supplementary material for variational ddpm paper """

    def __init__(
        self,
        *,
        log_snr_max,
        log_snr_min,
        hidden_dim = 1024,
        frac_gradient = 1.
    ):
        super().__init__()
        self.slope = log_snr_min - log_snr_max
        self.intercept = log_snr_max

        self.net = nn.Sequential(
            Rearrange('... -> ... 1'),
            MonotonicLinear(1, 1),
            Residual(nn.Sequential(
                MonotonicLinear(1, hidden_dim),
                nn.Sigmoid(),
                MonotonicLinear(hidden_dim, 1)
            )),
            Rearrange('... 1 -> ...'),
        )

        self.frac_gradient = frac_gradient

    def forward(self, x):
        frac_gradient = self.frac_gradient
        device = x.device

        out_zero = self.net(torch.zeros_like(x))
        out_one =  self.net(torch.ones_like(x))

        x = self.net(x)

        normed = self.slope * ((x - out_zero) / (out_one - out_zero)) + self.intercept
        return normed * frac_gradient + normed.detach() * (1 - frac_gradient)

class ContinuousTimeGaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        stats,
        *,
        noise_schedule = 'learned',
        num_sample_steps = 500,
        clip_sample_denoised = True,
        learned_schedule_net_hidden_dim = 1024,
        learned_noise_schedule_frac_gradient = 1.,   # between 0 and 1, determines what percentage of gradients go back, so one can update the learned noise schedule more slowly
        min_snr_loss_weight = False,
        min_snr_gamma = 1,
        LUT_path=None
    ):
        super().__init__()

        self.model = model

        # ------------ hpDIRC constants ------------

        self._allowed_x = torch.tensor(np.array(ALLOWED_X)).to(self.device)
        self._allowed_y = torch.tensor(np.array(ALLOWED_Y)).to(self.device)
                
        self.stats_ = stats

        if LUT_path is not None:
            print("Loading photon yield sampler.")
            dicte = np.load(LUT_path,allow_pickle=True)
            self.LUT = {k: v for k, v in dicte.items() if k != "global_values"}
            self.global_values = dicte['global_values']
            self.p_points = np.array(list(dicte.keys())[:-1]) 
            self.theta_points = np.array(list(dicte[self.p_points[0]].keys()))

        self.photons_generated = 0
        self.photons_resampled = 0
        self.gapx =  1.89216111455965 + 4.
        self.gapy = 1.3571428571428572 + 4.
        self.pixel_width = 3.3125
        self.pixel_height = 3.3125
        self.num_pixels = 16
        self.num_pmts_x = 6
        self.num_pmts_y = 4

        # continuous noise schedule related stuff
        self.noise_schedule = noise_schedule

        if noise_schedule == 'linear':
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == 'cosine':
            self.log_snr = alpha_cosine_log_snr
        elif noise_schedule == 'learned':
            log_snr_max, log_snr_min = [beta_linear_log_snr(torch.tensor([time])).item() for time in (0., 1.)]

            self.log_snr = learned_noise_schedule(
                log_snr_max = log_snr_max,
                log_snr_min = log_snr_min,
                hidden_dim = learned_schedule_net_hidden_dim,
                frac_gradient = learned_noise_schedule_frac_gradient
            )
        else:
            raise ValueError(f'unknown noise schedule {noise_schedule}')

        # sampling

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised

        # proposed https://arxiv.org/abs/2303.09556

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

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
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next, cond):
        # reviewer found an error in the equation in the paper (missing sigma)
        # following - https://openreview.net/forum?id=2LdBqxc1Yv&noteId=rIQgH0zKsRt

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        # print(f"log_snr: {log_snr}, log_snr_next: {log_snr_next}")

        c = -expm1(log_snr - log_snr_next)
        # print(f"c: {c}")

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])
        batch_cond = cond.repeat(x.shape[0], 1) # Adjusting for nphotons

        # print("x shape:",x.shape, "batch log_snr shape:", batch_log_snr.shape, "cond shape:", batch_cond.shape)
        pred_noise = self.model(x, batch_log_snr, batch_cond)
        # print(f"pred_noise:",pred_noise)

        if torch.isnan(pred_noise).any():
            print("NaN values detected in pred_noise")
            nanmask = torch.isnan(pred_noise)
            print("Total NaNs:", nanmask.sum().item())

        if self.clip_sample_denoised:
            x_start = (x - sigma * pred_noise) / alpha

            # in Imagen, this was changed to dynamic thresholding
            x_start.clamp_(-1., 1.)

            model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        else:
            model_mean = alpha_next / alpha * (x - c * sigma * pred_noise)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    # sampling related functions

    @torch.no_grad()
    def p_sample(self, x, time, time_next, cond):
        batch, num_inputs = x.shape
        device = x.device

        model_mean, model_variance = self.p_mean_variance(x = x, time = time, time_next = time_next, cond=cond)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, n_samples, input_dim, cond, unscale=True):

        sample = torch.randn((n_samples, input_dim), device = self.device)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device = self.device)

        for i in range(self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            sample = self.p_sample(sample, times, times_next, cond)

        # sample.clamp_(-1., 1.)
        # sample = unnormalize_to_zero_to_one(sample) # we normalized in forward pass, so unnormalize here? 

        # DIRC unscaling: 

        if unscale:
            assert input_dim >= 3

            x = self.unscale(sample[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min'])
            y = self.unscale(sample[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min'])
            t = self.unscale(sample[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])

            sample = torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1) 

        return sample


    @torch.no_grad()
    def sample(self, cond, n_samples, input_dim, unscale = True):
        return self.p_sample_loop(n_samples, input_dim, cond, unscale = unscale)
    
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

    # training related functions - noise prediction

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, times, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised =  x_start * alpha + noise * sigma # actual noise being added, with alpha controlling level of noise. 

        return x_noised, log_snr # size of t

    def random_times(self, batch_size):
        # times are now uniform from 0 to 1
        return torch.zeros((batch_size,), device = self.device).float().uniform_(0, 1)

    def p_losses(self, x_start, times, cond, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x, log_snr = self.q_sample(x_start = x_start, times = times, noise = noise)

        #full_conds = torch.cat([log_snr.unsqueeze(1),cond], dim=1) 
        #print("data shape:",x.shape,"logsnr shape:",log_snr.shape, "conditions shape:",full_conds.shape)
        model_out = self.model(x, log_snr, cond)

        # print(model_out.shape, noise.shape)
        losses = F.mse_loss(model_out, noise, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        if self.min_snr_loss_weight:
            snr = log_snr.exp()
            loss_weight = snr.clamp(min = self.min_snr_gamma) / snr
            losses = losses * loss_weight

        return losses.mean()

    def forward(self, x, cond, *args, **kwargs):
        b, *_, device = *x.shape, x.device
        #assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        times = self.random_times(b)
        # x = normalize_to_neg_one_to_one(x) # we do our normalization before forward pass
        return self.p_losses(x, times, cond, *args, **kwargs)

    # --------------- hpDIRC sampling ----------------

    def __get_track(self,n_samples, cond, input_dim=3):
        samples = self.sample(
                            cond, 
                            n_samples, 
                            input_dim,
                            unscale=False
                            )
        x = self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min'])#.round()
        y = self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min'])#.round()
        t = self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])

        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1)

    def __sample_photon_yield(self,p_value,theta_value):
        closest_p_idx = np.argmin(np.abs(self.p_points - p_value))
        closest_p = float(self.p_points[closest_p_idx])
        
        closest_theta_idx = np.argmin(np.abs(self.theta_points - theta_value))
        closest_theta = float(self.theta_points[closest_theta_idx])

        return int(np.random.choice(self.global_values,p=self.LUT[closest_p][closest_theta]))

    def _apply_mask(self, hits,fine_grained_prior):
        # Time > 0 
        mask = torch.where((hits[:,2] > 0) & (hits[:,2] < self.stats_['time_max']))
        hits = hits[mask]
        # Outter bounds
        mask = torch.where((hits[:, 0] > self.stats_['x_min']) & (hits[:, 0] < self.stats_['x_max']) & (hits[:, 1] > self.stats_['y_min']) & (hits[:, 1] < self.stats_['y_max']))[0] # Acceptance mask
        hits = hits[mask]

        # Can we make this faster? Currently 2x increase.
        if fine_grained_prior:
            # Spacings along x
            valid_x_mask = torch.ones(hits.shape[0], dtype=torch.bool, device=hits.device)
            for i in range(1, self.num_pmts_x): 
                x_low = self._allowed_x[i * self.num_pixels - 1] + self.pixel_width/2.0
                x_high = self._allowed_x[i * self.num_pixels] - self.pixel_width/2.0
                mask = (hits[:, 0] > x_low) & (hits[:, 0] < x_high)
                valid_x_mask &= ~mask  
                #print("x",i,x_low,x_high)

            hits = hits[valid_x_mask]
            
            # Spacings along y
            valid_y_mask = torch.ones(hits.shape[0], dtype=torch.bool, device=hits.device)
            for i in range(1, self.num_pmts_y): 
                y_low = self._allowed_y[i * self.num_pixels - 1] + self.pixel_height/2.0
                y_high = self._allowed_y[i * self.num_pixels] - self.pixel_height/2.0
                mask = (hits[:, 1] > y_low) & (hits[:, 1] < y_high)
                valid_y_mask &= ~mask
                #print("y",i,y_low,y_high)

            hits = hits[valid_y_mask]


        return hits


    def create_tracks(self,num_samples,context,p=None,theta=None,fine_grained_prior=True,dark_rate=None): # resampling logic
        if num_samples is None:
            assert p is not None and theta is not None, "p and theta must be provided if num_samples is None."
            num_samples = self.__sample_photon_yield(p,theta)
        
        hits = self.__get_track(num_samples,context)
        updated_hits = self._apply_mask(hits,fine_grained_prior=fine_grained_prior)
        n_resample = int(num_samples - len(updated_hits))
        

        self.photons_generated += len(hits)
        self.photons_resampled += n_resample
        while n_resample != 0:
            resampled_hits = self.__get_track(n_resample,context)
            updated_hits = torch.concat((updated_hits,resampled_hits),0)
            updated_hits = self._apply_mask(updated_hits,fine_grained_prior=fine_grained_prior)
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
        pixelID = 16 * (row - (pmtID // 6) * 16) + (col - (pmtID % 6) * 16)
        channel = pmtID * self.num_pixels**2 + pixelID

        assert(len(row) == num_samples)
        assert(len(col) == num_samples)
        assert(len(pmtID) == num_samples)

        P = self.unscale_conditions(context[0][0].detach().cpu().numpy(),self.stats_['P_max'],self.stats_['P_min'])
        Theta = self.unscale_conditions(context[0][1].detach().cpu().numpy(),self.stats_['theta_max'],self.stats_['theta_min'])
        Phi = 0.0

        return {"NHits":num_samples,"P":P,"Theta":Theta,"Phi":Phi,"x":x.numpy(),"y":y.numpy(),"leadTime":t.numpy(),"pmtID":pmtID.numpy(),"pixelID":pixelID.numpy(),"channel":channel.numpy()}