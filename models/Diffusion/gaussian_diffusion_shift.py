import math
import torch
import numpy as np

from functools import partial
import torch.nn as nn

from .ddim import DDIM

from utils.hpDIRC import ALLOWED_X,ALLOWED_Y

class ShiftGaussianDiffusion(nn.Module):
    def __init__(self, 
            stats, 
            denoise_fn,
            shift_predictor,
            timesteps,
            noise_schedule, 
            shift_type, 
            device,
            LUT_path=None
        ):
        super().__init__()
        self.device=device
        self.timesteps = timesteps

        if noise_schedule == "linear":
            betas = np.linspace(0.0001, 0.02, self.timesteps)
        elif noise_schedule == "cosine":
            betas = []
            alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            max_beta = 0.999
            for i in range(self.timesteps):
                t1 = i / self.timesteps
                t2 = (i + 1) / self.timesteps
                betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
            betas = np.array(betas)
        else:
            raise NotImplementedError
        
        # hpDIRC stuff
        self.stats_ = stats

        self._allowed_x = torch.tensor(np.array(ALLOWED_X))
        self._allowed_y = torch.tensor(np.array(ALLOWED_Y))

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

        # Models

        self.denoise_fn = denoise_fn
        self.shift_predictor = shift_predictor

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.)

        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)
        self.to_torch = to_torch

        self.alphas = to_torch(alphas)
        self.betas = to_torch(betas)
        self.alphas_cumprod = to_torch(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev)
        self.alphas_cumprod_next = to_torch(alphas_cumprod_next)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recip_alphas_cumprod_m1 = to_torch(np.sqrt(1. / alphas_cumprod - 1.))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = to_torch(posterior_variance)
        # clip the log because the posterior variance is 0 at the beginning of the diffusion chain
        posterior_log_variance_clipped = np.log(np.append(posterior_variance[1], posterior_variance[1:]))
        self.posterior_log_variance_clipped = to_torch(posterior_log_variance_clipped)

        self.x_0_posterior_mean_x_0_coef = to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.x_0_posterior_mean_x_t_coef = to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.noise_posterior_mean_x_t_coef = to_torch(np.sqrt(1. / alphas))
        self.noise_posterior_mean_noise_coef = to_torch(betas/(np.sqrt(alphas)*np.sqrt(1. - alphas_cumprod)))

        self.shift_type = shift_type
        if self.shift_type == "prior_shift":
            shift = 1. - np.sqrt(alphas_cumprod)
            # shift = np.array([(i+1)/1000 for i in range(1000)])
            # shift = np.array([((i+1)**2)/1000000 for i in range(1000)])
            # shift = np.array([np.sin((i+1)/timesteps*np.pi/2 - np.pi/2) + 1.0 for i in range(timesteps)])
        elif self.shift_type == "data_normalization":
            shift = - np.sqrt(alphas_cumprod)
        elif self.shift_type == "quadratic_shift":
            shift = np.sqrt(alphas_cumprod) * (1. - np.sqrt(alphas_cumprod))
            # def quadratic(timesteps, t):
            #     return - (1.0 / (timesteps / 2.0) ** 2) * (t - timesteps) * t
            # shift = np.array([quadratic(self.timesteps, i + 1) for i in range(1000)])
        elif self.shift_type == "early":
            shift = np.array([(i + 1) / 600 - 2. / 3. for i in range(1000)])
            shift[:400] = 0
        else:
            raise NotImplementedError

        self.shift = to_torch(shift)
        shift_prev = np.append(0., shift[:-1])
        self.shift_prev = to_torch(shift_prev)
        d = shift_prev - shift / np.sqrt(alphas)
        self.d = to_torch(d)

    @staticmethod
    def extract_coef_at_t(schedule, t, x_shape):
        return torch.gather(schedule, -1, t).reshape([x_shape[0]] + [1] * (len(x_shape) - 1))

    @staticmethod
    def get_ddim_betas_and_timestep_map(ddim_style, original_alphas_cumprod):
        original_timesteps = original_alphas_cumprod.shape[0]
        ddim_step = int(ddim_style[len("ddim"):])
        # data: x_{-1}  noisy latents: x_{0}, x_{1}, x_{2}, ..., x_{T-2}, x_{T-1}
        # encode: treat input x_{-1} as starting point x_{0}
        # sample: treat ending point x_{0} as output x_{-1}
        use_timesteps = set([int(s) for s in list(np.linspace(0, original_timesteps - 1, ddim_step + 1))])
        timestep_map = []

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(original_alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)

        return np.array(new_betas), torch.tensor(timestep_map, dtype=torch.long)

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

    # x_0: batch_size x input_dim
    # t: batch_size
    def q_sample(self, x_0, t, noise):
        shape = x_0.shape
        return (
            self.extract_coef_at_t(self.sqrt_alphas_cumprod, t, shape) * x_0
            + self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape) * noise
        )

    def q_posterior_mean(self, x_0, x_t, t):
        shape = x_t.shape
        return self.extract_coef_at_t(self.x_0_posterior_mean_x_0_coef, t, shape) * x_0 \
               + self.extract_coef_at_t(self.x_0_posterior_mean_x_t_coef, t, shape) * x_t

    # x_t: batch_size x image_channel x image_size x image_size
    # t: batch_size
    def noise_p_sample(self, x_t, t, predicted_noise):
        shape = x_t.shape
        predicted_mean = \
            self.extract_coef_at_t(self.noise_posterior_mean_x_t_coef, t, shape) * x_t - \
            self.extract_coef_at_t(self.noise_posterior_mean_noise_coef, t, shape) * predicted_noise
        log_variance_clipped = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)
        noise = torch.randn(shape, device=self.device)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape([shape[0]] + [1] * (len(shape) - 1))
        return predicted_mean + nonzero_mask * (0.5 * log_variance_clipped).exp() * noise

    # x_t: batch_size x image_channel x image_size x image_size
    # t: batch_size
    def x_0_clip_p_sample(self, x_t, t, predicted_noise, learned_range=None, clip_x_0=True):
        shape = x_t.shape

        predicted_x_0 = self.predicted_noise_to_predicted_x_0(x_t, t, predicted_noise)
        if clip_x_0:
            predicted_x_0.clamp_(-1,1)
        predicted_mean = self.q_posterior_mean(predicted_x_0, x_t, t)
        if learned_range is not None:
            log_variance = self.learned_range_to_log_variance(learned_range, t)
        else:
            log_variance = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)

        noise = torch.randn(shape, device=self.device)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape([shape[0]] + [1] * (len(shape) - 1))
        return predicted_mean + nonzero_mask * (0.5 * log_variance).exp() * noise

    def learned_range_to_log_variance(self, learned_range, t):
        shape = learned_range.shape
        min_log_variance = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)
        max_log_variance = self.extract_coef_at_t(torch.log(self.betas), t, shape)
        # The learned_range is [-1, 1] for [min_var, max_var].
        frac = (learned_range + 1) / 2
        return min_log_variance + frac * (max_log_variance - min_log_variance)

    def predicted_noise_to_predicted_x_0(self, x_t, t, predicted_noise):
        shape = x_t.shape
        return self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * x_t \
               - self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod_m1, t, shape) * predicted_noise

    def predicted_noise_to_predicted_mean(self, x_t, t, predicted_noise):
        shape = x_t.shape
        return self.extract_coef_at_t(self.noise_posterior_mean_x_t_coef, t, shape) * x_t - \
               self.extract_coef_at_t(self.noise_posterior_mean_noise_coef, t, shape) * predicted_noise

    def p_loss(self, noise, predicted_noise, weight=None, loss_type="l2"):
        if loss_type == 'l1':
            return (noise - predicted_noise).abs().mean()
        elif loss_type == 'l2':
            if weight is not None:
                return torch.mean(weight * (noise - predicted_noise) ** 2)
            else:
                return torch.mean((noise - predicted_noise) ** 2)
        else:
            raise NotImplementedError

    """
        test pretrained dpms
    """
    def test_pretrained_dpms(self, ddim_style, denoise_fn, x_T, condition=None):
        return self.ddim_sample(ddim_style, denoise_fn, x_T, condition)

    """
        ddim
    """
    def ddim_sample(self, ddim_style, denoise_fn, x_T, condition=None):
        new_betas, timestep_map = self.get_ddim_betas_and_timestep_map(ddim_style, self.alphas_cumprod.cpu().numpy())
        ddim = DDIM(new_betas, timestep_map, self.device)
        return ddim.ddim_sample_loop(denoise_fn, x_T, condition)

    """
        regular
    """
    def regular_train_one_batch(self, denoise_fn, x_0, condition=None):
        shape = x_0.shape
        batch_size = shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0=x_0, t=t, noise=noise)
        predicted_noise = denoise_fn(x_t, t, condition)

        prediction_loss = self.p_loss(noise, predicted_noise)

        return {
            'prediction_loss': prediction_loss,
        }

    def regular_ddpm_sample(self, denoise_fn, x_T, condition=None):
        shape = x_T.shape
        batch_size = shape[0]
        sample = x_T
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            predicted_noise = denoise_fn(sample, t, condition)
            sample = self.noise_p_sample(sample, t, predicted_noise)
        return sample

    def regular_ddim_sample(self, ddim_style, denoise_fn, x_T, condition=None):
        return self.ddim_sample(ddim_style, denoise_fn, x_T, condition)

    """
        shift
    """
    def shift_q_sample(self, x_0, u, t, noise):
        shape = x_0.shape
        return (
            self.extract_coef_at_t(self.sqrt_alphas_cumprod, t, shape) * x_0
            + self.extract_coef_at_t(self.shift, t, shape) * u
            + self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape) * noise
        )

    def shift_noise_p_sample(self, x_t, u, t, predicted_noise):
        shape = x_t.shape
        predicted_mean = \
            self.extract_coef_at_t(self.noise_posterior_mean_x_t_coef, t, shape) * x_t - \
            self.extract_coef_at_t(self.noise_posterior_mean_noise_coef, t, shape) * predicted_noise + \
            self.extract_coef_at_t(self.d, t, shape) * u
        log_variance_clipped = self.extract_coef_at_t(self.posterior_log_variance_clipped, t, shape)
        noise = torch.randn(shape, device=self.device)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape([shape[0]] + [1] * (len(shape) - 1))
        return predicted_mean + nonzero_mask * (0.5 * log_variance_clipped).exp() * noise

    def shift_train_one_batch(self, x_0, condition):
        shape = x_0.shape
        t = torch.randint(0, self.timesteps, (shape[0],), device=self.device).long()
        noise = torch.randn_like(x_0)
        u = self.shift_predictor(condition)
        x_t = self.shift_q_sample(x_0=x_0, u=u, t=t, noise=noise)
        tmp = self.extract_coef_at_t(self.shift, t, shape) * u / self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape)
        predicted_noise = self.denoise_fn(x_t, t, None) - tmp
        prediction_loss = self.p_loss(noise, predicted_noise)

        return {
            'prediction_loss': prediction_loss,
        }

    def shift_sample(self, x_T, condition, unscale = True):
        shape = x_T.shape
        u = self.shift_predictor(condition)
        if self.shift_type == "prior_shift" or self.shift_type == "early":
            sample = x_T + u
        elif self.shift_type == "data_normalization" or self.shift_type == "quadratic_shift":
            sample = x_T
        else:
            raise NotImplementedError
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            tmp = self.extract_coef_at_t(self.shift, t, shape) * u / self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape)
            predicted_noise = self.denoise_fn(sample, t, None) - tmp
            sample = self.shift_noise_p_sample(sample, u, t, predicted_noise)

        if unscale:

            x = self.unscale(sample[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min'])
            y = self.unscale(sample[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min'])
            t = self.unscale(sample[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])

            sample = torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1) 

        return sample

    # Resampling specific for FastDIRC
    @torch.no_grad()
    def resample(self, cond, n_samples, input_dim):
        # Resampling with the DIRC detector.
        samples = self.shift_sample(
                            #random normal
                            torch.randn((n_samples, input_dim), device=self.device), 
                            cond, 
                            unscale=True
                            )     
        
        x = self.set_to_closest(samples[:,0],self._allowed_x,cond.device)
        y = self.set_to_closest(samples[:,1],self._allowed_y,cond.device)
        t = samples[:,2]

        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1).detach().cpu().numpy()

    def shift_sample_interpolation(self, denoise_fn, x_T, u):
        shape = x_T.shape
        if self.shift_type == "prior_shift" or self.shift_type == "early":
            sample = x_T + u
        elif self.shift_type == "data_normalization" or self.shift_type == "quadratic_shift":
            sample = x_T
        else:
            raise NotImplementedError
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            tmp = self.extract_coef_at_t(self.shift, t, shape) * u / self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape)
            predicted_noise = denoise_fn(sample, t, None) - tmp
            sample = self.shift_noise_p_sample(sample, u, t, predicted_noise)
        return sample

    # --------------- hpDIRC sampling ----------------

    def __get_track(self, n_samples, cond, input_dim=3):
        xT = torch.randn((n_samples, input_dim), device=self.device)
        samples = self.shift_sample(
                            #random normal
                            xT, 
                            cond, 
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

    def create_tracks(self,num_samples,context,p=None,theta=None,fine_grained_prior=True): # resampling logic
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