import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResNet, EMA
import copy
import time
import numpy as np
import matplotlib.pyplot as plt


### Functions for sampling
def unscale_inputs(x,max_,min_):
    return x*0.5*(max_ - min_) + min_ + (max_-min_)/2

def _apply_mask(hits, stats):
        # Time > 0 
        mask = torch.where((hits[:,2] > 0) & (hits[:,2] < stats['time_max']))
        hits = hits[mask]
        # Outter bounds
        mask = torch.where((hits[:, 0] > 0) & (hits[:, 0] < 898) & (hits[:, 1] > 0) & (hits[:, 1] < 298))[0] # Acceptance mask
        hits = hits[mask]
        # PMTs OFF
        top_row_mask = torch.where(~((hits[:, 1] > 249) & (hits[:, 0] < 350)))[0] # rejection mask (keep everything not identified)
        hits = hits[top_row_mask]
        # PMTs OFF
        bottom_row_mask = torch.where(~((hits[:, 1] < 50) & (hits[:, 0] < 550)))[0] # rejection mask (keep everything not identified)
        hits = hits[bottom_row_mask]

        return hits

def _set_to_closest(x, allowed, device):
        x = x.unsqueeze(1)  # Adding a dimension to x for broadcasting
        diffs = torch.abs(x - allowed.to(device).float())
        closest_indices = torch.argmin(diffs, dim=1)
        closest_values = allowed[closest_indices]
        return closest_values
  

class DenseMonotone(nn.Module):
    '''Strictly increasing Dense layer.
    Ported directly from https://github.com/s-sahoo/MuLAN/tree/main. Converted from JAX to PyTorch
    '''
    
    def __init__(self, in_features, out_features, kernel_init='normal', bias_init='constant', use_bias=True, bias_value=0.0):
        """
        Initializes the DenseMonotone layer.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            kernel_init (str or callable): Initialization method for weights.
            bias_init (str or callable): Initialization method for biases.
            use_bias (bool): If set to False, the layer will not learn an additive bias.
            bias_value (float): Constant value for bias initialization if bias_init is 'constant'.
        """

        """
        Based off of the JAX code from https://github.com/s-sahoo/MuLAN/tree/main
        """
        super(DenseMonotone, self).__init__()
        self.use_bias = use_bias
        self.in_features = in_features
        self.out_features = out_features

        # Initialize raw weights (can be negative)
        self.raw_weight = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(kernel_init, bias_init, bias_value) # reset parameters should go after defining bias, otherwise error.

    def reset_parameters(self, kernel_init, bias_init, bias_value):
        """
        Initialize weights and biases.
        """
        # Initialize weights
        if isinstance(kernel_init, str):
            if kernel_init == 'normal':
                nn.init.normal_(self.raw_weight, mean=0.0, std=1.0)  # Adjust std as needed
            elif kernel_init == 'constant':
                nn.init.constant_(self.raw_weight, 1.0)  # Example constant
            else:
                raise ValueError(f"Unsupported kernel_init: {kernel_init}")
        elif callable(kernel_init):
            kernel_init(self.raw_weight)
        else:
            raise ValueError("kernel_init must be a string or a callable.")
        
        # Initialize bias if used
        if self.use_bias:
            if isinstance(bias_init, str):
                if bias_init == 'constant':
                    nn.init.constant_(self.bias, bias_value)
                elif bias_init == 'normal':
                    nn.init.normal_(self.bias, mean=0.0, std=1.0)
                else:
                    raise ValueError(f"Unsupported bias_init: {bias_init}")
            elif callable(bias_init):
                bias_init(self.bias)
            else:
                raise ValueError("bias_init must be a string or a callable.")

    def forward(self, inputs):
        """
        Forward pass of the DenseMonotone layer.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, in_features).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Ensure weights are non-negative
        weight = torch.abs(self.raw_weight)
        y = torch.matmul(inputs, weight)
        print(inputs.shape)
        print(weight.shape)
        if self.use_bias:
            y = y + self.bias
        return y


class NoiseSchedule_NNet(nn.Module):
    def __init__(self, gamma_min, gamma_max, n_features, nonlinear=False):
        """
        Initializes the NoiseSchedule_NNet model.
        Ported directly from https://github.com/s-sahoo/MuLAN/tree/main. Converted from JAX to PyTorch

        Args:
            gamma_min (float): Minimum initial bias for the noise schedule.
            gamma_max (float): Maximum scale for the noise schedule.
            n_features (int): Number of features (e.g., 32*32*3 for a 32x32 RGB image).
            nonlinear (bool): If True, includes nonlinear transformations.
        """
        super(NoiseSchedule_NNet, self).__init__()
        
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.n_features = n_features
        self.nonlinear = nonlinear
        
        # Calculate initial bias and scale for the first layer
        init_bias = gamma_min
        init_scale = gamma_max - gamma_min
        
        # Layer l1: Takes t as input and outputs a single value
        self.l1 = DenseMonotone(
            in_features=1,
            out_features=1,
            kernel_init='constant',  # Using 'constant' to set scale directly
            bias_init='constant',
            use_bias=True,
            bias_value=0.0
        )

        if self.nonlinear:
            # Layer l2: Processes the concatenated input
            # Assuming image_embedding is not part of this model; adjust as needed
            # Here, it's designed to take the scaled t as input
            self.l2 = DenseMonotone(
                in_features=1,
                out_features=n_features,
                kernel_init='normal',
                bias_init='constant',
                use_bias=True,
                bias_value=0.0  # Adjust as needed
            )
            
            # Layer l3: Final nonlinear transformation
            self.l3 = DenseMonotone(
                in_features=n_features,
                out_features=1,
                kernel_init='normal',
                bias_init='constant',
                use_bias=False
            )

    def forward(self, t): # model.foward() equivalent to model()
        """
        Forward pass of the NoiseSchedule_NNet model.
        
        Args:
            t (torch.Tensor): Input tensor representing the noise parameter. Can be scalar or 1D tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying the noise schedule.
        """
        # Ensure t has shape (batch_size, 1)
        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)  # Shape: (batch_size, 1)
        elif t.dim() == 2 and t.size(1) != 1:
            t = t.view(-1, 1)  # Shape: (batch_size, 1)
        elif t.dim() > 2:
            raise ValueError(f"Unsupported input shape: {t.shape}")
        
        # Apply linear transformation using l1
        linear = self.l1(t)  # Shape: (batch_size, 1)
        
        if self.nonlinear:
            # Scale input to [-1, +1]
            _h = 2. * (t - 0.5)  # Shape: (batch_size, 1)
            
            # Apply l2
            _h = self.l2(_h)  # Shape: (batch_size, n_features)
            
            # Apply sigmoid activation and scale
            _h = 2 * (torch.sigmoid(_h) - 0.5)  # Shape: (batch_size, n_features)
            
            # Apply l3 and normalize
            _h = self.l3(_h) / self.n_features  # Shape: (batch_size, 1)
            
            # Combine linear and nonlinear components
            output = linear + _h  # Shape: (batch_size, 1)
        else:
            output = linear  # Shape: (batch_size, 1)
        
        return output.squeeze(-1)  # Shape: (batch_size,)


class GSGM(nn.Module):
    "Score based generative model"
    def __init__(self, 
                 num_input, 
                 num_conds, 
                 device, 
                 stats,
                 num_layers=6, 
                 num_steps=500, 
                 num_embed = 32, 
                 mlp_dim=128, 
                 nonlinear_noise_schedule=False, 
                 learnedvar=False, name = 'SGM'
            ):
        super(GSGM, self).__init__() # invokes parent class constructor
        # Initializes underlying nn.Module, with all attributes
        # and methods
        self.activation = nn.LeakyReLU(0.01)
        self.num_input = num_input
        self.num_embed = num_embed
        self.num_conds = num_conds
        self.num_layers = num_layers

        # ------------ hpDIRC constants ------------

        self.device = device

        self._allowed_x = torch.tensor(np.array([  3.65625   ,   6.96875   ,  10.28125   ,  13.59375   ,
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
                                                   337.86705557, 341.17955557, 344.49205557, 347.80455557])).to(self.device)
        self._allowed_y = torch.tensor(np.array([  3.65625   ,   6.96875   ,  10.28125   ,  13.59375   ,
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
                                                   218.47767857, 221.79017857, 225.10267857, 228.41517857])).to(self.device) 
        
        self.stats_ = stats

        self.photons_generated = 0
        self.photons_resampled = 0
        self.gapx =  1.89216111455965 + 4.
        self.gapy = 1.3571428571428572 + 4.
        self.pixel_width = 3.3125
        self.pixel_height = 3.3125
        self.num_pixels = 16
        self.num_pmts_x = 6
        self.num_pmts_y = 4

        self.ema = 0.999
        self.minlogsnr = torch.tensor(-20.0, dtype=torch.float32)
        self.maxlogsnr = torch.tensor(20.0, dtype=torch.float32)
        self.lambda_smooth = 1e-3
        self.num_steps = num_steps
        self.mlp_dim = mlp_dim
        self.learnedvar = learnedvar


        self.projection = self.GaussianFourierProjection(scale = 16) # 
        #self.loss_tracker = keras.metrics.Mean(name="loss") # Need to convert to PyTorch

        self.conditional_transform = nn.Linear(self.num_input, self.num_embed)

        self.combine_embeds = nn.Linear(2* self.num_embed, self.num_embed)

        self.model = ResNet(
            input_dim = self.num_input,
            end_dim= self.num_input,
            cond_dim=self.num_conds,
            mlp_dim = self.mlp_dim,
            num_layer = self.num_layers
        )

        if self.learnedvar:
            self.noise_schedule_net = NoiseSchedule_NNet(
                gamma_min=self.minlogsnr,  # Using time_min as gamma_min
                gamma_max=self.maxlogsnr,  # Using time_max as gamma_max
                n_features=self.num_input,             # Adjust based on your requirement
                nonlinear=nonlinear_noise_schedule
            ).to(self.device)
        # is automatically included in the optimizer, now registered as a submodule of GSGM
        
        self.ema = EMA(beta=0.999) # EMA class to control smoothing process

        self.ema_model = copy.deepcopy(self.model) # to keep alongside regular model for stability

        for param in self.ema_model.parameters():
            param.requires_grad = False # not trained directly

        # Need to know in_feaures, out_features

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
    
    def GaussianFourierProjection(self, scale = 30):
        half_dim = self.num_embed // 2
        emb = torch.log(torch.tensor(10000.0, dtype=torch.float32)) / (half_dim - 1)
        freq = torch.exp(-emb* torch.arange(start=0, end=half_dim, dtype=torch.float32))
        return freq

    def Embedding(self,inputs,projection):
        projection = projection.to(inputs.device)
        angle = inputs*projection*1000
        embedding = torch.cat([torch.sin(angle),torch.cos(angle)],dim=1)
        return embedding
    
    def prior_sde(self,dimensions):
        return torch.randn(dimensions,dtype=torch.float32, device=self.device)
    
    def logsnr_schedule_cosine(self,t, logsnr_min, logsnr_max):
        b = torch.atan(torch.exp(-0.5 * logsnr_max))
        a = torch.atan(torch.exp(-0.5 * logsnr_min)) - b
        return -2. * torch.log(torch.tan(a * t.float() + b))

    def logsnr_schedule_linear(self, t, logsnr_min, logsnr_max):
        return logsnr_min + t * (logsnr_max - logsnr_min)

    
    def get_logsnr_alpha_sigma(self,time,shape=None):
        if self.learnedvar:
            logsnr = self.noise_schedule_net(time)  # Shape: (batch_size,)
            logsnr = logsnr.view(-1, 1)         # Shape: (batch_size, 1)
        else:
            logsnr = self.logsnr_schedule_cosine(time,logsnr_min=self.minlogsnr, logsnr_max=self.maxlogsnr)
            #logsnr = self.logsnr_schedule_linear(time, logsnr_min=self.minlogsnr, logsnr_max=self.maxlogsnr) 
        alpha = torch.sqrt(torch.sigmoid(logsnr))
        sigma = torch.sqrt(torch.sigmoid(-logsnr))
        if shape is not None:
            alpha = alpha.view(shape)
            sigma = sigma.view(shape)
        return logsnr, alpha, sigma

    def inv_logsnr_schedule_cosine(self,logsnr, logsnr_min, logsnr_max):
        b = torch.atan(torch.exp(-0.5 * logsnr_max))
        a = torch.atan(torch.exp(-0.5 * logsnr_min)) - b
        return torch.atan(torch.exp(-0.5 * torch.cast(logsnr,torch.float32)))/a -b/a
    
    def train_step(self, x, cond, optimizer_noise = None):
        eps=1e-5
        random_t = torch.rand((x.shape[0],1), device=self.device)

        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(random_t) # ∇ x t logq(x_t ∣ x)= α/σ^2 \cdot (z− σx/α) if q(x_t ∣ x) is Gaussian
        alpha.to(self.device)
        sigma.to(self.device)

        z = torch.randn((x.shape),dtype=torch.float32, device=self.device)
        
        perturbed_x = alpha*x + z * sigma
        pred_noise = self.model.forward(perturbed_x,random_t,cond)
        v = alpha* z - sigma* x
        losses = torch.square(pred_noise - v) # just MSE                 
        loss = torch.mean(losses.view(losses.size(0),-1))

        if optimizer_noise:
            smoothness_penalty = self.compute_smoothness_penalty()
            loss = loss + (self.lambda_smooth * smoothness_penalty)

            torch.nn.utils.clip_grad_norm_(self.noise_schedule_net.parameters(), max_norm=1.0)

            optimizer_noise.step()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # take out gradient clipping
            
        self.ema.step_ema(self.ema_model, self.model, step_start_ema=0)

        # Find a loss_tracker to use
        # self.loss_tracker.update(loss.item(), n=batch_inputs.size(0))

        # Loss tracker, figure this out

        # return {
        #     "loss": self.loss_tracker.result(), 
        # }

        return loss, logsnr


    def test_step(self, x, cond):
        eps=1e-5              
        random_t = torch.rand((x.shape[0],1), device = self.device)            
        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)
        alpha.to(self.device)
        sigma.to(self.device)

        z = torch.randn((x.shape),dtype=torch.float32, device = self.device)
        
        perturbed_x = alpha*x + z * sigma
        pred_noise = self.model.forward(perturbed_x,random_t,cond)
        v = alpha* z - sigma* x

        losses = torch.square(pred_noise - v)                
        loss = torch.mean(torch.reshape(losses,(losses.shape[0], -1)))
        
        # Loss tracker, figure this out

        #self.loss_tracker.update_state(loss) 
        # return {
        #     "loss": self.loss_tracker.result(), 
        # }

        return loss, logsnr


    def call(self,x):        
        return self.model(x)
    
    def set_to_closest(self, x, allowed):
        x = x.unsqueeze(1)  # Adding a dimension to x for broadcasting
        diffs = torch.abs(x - allowed.to(self.device).float())
        closest_indices = torch.argmin(diffs, dim=1)
        closest_values = allowed[closest_indices]
        return closest_values

    def resample(self, cond, num_samples):
        # Modified to do resampling as needed with the DIRC detector
        
        # initial_cond = cond.repeat(num_samples, 1) # Creating "batch size", which is just the number of photons we want to reproduce

        samples = self.DDPMSampler(cond.repeat(num_samples, 1),
                                self.model,
                                data_shape=[num_samples,self.num_input],
                                unscale=True
                                ) # this is unscaled
                                #    const_shape = (-1,1)).numpy() 

        x = self.set_to_closest(samples[:,0],self.allowed_x)
        y = self.set_to_closest(samples[:,1],self.allowed_y)
        t = samples[:,2]

        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1).detach().cpu().numpy()
    
    def DDPMSampler(self,
                    cond,
                    model,
                    data_shape=None,
                    unscale=False):
        """Generate samples from score-based models with Predictor-Corrector method.
        
        Args:
        cond: Conditional input
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        eps: The smallest time step for numerical stability.
        
        Returns: 
        Samples.
        """
        model.eval()
        batch_size = cond.shape[0]
        x = self.prior_sde(data_shape)
        
        # print("x:",x.shape)
        
        # print("cond:",cond.shape)

        with torch.no_grad():
            for time_step in torch.arange(self.num_steps, 0, -1):
                random_t = torch.ones((batch_size, 1), dtype=torch.float, device=self.device) * time_step / self.num_steps
                logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)
                random_t_prev = torch.ones((batch_size, 1), dtype=torch.float, device=self.device) * (time_step - 1) / self.num_steps
                logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(random_t_prev)

                v = model.forward(x,random_t,cond) # During inference, training mode is off

                mean = alpha * x - sigma * v
                eps = sigma * x + alpha * v            
                x = alpha_ * mean + sigma_ * eps
                
            # The last step does not include any noise
        # print("v:",v.shape)
        if unscale:
            x = unscale_inputs(mean[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min'])
            y = unscale_inputs(mean[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min'])
            t = unscale_inputs(mean[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])

            mean = torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1) 
        # return the concatenation of the rescaled data

        return mean
    
    # --------------- hpDIRC sampling ----------------

    def __get_track(self,n_samples, cond):
        samples = self.DDPMSampler(cond.repeat(n_samples, 1),
                                self.model,
                                data_shape=[n_samples,self.num_input],
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