import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from typing import Callable
import numpy as np
import copy 
from nflows.utils import torchutils
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
from nflows.utils import torchutils
from models.MADE import MixtureOfGaussiansMADE
from torch import Tensor

def antiderivTanh(x): # activation function aka the antiderivative of tanh
    return torch.abs(x) + torch.log(1+torch.exp(-2.0*torch.abs(x)))

def derivTanh(x): # act'' aka the second derivative of the activation function antiderivTanh
    return 1 - torch.pow( torch.tanh(x) , 2 )

class ResNN(nn.Module):
    def __init__(self, d, m, nTh=2,conditional_dim=2):
        """
            ResNet N portion of Phi
        :param d:   int, dimension of space input (expect inputs to be d+1 for space-time)
        :param m:   int, hidden dimension
        :param nTh: int, number of resNet layers , (number of theta layers)
        """
        super().__init__()

        if nTh < 2:
            print("nTh must be an integer >= 2")
            exit(1)

        self.d = d
        self.m = m
        self.nTh = nTh
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(d + 1 + conditional_dim, m, bias=True)) # opening layer
        self.layers.append(nn.Linear(m,m, bias=True)) # resnet layers
        for i in range(nTh-2):
            self.layers.append(copy.deepcopy(self.layers[1]))
        self.act = antiderivTanh
        self.h = 1.0 / (self.nTh-1) # step size for the ResNet
        self.output_layer = nn.Linear(m,d)

    def forward(self, x,t, c=None):
        t = t.reshape(-1,1).float()

        if c is None:
            x = self.act(self.layers[0].forward(torch.concat([x,t],-1)))
        else:
            x = self.act(self.layers[0].forward(torch.concat([x,c,t],dim=-1)))

        for i in range(1,self.nTh):
            x = x + self.h * self.act(self.layers[i](x))

        return self.output_layer(x)

    def step(self, x,t,c=None):
        t = t.reshape(-1, 1).expand(x.shape[0], 1)

        if c is None:
            x = self.act(self.layers[0].forward(torch.concat([x,t],-1)))
        else:
            x = self.act(self.layers[0].forward(torch.concat([x,c,t],dim=-1)))

        for i in range(1,self.nTh):
            x = x + self.h * self.act(self.layers[i](x))

        return self.output_layer(x)

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        c = extras.get("model_extras", {}).get("c", None)
        return self.model.step(x, t, c=c)


class FlowMatching(nn.Module):
    def __init__(self,input_shape,layers,context_shape,embedding=False,hidden_units=512,stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":380.00,"time_min":0.0,
            "P_max":8.5 ,"P_min":0.95 , "theta_max": 11.63,"theta_min": 0.90,"phi_max": 175.5, "phi_min":-176.0 },device='cuda',LUT_path=None):
        super(FlowMatching, self).__init__()
        self.input_shape = input_shape
        self.layers = layers
        self.context_shape = context_shape
        self.embedding = embedding
        self.hidden_units = hidden_units
        self.photons_generated = 0
        self.photons_resampled = 0
        self.device = device
        self.gapx =  1.89216111455965 + 4.
        self.gapy = 1.3571428571428572 + 4.
        self.pixel_width = 3.3125
        self.pixel_height = 3.3125
        self.num_pixels = 16
        self.num_pmts_x = 6
        self.num_pmts_y = 4

        self.NN = ResNN(nTh=self.layers,m=self.hidden_units,d=self.input_shape,conditional_dim=self.context_shape)
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        # self.distribution = MixtureOfGaussiansMADE(
        #                 features=input_shape,
        #                 hidden_features=128,
        #                 context_features=context_shape,
        #                 num_blocks=2,
        #                 num_mixture_components=40,
        #                 use_residual_blocks=True,
        #                 random_mask=False,
        #                 activation=F.relu,
        #                 dropout_probability=0.0,
        #                 use_batch_norm=False,
        #                 epsilon=1e-2,
        #                 custom_initialization=True,
        #                 ).to(self.device)
        # print("Using MOG 40.")

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

        if LUT_path is not None:
            print("Loading photon yield sampler.")
            dicte = np.load(LUT_path,allow_pickle=True)
            self.LUT = {k: v for k, v in dicte.items() if k != "global_values"}
            self.global_values = dicte['global_values']
            self.p_points = np.array(list(dicte.keys())[:-1]) 
            self.theta_points = np.array(list(dicte[self.p_points[0]].keys()))


    def compute_loss(self, x_1, context):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context

        x_0 = torch.randn_like(x_1).to(x_1.device)
        #x_0 = self.distribution.sample(x_1.shape[0],context=embedded_context,inference=False).squeeze(0)

        t = torch.rand(x_1.shape[0]).to(x_1.device) 

        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        loss = torch.pow( self.NN(x=path_sample.x_t,t=path_sample.t,c=embedded_context) - path_sample.dx_t, 2).mean() 
        
        return loss
        

    def __sample(self, num_samples,nt, context=None):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context

        step_size = 1.0 / (2*nt) 
        wrapped_vf = WrappedModel(self.NN)
        T = torch.linspace(0,1,nt).to(self.device)
        #x_init = self.distribution.sample(num_samples,context=embedded_context,inference=True)

        if embedded_context is not None:
            #x_init = torchutils.merge_leading_dims(x_init, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        x_init = torch.randn((num_samples, self.input_shape), dtype=torch.float32, device=self.device)
        solver = ODESolver(velocity_model=wrapped_vf) 
        sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint', step_size=step_size, return_intermediates=False,model_extras={"c": embedded_context})

        return sol

    def log_prob(self,inputs,context,nt=18):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context

        step_size = 1.0 / (2*nt)
        T = torch.linspace(1,0,nt).to(self.device)

        def log_p(x: Tensor) -> Tensor:
            neg_energy = -0.5 * \
            torchutils.sum_except_batch(inputs ** 2, num_batch_dims=1)
            _log_z = 0.5 * x.shape[-1] * torch.log(2 * torch.tensor(math.pi))
            return neg_energy - _log_z

        wrapped_vf = WrappedModel(self.NN)
        solver = ODESolver(velocity_model=wrapped_vf)

        _, log_likelihood = solver.compute_likelihood(time_grid=T, x_1=inputs,log_p0=log_p,
                                                                 method='midpoint', step_size=step_size, return_intermediates=False,
                                                                 model_extras={"c": embedded_context}, exact_divergence=True)

        assert(log_likelihood.shape[0] == inputs.shape[0])

        return log_likelihood 

    def unscale(self,x,max_,min_):
        return x*0.5*(max_ - min_) + min_ + (max_-min_)/2

    def unscale_conditions(self,x,max_,min_):
        return x * 0.5 * (max_ - min_) + (max_ - min_)/2

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

    def _sample(self,num_samples,context,nt):
        samples = self.__sample(num_samples,context,nt=nt)

        x = self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min'])
        y = self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min'])
        t = self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])

        x = self.set_to_closest(x,self._allowed_x)
        y = self.set_to_closest(y,self._allowed_y)

        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1).detach().cpu().numpy()

    def __get_track(self,num_samples,context,nt):
        samples = self.__sample(num_samples,nt=nt,context=context)
        x = self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min'])#.round()
        y = self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min'])#.round()
        t = self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])

        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1)

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

    def __sample_photon_yield(self,p_value,theta_value):
        closest_p_idx = np.argmin(np.abs(self.p_points - p_value))
        closest_p = float(self.p_points[closest_p_idx])
        
        closest_theta_idx = np.argmin(np.abs(self.theta_points - theta_value))
        closest_theta = float(self.theta_points[closest_theta_idx])

        return int(np.random.choice(self.global_values,p=self.LUT[closest_p][closest_theta]))

    def create_tracks(self,num_samples,context,p=None,theta=None,nt=5,fine_grained_prior=True):
        if num_samples is None:
            assert p is not None and theta is not None, "p and theta must be provided if num_samples is None."
            num_samples = self.__sample_photon_yield(p,theta)

        hits = self.__get_track(num_samples,context,nt=nt)
        updated_hits = self._apply_mask(hits,fine_grained_prior=fine_grained_prior)
        n_resample = int(num_samples - len(updated_hits))

        self.photons_generated += len(hits)
        self.photons_resampled += n_resample
        while n_resample != 0:
            resampled_hits = self.__get_track(n_resample,context,nt=nt)
            updated_hits = torch.concat((updated_hits,resampled_hits),0)
            updated_hits = self._apply_mask(updated_hits,fine_grained_prior=fine_grained_prior)
            n_resample = int(num_samples - len(updated_hits))
            self.photons_resampled += n_resample
            self.photons_generated += len(resampled_hits)
            
        # Use euclidean distance
        x,y = self.set_to_closest_2d(updated_hits[:,:-1])
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

