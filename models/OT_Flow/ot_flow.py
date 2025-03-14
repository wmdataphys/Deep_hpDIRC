import torch.nn as nn
import warnings
from typing import Callable
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows.distributions.base import Distribution
from nflows.utils import torchutils
from typing import Union, Iterable, Tuple
import scipy
from models.OT_Flow.OTFlowProblem import *
from models.OT_Flow.Phi import Phi
from models.MADE import MixtureOfGaussiansMADE
from utils.hpDIRC import ALLOWED_X,ALLOWED_Y

class OT_Flow(nn.Module):
    def __init__(self,input_shape,layers,context_shape,embedding=False,hidden_units=512,stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":380.00,"time_min":0.0,
            "P_max":8.5 ,"P_min":0.95 , "theta_max": 11.63,"theta_min": 0.90,"phi_max": 175.5, "phi_min":-176.0 },stepper='rk4',nt=6,nt_val=10,alph=[1.0,10.0,1.50],device='cuda',train_T=False,LUT_path=None):
        super(OT_Flow, self).__init__()
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
        self.stepper = stepper
        self.nt = nt
        self.nt_val = nt_val
        self.alph = alph
        self.train_T = train_T
        self.num_pixels = 16
        self.num_pmts_x = 6
        self.num_pmts_y = 4

        print("Using alph: ",self.alph)
        if train_T:
            T = 1.0
            self.register_parameter("end_time", nn.Parameter(torch.tensor(T),requires_grad=True))

        self.Phi = Phi(nTh=self.layers,m=self.hidden_units,d=self.input_shape,alph=self.alph,conditional_dim=self.context_shape).to(self.device)

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

        if self.embedding:
            self.context_embedding = nn.Sequential(*[nn.Linear(context_shape,16),nn.ReLU(),nn.Linear(16,input_shape)])

        context_encoder =  nn.Sequential(*[nn.Linear(context_shape,16),nn.ReLU(),nn.Linear(16,input_shape*2)])

        self.distribution = MixtureOfGaussiansMADE(
                                features=input_shape,
                                hidden_features=128,
                                context_features=context_shape,
                                num_blocks=2,
                                num_mixture_components=40,
                                use_residual_blocks=True,
                                random_mask=False,
                                activation=F.relu,
                                dropout_probability=0.0,
                                use_batch_norm=False,
                                epsilon=1e-2,
                                custom_initialization=True,
                                ).to(self.device)

       

    def log_prob(self, inputs, context,nt=12, tspan=[0,1],alpha=[1.0,1.0,1.0]):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context

        if self.train_T:
            tspan = [0,self.end_time]

        _,costs = OTFlowProblem(inputs,self.Phi,tspan,nt,alph=self.alph,conds=embedded_context,dist_func=self.distribution,training=False)

        return -1*costs[1]


    def compute_loss(self,inputs,context,nt,tspan=[0,1],alpha=[1.0,1.0,1.0]):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context

        if self.train_T:
            tspan = [0,self.end_time]

        cs,costs = OTFlowProblem(inputs,self.Phi,tspan,nt,alph=self.alph,conds=embedded_context,dist_func=self.distribution)

        return cs,costs

        

    def __sample(self, num_samples, context,nt, tspan=[1.0,0.0]):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context

        z = self.distribution.sample(num_samples,context=embedded_context)

        if embedded_context is not None:
            z = torchutils.merge_leading_dims(z, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        if self.train_T:
            tspan = [self.end_time.detach().item(),0]
        
        d = 3
        samples = integrate(z[:, 0:d],net=self.Phi,tspan=tspan,nt=nt,intermediates=False,conds=embedded_context,alph=self.alph)

        return samples

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
            
    def _sample(self,num_samples,context,nt=20):
        samples = self.__sample(num_samples,context,nt=nt)
        x = self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min'])
        y = self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min'])
        t = self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])


        x = self.set_to_closest(x,self._allowed_x)
        y = self.set_to_closest(y,self._allowed_y)
        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1).detach().cpu().numpy()

    def __get_track(self,num_samples,context,nt):
        samples = self.__sample(num_samples,context,nt=nt)
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

    def create_tracks(self,num_samples,context,nt=20,p=None,theta=None,fine_grained_prior=True):
        if num_samples is None:
            assert p is not None and theta is not None, "p and theta must be provided if num_samples is None."
            num_samples = self.__sample_photon_yield(p,theta)

        hits = self.__get_track(num_samples,context,nt)
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


    def to_noise(self,inputs,context):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context
        noise,_ = self.sequence.forward(inputs,rev=False,c=[embedded_context])

        return noise

    def sample_and_log_prob(self,num_samples,context):
        if self.embedding:
            embedded_context = self._embedding_net(context)
        else:
            embedded_context = context

        noise, log_prob = self.distribution.sample_and_log_prob(
            num_samples, context=embedded_context
        )
 
        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )
        samples, logabsdet = self.sequence.forward(noise,rev=True,c=[embedded_context])

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet