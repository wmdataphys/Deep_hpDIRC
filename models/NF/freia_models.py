import torch.nn as nn
from FrEIA.modules.base import InvertibleModule
import warnings
from typing import Callable
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import special_ortho_group
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from nflows.distributions.base import Distribution
from nflows.utils import torchutils
from typing import Union, Iterable, Tuple
import scipy
from models.MADE import MixtureOfGaussiansMADE
from utils.hpDIRC import ALLOWED_X,ALLOWED_Y,gapx,gapy,pixel_width,pixel_height,npix,npmt

class ResNetBlock(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.linear1 = nn.Linear(hidden_units, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.activation = nn.ReLU()

    def forward(self, x):
        #x = self.activation(self.linear1(x) + x)
        #x = self.activation(self.linear2(x) + x)
        inputs = x
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x) + inputs)
        return x


class FreiaNet(nn.Module):
    def __init__(self,input_shape,layers,context_shape,embedding=False,hidden_units=512,num_blocks=2,stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":380.00,"time_min":0.0,
            "P_max":8.5 ,"P_min":0.95 , "theta_max": 11.63,"theta_min": 0.90,"phi_max": 175.5, "phi_min":-176.0 },device='cuda',LUT_path=None):
        super(FreiaNet, self).__init__()
        self.input_shape = input_shape
        self.layers = layers
        self.context_shape = context_shape
        self.embedding = embedding
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
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

        self.distribution = MixtureOfGaussiansMADE(
                                features=input_shape,
                                hidden_features=128,
                                context_features=context_shape,
                                num_blocks=6,
                                num_mixture_components=40,
                                use_residual_blocks=True,
                                random_mask=False,
                                activation=F.relu,
                                dropout_probability=0.0,
                                use_batch_norm=False,
                                epsilon=1e-2,
                                custom_initialization=True,
                                ).to(self.device)


        def create_freai(input_shape,layer,cond_shape):
            inn = Ff.SequenceINN(input_shape)
            for k in range(layers):
                inn.append(Fm.AllInOneBlock,cond=0,cond_shape=(cond_shape,),subnet_constructor=resnet_subnet, permute_soft=True)

            return inn

        def block(hidden_units):
            return [nn.Linear(hidden_units,hidden_units),nn.BatchNorm1d(hidden_units),nn.ReLU(),
                    nn.Linear(hidden_units,hidden_units),nn.BatchNorm1d(hidden_units),nn.ReLU()]

        def subnet_fc(c_in, c_out):
            blks = [nn.Linear(c_in,hidden_units)]
            for _ in range(num_blocks):
                blks += block(hidden_units)

            blks += [nn.Linear(hidden_units,c_out)]
            return nn.Sequential(*blks)

        def resnet_subnet(c_in, c_out):
            layers = [nn.Linear(c_in,hidden_units)]
            
            # Stack residual blocks
            for _ in range(num_blocks):
                layers.append(ResNetBlock(hidden_units))
            
            layers += [nn.Linear(hidden_units, c_out)]
            return nn.Sequential(*layers)

        self.sequence = create_freai(self.input_shape,self.layers,self.context_shape)

    def log_prob(self,inputs,context):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context

        noise,logabsdet = self.sequence.forward(inputs,rev=False,c=[embedded_context])
        log_prob = self.distribution.log_prob(noise,context=embedded_context)

        return log_prob + logabsdet

    def loss_function(self,inputs,context):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context

        inputs.requires_grad_(True)
        embedded_context.requires_grad_(True)

        noise,logabsdet = self.sequence.forward(inputs,rev=False,c=[embedded_context])
        log_prob = self.distribution.log_prob(noise,context=embedded_context)

        grad_log_prob = torch.autograd.grad(
            outputs=log_prob, 
            inputs=inputs, 
            grad_outputs=torch.ones_like(log_prob), 
            create_graph=True)[0]

        grad_log_prob = grad_log_prob.view(grad_log_prob.size(0), -1)

        grad_log_prob = torch.norm(grad_log_prob, dim=1, p=1) 

        return log_prob + logabsdet, grad_log_prob

    def __sample(self,num_samples,context):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context

        noise = self.distribution.sample(num_samples,context=embedded_context,device=self.device)

        if embedded_context is not None:
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )
    
        samples, _= self.sequence.forward(noise,rev=True,c=[embedded_context])

        return samples

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
            
    def __get_track(self,num_samples,context):
        samples = self.__sample(num_samples,context)
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

    def create_tracks(self,num_samples,context,p=None,theta=None,fine_grained_prior=True,dark_rate=None):
        if num_samples is None:
            assert p is not None and theta is not None, "p and theta must be provided if num_samples is None."
            num_samples = self.__sample_photon_yield(p,theta)

        hits = self.__get_track(num_samples,context)
        updated_hits = self._apply_mask(hits,fine_grained_prior=fine_grained_prior)
        n_resample = int(num_samples - len(updated_hits))
        #print(n_resample,num_samples)
        
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

        if dark_rate is not None:
            x,y,t,pmtID,pixelID,channel,dn_hits = self.add_dark_noise(np.concatenate([np.c_[x],np.c_[y],np.c_[t],np.c_[pmtID],np.c_[pixelID],np.c_[channel]],axis=1),dark_noise_pmt=dark_rate)
            num_samples += dn_hits
            return {"NHits":num_samples,"P":P,"Theta":Theta,"Phi":Phi,"x":x,"y":y,"leadTime":t,"pmtID":pmtID,"pixelID":pixelID,"channel":channel}
        else:
            return {"NHits":num_samples,"P":P,"Theta":Theta,"Phi":Phi,"x":x.numpy(),"y":y.numpy(),"leadTime":t.numpy(),"pmtID":pmtID.numpy(),"pixelID":pixelID.numpy(),"channel":channel.numpy()}





    # Based off of: https://github.com/rdom/eicdirc/blob/996e031d40825ce14292d1379fc173c54594ec5f/src/PrtPixelSD.cxx#L212
    # Dark rate coincides with -c 2031 in standalone simulation
    def add_dark_noise(self,hits,dark_noise_pmt=28000):
        # probability to have a noise hit in 100 ns window
        prob = dark_noise_pmt * 100 / 1e9
        new_hits = []
        for p in range(npmt):
            for i in range(int(prob) + 1):
                if(i == 0) and (prob - int(prob) < np.random.uniform()):
                    continue

                dn_time = 100 * np.random.uniform() # [1,100] ns
                dn_pix = int(npix * np.random.uniform())
                dn_channel = int(dn_pix * p)
                row = (p//6) * 16 + dn_pix//16 
                col = (p%6) * 16 + dn_pix%16
                x = 2 + col * pixel_width + (p % 6) * gapx + (pixel_width) / 2. # Center at middle
                y = 2 + row * pixel_height + (p // 6) * gapy + (pixel_height) / 2. # Center at middle
                # x,y,t,pmtID,pixelID,channel
                h = [x,y,dn_time,p,dn_pix,dn_channel]
                new_hits.append(h)

        if new_hits:
            new_hits = np.array(new_hits)
            hits = np.vstack([hits,new_hits])
            return hits[:,0],hits[:,1],hits[:,2],hits[:,3],hits[:,4],hits[:,5],hits.shape[0]
        else:
            return hits[:,0],hits[:,1],hits[:,2],hits[:,3],hits[:,4],hits[:,5],0

    def to_noise(self,inputs,context):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context
        noise,_ = self.sequence.forward(inputs,rev=False,c=[embedded_context])

        return noise


        
 
