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
#from models.torch_mnf.layers import MNFLinear
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.utils import torchutils
from typing import Union, Iterable, Tuple



# Domain of inverse tangent is (-1,1) -> Careful
class InvertibleTanh(InvertibleModule):
    def __init__(self, dims_in, **kwargs):
        super().__init__(dims_in, **kwargs)

    def output_dims(self, dims_in):
        return dims_in

    def forward(self, x_or_z: Iterable[torch.Tensor], c: Iterable[torch.Tensor] = None,
                rev: bool = False, jac: bool = True) \
            -> Tuple[Tuple[torch.Tensor], torch.Tensor]:
        x_or_z = x_or_z[0]
        if not rev:
            # S(x)
            result = torch.tanh(x_or_z)
            logabsdet = torch.log(1 - result ** 2)
            logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        else:
            # S^-1(z)
            result = 0.5 * torch.log((1 + x_or_z) / (1.0 - x_or_z))
            logabsdet = -torch.log(1 - x_or_z ** 2)
            logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        if not jac:
            return (result, )


        return ((result, ), logabsdet)

class FreiaNet(nn.Module):
    def __init__(self,input_shape,layers,context_shape,embedding=False,hidden_units=512,num_blocks=2,log_time=False
    ,stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":380.00,"time_min":0.0,
            "P_max":8.5 ,"P_min":0.95 , "theta_max": 11.63,"theta_min": 0.90,"phi_max": 175.5, "phi_min":-176.0 }
    ,device='cuda'):
        #     conditional_maxes = np.array([8.5,11.63,175.5])
        #     conditional_mins = np.array([0.95,0.90,-176.])
        super(FreiaNet, self).__init__()
        self.input_shape = input_shape
        self.layers = layers
        self.context_shape = context_shape
        self.embedding = embedding
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.burn_in = 1000
        self.photons_generated = 0
        self.photons_resampled = 0
        self.device = device

        self._allowed_x = torch.tensor(np.array([  3.,   9.,  15.,  21.,  27.,  33.,  39.,  45.,   # 0
                                                    53.,  59.,  65., 71.,  77.,  83.,  89.,  95.,  # 1
                                                    103., 109., 115., 121., 127., 133., 139., 145.,# 2
                                                    153., 159., 165., 171., 177., 183., 189., 195.,# 3
                                                    203., 209., 215., 221., 227., 233., 239., 245.,# 4 
                                                    253., 259., 265., 271.,277., 283., 289.,  295.,# 5
                                                    303., 309., 315., 321., 327., 333., 339., 345.,# 6
                                                    353., 359., 365., 371., 377., 383., 389., 395.,# 7 
                                                    403., 409., 415., 421., 427., 433., 439., 445.,# 8
                                                    453., 459., 465., 471., 477., 483., 489., 495.,# 9 
                                                    503., 509., 515., 521., 527., 533., 539., 545.,# 10
                                                    553., 559., 565., 571., 577., 583., 589., 595.,# 11
                                                    603., 609., 615., 621., 627., 633., 639., 645.,# 12 
                                                    653., 659., 665., 671., 677., 683., 689., 695.,# 13
                                                    703., 709., 715., 721., 727., 733., 739., 745.,# 14 
                                                    753., 759., 765., 771., 777., 783., 789., 795.,# 15 
                                                    803., 809., 815., 821., 827., 833., 839., 845.,# 16
                                                    853., 859., 865., 871., 877., 883., 889., 895.])).to(self.device) # 17
        self._allowed_y = torch.tensor(np.array([  3.,   9.,  15.,  21.,  27.,  33.,  39.,  45.,  # 0
                                                   53.,  59.,  65.,71.,  77.,  83.,  89.,  95.,   # 1
                                                   103., 109., 115., 121., 127., 133.,139., 145., # 2
                                                   153., 159., 165., 171., 177., 183., 189., 195.,# 3  
                                                   203., 209., 215., 221., 227., 233., 239., 245.,# 4 
                                                   253., 259., 265., 271.,277., 283., 289., 295.])).to(self.device) # 5
        self.stats_ = stats
        self.log_time = log_time

        if self.log_time:
            self.stats_['time_max'] = 5.931767619849855
            self.stats_['time_min'] = -10.870140433500834
            self.stats_['x_max'] = 6.800170048114738
            self.stats_['x_min'] = -11.639826026001888
            self.stats_['y_max'] = 5.697093360008697
            self.stats_['y_min'] = -11.012369390162362

        if self.embedding:
            self.context_embedding = nn.Sequential(*[nn.Linear(context_shape,16),nn.ReLU(),nn.Linear(16,input_shape)])

        context_encoder =  nn.Sequential(*[nn.Linear(context_shape,16),nn.ReLU(),nn.Linear(16,input_shape*2)])

        self.distribution = ConditionalDiagonalNormal(shape=[input_shape],context_encoder=context_encoder)

        def create_freai(input_shape,layer,cond_shape):
            inn = Ff.SequenceINN(input_shape)
            #inn.append(InvertibleTanh)
            for k in range(layers):
                inn.append(Fm.AllInOneBlock,cond=0,cond_shape=(cond_shape,),subnet_constructor=subnet_fc, permute_soft=True)

            return inn

        def block(hidden_units):
            return [nn.Linear(hidden_units,hidden_units),nn.ReLU(),nn.Linear(hidden_units,hidden_units),nn.ReLU()]

        def subnet_fc(c_in, c_out):
            blks = [nn.Linear(c_in,hidden_units)]
            for _ in range(num_blocks):
                blks += block(hidden_units)

            blks += [nn.Linear(hidden_units,c_out)]
            return nn.Sequential(*blks)

        self.sequence = create_freai(self.input_shape,self.layers,self.context_shape)

    def log_prob(self,inputs,context):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context

        noise,logabsdet = self.sequence.forward(inputs,rev=False,c=[embedded_context])
        log_prob = self.distribution.log_prob(noise,context=embedded_context)

        return log_prob + logabsdet


    def __sample(self,num_samples,context):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context

        noise = self.distribution.sample(num_samples,context=embedded_context)

        if embedded_context is not None:
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, _ = self.sequence.forward(noise,rev=True,c=[embedded_context])

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
            
    def _sample(self,num_samples,context):
        samples = self.__sample(num_samples,context)
        if self.log_time:
            x = torch.exp(self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min']))
            y = torch.exp(self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min']))
            t = torch.exp(self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min']))
        else:
            x = self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min'])
            y = self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min'])
            t = self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])

        x = self.set_to_closest(x,self._allowed_x)
        y = self.set_to_closest(y,self._allowed_y)
        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1).detach().cpu().numpy()

    def __get_track(self,num_samples,context):
        samples = self.__sample(num_samples,context)
        if self.log_time:
            x = torch.exp(self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min']))
            y = torch.exp(self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min']))
            t = torch.exp(self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min']))
        else:
            x = self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min'])
            y = self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min'])
            t = self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])

        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1)

    def __apply_mask(self, hits):
        mask = torch.where((hits[:, 0] > 0) & (hits[:, 0] < 898) & (hits[:, 1] > 0) & (hits[:, 1] < 298))[0] # Acceptance mask
        hits = hits[mask]
        
        top_row_mask = torch.where(~((hits[:, 1] > 249) & (hits[:, 0] < 351)))[0] # rejection mask (keep everything not identified)
        hits = hits[top_row_mask]
        
        bottom_row_mask = torch.where(~((hits[:, 1] < 51) & (hits[:, 0] < 551)))[0] # rejection mask (keep everything not identified)
        hits = hits[bottom_row_mask]

        return hits

    def create_tracks(self,num_samples,context):
        hits = self.__get_track(num_samples,context)
        updated_hits = self.__apply_mask(hits)
        n_resample = int(num_samples - len(updated_hits))
        

        self.photons_generated += len(hits)
        self.photons_resampled += n_resample
        while n_resample != 0:
            resampled_hits = self.__get_track(n_resample,context)
            updated_hits = torch.concat((updated_hits,resampled_hits),0)
            updated_hits = self.__apply_mask(updated_hits)
            n_resample = int(num_samples - len(updated_hits))
            

        x = self.set_to_closest(updated_hits[:,0],self._allowed_x).detach().cpu()
        y = self.set_to_closest(updated_hits[:,1],self._allowed_y).detach().cpu()
        t = updated_hits[:,2].detach().cpu()

        pmtID = torch.div(x,torch.tensor(50,dtype=torch.int),rounding_mode='floor') + torch.div(y, torch.tensor(50,dtype=torch.int),rounding_mode='floor') * 18
        row = (1.0/6.0) * ( y - 3 - 2* torch.div(pmtID,torch.tensor(18,dtype=torch.int),rounding_mode='floor'))
        col = (1.0/6.0) * ( x - 3 - 2*(pmtID % 18))

        assert(len(row) == num_samples)
        assert(len(col) == num_samples)
        assert(len(pmtID) == num_samples)

        P = self.unscale_conditions(context[0][0].detach().cpu().numpy(),self.stats_['P_max'],self.stats_['P_min'])
        Theta = self.unscale_conditions(context[0][1].detach().cpu().numpy(),self.stats_['theta_max'],self.stats_['theta_min'])
        Phi = self.unscale_conditions(context[0][2].detach().cpu().numpy(),self.stats_['phi_max'],self.stats_['phi_min'])
        #data_dict = {"NHits: ",len(row),"P":,"Theta":Theta,"Phi":Phi,"row":row,"column":col,"leadTime":t}
        return {"NHits":num_samples,"P":P,"Theta":Theta,"Phi":Phi,"row":row.numpy(),"column":col.numpy(),"leadTime":t.numpy(),"pmtID":pmtID.numpy()}
        #return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1)

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

    def prior(self,hits,ux=898,lx=0,uy=298,ly=0,pmt_off=None): # pmt_off is a list of turned off readout sections 
        prior = torch.ones_like(hits[:,0])
        prior = prior * (hits[:,0] > lx).float() * (hits[:,0] < ux).float() * (hits[:,1] > ly).float() * (hits[:,1] < uy) * (hits[:,2] > 0.0).float()
        return prior
        
    def trace_path(self,matrix, current_state, chain,photon_yield):
        if len(chain) < photon_yield:
            if current_state < matrix.size(0):
                idx = (matrix[current_state] == True).nonzero()
                if idx.numel() == 0:
                    if len(chain) > 0:
                        chain[-1] = current_state
                    return
                #print(current_state)
                elif idx.numel() > 0:
                    #print(current_state)
                    idx = idx[0].item()
                    n_stays = idx - 1  # Number of times current state remains the same
                    chain.extend([current_state] * n_stays)  # Append the current state to the chain multiple times
                    self.trace_path(matrix, idx, chain,photon_yield)  # Continue tracing from the next state
                elif current_state != len(chain):
                    # If no transitions are found from the current state and it's not the last state in the chain, terminate the recursion
                    return

        
    def probabalistic_sample(self,pre_compute_dist,context,photon_yield):
            samples, log_prob = self.sample_and_log_prob(pre_compute_dist,context)
            samples = samples.squeeze(0)
            log_prob = log_prob.squeeze(0)

            if self.log_time:
                x = torch.exp(self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min'])).round()
                y = torch.exp(self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min'])).round()
                t = torch.exp(self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min']))
            else:
                x = self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min']).round()
                y = self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min']).round()
                t = self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])
            
            h = torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1)
            prior = self.prior(h)
            # Restrict sampling from pior
            non_zero_p = torch.where(prior != 0.0)
            h = h[non_zero_p]
            log_prob = log_prob[non_zero_p]

            transition_matrix = log_prob.unsqueeze(1) - log_prob.unsqueeze(0)
            # Only forward sampling, its stochastic so its fine
            accept_ = torch.triu(torch.rand(transition_matrix.shape).to(self.device) < torch.min(torch.ones_like(transition_matrix),transition_matrix),diagonal=1)
            
            chain = []
            current_state = 0
            self.trace_path(accept_, current_state, chain,photon_yield)
            #print(chain)
            chain_hits = h[np.array(chain)]
            
            x = self.set_to_closest(chain_hits[:,0],self._allowed_x)
            y = self.set_to_closest(chain_hits[:,1],self._allowed_y)
            chain = torch.concat((x.unsqueeze(1),y.unsqueeze(1),chain_hits[:,2].unsqueeze(1).detach().cpu()),axis=1).numpy()
            return chain

    # def prior(self,hits,ux=898,lx=0,uy=298,ly=0,pmt_off=None): # pmt_off is a list of turned off readout sections 
    #     if (hits[:,0] < ux) and (hits[:,0] > lx):
    #         if (hits[:,1] < uy) and (hits[:,1] > ly):
    #             return torch.ones_like(hits[:,0])
    #         else:
    #             return torch.zeros_like(hits[:,0])
    #     else:
    #         return torch.zeros_like(hits[:,0])
        
        
    # def probabalistic_sample(self,pre_compute_dist,context,photon_yield):
    #     samples, log_prob = self.sample_and_log_prob(pre_compute_dist,context)
    #     samples = samples.squeeze(0)
    #     log_prob = log_prob.squeeze(0)
    #     log_prob = torch.exp(log_prob) 

    #     if self.log_time:
    #         x = torch.exp(self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min'])).round()
    #         y = torch.exp(self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min'])).round()
    #         t = torch.exp(self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min']))
    #     else:
    #         x = self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min']).round()
    #         y = self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min']).round()
    #         t = self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])

    #     h = torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1)
    #     #print('h',h.shape)
    #     start_idx = np.random.randint(0,h.shape[0])

    #     init_prior = torch.tensor(0.0)
    #     while init_prior == 0:
    #         init_sample = h[start_idx]
    #         init_prob = log_prob[start_idx]
    #         init_prior = self.prior(init_sample)
    #         start_idx = np.random.randint(0,h.shape[0])

    #     chain = torch.zeros((photon_yield,3)).to('cuda')
    #     probabilities = []
    #     i = 0
    #     c_len = 0
    #     while (c_len < photon_yield):
    #         proposed_idx = np.random.randint(0,h.shape[0])
    #         proposed_sample = h[proposed_idx]
    #         proposed_prob = log_prob[proposed_idx]
    #         numerator = proposed_prob * self.prior(proposed_sample)
    #         denominator = init_prob * self.prior(init_sample)

    #         delta_log = torch.log(numerator+1e-50) - torch.log(denominator+1e-50)
    #         acceptance_ratio = torch.min(torch.stack((torch.ones_like(delta_log), torch.exp(delta_log)), dim=0))
    #         if torch.rand(1) < acceptance_ratio.detach().cpu():
    #             init_sample = proposed_sample
    #             init_prob = proposed_prob
    #             if i > self.burn_in:
    #                 chain[c_len] = proposed_sample
    #                 c_len += 1
    #         else:
    #             if i > self.burn_in:
    #                 chain[c_len] = init_sample
    #                 c_len += 1

    #         i += 1
        
    #     x = self.set_to_closest(chain[:,0],self._allowed_x)
    #     y = self.set_to_closest(chain[:,1],self._allowed_y)
    #     chain = torch.concat((x.unsqueeze(1),y.unsqueeze(1),chain[:,2].unsqueeze(1).detach().cpu()),axis=1).numpy()
    #     return chain


        # def probabalistic_sample(self,pre_compute_dist,context,photon_yield):
        #     samples, log_prob = self.sample_and_log_prob(pre_compute_dist,context)
        #     samples = samples.squeeze(0)
        #     log_prob = log_prob.squeeze(0)

        #     if self.log_time:
        #         x = torch.exp(self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min'])).round()
        #         y = torch.exp(self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min'])).round()
        #         t = torch.exp(self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min']))
        #     else:
        #         x = self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min']).round()
        #         y = self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min']).round()
        #         t = self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])
            
        #     h = torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1)
        #     prior = self.prior(h)
        #     # Restrict sampling from pior
        #     non_zero_p = torch.where(prior != 0.0)
        #     h = h[non_zero_p]
        #     log_prob = log_prob[non_zero_p]
        #     prior = prior[non_zero_p]

        #     chain = torch.zeros((photon_yield,3)).to('cuda')
        #     probabilities = []
        #     i = 0
        #     c_len = 0
        #     while (c_len < photon_yield):
        #         proposed_idx = np.random.randint(0,h.shape[0])
        #         proposed_sample = h[proposed_idx]
        #         proposed_prob = log_prob[proposed_idx]

        #         delta_log  = proposed_prob - init_prob
        #         acceptance_ratio = torch.min(torch.stack((torch.ones_like(delta_log), torch.exp(delta_log)), dim=0))
        #         if torch.rand(1) < acceptance_ratio.detach().cpu():
        #             init_sample = proposed_sample
        #             init_prob = proposed_prob
        #             if i > self.burn_in:
        #                 chain[c_len] = proposed_sample
        #                 c_len += 1
        #         else:
        #             if i > self.burn_in:
        #                 chain[c_len] = init_sample
        #                 c_len += 1

        #         i += 1
            
        #     x = self.set_to_closest(chain[:,0],self._allowed_x)
        #     y = self.set_to_closest(chain[:,1],self._allowed_y)
        #     chain = torch.concat((x.unsqueeze(1),y.unsqueeze(1),chain[:,2].unsqueeze(1).detach().cpu()),axis=1).numpy()
        #     return chain


