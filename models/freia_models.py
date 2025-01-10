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
from models.distributions import ConditionalDiagonalStudentT
from nflows.distributions.base import Distribution
from nflows.utils import torchutils
from typing import Union, Iterable, Tuple
import scipy
#from nflows.nn.nde.made import MixtureOfGaussiansMADE
from models.MADE import MixtureOfGaussiansMADE

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
            "P_max":8.5 ,"P_min":0.95 , "theta_max": 11.63,"theta_min": 0.90,"phi_max": 175.5, "phi_min":-176.0 },device='cuda'):
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

        if self.embedding:
            self.context_embedding = nn.Sequential(*[nn.Linear(context_shape,16),nn.ReLU(),nn.Linear(16,input_shape)])

        context_encoder =  nn.Sequential(*[nn.Linear(context_shape,16),nn.ReLU(),nn.Linear(16,input_shape*2)])

        #self.distribution = ConditionalDiagonalNormal(shape=[input_shape],context_encoder=context_encoder)
        #self.distribution = ConditionalDiagonalStudentT(shape=[input_shape],context_encoder=context_encoder)
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


        def create_freai(input_shape,layer,cond_shape):
            inn = Ff.SequenceINN(input_shape)
            #inn.append(InvertibleTanh)
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

        #print(embedded_context)
        #print(self.distribution)

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
        x = self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min'])
        y = self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min'])
        t = self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])

        x = self.set_to_closest(x,self._allowed_x)
        y = self.set_to_closest(y,self._allowed_y)
        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1).detach().cpu().numpy()

    def __get_track(self,num_samples,context):
        samples = self.__sample(num_samples,context)
        x = self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min']).round()
        y = self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min']).round()
        t = self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])

        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1)

    def _apply_mask(self, hits):
        # Time > 0 
        mask = torch.where((hits[:,2] > 0) & (hits[:,2] < self.stats_['time_max']))
        hits = hits[mask]
        # Outter bounds
        mask = torch.where((hits[:, 0] > self.stats_['x_min']) & (hits[:, 0] < self.stats_['x_max']) & (hits[:, 1] > self.stats_['y_min']) & (hits[:, 1] < self.stats_['y_max']))[0] # Acceptance mask
        hits = hits[mask]

        return hits

    def create_tracks(self,num_samples,context,plotting=False):
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
            

        x = self.set_to_closest(updated_hits[:,0],self._allowed_x).detach().cpu()
        y = self.set_to_closest(updated_hits[:,1],self._allowed_y).detach().cpu()
        t = updated_hits[:,2].detach().cpu()

        #pmtID = torch.div(x,torch.tensor(50,dtype=torch.int),rounding_mode='floor') + torch.div(y, torch.tensor(50,dtype=torch.int),rounding_mode='floor') * 18
        #row = (1.0/6.0) * ( y - 3 - 2* torch.div(pmtID,torch.tensor(18,dtype=torch.int),rounding_mode='floor'))
        #col = (1.0/6.0) * ( x - 3 - 2*(pmtID % 18))
        pmtID = torch.div(x,torch.tensor(58,dtype=torch.int),rounding_mode='floor') + torch.div(y, torch.tensor(58,dtype=torch.int),rounding_mode='floor') * 6
        col = (1.0/self.pixel_width) * (x - 2 - self.pixel_width/2. - (pmtID%6)*self.gapx)
        row = (1.0/self.pixel_height) * (y - 2 - self.pixel_height/2. - self.gapy * torch.div(pmtID,torch.tensor(6,dtype=torch.int),rounding_mode='floor'))

        assert(len(row) == num_samples)
        assert(len(col) == num_samples)
        assert(len(pmtID) == num_samples)

        P = self.unscale_conditions(context[0][0].detach().cpu().numpy(),self.stats_['P_max'],self.stats_['P_min'])
        Theta = self.unscale_conditions(context[0][1].detach().cpu().numpy(),self.stats_['theta_max'],self.stats_['theta_min'])
        #Theta = self.unscale_conditions(context[0].detach().cpu().numpy(),self.stats_['theta_max'],self.stats_['theta_min'])
        #Phi = self.unscale_conditions(context[0][2].detach().cpu().numpy(),self.stats_['phi_max'],self.stats_['phi_min'])
        Phi = 0.0

        if not plotting:
            return {"NHits":num_samples,"P":P,"Theta":Theta,"Phi":Phi,"x":x.numpy(),"y":y.numpy(),"leadTime":t.numpy(),"pmtID":pmtID.numpy()}
        else:
            return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1)

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



class ConditionalTStudent(Distribution):
    """A diagonal multivariate Normal whose parameters are functions of a context."""

    def __init__(self, shape, context_encoder=None,nu=1):
        """Constructor.

        Args:
            shape: list, tuple or torch.Size, the shape of the input variables.
            context_encoder: callable or None, encodes the context to the distribution parameters.
                If None, defaults to the identity function.
        """
        super().__init__()
        self._shape = torch.Size(shape)
        self._nu = nu
        if context_encoder is None:
            self._context_encoder = lambda x: x
        else:
            self._context_encoder = context_encoder
        self._sample_dist = torch.distributions.studentT.StudentT(df=torch.tensor([self._nu]),loc=torch.zeros(self._shape),scale=torch.ones(self._shape))

        self.const_ = scipy.special.loggamma(0.5*(self._nu + self._shape[0])) - scipy.special.loggamma(0.5 * nu) - 0.5 * self._shape[0] * np.log(np.pi * self._nu)

    def _compute_params(self, context):
        """Compute the means and log stds form the context."""
        if context is None:
            raise ValueError("Context can't be None.")

        params = self._context_encoder(context)
        if params.shape[-1] % 2 != 0:
            raise RuntimeError(
                "The context encoder must return a tensor whose last dimension is even."
            )
        if params.shape[0] != context.shape[0]:
            raise RuntimeError(
                "The batch dimension of the parameters is inconsistent with the input."
            )

        split = params.shape[-1] // 2
        means = params[..., :split].reshape(params.shape[0], *self._shape)
        log_stds = params[..., split:].reshape(params.shape[0], *self._shape)
        return means, log_stds

    def _log_prob(self, inputs, context):
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )

        # Compute parameters.
        means, log_stds = self._compute_params(context)
        assert means.shape == inputs.shape and log_stds.shape == inputs.shape

        # Compute log prob.
        norm_inputs = torchutils.sum_except_batch((inputs - means)**2,num_batch_dims=1)
        log_prob = self.const_ - 0.5 * (self._nu + self._shape[0])*torch.log(1.0 + (1.0/self._nu) * norm_inputs)
        return log_prob
        #norm_inputs = (inputs - means) * torch.exp(-log_stds)
        #log_prob = -0.5 * torchutils.sum_except_batch(
        #    norm_inputs ** 2, num_batch_dims=1
        #)
        # log_prob -= torchutils.sum_except_batch(log_stds, num_batch_dims=1)
        # log_prob -= self._log_z

        return log_prob

    def _sample(self, num_samples, context):
        # Compute parameters.
        means, log_stds = self._compute_params(context)
        stds = torch.exp(log_stds)
        means = torchutils.repeat_rows(means, num_samples)
        stds = torchutils.repeat_rows(stds, num_samples)

        # Generate samples.
        context_size = context.shape[0]
        #noise = torch.randn(context_size * num_samples, *
         #                   self._shape, device=means.device)
        noise = self._sample_dist.sample((context_size * num_samples,)).to(means.device)
        samples = means + stds * noise
        return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        means, _ = self._compute_params(context)
        return means