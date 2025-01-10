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
import torchdiffeq
from torchdiffeq import odeint_adjoint as odeint
#from torchdiffeq import odeint 
from models.diffeq_layers import ConcatSquashLinear
from models.cnf_reg import RegularizedODEfunc,l2_regularzation_fn

def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()

def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


class ODENetwork(nn.Module):
    def __init__(self,input_shape,hidden_units,hidden_layers=1,activation_fn=torch.nn.Softplus):
        super().__init__()
        self.activation_fn = activation_fn()
        self.hidden_units = hidden_units
        self.input_shape = input_shape

        layers_ = [ConcatSquashLinear(self.input_shape,self.hidden_units)]

        for _ in range(hidden_layers):
            layers_.append(ConcatSquashLinear(self.hidden_units,self.hidden_units))

        layers_.append(ConcatSquashLinear(self.hidden_units,self.input_shape))

        self.layers = nn.ModuleList(layers_)

    def forward(self,t,y):
        dx = y

        for l,layer in enumerate(self.layers):
            dx = layer(t,dx)

            if l < len(self.layers) - 1:
                dx = self.activation_fn(dx)
            
        return dx
        
        

class CNFOdeFunc(nn.Module):
    def __init__(self, input_shape, context_shape, hidden_units=512, num_layers=2, embed_out=1, context_embedding=None, divergence_fn=divergence_approx):
        super(CNFOdeFunc, self).__init__()

        self.context_embedding = context_embedding
        self.divergence_fn = divergence_fn
        self.rademacher = False  # Default to Gaussian noise
        self._e = None  # Placeholder for noise
        self.residual = False  # Optional: enable residual connections
        self.register_buffer("_num_evals", torch.tensor(0.))

        if self.context_embedding is None:
            self.context_shape = context_shape
        else:
            self.context_shape = embed_out

        self.ode_network = ODENetwork(input_shape,hidden_units,hidden_layers=num_layers)


    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) >= 2, "States must include [y, log_det] and optionally more context."
        y = states[0]
        log_det = states[1]
        batch_size = y.shape[0]

        if t.dim() == 0:  # Scalar time value
            t = t.unsqueeze(0)

        self._num_evals += 1

        # Sample and fix noise for divergence computation
        if self._e is None:
            if self.rademacher:
                self._e = sample_rademacher_like(y)
            else:
                self._e = sample_gaussian_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)

            # Handle additional states
            for s_ in states[2:]:
                s_.requires_grad_(True)

            # Compute dynamics
            dy = self.ode_network(t,y)

            # Compute divergence
            if self.divergence_fn is not None:
                divergence = self.divergence_fn(dy, y, e=self._e).view(batch_size, 1)
            else:
                raise ValueError("A divergence function must be provided.")

        # Handle residual dynamics if enabled
        if self.residual:
            dy = dy - y
            divergence -= torch.tensor(np.prod(y.shape[1:]), dtype=torch.float32).to(divergence)

        # Return tuple with dy, updated log_det, and other states
        return (dy, -divergence) + tuple(torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:])



def get_regularization(model, regularization_coeffs):
    if len(regularization_coeffs) == 0:
        return None

    acc_reg_states = tuple([0.] * len(regularization_coeffs))
    for module in model.modules():
        if isinstance(module, CNF_Module):
            acc_reg_states = tuple(acc + reg for acc, reg in zip(acc_reg_states, module.get_regularization_states()))
    return acc_reg_states

class CNF_Module(nn.Module):
    def __init__(self,ode,T=1.0,train_T=False,method='rk4',atol=1e-5,rtol=1e-5,reg_funcs=[l2_regularzation_fn]):
        super(CNF_Module,self).__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))
        
        self.cnf_ode = RegularizedODEfunc(ode,regularization_fns=reg_funcs)
        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.n_reg = len(reg_funcs)
        self.regularization_states = None

    def get_regularization_states(self):
        reg_states = self.regularization_states
        self.regularization_states = None
        return reg_states

    def num_evals(self,):
        return self.cnf_ode._num_evals.item()


    def forward(self,z,log_t0=None,integration_times=None,reverse=False):
        b_ = z.shape[0]

        if log_t0 == None:
            log_t0  = torch.zeros(b_, 1).type(torch.float32).to(z.device)

        else:
            pass

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z.device)

        if reverse:
            integration_times = _flip(integration_times,0)

        self.cnf_ode.before_odeint()

        reg_states = tuple(torch.tensor(0).to(z) for _ in range(self.n_reg))

        if self.training:
            state = odeint(self.cnf_ode,(z,log_t0) + reg_states,integration_times,atol=self.atol,rtol=self.rtol,method=self.method)
        else:
            state = odeint(self.cnf_ode,(z,log_t0),integration_times,atol=self.atol,rtol=self.rtol,method=self.method)

        if len(integration_times) == 2:
            state = tuple(s[1] for s in state)

        z_t,logp_z_t = state[:2]
        self.regularization_states = state[2:]

        if log_t0 is not None:
            return z_t, logp_z_t
        else:
            return z_t
        

class CNFFlow(nn.Module):
    def __init__(self, layersList):
        super(CNFFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, reverse=reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, logpx, reverse=reverse)
            return x, logpx


class CNF(nn.Module):
    def __init__(self,input_shape,layers,context_shape,embedding=False,hidden_units=512,num_blocks=2,stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":380.00,"time_min":0.0,
            "P_max":8.5 ,"P_min":0.95 , "theta_max": 11.63,"theta_min": 0.90,"phi_max": 175.5, "phi_min":-176.0 },device='cuda',train_T=False,T=1.0):
        super(CNF, self).__init__()
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
        self.int_method = "dopri5"
        self.T = T

        layers_ = []

        for _ in range(layers):
            ode_ = CNFOdeFunc(input_shape, context_shape, hidden_units=hidden_units,num_layers=num_blocks,embed_out=1,context_embedding=None)
            layers_.append(CNF_Module(ode_,T=1.0,train_T=train_T,method=self.int_method,atol=1e-5,rtol=1e-5))

        self.cnf_flows = CNFFlow(layers_)

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
                                                   37.86705557, 341.17955557, 344.49205557, 347.80455557])).to(self.device)
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

        self.distribution = ConditionalDiagonalNormal(shape=[input_shape],context_encoder=context_encoder)

        #self.cnf_ode = CNFOdeFunc(input_shape, context_shape, hidden_units=hidden_units,num_layers=layers,embed_out=1,context_embedding=None)

    def log_prob(self, inputs, context, t_span=(0.0, 1.0), num_steps=100,integration_times=None):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context

        zero = torch.zeros(inputs.shape[0], 1).to(inputs.device)

        z, log_det = self.cnf_flows(inputs,zero)
        log_prob = self.distribution.log_prob(z,context=embedded_context)

        return log_prob - log_det

    def __sample(self, num_samples, context, t_span=(0.0, 1.0), num_steps=100,integration_times=None):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context

        z, log_p_t0 = self.distribution.sample_and_log_prob(num_samples, context=embedded_context)


        if embedded_context is not None:
            z = torchutils.merge_leading_dims(z, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples,_ = self.cnf_flows(z,log_p_t0,reverse=True)

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
        counter = 0
        hits = self.__get_track(num_samples,context)
        #print("hits",hits.shape)
        updated_hits = self._apply_mask(hits)
        n_resample = int(num_samples - len(updated_hits))

        #print(updated_hits)
        

        self.photons_generated += len(hits)
        self.photons_resampled += n_resample
        while n_resample != 0:
            counter += 1
            resampled_hits = self.__get_track(n_resample,context)
            updated_hits = torch.concat((updated_hits,resampled_hits),0)
            updated_hits = self._apply_mask(updated_hits)
            n_resample = int(num_samples - len(updated_hits))
            self.photons_resampled += n_resample
            self.photons_generated += len(resampled_hits)
            # if counter == 10:
            #     break
            

        x = self.set_to_closest(updated_hits[:,0],self._allowed_x).detach().cpu()
        y = self.set_to_closest(updated_hits[:,1],self._allowed_y).detach().cpu()
        t = updated_hits[:,2].detach().cpu()

        ##print(updated_hits.shape)

        #pmtID = torch.div(x,torch.tensor(50,dtype=torch.int),rounding_mode='floor') + torch.div(y, torch.tensor(50,dtype=torch.int),rounding_mode='floor') * 18
        #row = (1.0/6.0) * ( y - 3 - 2* torch.div(pmtID,torch.tensor(18,dtype=torch.int),rounding_mode='floor'))
        #col = (1.0/6.0) * ( x - 3 - 2*(pmtID % 18))
        pmtID = torch.div(x,torch.tensor(58,dtype=torch.int),rounding_mode='floor') + torch.div(y, torch.tensor(58,dtype=torch.int),rounding_mode='floor') * 6
        col = (1.0/self.pixel_width) * (x - 2 - self.pixel_width/2. - (pmtID%6)*self.gapx)
        row = (1.0/self.pixel_height) * (y - 2 - self.pixel_height/2. - self.gapy * torch.div(pmtID,torch.tensor(6,dtype=torch.int),rounding_mode='floor'))

        #assert(len(row) == num_samples)
        #assert(len(col) == num_samples)
        #assert(len(pmtID) == num_samples)

        P = self.unscale_conditions(context[0][0].detach().cpu().numpy(),self.stats_['P_max'],self.stats_['P_min'])
        Theta = self.unscale_conditions(context[0][1].detach().cpu().numpy(),self.stats_['theta_max'],self.stats_['theta_min'])
        #Theta = self.unscale_conditions(context[0].detach().cpu().numpy(),self.stats_['theta_max'],self.stats_['theta_min'])
        #Phi = self.unscale_conditions(context[0][2].detach().cpu().numpy(),self.stats_['phi_max'],self.stats_['phi_min'])
        Phi = 0.0

        #print(num_samples / len(x))

        #print('here',len(pmtID))

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