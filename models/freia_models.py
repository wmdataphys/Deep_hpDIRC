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
from models.torch_mnf.layers import MNFLinear
from sequence_inn import SequenceINN

class SimpleAbsSurjection():
    '''
    An absolute value layer.
    Uses a fixed inverse which flips the sign with probability 0.5.
    This enforces symmetry across all axes.
    '''

    stochastic_forward = False

    def forward(self, x):
        z = x.abs()
        ldj = - x.new_ones(x.shape[0]) * math.log(2) * x.shape[1:].numel()
        return z, ldj

    def inverse(self, z):
        s = torch.bernoulli(0.5*torch.ones_like(z))
        x = (2*s-1)*z
        return x

class SimpleSortSurjection(nn.Module):
    '''
    A sorting layer. Sorts along `dim` for element extracted using `lambd`.
    The inverse is a random permutation.

    Args:
        dim: int, the dimension along which the tensor is sorted.
        lambd: callable, a callable which extracts a subset of x which is used to determine the sorting order.

    Example for (1,4) tensor using (dim=1, lambd=lambda x: x):
    # Input x, output z
    tensor([[0.6268, 0.0913, 0.8587, 0.2548]])
    tensor([[0.0913, 0.2548, 0.6268, 0.8587]])

    Example for (1,4,2) tensor using (dim=1, lambd=lambda x: x[:,:,0]):
    # Input x, output z
    tensor([[[0.6601, 0.0948],
             [0.9293, 0.1715],
             [0.5511, 0.7153],
             [0.3567, 0.7232]]])
    tensor([[[0.3567, 0.7232],
             [0.5511, 0.7153],
             [0.6601, 0.0948],
             [0.9293, 0.1715]]])

    '''
    stochastic_forward = False

    def __init__(self, dim=1, lambd=lambda x: x):
        super(SimpleSortSurjection, self).__init__()
        self.register_buffer('buffer', torch.zeros(1))
        self.dim = dim
        self.lambd = lambd

    def forward(self, x):
        x_order = self.lambd(x)
        assert x_order.dim() == 2, 'lambd should return a tensor of shape (batch_size, dim_size) = ({}, {}), not {}'.format(x.shape[0], x.shape[self.dim], x_order.shape)
        assert x_order.shape[0] == x.shape[0], 'lambd should return a tensor of shape (batch_size, dim_size) = ({}, {}), not {}'.format(x.shape[0], x.shape[self.dim], x_order.shape)
        assert x_order.shape[1] == x.shape[self.dim], 'lambd should return a tensor of shape (batch_size, dim_size) = ({}, {}), not {}'.format(x.shape[0], x.shape[self.dim], x_order.shape)
        permutation = torch.argsort(x_order, dim=1)
        for d in range(1, self.dim):
            permutation = permutation.unsqueeze(1)
        for d in range(self.dim+1, x.dim()):
            permutation = permutation.unsqueeze(-1)
        permutation = permutation.expand_as(x)
        z = torch.gather(x, self.dim, permutation)
        #ldj = - self.buffer.new_ones(x.shape[0]) * torch.arange(1, 1+x.shape[self.dim]).type(self.buffer.dtype).log().sum()
        ldj = - self.buffer.new_ones(x.shape[0]) * torch.arange(1, 1+x.shape[self.dim]).type(self.buffer.dtype).log().sum()
        return z, ldj

    def inverse(self, z):
        rand = torch.rand(z.shape[0], z.shape[self.dim], device=z.device)
        permutation = rand.argsort(dim=1)
        for d in range(1, self.dim):
            permutation = permutation.unsqueeze(1)
        for d in range(self.dim+1, z.dim()):
            permutation = permutation.unsqueeze(-1)
        permutation = permutation.expand_as(z)
        x = torch.gather(z, self.dim, permutation)
        return x,torch.tensor(1.0)

class VariationalLayer(nn.Module):
    def __init__(self,):
        super(VariationalLayer, self).__init__()
        self.in_c = 6
        #self.lower_dim = 21
        self.mu_net  = nn.Sequential(*[nn.Linear(self.in_c,128),nn.ReLU(),nn.Linear(128,self.in_c)])
        self.diag_net = nn.Sequential(*[nn.Linear(self.in_c,128),nn.ReLU(),nn.Linear(128,self.in_c)])

    def forward(self, x,temp=0.1):
        mu = torch.tanh(self.mu_net(x))
        diag_cov = self.diag_net(x) + 1e-10
        diag_cov = torch.clip(diag_cov,min=-9.21,max=4.61)
        diag_cov = torch.exp(diag_cov).sqrt()
        noise = torch.distributions.Normal(mu, diag_cov).sample()
        z = x + temp*noise
        return z

# Taken from Freia - modified to add KL divergence function for the sub-models.
class AllInOneBlock(InvertibleModule):
    '''Module combining the most common operations in a normalizing flow or similar model.

    It combines affine coupling, permutation, and global affine transformation
    ('ActNorm'). It can also be used as GIN coupling block, perform learned
    householder permutations, and use an inverted pre-permutation. The affine
    transformation includes a soft clamping mechanism, first used in Real-NVP.
    The block as a whole performs the following computation:

    .. math::

        y = V\\,R \\; \\Psi(s_\\mathrm{global}) \\odot \\mathrm{Coupling}\\Big(R^{-1} V^{-1} x\\Big)+ t_\\mathrm{global}

    - The inverse pre-permutation of x (i.e. :math:`R^{-1} V^{-1}`) is optional (see
      ``reverse_permutation`` below).
    - The learned householder reflection matrix
      :math:`V` is also optional all together (see ``learned_householder_permutation``
      below).
    - For the coupling, the input is split into :math:`x_1, x_2` along
      the channel dimension. Then the output of the coupling operation is the
      two halves :math:`u = \\mathrm{concat}(u_1, u_2)`.

      .. math::

          u_1 &= x_1 \\odot \\exp \\Big( \\alpha \\; \\mathrm{tanh}\\big( s(x_2) \\big)\\Big) + t(x_2) \\\\
          u_2 &= x_2

      Because :math:`\\mathrm{tanh}(s) \\in [-1, 1]`, this clamping mechanism prevents
      exploding values in the exponential. The hyperparameter :math:`\\alpha` can be adjusted.

    '''

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 affine_clamping: float = 2.,
                 gin_block: bool = False,
                 global_affine_init: float = 1.,
                 global_affine_type: str = 'SOFTPLUS',
                 permute_soft: bool = False,
                 learned_householder_permutation: int = 0,
                 reverse_permutation: bool = False):
        '''
        Args:
          subnet_constructor:
            class or callable ``f``, called as ``f(channels_in, channels_out)`` and
            should return a torch.nn.Module. Predicts coupling coefficients :math:`s, t`.
          affine_clamping:
            clamp the output of the multiplicative coefficients before
            exponentiation to +/- ``affine_clamping`` (see :math:`\\alpha` above).
          gin_block:
            Turn the block into a GIN block from Sorrenson et al, 2019.
            Makes it so that the coupling operations as a whole is volume preserving.
          global_affine_init:
            Initial value for the global affine scaling :math:`s_\mathrm{global}`.
          global_affine_init:
            ``'SIGMOID'``, ``'SOFTPLUS'``, or ``'EXP'``. Defines the activation to be used
            on the beta for the global affine scaling (:math:`\\Psi` above).
          permute_soft:
            bool, whether to sample the permutation matrix :math:`R` from :math:`SO(N)`,
            or to use hard permutations instead. Note, ``permute_soft=True`` is very slow
            when working with >512 dimensions.
          learned_householder_permutation:
            Int, if >0, turn on the matrix :math:`V` above, that represents
            multiple learned householder reflections. Slow if large number.
            Dubious whether it actually helps network performance.
          reverse_permutation:
            Reverse the permutation before the block, as introduced by Putzky
            et al, 2019. Turns on the :math:`R^{-1} V^{-1}` pre-multiplication above.
        '''

        super().__init__(dims_in, dims_c)
        channels = dims_in[0][0]
        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = len(dims_in[0]) - 1
        # tuple containing all dims except for batch-dim (used at various points)
        self.sum_dims = tuple(range(1, 2 + self.input_rank))
        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        else:
            assert tuple(dims_c[0][1:]) == tuple(dims_in[0][1:]), \
                F"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}."
            self.conditional = True
            self.condition_channels = sum(dc[0] for dc in dims_c)

        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]

        try:
            if permute_soft or learned_householder_permutation:
                self.permute_function = {0: F.linear,
                                        1: F.conv1d,
                                        2: F.conv2d,
                                        3: F.conv3d}[self.input_rank]
            else:
                self.permute_function = lambda x, p: x[:, p]
        except KeyError:
            raise ValueError(f"Data is {1 + self.input_rank}D. Must be 1D-4D.")

        self.in_channels         = channels
        self.clamp               = affine_clamping
        self.GIN                 = gin_block
        self.reverse_pre_permute = reverse_permutation
        self.householder         = learned_householder_permutation

        if permute_soft and channels > 512:
            warnings.warn(("Soft permutation will take a very long time to initialize "
                           f"with {channels} feature channels. Consider using hard permutation instead."))

        # global_scale is used as the initial value for the global affine scale
        # (pre-activation). It is computed such that
        # global_scale_activation(global_scale) = global_affine_init
        # the 'magic numbers' (specifically for sigmoid) scale the activation to
        # a sensible range.
        if global_affine_type == 'SIGMOID':
            global_scale = 2. - np.log(10. / global_affine_init - 1.)
            self.global_scale_activation = self._sigmoid_global_scale_activation
        elif global_affine_type == 'SOFTPLUS':
            global_scale = 2. * np.log(np.exp(0.5 * 10. * global_affine_init) - 1)
            self.softplus = nn.Softplus(beta=0.5)
            self.global_scale_activation = self._softplus_global_scale_activation
        elif global_affine_type == 'EXP':
            global_scale = np.log(global_affine_init)
            self.global_scale_activation = self._exp_global_scale_activation
        else:
            raise ValueError('Global affine activation must be "SIGMOID", "SOFTPLUS" or "EXP"')

        self.global_scale = nn.Parameter(torch.ones(1, self.in_channels, *([1] * self.input_rank)) * float(global_scale))
        self.global_offset = nn.Parameter(torch.zeros(1, self.in_channels, *([1] * self.input_rank)))

        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            w_index = torch.randperm(channels, requires_grad=False)

        if self.householder:
            # instead of just the permutation matrix w, the learned housholder
            # permutation keeps track of reflection vectors vk, in addition to a
            # random initial permutation w_0.
            self.vk_householder = nn.Parameter(0.2 * torch.randn(self.householder, channels), requires_grad=True)
            self.w_perm = None
            self.w_perm_inv = None
            self.w_0 = nn.Parameter(torch.from_numpy(w).float(), requires_grad=False)
        elif permute_soft:
            self.w_perm = nn.Parameter(torch.from_numpy(w).float().view(channels, channels, *([1] * self.input_rank)).contiguous(),
                                       requires_grad=False)
            self.w_perm_inv = nn.Parameter(torch.from_numpy(w.T).float().view(channels, channels, *([1] * self.input_rank)).contiguous(),
                                           requires_grad=False)
        else:
            self.w_perm = nn.Parameter(w_index, requires_grad=False)
            self.w_perm_inv = nn.Parameter(torch.argsort(w_index), requires_grad=False)

        if subnet_constructor is None:
            raise ValueError("Please supply a callable subnet_constructor "
                             "function or object (see docstring)")
        self.subnet = subnet_constructor(self.splits[0] + self.condition_channels, 2 * self.splits[1])
        self.last_jac = None
        #self.var_net = VariationalLayer()
        #self.abs_sur = SimpleAbsSurjection()
        self.stochastic_layer = VariationalLayer()

    def kl_div(self,):
        return self.subnet.kl_div()

    def _sigmoid_global_scale_activation(self, a):
        return 10 * torch.sigmoid(a - 2.)

    def _softplus_global_scale_activation(self, a):
        return 0.1 * self.softplus(a)

    def _exp_global_scale_activation(self, a):
        return torch.exp(a)

    def _construct_householder_permutation(self):
        '''Computes a permutation matrix from the reflection vectors that are
        learned internally as nn.Parameters.'''
        w = self.w_0
        for vk in self.vk_householder:
            w = torch.mm(w, torch.eye(self.in_channels).to(w.device) - 2 * torch.ger(vk, vk) / torch.dot(vk, vk))

        for i in range(self.input_rank):
            w = w.unsqueeze(-1)
        return w

    def _permute(self, x, rev=False):
        '''Performs the permutation and scaling after the coupling operation.
        Returns transformed outputs and the LogJacDet of the scaling operation.'''
        if self.GIN:
            scale = 1.
            perm_log_jac = 0.
        else:
            scale = self.global_scale_activation(self.global_scale)
            perm_log_jac = torch.sum(torch.log(scale))

        if rev:
            return ((self.permute_function(x, self.w_perm_inv) - self.global_offset) / scale,
                    perm_log_jac)
        else:
            return (self.permute_function(x * scale + self.global_offset, self.w_perm),
                    perm_log_jac)

    def _pre_permute(self, x, rev=False):
        '''Permutes before the coupling block, only used if
        reverse_permutation is set'''
        if rev:
            return self.permute_function(x, self.w_perm)
        else:
            return self.permute_function(x, self.w_perm_inv)

    def _affine(self, x, a, rev=False):
        '''Given the passive half, and the pre-activation outputs of the
        coupling subnetwork, perform the affine coupling operation.
        Returns both the transformed inputs and the LogJacDet.'''

        # the entire coupling coefficient tensor is scaled down by a
        # factor of ten for stability and easier initialization.
        a = a * 0.1
        ch = x.shape[1]

        sub_jac = self.clamp * torch.tanh(a[:, :ch]/self.clamp)
        if self.GIN:
            sub_jac = sub_jac - torch.mean(sub_jac, dim=self.sum_dims, keepdim=True)

        if not rev:
            return (x * torch.exp(sub_jac) + a[:, ch:],
                    torch.sum(sub_jac, dim=self.sum_dims))
        else:
            return ((x - a[:, ch:]) * torch.exp(-sub_jac),
                    -torch.sum(sub_jac, dim=self.sum_dims))

    def forward(self, x, c=[], rev=False, jac=True):
        '''See base class docstring'''
        if tuple(x[0].shape[1:]) != self.dims_in[0]:
            raise RuntimeError(f"Expected input of shape {self.dims_in[0]}, "
                             f"got {tuple(x[0].shape[1:])}.")

        if self.householder:
            self.w_perm = self._construct_householder_permutation()
            if rev or self.reverse_pre_permute:
                self.w_perm_inv = self.w_perm.transpose(0, 1).contiguous()

        if rev:
            x, global_scaling_jac = self._permute(x[0], rev=True)
            x = (x,)
        elif self.reverse_pre_permute:
            x = (self._pre_permute(x[0], rev=False),)

        # if not rev: # Rev = False -> Forward
        #     x0,_ = self.sort_sur.forward(x[0])
        #
        # if rev:
        #     x0,_ = self.sort_sur.inverse(x[0])

        x1, x2 = torch.split(x[0], self.splits, dim=1)

        if self.conditional:
            x1c = torch.cat([x1, *c], 1)
        else:
            x1c = x1

        if not rev: # Rev = False
            a1 = self.subnet(x1c)
            x2, j2 = self._affine(x2, a1)
        else: # Rev = True
            a1 = self.subnet(x1c)
            x2, j2 = self._affine(x2, a1, rev=True)

        log_jac_det = j2
        x_out = torch.cat((x1, x2), 1)

        if not rev: # Rev = False
            x_out, global_scaling_jac = self._permute(x_out, rev=False)
        elif self.reverse_pre_permute:
            print('Triggered')
            x_out = self._pre_permute(x_out, rev=True)

        # add the global scaling Jacobian to the total.
        # trick to get the total number of non-channel dimensions:
        # number of elements of the first channel of the first batch member
        n_pixels = x_out[0, :1].numel()
        log_jac_det = log_jac_det + (-1)**rev * n_pixels * global_scaling_jac
        #x_out = self.stochastic_layer(x_out)
        return (x_out,), log_jac_det

    def output_dims(self, input_dims):
        return input_dims


def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 512),nn.ReLU(),
                         nn.Linear(512,c_out))


class MNF_Subnet(nn.Sequential):
    def __init__(self,c_in,c_out):
        layers = [MNFLinear(c_in, 512), nn.ReLU(),
                 MNFLinear(512,c_out)]

        super().__init__(*layers)

    def kl_div(self) -> float:
        """Compute current KL divergence of the whole model. Given by the sum
        of KL divs. from each MNF layer. Use as a regularization term during training.
        """
        return sum(lyr.kl_div() for lyr in self if hasattr(lyr, "kl_div"))

def create_freai(input_shape,layers):
    inn = SequenceINN(input_shape)
    for k in range(layers):
        inn.append(AllInOneBlock,subnet_constructor=subnet_fc, permute_soft=True)

    return inn
