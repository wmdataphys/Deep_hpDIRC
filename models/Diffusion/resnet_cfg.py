import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange


from einops import rearrange, repeat, reduce

 # Helper functions

def divisible_by(numer, denom):
    return (numer % denom) == 0

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


class ResNetDenseBlock(nn.Module):
    def __init__(self, hidden_size, nlayers=1, dropout_rate=0.1, activation=None):
        super(ResNetDenseBlock, self).__init__()
        self.nlayers = nlayers
        self.activation = activation or nn.SiLU()
        self.dropout_rate = dropout_rate

        self.residual_layer = nn.Linear(hidden_size, hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=True),
                self.activation,
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.Dropout(dropout_rate)
            ))

    def forward(self, x):
        residual = self.residual_layer(x)
        out = x
        for layer_block in self.layers:
            out = layer_block(out)
        out = out + residual
        return out

class ResNet(nn.Module):
    def __init__(self, 
                 input_dim, 
                 end_dim, 
                 cond_dim = None, 
                 mlp_dim=128, 
                 num_layer=3):
        
        super(ResNet, self).__init__()
        self.activation = nn.LeakyReLU(0.01)

        self.input_dim = input_dim # Need this for sampling
        self.cond_dim = cond_dim

        # Initial layers for processing input and time embedding
        self.input_dense = nn.Linear(input_dim, mlp_dim)

        # from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py

        # self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        # if self.random_or_learned_sinusoidal_cond:
        #     sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
        #     fourier_dim = learned_sinusoidal_dim + 1
        # else:
    
        sinu_pos_emb = SinusoidalPosEmb(mlp_dim // 4)
        fourier_dim = mlp_dim // 4
    
        self.time_embed = nn.Sequential( 
            sinu_pos_emb,
            nn.Linear(fourier_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim)
        )

        # from https://arxiv.org/pdf/2407.07376

        if cond_dim:
            self.context_encoder =  nn.Sequential(*[nn.Linear(cond_dim,16),
                                                    nn.ReLU(),
                                                    nn.Linear(16,mlp_dim)]) # needs to be same size as time embedding to add together, 
            self.null_classes_emb = nn.Parameter(torch.zeros(mlp_dim)) # for cfg, should be size of mlp. Same as cond_embedding
            # so we project to mlp_dim 

        # Residual connection after combining input and time embedding
        self.initial_residual_dense1 = nn.Linear(mlp_dim, mlp_dim)
        self.initial_residual_dense2 = nn.Linear(mlp_dim, mlp_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResNetDenseBlock(hidden_size=mlp_dim, nlayers=1, dropout_rate=0.1, activation=self.activation)
            for _ in range(num_layer - 1)
        ])

        # Final layers
        self.final_dense = nn.Linear(mlp_dim, 2 * mlp_dim)
        self.output_dense = nn.Linear(2 * mlp_dim, end_dim)

    def forward(self, 
                inputs, 
                t, 
                cond=None, 
                cond_drop_prob = 0):
        # Process time embedding and conds

        time_embed = self.activation(self.time_embed(t))

        # cfg code taken from LucidRains https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py
        batch, device = inputs.shape[0], inputs.device
        # .squeeze removes the singleton dimension at index 1 if it is of size 1
       
        # print("time_embed:",time_embed.shape)

        if cond is None: 
            embed = time_embed
        else:
            # cond = cond.view(cond.size(0),-1)
            conds_embed = self.activation(self.context_encoder(cond)) if hasattr(self, 'context_encoder') else torch.zeros_like(embed)
            
            if cond_drop_prob > 0:
                keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
                null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)
                conds_embed = torch.where( # randomly masking
                    rearrange(keep_mask, 'b -> b 1'),
                    conds_embed,
                    null_classes_emb
                )
            
            # print("cond_embed:",conds_embed.shape)

            embed = time_embed + conds_embed
        
        # Check shapes here for conditional embedding        

        # Process inputs 
        inputs_dense = self.activation(self.input_dense(inputs))

        # Initial residual connection
        residual = self.activation(self.initial_residual_dense1(inputs_dense + embed))
        residual = self.initial_residual_dense2(residual)

        # Apply residual blocks
        layer = residual
        for block in self.blocks: # num_layer-1
            layer = block(layer)

        # Final processing
        layer = self.activation(self.final_dense(residual + layer))
        outputs = self.output_dense(layer)

        return outputs
    
    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        rescaled_phi = 0.,
        remove_parallel_component = True,
        keep_parallel_frac = 0.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob=0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs) # cfg
        update = logits - null_logits

        if remove_parallel_component:
            parallel, orthog = project(update, logits)
            update = orthog + parallel * keep_parallel_frac

        scaled_logits = logits + update * (cond_scale - 1.)

        if rescaled_phi == 0.:
            return scaled_logits, null_logits

        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))
        interpolated_rescaled_logits = rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

        return interpolated_rescaled_logits, null_logits