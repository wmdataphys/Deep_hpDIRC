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

        # self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        # if self.random_or_learned_sinusoidal_cond:
        #     sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
        #     fourier_dim = learned_sinusoidal_dim + 1
        # else:
        self.time_dense = nn.Linear(1, mlp_dim)
        # self.projection = self.GaussianFourierProjection(scale = 16)
        # self.time_dense = nn.Linear(32, mlp_dim)

        if cond_dim:
            self.context_encoder =  nn.Sequential(*[nn.Linear(cond_dim,16),
                                                    nn.ReLU(),
                                                    nn.Linear(16,mlp_dim)]) # needs to be same size as time embedding to add together, 

        # combining input and time embedding
        self.initial_residual_dense1 = nn.Linear(mlp_dim, mlp_dim)
        self.initial_residual_dense2 = nn.Linear(mlp_dim, mlp_dim)

        self.blocks = nn.ModuleList([
            ResNetDenseBlock(hidden_size=mlp_dim, nlayers=1, dropout_rate=0.1, activation=self.activation)
            for _ in range(num_layer - 1)
        ])

        # Final layers
        # self.final_dense = nn.Linear(mlp_dim, end_dim)
        self.final_dense = nn.Linear(mlp_dim, 2 * mlp_dim)
        self.output_dense = nn.Linear(2 * mlp_dim, end_dim)

    def GaussianFourierProjection(self, scale = 30):
        half_dim = 16
        emb = torch.log(torch.tensor(10000.0, dtype=torch.float32)) / (half_dim - 1)
        freq = torch.exp(-emb* torch.arange(start=0, end=half_dim, dtype=torch.float32))
        return freq

    def Embedding(self,inputs,projection):
        projection = projection.to(inputs.device)
        angle = inputs * projection.unsqueeze(0) * 1000
        embedding = torch.cat([torch.sin(angle),torch.cos(angle)],dim=1)

        embedding = self.time_dense(embedding)
        embedding = self.activation(embedding)
        return embedding

    def forward(self, 
                inputs, 
                t, 
                cond=None):
        # Process time embedding and conds

        time_embed = self.activation(self.time_dense(t.view(-1,1).float()))
        # time_embed = self.Embedding(t, self.projection)

        batch, device = inputs.shape[0], inputs.device
        # .squeeze removes the singleton dimension at index 1 if it is of size 1
       
        # print("time_embed:",time_embed.shape)

        if cond is None: 
            embed = time_embed
        else:
            cond = cond.view(cond.size(0),-1)
            conds_embed = self.activation(self.context_encoder(cond)) if hasattr(self, 'context_encoder') else torch.zeros_like(embed)

            embed = time_embed + conds_embed
        
        # Check shapes here for conditional embedding        

        inputs_dense = self.activation(self.input_dense(inputs))

        residual = self.activation(self.initial_residual_dense1(inputs_dense + embed))
        residual = self.initial_residual_dense2(residual)

        layer = residual
        for block in self.blocks: # num_layer-1
            layer = block(layer)

        layer = self.activation(self.final_dense(residual + layer))
        
        outputs = self.output_dense(layer)

        return outputs
