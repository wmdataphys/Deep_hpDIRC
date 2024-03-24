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
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.utils import torchutils


class FreiaNet(nn.Module):
    def __init__(self,input_shape,layers,context_shape,embedding=False):
        super(FreiaNet, self).__init__()
        self.input_shape = input_shape
        self.layers = layers
        self.context_shape = context_shape
        self.embedding = embedding
        if self.embedding:
            self.context_embedding = nn.Sequential(*[nn.Linear(context_shape,16),nn.ReLU(),nn.Linear(16,input_shape)])

        context_encoder =  nn.Sequential(*[nn.Linear(context_shape,16),nn.ReLU(),nn.Linear(16,input_shape*2)])

        self.distribution = ConditionalDiagonalNormal(shape=[input_shape],context_encoder=context_encoder)

        def create_freai(input_shape,layer,cond_shape):
            inn = Ff.SequenceINN(input_shape)
            for k in range(layers):
                inn.append(Fm.AllInOneBlock,cond=0,cond_shape=(cond_shape,),subnet_constructor=subnet_fc, permute_soft=True)

            return inn

        def subnet_fc(c_in, c_out):
            return nn.Sequential(nn.Linear(c_in, 512),nn.ReLU(),
                                 nn.Linear(512,c_out))

        self.sequence = create_freai(self.input_shape,self.layers,self.context_shape)

    def log_prob(self,inputs,context):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context

        noise,logabsdet = self.sequence.forward(inputs,rev=False,c=[embedded_context])
        log_prob = self.distribution.log_prob(noise,context=embedded_context)

        return log_prob + logabsdet


    def _sample(self,num_samples,context):
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


    def to_noise(self,inputs,context):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context
        noise,_ = self.sequence.forward(inputs,rev=False,c=[embedded_context])
