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
    def __init__(self,input_shape,layers,context_shape,embedding=False,hidden_units=512,num_blocks=2):
        super(FreiaNet, self).__init__()
        self.input_shape = input_shape
        self.layers = layers
        self.context_shape = context_shape
        self.embedding = embedding
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self._allowed_x = torch.tensor(np.array([  3.,   9.,  15.,  21.,  27.,  33.,  39.,  45.,  53.,  59.,  65.,
                                                    71.,  77.,  83.,  89.,  95., 103., 109., 115., 121., 127., 133.,
                                                    139., 145., 153., 159., 165., 171., 177., 183., 189., 195., 203.,
                                                    209., 215., 221., 227., 233., 239., 245., 253., 259., 265., 271.,
                                                    277., 283., 289., 295., 303., 309., 315., 321., 327., 333., 339.,
                                                    345., 353., 359., 365., 371., 377., 383., 389., 395., 403., 409.,
                                                    415., 421., 427., 433., 439., 445., 453., 459., 465., 471., 477.,
                                                    483., 489., 495., 503., 509., 515., 521., 527., 533., 539., 545.,
                                                    553., 559., 565., 571., 577., 583., 589., 595., 603., 609., 615.,
                                                    621., 627., 633., 639., 645., 653., 659., 665., 671., 677., 683.,
                                                    689., 695., 703., 709., 715., 721., 727., 733., 739., 745., 753.,
                                                    759., 765., 771., 777., 783., 789., 795., 803., 809., 815., 821.,
                                                    827., 833., 839., 845., 853., 859., 865., 871., 877., 883., 889.,
                                                    895.]))
        self._allowed_y = torch.tensor(np.array([  3.,   9.,  15.,  21.,  27.,  33.,  39.,  45.,  53.,  59.,  65.,
                                                    71.,  77.,  83.,  89.,  95., 103., 109., 115., 121., 127., 133.,
                                                    139., 145., 153., 159., 165., 171., 177., 183., 189., 195., 203.,
                                                    209., 215., 221., 227., 233., 239., 245., 253., 259., 265., 271.,
                                                    277., 283., 289., 295.]))
        self.stats_ = {"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":380.00,"time_min":0.0}
        if self.embedding:
            self.context_embedding = nn.Sequential(*[nn.Linear(context_shape,16),nn.ReLU(),nn.Linear(16,input_shape)])

        context_encoder =  nn.Sequential(*[nn.Linear(context_shape,16),nn.ReLU(),nn.Linear(16,input_shape*2)])

        self.distribution = ConditionalDiagonalNormal(shape=[input_shape],context_encoder=context_encoder)

        def create_freai(input_shape,layer,cond_shape):
            inn = Ff.SequenceINN(input_shape)
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

    def set_to_closest(self, x, allowed):
        x = x.unsqueeze(1)  # Adding a dimension to x for broadcasting
        diffs = torch.abs(x - allowed.to('cuda').float())
        closest_indices = torch.argmin(diffs, dim=1)
        closest_values = allowed[closest_indices]
        return closest_values
            
    def _sample(self,num_samples,context):
        samples = self.__sample(num_samples,context)
        x = self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min'])
        y = self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min'])
        t = self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min']).detach().cpu()
        x = self.set_to_closest(x,self._allowed_x)
        y = self.set_to_closest(y,self._allowed_y)
        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1).numpy()

    def to_noise(self,inputs,context):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context
        noise,_ = self.sequence.forward(inputs,rev=False,c=[embedded_context])
