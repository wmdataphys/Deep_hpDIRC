import torch
from torch import nn
import numpy as np
from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.nn.nets import ResidualNet
import torch.nn.functional as F
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.nn import nets as nets

def create_nflows(input_shape,context_shape,num_layers,autoregressive=False):
    context_encoder = nn.Sequential(*[nn.Linear(context_shape,64),nn.ReLU(),nn.Linear(64,input_shape*2)])
    base_dist = ConditionalDiagonalNormal(shape=[input_shape],context_encoder=context_encoder)
    transforms = []
    def create_resnet(in_features, out_features):
        return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=512,
                num_blocks=2,
                activation=F.relu,
                dropout_probability=0,
                use_batch_norm=False,
            )
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=3))
        if autoregressive:
            transforms.append(MaskedAffineAutoregressiveTransform(features=3,hidden_features=512,
                                                                context_features=context_shape,num_blocks=2))
        else:
            transforms.append(AffineCouplingTransform(mask=[1,1,0],context_features=context_shape,transform_net_create_fn=create_resnet))

    transform = CompositeTransform(transforms)
    flow = Flow(transform,base_dist)

    return flow



class MAAF(nn.Module):
    def __init__(self,input_shape,layers,context_shape,embedding=False,hidden_units=512,num_blocks=2):
        super(MAAF, self).__init__()
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
        self.stats_ = {"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":500.00,"time_min":0.0}

        if self.embedding:
            self.context_embedding = nn.Sequential(*[nn.Linear(context_shape,16),nn.ReLU(),nn.Linear(16,input_shape)])


        def create_transform(input_shape,layers,context_shape,hidden_features,num_blocks):
            transforms = []
            context_encoder =  nn.Sequential(*[nn.Linear(context_shape,16),nn.ReLU(),nn.Linear(16,input_shape*2)])
            distribution = ConditionalDiagonalNormal(shape=[input_shape],context_encoder=context_encoder)
            for k in range(layers):
                transforms.append(ReversePermutation(features=input_shape))
                transforms.append(MaskedAffineAutoregressiveTransform(features=input_shape,hidden_features=hidden_features,
                                                                context_features=context_shape,num_blocks=num_blocks))

            transform = CompositeTransform(transforms)
            flow = Flow(transform,distribution)

            return flow

        self.sequence = create_transform(input_shape,layers,context_shape,hidden_units,num_blocks)

    def log_prob(self,inputs,context):
        return self.sequence.log_prob(inputs,context=context)

    def unscale(self,x,max_,min_):
        return x*0.5*(max_ - min_) + min_ + (max_-min_)/2

    def set_to_closest(self, x, allowed):
        x = x.unsqueeze(1)  # Adding a dimension to x for broadcasting
        diffs = torch.abs(x - allowed.to('cuda').float())
        closest_indices = torch.argmin(diffs, dim=1)
        closest_values = allowed[closest_indices]
        return closest_values
            
    def _sample(self,num_samples,context):
        samples = self.sequence._sample(num_samples,context)
        x = self.unscale(samples[:,:,0].flatten(),self.stats_['x_max'],self.stats_['x_min'])
        y = self.unscale(samples[:,:,1].flatten(),self.stats_['y_max'],self.stats_['y_min'])
        t = self.unscale(samples[:,:,2].flatten(),self.stats_['time_max'],self.stats_['time_min']).detach().cpu()
        x = self.set_to_closest(x,self._allowed_x)
        y = self.set_to_closest(y,self._allowed_y)
        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1).numpy()

    def to_noise(self,inputs,context):
        return self.sequence.transform_to_noise(inputs,context=context)
