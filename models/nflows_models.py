import torch
from torch import nn

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
