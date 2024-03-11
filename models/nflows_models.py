import torch
from torch import nn

from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.nn.nets import ResidualNet
import torch.nn.functional as F


def create_nflows(input_shape,context_shape,num_layers):
    context_encoder = nn.Sequential(*[nn.Linear(context_shape,16),nn.ReLU(),nn.Linear(16,6)])
    base_dist = ConditionalDiagonalNormal(shape=[input_shape],context_encoder=context_encoder)
    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=3))
        transforms.append(MaskedAffineAutoregressiveTransform(features=3,hidden_features=512,
                                                                context_features=context_shape,num_blocks=3))

    transform = CompositeTransform(transforms)
    flow = Flow(transform,base_dist)

    return flow
