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
    def __init__(self,input_shape,layers,context_shape,embedding=False,hidden_units=512,num_blocks=2,stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":380.00,"time_min":0.0,
            "P_max":8.5 ,"P_min":0.95 , "theta_max": 11.63,"theta_min": 0.90,"phi_max": 175.5, "phi_min":-176.0 },device='cuda'):
        super(MAAF, self).__init__()
        self.input_shape = input_shape
        self.layers = layers
        self.context_shape = context_shape
        self.embedding = embedding
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.photons_generated = 0
        self.photons_resampled = 0
        self.device = device
        self._allowed_x = torch.tensor(np.array([  3.,   9.,  15.,  21.,  27.,  33.,  39.,  45.,   # 0
                                                    53.,  59.,  65., 71.,  77.,  83.,  89.,  95.,  # 1
                                                    103., 109., 115., 121., 127., 133., 139., 145.,# 2
                                                    153., 159., 165., 171., 177., 183., 189., 195.,# 3
                                                    203., 209., 215., 221., 227., 233., 239., 245.,# 4 
                                                    253., 259., 265., 271.,277., 283., 289.,  295.,# 5
                                                    303., 309., 315., 321., 327., 333., 339., 345.,# 6
                                                    353., 359., 365., 371., 377., 383., 389., 395.,# 7 
                                                    403., 409., 415., 421., 427., 433., 439., 445.,# 8
                                                    453., 459., 465., 471., 477., 483., 489., 495.,# 9 
                                                    503., 509., 515., 521., 527., 533., 539., 545.,# 10
                                                    553., 559., 565., 571., 577., 583., 589., 595.,# 11
                                                    603., 609., 615., 621., 627., 633., 639., 645.,# 12 
                                                    653., 659., 665., 671., 677., 683., 689., 695.,# 13
                                                    703., 709., 715., 721., 727., 733., 739., 745.,# 14 
                                                    753., 759., 765., 771., 777., 783., 789., 795.,# 15 
                                                    803., 809., 815., 821., 827., 833., 839., 845.,# 16
                                                    853., 859., 865., 871., 877., 883., 889., 895.])).to(self.device) # 17
        self._allowed_y = torch.tensor(np.array([  3.,   9.,  15.,  21.,  27.,  33.,  39.,  45.,  # 0
                                                   53.,  59.,  65.,71.,  77.,  83.,  89.,  95.,   # 1
                                                   103., 109., 115., 121., 127., 133.,139., 145., # 2
                                                   153., 159., 165., 171., 177., 183., 189., 195.,# 3  
                                                   203., 209., 215., 221., 227., 233., 239., 245.,# 4 
                                                   253., 259., 265., 271.,277., 283., 289., 295.])).to(self.device)
        self.stats_ = stats

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


        self._initialize_weights()

    def _initialize_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.apply(init_weights)

    def log_prob(self,inputs,context):
        return self.sequence.log_prob(inputs,context=context)

    def unscale(self,x,max_,min_):
        return x*0.5*(torch.tensor(max_,device=x.device) - torch.tensor(min_,device=x.device)) + torch.tensor(min_,device=x.device) + torch.tensor((max_-min_)/2,device=x.device)

    def unscale_conditions(self,x,max_,min_):
        return x * (max_ - min_) + max_

    def set_to_closest(self, x, allowed):
        x = x.unsqueeze(1)  # Adding a dimension to x for broadcasting
        diffs = torch.abs(x - allowed.to(self.device).float())
        closest_indices = torch.argmin(diffs, dim=1)
        closest_values = allowed[closest_indices]
        return closest_values
            
    def _sample(self,num_samples,context):
        samples = self.sequence._sample(num_samples,context)
        x = self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min'])
        y = self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min'])
        t = self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])

        x = self.set_to_closest(x,self._allowed_x)
        y = self.set_to_closest(y,self._allowed_y)
        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1).detach().cpu().numpy()

    def to_noise(self,inputs,context):
        return self.sequence.transform_to_noise(inputs,context=context)

    def __get_track(self,num_samples,context):
        samples = self.sequence._sample(num_samples,context).squeeze(0)
        x = self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min']).round()
        y = self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min']).round()
        t = self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])

        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1)

    def _apply_mask(self, hits):
        # Time > 0 
        mask = torch.where((hits[:,2] > 0) & (hits[:,2] < self.stats_['time_max']))
        hits = hits[mask]
        # Outter bounds
        mask = torch.where((hits[:, 0] > 0) & (hits[:, 0] < 898) & (hits[:, 1] > 0) & (hits[:, 1] < 298))[0] # Acceptance mask
        hits = hits[mask]
        # PMTs OFF
        top_row_mask = torch.where(~((hits[:, 1] > 249) & (hits[:, 0] < 350)))[0] # rejection mask (keep everything not identified)
        hits = hits[top_row_mask]
        # PMTs OFF
        bottom_row_mask = torch.where(~((hits[:, 1] < 50) & (hits[:, 0] < 550)))[0] # rejection mask (keep everything not identified)
        hits = hits[bottom_row_mask]

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

        pmtID = torch.div(x,torch.tensor(50,dtype=torch.int),rounding_mode='floor') + torch.div(y, torch.tensor(50,dtype=torch.int),rounding_mode='floor') * 18
        row = (1.0/6.0) * ( y - 3 - 2* torch.div(pmtID,torch.tensor(18,dtype=torch.int),rounding_mode='floor'))
        col = (1.0/6.0) * ( x - 3 - 2*(pmtID % 18))

        assert(len(row) == num_samples)
        assert(len(col) == num_samples)
        assert(len(pmtID) == num_samples)

        P = self.unscale_conditions(context[0][0].detach().cpu().numpy(),self.stats_['P_max'],self.stats_['P_min'])
        Theta = self.unscale_conditions(context[0][1].detach().cpu().numpy(),self.stats_['theta_max'],self.stats_['theta_min'])
        Phi = self.unscale_conditions(context[0][2].detach().cpu().numpy(),self.stats_['phi_max'],self.stats_['phi_min'])

        if not plotting:
            return {"NHits":num_samples,"P":P,"Theta":Theta,"Phi":Phi,"row":row.numpy(),"column":col.numpy(),"leadTime":t.numpy(),"pmtID":pmtID.numpy()}
        else:
            return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1)