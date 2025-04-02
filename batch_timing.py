import os
import json
import argparse
import torch
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from dataloader.dataloader import CreateLoaders
import pkbar
import torch.nn as nn
from datetime import datetime
import itertools
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
import pickle
import warnings
from make_plots import make_ratios
import itertools
import copy
import re
import math

warnings.filterwarnings("ignore", message=".*weights_only.*")

from models.NF.freia_models import FreiaNet
from models.OT_Flow.ot_flow import OT_Flow
from models.FlowMatching.flow_matching import FlowMatching
from models.Diffusion.resnet import ResNet
from models.Diffusion.continuous_diffusion import ContinuousTimeGaussianDiffusion
from models.Diffusion.gsgm import GSGM
from models.Diffusion.gaussian_diffusion import GaussianDiffusion

def calculate_splits(gpu_mem, total_samples):
    events_per_24GB = int(3.5e5)  # roughly 350k events for 24GB VRAM at generation
    vram_per_event = 24 / events_per_24GB
    available_events = gpu_mem / vram_per_event
    num_splits = math.ceil(total_samples / available_events)
    
    if num_splits < 1:
        num_splits = 1

    events_per_split = [int(total_samples / num_splits)] * num_splits
    events_per_split[-1] += total_samples - sum(events_per_split)

    return num_splits, events_per_split

def combine_photons(pions,kaons):
    new_pions = {"pmtID": np.concatenate([pion['pmtID'] for pion in pions]),
                "pixelID": np.concatenate([pion['pixelID'] for pion in pions]),
                "leadTime": np.concatenate([pion['leadTime'] for pion in pions])}
    new_kaons = {"pmtID": np.concatenate([kaon['pmtID'] for kaon in kaons]),
            "pixelID": np.concatenate([kaon['pixelID'] for kaon in kaons]),
            "leadTime": np.concatenate([kaon['leadTime'] for kaon in kaons])}

    return new_pions,new_kaons

def main(config,args):

    # Setup random seed
    print("Using model type: ",str(args.model_type))
    # Remove seeding, make it random.
    seed = int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

    gpu_mem, _ = torch.cuda.mem_get_info()
    gpu_mem = gpu_mem / (1024 ** 3) # GB

    NUM_SAMPLES = int(3.5e5)

    print("You currently have {0:.2f} GB of free GPU memory. Our generation procedure will scale accordingly.".format(gpu_mem))

    Kdicte = torch.load(config['Inference']['kaon_model_path_'+str(args.model_type)])
    Pdicte = torch.load(config['Inference']['pion_model_path_'+str(args.model_type)])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError("No GPU was found! Exiting the program.")

    num_layers = int(config['model_'+str(args.model_type)]['num_layers'])
    input_shape = int(config['model_'+str(args.model_type)]['input_shape'])
    cond_shape = int(config['model_'+str(args.model_type)]['cond_shape'])
    hidden_nodes = int(config['model_'+str(args.model_type)]['hidden_nodes'])
    stats = config['stats']
    
    if args.model_type == "NF":
        num_blocks = int(config['model_'+str(args.model_type)]['num_blocks'])
        kaon_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats)
        pion_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats)
    elif args.model_type == "CNF":
        num_blocks = int(config['model_'+str(args.model_type)]['num_blocks'])
        alph = config['model_'+str(args.model_type)]['alph']
        train_T = bool(config['model_'+str(args.model_type)]['train_T'])
        kaon_net = OT_Flow(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,train_T=train_T,alph=alph,LUT_path=None)
        pion_net = OT_Flow(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,train_T=train_T,alph=alph,LUT_path=None)
    elif args.model_type == "FlowMatching":
        kaon_net = FlowMatching(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,LUT_path=None)
        pion_net = FlowMatching(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,LUT_path=None)
    elif args.model_type == 'Score':
        num_steps = int(config['model_Score']['num_steps'])
        noise_schedule = config['model_Score']['noise_schedule']
        learned_schedule_net_hidden_dim = int(config['model_Score']['learned_schedule_net_hidden_dim'])
        gamma = int(config['model_Score']['gamma'])
        model = ResNet(input_dim=input_shape, end_dim=input_shape, cond_dim=cond_shape, mlp_dim=hidden_nodes, num_layer=num_layers)
        kaon_net = ContinuousTimeGaussianDiffusion(model=model, stats=stats,num_sample_steps=num_steps, noise_schedule=noise_schedule, learned_schedule_net_hidden_dim=learned_schedule_net_hidden_dim, min_snr_loss_weight = True, min_snr_gamma=gamma,LUT_path=None)
        pion_net = ContinuousTimeGaussianDiffusion(model=model, stats=stats,num_sample_steps=num_steps, noise_schedule=noise_schedule, learned_schedule_net_hidden_dim=learned_schedule_net_hidden_dim, min_snr_loss_weight = True, min_snr_gamma=gamma,LUT_path=None)
    elif args.model_type == 'GSGM':
        num_steps = int(config['model_GSGM']['num_steps'])
        num_embed = int(config['model_GSGM']['num_embed'])
        learned_variance = config['model_GSGM']['learned_variance']
        nonlinear_noise_schedule = int(config['model_GSGM']['nonlinear_noise_schedule'])

        kaon_net = GSGM(num_input=input_shape, num_conds=cond_shape, device=device, stats=stats, num_layers=num_layers, num_steps=num_steps, num_embed=num_embed, mlp_dim=hidden_nodes, nonlinear_noise_schedule=nonlinear_noise_schedule, learnedvar=learned_variance,LUT_path=None)
        pion_net = GSGM(num_input=input_shape, num_conds=cond_shape, device=device, stats=stats, num_layers=num_layers, num_steps=num_steps, num_embed=num_embed, mlp_dim=hidden_nodes, nonlinear_noise_schedule=nonlinear_noise_schedule, learnedvar=learned_variance,LUT_path=None)
    elif args.model_type == 'DDPM':
        num_steps = int(config['model_DDPM']['num_steps'])

        model = ResNet(input_dim=input_shape, end_dim=input_shape, cond_dim=cond_shape, mlp_dim=hidden_nodes, num_layer=num_layers)
        kaon_net = GaussianDiffusion(denoise_fn=model, device=device, stats=stats, timesteps=num_steps, loss_type='l2',LUT_path=None)
        pion_net = GaussianDiffusion(denoise_fn=model, device=device, stats=stats, timesteps=num_steps, loss_type='l2',LUT_path=None)
    else:
        raise ValueError("Model type not found.")
    
    kaon_net.to('cuda')
    pion_net.to('cuda')

    if args.fine_grained_prior:
        print("Using fine grained prior for PMTs. Consider disabling for increased generation speed.")
    else:
        print("Fine grained prior disabled. Consider enabling for high fidelity.")

    kaon_net.load_state_dict(Kdicte['net_state_dict'], strict=False)
    pion_net.load_state_dict(Pdicte['net_state_dict'], strict=False)

    torch.cuda.empty_cache()
    # print("Running inference for momentum = {0}, theta = {1}, for {2} pions and {3} kaons.".format(args.momentum,theta_,len(inference_pions),len(inference_kaons)))

    print("--------------- Fast Simulation -----------------\n")

    with torch.set_grad_enabled(False):
        p = np.random.uniform(5,6)
        p_scaled = (p - stats['P_max'])  / (stats['P_max'] - stats['P_min'])

        theta = np.random.uniform(50,60)
        theta_scaled = theta = (theta - stats['theta_max']) / (stats['theta_max'] - stats['theta_min'])

        k = torch.tensor(np.array([p_scaled,theta_scaled])).to('cuda').float()

        start = time.time()
        num_splits,batches = calculate_splits(gpu_mem, NUM_SAMPLES) # should be about 3.5e5 per batch.
        if num_splits > 1:
            print("Batching generations into {0} runs.".format(num_splits)," ",batches)
            fs_kaons = []
            fs_pions = []
            start = time.time()
            for i in range(num_splits):
                fs_kaons.append(kaon_net.create_tracks(num_samples=batches[i],context=k.unsqueeze(0),fine_grained_prior=args.fine_grained_prior))
                fs_pions.append(pion_net.create_tracks(num_samples=batches[i],context=k.unsqueeze(0),fine_grained_prior=args.fine_grained_prior))
                torch.cuda.empty_cache()

            fs_pions,fs_kaons = combine_photons(fs_pions,fs_kaons)

            end = time.time()
        
        else:
            print('Generating {} samples...'.format(NUM_SAMPLES*2))

            # Essentially generate 1 track with N photons.
            start = time.time()
            fs_kaons = kaon_net.create_tracks(num_samples=NUM_SAMPLES,context=k.unsqueeze(0),fine_grained_prior=args.fine_grained_prior)
            fs_pions = pion_net.create_tracks(num_samples=NUM_SAMPLES,context=k.unsqueeze(0),fine_grained_prior=args.fine_grained_prior)
            end = time.time()

    print("Time to create both PDFs: ",end - start)
    print("Time / photon: {0}.\n".format((end - start)/ (2*NUM_SAMPLES)))


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FastSim Generation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-mt','--model_type',default="NF",type=str,help='Which model to use.')
    parser.add_argument('-f','--fine_grained_prior',action='store_true',help="Enable fine grained prior, default False.")
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args)