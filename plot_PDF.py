import os
import json
import math
import time
import pkbar
import torch
import pickle
import random
import warnings
import argparse
import itertools
import numpy as np
from datetime import datetime

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from dataloader.dataloader import CreateLoaders

from utils.KDE_utils import perform_fit_KDE,gaussian_normalized,gaussian_unnormalized,plot,FastDIRC
from utils.hpDIRC import gapx,gapy,pixel_width,pixel_height,npix,npmt
from make_plots import make_PDFs

from models.NF.freia_models import FreiaNet
from models.OT_Flow.ot_flow import OT_Flow
from models.FlowMatching.flow_matching import FlowMatching
from models.Diffusion.resnet import ResNet
from models.Diffusion.continuous_diffusion import ContinuousTimeGaussianDiffusion
from models.Diffusion.gaussian_diffusion import GaussianDiffusion

warnings.filterwarnings("ignore", message=".*weights_only.*")
# This is a bandwith warning. Occurs with low number of support photons.
warnings.filterwarnings("ignore", message=".*Grid size 1 will likely result in GPU under-utilization.*")


def create_supports_fs(pions,kaons):
    x = pions['x']
    y = pions['y']
    t = pions['leadTime']
    support_pions = np.vstack((x, y,t)).T

    x = kaons['x']
    y = kaons['y']
    t = kaons['leadTime']
    support_kaons = np.vstack((x, y,t)).T  

    return support_pions,support_kaons


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
    new_pions = {"x": np.concatenate([pion['x'] for pion in pions]),
                "y": np.concatenate([pion['y'] for pion in pions]),
                "leadTime": np.concatenate([pion['leadTime'] for pion in pions])}
    new_kaons = {"x": np.concatenate([kaon['x'] for kaon in kaons]),
            "y": np.concatenate([kaon['y'] for kaon in kaons]),
            "leadTime": np.concatenate([kaon['leadTime'] for kaon in kaons])}

    return new_pions,new_kaons

def main(config,args):

    seed = int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    gpu_mem, _ = torch.cuda.mem_get_info()
    gpu_mem = gpu_mem / (1024 ** 3) # GB
    assert torch.cuda.is_available()
    device = 'cuda'

    assert args.momentum >= 1.0 and args.momentum <= 10.0

    print("---------------- PDF Stats ------------------")
    print("Using model type: ",str(args.model_type))
    print("Fast Simulated Support Photons: ",args.fs_support)
    print("Important note: Generating 350k photons requires ~ 24GB of VRAM.")
    print("You currently have {0:.2f} GB of free GPU memory. Our generation procedure will scale accorindgly.".format(gpu_mem))
    print("---------------------------------------------")


    num_layers = int(config['model_'+str(args.model_type)]['num_layers'])
    input_shape = int(config['model_'+str(args.model_type)]['input_shape'])
    cond_shape = int(config['model_'+str(args.model_type)]['cond_shape'])
    hidden_nodes = int(config['model_'+str(args.model_type)]['hidden_nodes'])
    stats = config['stats']
    
    if args.model_type == "NF":
        num_blocks = int(config['model_'+str(args.model_type)]['num_blocks'])
        kaon_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats,LUT_path=None)
        pion_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats,LUT_path=None)
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
    elif args.model_type == 'DDPM':
        num_steps = int(config['model_DDPM']['num_steps'])
        model = ResNet(input_dim=input_shape, end_dim=input_shape, cond_dim=cond_shape, mlp_dim=hidden_nodes, num_layer=num_layers)
        kaon_net = GaussianDiffusion(denoise_fn=model, device=device, stats=stats, timesteps=num_steps, loss_type='l2',LUT_path=None)
        pion_net = GaussianDiffusion(denoise_fn=model, device=device, stats=stats, timesteps=num_steps, loss_type='l2',LUT_path=None)
    else:
        raise ValueError("Model type not found.")

    Kdicte = torch.load(config['Inference']['kaon_model_path_'+str(args.model_type)])
    Pdicte = torch.load(config['Inference']['pion_model_path_'+str(args.model_type)])
    kaon_net.load_state_dict(Kdicte['net_state_dict'])
    pion_net.load_state_dict(Pdicte['net_state_dict'])

    kaon_net.to(device)
    pion_net.to(device)
    kaon_net.eval()
    pion_net.eval()


    os.makedirs("PDFs",exist_ok=True)
    out_folder = os.path.join("PDFs",config['Inference']['pdf_dir'])
    os.makedirs(out_folder,exist_ok=True)
    print("Outputs can be found in " + str(out_folder))

    if args.fine_grained_prior:
        print("Using fine grained prior for PMTs. Consider disabling for increased generation speed.")
    else:
        print("Fine grained prior disabled. Consider enabling for high fidelity.")


    thetas =  [25,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,95.,100.,105.,110.,115.,120.,125.,130.,135.,140.,145.,150.,155.] 

    for theta_ in thetas:
        print(" ")
        torch.cuda.empty_cache()
        print("--------------- Fast Simulation -----------------")
        print("Genearting pion and kaon PDFs for momentum = {0}, theta = {1}, of size {2}.".format(args.momentum,theta_,args.fs_support))

        with torch.set_grad_enabled(False):
            p = (args.momentum - stats['P_max'])  / (stats['P_max'] - stats['P_min'])
            theta = (theta_ - stats['theta_max']) / (stats['theta_max'] - stats['theta_min'])
            k = torch.tensor(np.array([p,theta])).to(device).float().unsqueeze(0)

            
            start = time.time()
            num_splits,batches = calculate_splits(gpu_mem, args.fs_support)
            if num_splits > 1:
                print("Batching generations into {0} runs.".format(num_splits)," ",batches)
                fs_kaons = []
                fs_pions = []
                start = time.time()
                for i in range(num_splits):
                    fs_kaons.append(kaon_net.create_tracks(num_samples=batches[i],context=k,fine_grained_prior=args.fine_grained_prior))
                    fs_pions.append(pion_net.create_tracks(num_samples=batches[i],context=k,fine_grained_prior=args.fine_grained_prior))
                    torch.cuda.empty_cache()

                fs_pions,fs_kaons = combine_photons(fs_pions,fs_kaons)

                end = time.time()
            else:
                # Essentially generate 1 track with N photons.
                start = time.time()
                fs_kaons = kaon_net.create_tracks(num_samples=args.fs_support,context=k,fine_grained_prior=args.fine_grained_prior)
                fs_pions = pion_net.create_tracks(num_samples=args.fs_support,context=k,fine_grained_prior=args.fine_grained_prior)
                end = time.time()

            torch.cuda.empty_cache() # This is a must.
            support_pions,support_kaons = create_supports_fs(fs_pions,fs_kaons)
            print("Number of pions: ",len(support_pions)," Number of Kaons: ",len(support_kaons))
            del fs_kaons,fs_pions

        print("Time to create both PDFs: ",end - start)
        print("Time / photon: {0}.\n".format((end - start)/ (2*args.fs_support)))

        outpath = os.path.join(out_folder,f"Example_PDFs_theta_{theta_}_p_{args.momentum}.pdf")
        make_PDFs(support_pions, support_kaons, outpath, momentum=args.momentum, theta=theta_, log_norm=True)
 

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FastSim Generation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-p', '--momentum', default=6.0,type=float,help='Particle Momentum.')
    parser.add_argument('-mt','--model_type',default="NF",type=str,help='Which model to use.')
    parser.add_argument('-f','--fine_grained_prior',action='store_true',help="Enable fine grained prior, default False.")
    parser.add_argument('-fs','--fs_support', default=2e5,type=float,help='Number of Fast Simulated support photons.')
    args = parser.parse_args()

    args.fs_support = int(args.fs_support)

    config = json.load(open(args.config))

    main(config,args)
