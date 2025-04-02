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

warnings.filterwarnings("ignore", message=".*weights_only.*")

from models.NF.freia_models import FreiaNet
from models.OT_Flow.ot_flow import OT_Flow
from models.FlowMatching.flow_matching import FlowMatching
from models.Diffusion.resnet import ResNet
from models.Diffusion.resnet_cfg import ResNet as CFGResNet
from models.Diffusion.continuous_diffusion import ContinuousTimeGaussianDiffusion
from models.Diffusion.gsgm import GSGM
from models.Diffusion.classifier_free_guidance import CFGDiffusion
from models.Diffusion.gaussian_diffusion import GaussianDiffusion


def main(config,args):

    # Setup random seed
    print("Using model type: ",str(args.model_type))
    # Remove seeding, make it random.
    seed = int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

    if args.method in ['Kaon','MixPK']:
        print("Generation for kaons.")
        Kdicte = torch.load(config['Inference']['kaon_model_path_'+str(args.model_type)])
        Ksampler_path = config['Photon_Sampler']['Kaon_LUT_path']
        PID = 321
    if args.method == "Pion" or "MixPK":
        print("Generating for pions.")
        Pdicte = torch.load(config['Inference']['pion_model_path_'+str(args.model_type)])
        Psampler_path = config['Photon_Sampler']['Pion_LUT_path']
        PID = 211
    else:
        print("Specify particle to generate in config file")
        exit()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError("No GPU was found! Exiting the program.")
    
    if not args.n_dump:
        print('No value found for n_dump. Setting it equal to {}'.format(args.n_tracks))
        args.n_dump = args.n_tracks
    
    assert args.n_dump <= args.n_tracks, "total n_tracks must be at least n_dump, the number of tracks to dump per .pkl file. Got n_tracks of {} and n_dump of {} instead.".format(args.n_tracks, args.n_dump)

    num_layers = int(config['model_'+str(args.model_type)]['num_layers'])
    input_shape = int(config['model_'+str(args.model_type)]['input_shape'])
    cond_shape = int(config['model_'+str(args.model_type)]['cond_shape'])
    hidden_nodes = int(config['model_'+str(args.model_type)]['hidden_nodes'])
    stats = config['stats']
    
    if args.model_type == "NF":
        num_blocks = int(config['model_'+str(args.model_type)]['num_blocks'])
        if args.method in ['Kaon','MixPK']:
            kaon_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats,LUT_path=Ksampler_path)
        if args.method in ['Pion','MixPK']:
            pion_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats,LUT_path=Psampler_path)
    elif args.model_type == "CNF":
        num_blocks = int(config['model_'+str(args.model_type)]['num_blocks'])
        alph = config['model_'+str(args.model_type)]['alph']
        train_T = bool(config['model_'+str(args.model_type)]['train_T'])
        if args.method in ['Kaon','MixPK']:
            kaon_net = OT_Flow(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,train_T=train_T,alph=alph,LUT_path=Ksampler_path)
        if args.method in ['Pion','MixPK']:
            pion_net = OT_Flow(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,train_T=train_T,alph=alph,LUT_path=Psampler_path)
    elif args.model_type == "FlowMatching":
        if args.method in ['Kaon','MixPK']:
            kaon_net = FlowMatching(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,LUT_path=Ksampler_path)
        if args.method in ['Pion','MixPK']:
            pion_net = FlowMatching(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,LUT_path=Psampler_path)
    elif args.model_type == 'Score':
        num_steps = int(config['model_Score']['num_steps'])
        noise_schedule = config['model_Score']['noise_schedule']
        learned_schedule_net_hidden_dim = int(config['model_Score']['learned_schedule_net_hidden_dim'])
        gamma = int(config['model_Score']['gamma'])

        model = ResNet(input_dim=input_shape, end_dim=input_shape, cond_dim=cond_shape, mlp_dim=hidden_nodes, num_layer=num_layers)
        if args.method in ['Kaon','MixPK']:
            kaon_net = ContinuousTimeGaussianDiffusion(model=model, stats=stats,num_sample_steps=num_steps, noise_schedule=noise_schedule, learned_schedule_net_hidden_dim=learned_schedule_net_hidden_dim, min_snr_loss_weight = True, min_snr_gamma=gamma,LUT_path=Ksampler_path)
        if args.method in ['Pion','MixPK']:
            pion_net = ContinuousTimeGaussianDiffusion(model=model, stats=stats,num_sample_steps=num_steps, noise_schedule=noise_schedule, learned_schedule_net_hidden_dim=learned_schedule_net_hidden_dim, min_snr_loss_weight = True, min_snr_gamma=gamma,LUT_path=Psampler_path)
    elif args.model_type == 'GSGM':
        num_steps = int(config['model_GSGM']['num_steps'])
        num_embed = int(config['model_GSGM']['num_embed'])
        learned_variance = config['model_GSGM']['learned_variance']
        nonlinear_noise_schedule = int(config['model_GSGM']['nonlinear_noise_schedule'])

        if args.method in ['Kaon','MixPK']:
            kaon_net = GSGM(num_input=input_shape, num_conds=cond_shape, device=device, stats=stats, num_layers=num_layers, num_steps=num_steps, num_embed=num_embed, mlp_dim=hidden_nodes, nonlinear_noise_schedule=nonlinear_noise_schedule, learnedvar=learned_variance,LUT_path=Ksampler_path)
        if args.method in ['Pion','MixPK']:
            pion_net = GSGM(num_input=input_shape, num_conds=cond_shape, device=device, stats=stats, num_layers=num_layers, num_steps=num_steps, num_embed=num_embed, mlp_dim=hidden_nodes, nonlinear_noise_schedule=nonlinear_noise_schedule, learnedvar=learned_variance,LUT_path=Psampler_path)
    elif args.model_type == 'DDPM':
        num_steps = int(config['model_DDPM']['num_steps'])

        model = ResNet(input_dim=input_shape, end_dim=input_shape, cond_dim=cond_shape, mlp_dim=hidden_nodes, num_layer=num_layers)
        if args.method in ['Kaon','MixPK']:
            kaon_net = GaussianDiffusion(denoise_fn=model, device=device, stats=stats, timesteps=num_steps, loss_type='l2',LUT_path=Ksampler_path)
        if args.method in ['Pion','MixPK']:
            pion_net = GaussianDiffusion(denoise_fn=model, device=device, stats=stats, timesteps=num_steps, loss_type='l2',LUT_path=Psampler_path)
    else:
        raise ValueError("Model type not found.")
    
    if args.method in ['Kaon','MixPK']:
        kaon_net.to('cuda')
    if args.method in ['Pion','MixPK']:
        pion_net.to('cuda')

    print("Using LUT photon yield sampling.")
    
    if args.fine_grained_prior:
        print("Using fine grained prior for PMTs. Consider disabling for increased generation speed.")
    else:
        print("Fine grained prior disabled. Consider enabling for high fidelity.")

    if args.method in ['Kaon','MixPK']:
        kaon_net.load_state_dict(Kdicte['net_state_dict'], strict=False)
    if args.method in ['Pion','MixPK']:
        pion_net.load_state_dict(Pdicte['net_state_dict'], strict=False)

    generations = []
    kbar = pkbar.Kbar(target=args.n_tracks, width=20, always_stateful=False)
    start = time.time()

    if re.fullmatch(r"\d+(?:\.\d+)?", args.momentum):
        p_low = float(args.momentum)
        p_high = float(args.momentum)
    elif re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", args.momentum):
        match = re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", args.momentum)
        p_low = float(match.group(1))
        p_high = float(match.group(2))
    else:
        raise ValueError('Momentum format is p1-p2, where p1, p2 are between 1 and 10 (e.g. 3.5-10).')
    
    if re.fullmatch(r"\d+(?:\.\d+)?", args.theta):
        theta_low = int(args.theta)
        theta_high = int(args.theta)
    elif re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", args.theta):
        match = re.fullmatch(r"(\d+)-(\d+)", args.theta)
        theta_low = int(match.group(1))
        theta_high = int(match.group(2))
    else:
        raise ValueError('Theta format is theta1-theta2, where theta1, theta2 are between 25 and 155 (e.g. 70.5-80)')
    
    assert 1 <= p_low <= 10 and 1 <= p_high <= 10, "momentum range must be between 0 and 10, got {}, {} instead.".format(p_low, p_high)
    assert 25 <= theta_low <= 155 and 25 <= theta_high <= 155, "theta range must be between 25 and 155, got {}, {} instead.".format(theta_low, theta_high)
    
    running_gen = 0
    file_count = 1
    while running_gen < args.n_tracks:
        if args.method in ['Kaon','MixPK']:
            with torch.set_grad_enabled(False):
                p = np.random.uniform(low = p_low, high = p_high, size = None)
                p_scaled = (p - stats['P_max'])  / (stats['P_max'] - stats['P_min'])

                theta = np.random.uniform(low = theta_low, high = theta_high, size = None)
                theta_scaled = theta = (theta - stats['theta_max']) / (stats['theta_max'] - stats['theta_min'])

                k = torch.tensor(np.array([p_scaled,theta_scaled])).to('cuda').float()

                gen = kaon_net.create_tracks(num_samples=None,context=k.unsqueeze(0),p=p,theta=theta,fine_grained_prior=args.fine_grained_prior)

            generations.append(gen)
            running_gen +=1 
            kbar.update(running_gen)

            if running_gen >= args.n_tracks:
                break

            if len(generations) == args.n_dump:
                print(f'Generated {args.n_dump} samples. Making .pkl file...')
                gen_dict = {}
                gen_dict['fast_sim'] = generations

                os.makedirs("Generations",exist_ok=True)
                out_folder = os.path.join("Generations",config['Inference']['fixed_point_dir'])
                os.makedirs(out_folder,exist_ok=True)

                out_path_ = os.path.join(out_folder,str(args.method)+f"_p_{args.momentum}_theta_{args.theta}_PID_{args.method}_ntracks_{len(generations)}_{file_count}.pkl")
                
                with open(out_path_,"wb") as file:
                    pickle.dump(gen_dict,file)

                # reset lists
                generations = []
                gen_dict = {}
                file_count += 1

        if args.method in ['Pion','MixPK']:
            with torch.set_grad_enabled(False):
                p = np.random.uniform(low = p_low, high = p_high, size = None)
                p_scaled = (p - stats['P_max'])  / (stats['P_max'] - stats['P_min'])

                theta = np.random.uniform(low = theta_low, high = theta_high, size = None)
                theta_scaled = theta = (theta - stats['theta_max']) / (stats['theta_max'] - stats['theta_min'])

                k = torch.tensor(np.array([p_scaled,theta_scaled])).to('cuda').float()

                gen = pion_net.create_tracks(num_samples=None,context=k.unsqueeze(0),p=p,theta=theta,fine_grained_prior=args.fine_grained_prior)

            generations.append(gen)
            running_gen +=1 
            kbar.update(running_gen)

            if running_gen >= args.n_tracks:
                break

            if len(generations) == args.n_dump:
                print(f'Generated {args.n_dump} samples. Making .pkl file...')
                gen_dict = {}
                gen_dict['fast_sim'] = generations

                os.makedirs("Generations",exist_ok=True)
                out_folder = os.path.join("Generations",config['Inference']['fixed_point_dir'])
                os.makedirs(out_folder,exist_ok=True)

                out_path_ = os.path.join(out_folder,str(args.method)+f"_p_{args.momentum}_theta_{args.theta}_PID_{args.method}_ntracks_{len(generations)}_{file_count}.pkl")
                
                with open(out_path_,"wb") as file:
                    pickle.dump(gen_dict,file)

                # reset lists
                generations = []
                gen_dict = {}
        
    end = time.time()

    n_photons = 0
    n_gamma = 0

    for i in range(len(generations)):
        n_gamma += generations[i]['NHits']

    time_per_photon = (end - start) / n_gamma if n_gamma > 0 else r'N/A'
    time_per_track = (end - start) / len(generations) if len(generations) > 0 else r'N/A'
    
    if args.method == 'Kaon' or args.method == 'MixPK':
        print(" ")
        print("Sampling statistics for kaon net:")
        print("Number of tracks generated: ",len(generations))
        print("Number of photons generated: ",kaon_net.photons_generated)
        print("Number of photons resampled: ",kaon_net.photons_resampled)
        kaon_percentage_effect = kaon_net.photons_resampled * 100 / kaon_net.photons_generated if kaon_net.photons_generated != 0 else r'N/A' 
        print('Percentage effect: ',kaon_percentage_effect)
        print("Elapsed Time: ", end - start)
        print("Time / photon: ",time_per_photon)
        print("Average time / track: ",time_per_track)
        print("Generated photon yield: ",n_gamma)
        print(" ")

    if args.method == 'Pion' or args.method == 'MixPK':
        print(" ")
        print("Sampling statistics for pion net:")
        print("Number of tracks generated: ",len(generations))
        print("Number of photons generated: ",pion_net.photons_generated)
        print("Number of photons resampled: ",pion_net.photons_resampled)
        pion_percentage_effect = pion_net.photons_resampled * 100 / pion_net.photons_generated if pion_net.photons_generated != 0 else r'N/A' 
        print('Percentage effect: ',pion_percentage_effect)
        print("Elapsed Time: ", end - start)
        print("Time / photon: ",time_per_photon)
        print("Average time / track: ",time_per_track)
        print("Generated photon yield: ",n_gamma)
        print(" ")
    
    print(f'Generating final samples. Making .pkl file...')
    gen_dict = {}
    gen_dict['fast_sim'] = generations

    os.makedirs("Generations",exist_ok=True)
    out_folder = os.path.join("Generations",config['Inference']['fixed_point_dir'])
    os.makedirs(out_folder,exist_ok=True)
    print("Outputs can be found in " + str(out_folder))

    out_path_ = os.path.join(out_folder,str(args.method)+f"_p_{args.momentum}_theta_{args.theta}_PID_{args.method}_ntracks_{len(generations)}_{file_count}.pkl")
    
    with open(out_path_,"wb") as file:
        pickle.dump(gen_dict,file)
    

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FastSim Generation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-nt', '--n_tracks', default=1e4,type=int,help='Number of particles to generate. Take the first n_tracks.')
    parser.add_argument('-nd', '--n_dump', default=None, type=int, help='Number of particles to dump per .pkl file.')
    parser.add_argument('-m', '--method',default="MixPK",type=str,help='Generated particle type, Kaon, Pion, or MixPK.')
    parser.add_argument('-rho','--momentum',default=3,type=str,help='Which momentum to generate for.')
    parser.add_argument('-th','--theta',default=30,type=str,help='Which theta angle to generate for.')
    parser.add_argument('-mt','--model_type',default="NF",type=str,help='Which model to use.')
    # parser.add_argument('-sp','--sample_photons', action='store_true', help="Enable verbose mode")
    parser.add_argument('-f','--fine_grained_prior',action='store_true',help="Enable fine grained prior, default False.")
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args)
