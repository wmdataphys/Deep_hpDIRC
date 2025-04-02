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

import re
import sys

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
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # Setting up directory structure
    os.makedirs("Generations",exist_ok=True)
    out_folder = os.path.join("Generations",config['Inference']['full_phase_space_dir'])
    os.makedirs(out_folder,exist_ok=True)

    # Checking for existing data files
    exists = False
    out_files = os.listdir(out_folder)
    for file in out_files:
        match = re.search(rf'{args.method}_ntracks_{args.n_particles}\.pkl', file)
        if match:
            exists = True
            break
    
    if exists:
        print(f"Found existing .pkl files for {args.method} {args.model_type} for {args.n_particles} particles.")
        print("Making ratio plots.")
        make_ratios(path_=out_folder,label=args.method,momentum="1-10",outpath=os.path.join(out_folder,f"Ratios_{args.method}.pdf"))
        sys.exit()

    config['method'] = args.method

    if config['method'] == "Pion":
        print("Generating for pions.")
        dicte = torch.load(config['Inference']['pion_model_path_'+str(args.model_type)])
        if args.sample_photons:
            sampler_path = config['Photon_Sampler']['Pion_LUT_path']
        PID = 211
    elif config['method'] == 'Kaon':
        print("Generation for kaons.")
        dicte = torch.load(config['Inference']['kaon_model_path_'+str(args.model_type)])
        if args.sample_photons:
            sampler_path = config['Photon_Sampler']['Kaon_LUT_path']
        PID = 321
    else:
        print("Specify particle to generate in config file")
        exit()

    if not args.sample_photons:
        sampler_path = None

    if args.sample_photons:
        print("Using LUT photon yield sampling.")
    else:
        print("Using photon yield associated to ground truth track.")

    if args.fine_grained_prior:
        print("Using fine grained prior for PMTs. Consider disabling for increased generation speed.")
    else:
        print("Fine grained prior disabled. Consider enabling for high fidelity.")
        

    num_layers = int(config['model_'+str(args.model_type)]['num_layers'])
    input_shape = int(config['model_'+str(args.model_type)]['input_shape'])
    cond_shape = int(config['model_'+str(args.model_type)]['cond_shape'])
    hidden_nodes = int(config['model_'+str(args.model_type)]['hidden_nodes'])
    stats = config['stats']
    
    device = torch.device('cuda')
    
    if args.model_type == "NF":
        num_blocks = int(config['model_'+str(args.model_type)]['num_blocks'])
        net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats,LUT_path=sampler_path)
    elif args.model_type == "CNF":
        alph = config['model_'+str(args.model_type)]['alph']
        train_T = bool(config['model_'+str(args.model_type)]['train_T'])
        net = net = OT_Flow(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,train_T=train_T,alph=alph,LUT_path=sampler_path)
    elif args.model_type == "FlowMatching":
        net = FlowMatching(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,LUT_path=sampler_path)
    elif args.model_type == 'Score':
        num_steps = int(config['model_Score']['num_steps'])
        noise_schedule = config['model_Score']['noise_schedule']
        learned_schedule_net_hidden_dim = int(config['model_Score']['learned_schedule_net_hidden_dim'])
        gamma = int(config['model_Score']['gamma'])

        model = ResNet(input_dim=input_shape, end_dim=input_shape, cond_dim=cond_shape, mlp_dim=hidden_nodes, num_layer=num_layers)
        net = ContinuousTimeGaussianDiffusion(model=model, stats=stats,num_sample_steps=num_steps, noise_schedule=noise_schedule, learned_schedule_net_hidden_dim=learned_schedule_net_hidden_dim, min_snr_loss_weight = True, min_snr_gamma=gamma,LUT_path=sampler_path)
    elif args.model_type == 'GSGM':
        num_steps = int(config['model_GSGM']['num_steps'])
        num_embed = int(config['model_GSGM']['num_embed'])
        learned_variance = config['model_GSGM']['learned_variance']
        nonlinear_noise_schedule = int(config['model_GSGM']['nonlinear_noise_schedule'])
        
        net = GSGM(num_input=input_shape, num_conds=cond_shape, device=device, stats=stats, num_layers=num_layers, num_steps=num_steps, num_embed=num_embed, mlp_dim=hidden_nodes, nonlinear_noise_schedule=nonlinear_noise_schedule, learnedvar=learned_variance,LUT_path=sampler_path)
    elif args.model_type == 'CFG':
        num_steps = int(config['model_CFG']['num_steps'])
        noise_schedule = config['model_CFG']['noise_schedule']
        sampling_timesteps = config['model_CFG']['sampling_timesteps']
        noising_level = config['model_CFG']['noising_level']
        gamma = config['model_CFG']['gamma']

        model = CFGResNet(input_dim=input_shape, end_dim=input_shape, cond_dim=cond_shape, mlp_dim=hidden_nodes, num_layer=num_layers)
        net = CFGDiffusion(model=model, stats=stats, timesteps=num_steps, sampling_timesteps=sampling_timesteps, beta_schedule=noise_schedule, ddim_sampling_eta = noising_level, min_snr_loss_weight = True,min_snr_gamma=gamma,LUT_path=sampler_path)
    elif args.model_type == 'DDPM':
        num_steps = int(config['model_DDPM']['num_steps'])

        model = ResNet(input_dim=input_shape, end_dim=input_shape, cond_dim=cond_shape, mlp_dim=hidden_nodes, num_layer=num_layers)
        net = GaussianDiffusion(denoise_fn=model, device=device, stats=stats, timesteps=num_steps, loss_type='l2',LUT_path=sampler_path)
    else:
        raise ValueError("Model type not found.")

    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    net.to('cuda')
    net.load_state_dict(dicte['net_state_dict'])
    n_samples = int(config['Inference']['samples'])

    if config['method'] == 'Pion':
        print("Generating Pions over entire phase space.")
        print("Using first {0} pions for comparison.".format(args.n_particles))
        datapoints = np.load(config['dataset']['full_phase_space']["pion_data_path"],allow_pickle=True)[:args.n_particles]

    elif config['method'] == 'Kaon':
        print("Generating Kaons over entire phase space.")
        print("Using first {0} kaons for comparison.".format(args.n_particles))
        datapoints = np.load(config['dataset']['full_phase_space']["kaon_data_path"],allow_pickle=True)[:args.n_particles]
            
    else:
        raise ValueError("Method not found.")

    generations = []
    kbar = pkbar.Kbar(target=len(datapoints), width=20, always_stateful=False)
    start = time.time()
    n_photons = 0
    truth = []
    for i in range(len(datapoints)):   
        with torch.set_grad_enabled(False):
            if datapoints[i]['Phi'] != 0 or datapoints[i]['P'] > stats['P_max'] or datapoints[i]['P'] < stats['P_min'] or datapoints[i]['Theta'] > stats['theta_max'] or datapoints[i]['Theta'] < stats['theta_min'] or datapoints[i]['Phi'] != 0.0 or datapoints[i]['NHits'] <= 0:
                continue

            else:
                p = (datapoints[i]['P'] - stats['P_max'])  / (stats['P_max'] - stats['P_min'])
                theta = (datapoints[i]['Theta'] - stats['theta_max']) / (stats['theta_max'] - stats['theta_min'])
                k = torch.tensor(np.array([p,theta])).to('cuda').float()
                if not args.sample_photons:
                    if datapoints[i]['NHits'] > 0 and datapoints[i]['NHits'] < 300:
                        gen = net.create_tracks(num_samples=datapoints[i]['NHits'],context=k.unsqueeze(0),fine_grained_prior=args.fine_grained_prior)
                        truth.append(datapoints[i])
                    else:
                        continue
                else:
                    if datapoints[i]['NHits'] > 0 and datapoints[i]['NHits'] < 300:
                        gen = net.create_tracks(num_samples=None,context=k.unsqueeze(0),p=datapoints[i]['P'],theta=datapoints[i]['Theta'],fine_grained_prior=args.fine_grained_prior)
                        truth.append(datapoints[i])
                    else:
                        continue
  
        
        generations.append(gen)
        kbar.update(i)
    end = time.time()

    n_photons = 0
    n_gamma = 0

    for i in range(len(truth)):
        if truth[i]['NHits'] < 300:
            n_photons += truth[i]['NHits']

    if args.sample_photons:
        for i in range(len(generations)):
            n_gamma += generations[i]['NHits']

    else:
        n_gamma = n_photons


    print(" ")
    print("Number of tracks generated: ",len(generations))
    print("Number of photons generated: ",net.photons_generated)
    print("Number of photons resampled: ",net.photons_resampled)
    print('Percentage effect: ',net.photons_resampled * 100 / net.photons_generated)
    print("Elapsed Time: ", end - start)
    print("Time / photon: ",(end - start) / n_gamma)
    print("Average time / track: ",(end - start) / len(truth))
    if args.sample_photons:
        print("True photon yield: ",n_photons," Generated photon yield: ",n_gamma)
    print(" ")
    gen_dict = {}
    gen_dict['fast_sim'] = generations
    gen_dict['truth'] = truth


    print("Outputs can be found in " + str(out_folder))

    if args.sample_photons:
        out_path_ = os.path.join(out_folder,str(config['method'])+f"_ntracks_{len(datapoints)}_LUT.pkl")
    else:
        out_path_ = os.path.join(out_folder,str(config['method'])+f"_ntracks_{len(datapoints)}.pkl")
    
    with open(out_path_,"wb") as file:
        pickle.dump(gen_dict,file)

    print("Making ratio plots.")
    del gen_dict,datapoints 

    ratio_title = f"Ratios_{args.method}_{args.n_particles}_LUT.pdf" if args.sample_photons else f"Ratios_{args.method}_{args.n_particles}.pdf"
    # pkl_label = args.method + 

    make_ratios(path_=out_folder,label=args.method,momentum="1-10",outpath=os.path.join(out_folder,ratio_title))


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FastSim Generation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-nt', '--n_particles', default=2e5,type=int,help='Number of particles to generate. Take the first n_particles. -1 is all tracks available.')
    parser.add_argument('-m', '--method',default="Kaon",type=str,help='Generated particle type, Kaon or Pion.')
    parser.add_argument('-mt','--model_type',default="NF",type=str,help='Which model to use.')
    parser.add_argument('-sp','--sample_photons', action='store_true', help="Enable verbose mode")
    parser.add_argument('-f','--fine_grained_prior',action='store_true',help="Enable fine grained prior, default False.")
    args = parser.parse_args()

    args.n_particles = int(args.n_particles)

    config = json.load(open(args.config))

    main(config,args)
