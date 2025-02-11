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

from models.NF.freia_models import FreiaNet
from models.OT_Flow.ot_flow import OT_Flow
from models.FlowMatching.flow_matching import FlowMatching
from models.Diffusion.resnet import ResNet
from models.Diffusion.resnet_cfg import ResNet as CFGResNet
from models.Diffusion.continuous_diffusion import ContinuousTimeGaussianDiffusion
from models.Diffusion.gsgm import GSGM
from models.Diffusion.classifier_free_guidance import CFGDiffusion
from models.Diffusion.gaussian_diffusion import GaussianDiffusion

import matplotlib.colors as mcolors
import pickle
import warnings

warnings.filterwarnings("ignore", message=".*weights_only.*")


def main(config,args):

    # Setup random seed
    print("Using model type: ",str(args.model_type))
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    config['method'] = args.method

    if config['method'] == "Pion":
        print("Generating for pions.")
        dicte = torch.load(config['Inference']['pion_model_path_'+str(args.model_type)])
        PID = 211
    elif config['method'] == 'Kaon':
        print("Generation for kaons.")
        dicte = torch.load(config['Inference']['kaon_model_path_'+str(args.model_type)])
        PID = 321
    else:
        print("Specify particle to generate in config file")
        exit()

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
        net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats)
    elif args.model_type == "CNF":
        num_blocks = int(config['model_'+str(args.model_type)]['num_blocks'])
        alph = config['model_'+str(args.model_type)]['alph']
        train_T = bool(config['model_'+str(args.model_type)]['train_T'])
        net = net = OT_Flow(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,train_T=train_T,alph=alph)
    elif args.model_type == "FlowMatching":
        net = FlowMatching(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats)
    elif args.model_type == 'Score':
        num_steps = int(config['model_Score']['num_steps'])
        noise_schedule = config['model_Score']['noise_schedule']
        learned_schedule_net_hidden_dim = int(config['model_Score']['learned_schedule_net_hidden_dim'])
        gamma = int(config['model_Score']['gamma'])

        model = ResNet(input_dim=input_shape, end_dim=input_shape, cond_dim=cond_shape, mlp_dim=hidden_nodes, num_layer=num_layers)
        net = ContinuousTimeGaussianDiffusion(model=model, stats=stats,num_sample_steps=num_steps, noise_schedule=noise_schedule, learned_schedule_net_hidden_dim=learned_schedule_net_hidden_dim, min_snr_loss_weight = True, min_snr_gamma=gamma)
    elif args.model_type == 'GSGM':
        num_steps = int(config['model_GSGM']['num_steps'])
        num_embed = int(config['model_GSGM']['num_embed'])
        learned_variance = config['model_GSGM']['learned_variance']
        nonlinear_noise_schedule = int(config['model_GSGM']['nonlinear_noise_schedule'])
        
        net = GSGM(num_input=input_shape, num_conds=cond_shape, device=device, stats=stats, num_layers=num_layers, num_steps=num_steps, num_embed=num_embed, mlp_dim=hidden_nodes, nonlinear_noise_schedule=nonlinear_noise_schedule, learnedvar=learned_variance)
    elif args.model_type == 'CFG':
        num_steps = int(config['model_CFG']['num_steps'])
        noise_schedule = config['model_CFG']['noise_schedule']
        sampling_timesteps = config['model_CFG']['sampling_timesteps']
        noising_level = config['model_CFG']['noising_level']
        gamma = config['model_CFG']['gamma']

        model = CFGResNet(input_dim=input_shape, end_dim=input_shape, cond_dim=cond_shape, mlp_dim=hidden_nodes, num_layer=num_layers)
        net = CFGDiffusion(model=model, stats=stats, timesteps=num_steps, sampling_timesteps=sampling_timesteps, beta_schedule=noise_schedule, ddim_sampling_eta = noising_level, min_snr_loss_weight = True,min_snr_gamma=gamma)
    elif args.model_type == 'DDPM':
        num_steps = int(config['model_DDPM']['num_steps'])

        model = ResNet(input_dim=input_shape, end_dim=input_shape, cond_dim=cond_shape, mlp_dim=hidden_nodes, num_layer=num_layers)
        net = GaussianDiffusion(denoise_fn=model, device=device, stats=stats, timesteps=num_steps, loss_type='l2')
    else:
        raise ValueError("Model type not found.")

    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    
    net.to('cuda')

    # For cfg testing
    if 'model.null_classes_emb' in dicte['net_state_dict']:
        del dicte['net_state_dict']['model.null_classes_emb']
    
    net.load_state_dict(dicte['net_state_dict'], strict=False)
    n_samples = int(config['Inference']['samples'])

    if config['method'] == 'Pion':
        print("Generating pions with momentum of {0} GeV/c".format(args.momentum))
        if args.momentum == 1.0:
            datapoints = np.load(config['dataset']['fixed_point']["pion_data_path_1GeV"],allow_pickle=True)
        elif args.momentum == 3.0:
            datapoints = np.load(config['dataset']['fixed_point']["pion_data_path_3GeV"],allow_pickle=True)
        elif args.momentum == 6.0:
            datapoints = np.load(config['dataset']['fixed_point']["pion_data_path_6GeV"],allow_pickle=True)
        elif args.momentum == 9.0:
            datapoints = np.load(config['dataset']['fixed_point']["pion_data_path_9GeV"],allow_pickle=True)
        else:
            raise ValueError("Value of momentum does correspond to a dataset. Check if the path is correct, or simulate and processes.")

    elif config['method'] == 'Kaon':
        print("Generating kaons with momentum of {0} GeV/c".format(args.momentum))
        if args.momentum == 1.0:
            datapoints = np.load(config['dataset']['fixed_point']["kaon_data_path_1GeV"],allow_pickle=True)
        elif args.momentum == 3.0:
            datapoints = np.load(config['dataset']['fixed_point']["kaon_data_path_3GeV"],allow_pickle=True)
        elif args.momentum == 6.0:
            datapoints = np.load(config['dataset']['fixed_point']["kaon_data_path_6GeV"],allow_pickle=True)
        elif args.momentum == 9.0:
            datapoints = np.load(config['dataset']['fixed_point']["kaon_data_path_9GeV"],allow_pickle=True)
        else:
            raise ValueError("Value of momentum does correspond to a dataset. Check if the path is correct, or simulate and processes.")       
        
    else:
        raise ValueError("Method not found.")


    list_to_gen = []

    for i in range(len(datapoints)):
        if (datapoints[i]['Theta'] == args.theta) and (datapoints[i]['P'] == args.momentum):
            list_to_gen.append(datapoints[i])

    print("Generating {0} tracks, with p={1} and theta={2}.".format(len(list_to_gen),args.momentum,args.theta))
       
    generations = []
    kbar = pkbar.Kbar(target=len(list_to_gen), width=20, always_stateful=False)
    start = time.time()
    for i in range(len(list_to_gen)):   
        with torch.set_grad_enabled(False):
            p = (list_to_gen[i]['P'] - stats['P_max'])  / (stats['P_max'] - stats['P_min'])
            theta = (list_to_gen[i]['Theta'] - stats['theta_max']) / (stats['theta_max'] - stats['theta_min'])
            k = torch.tensor(np.array([p,theta])).to('cuda').float()
            
            if list_to_gen[i]['NHits'] > 0:
                gen = net.create_tracks(num_samples=list_to_gen[i]['NHits'],context=k.unsqueeze(0))
            else:
                print("Error with number of hits to generate: ",list_to_gen[i]['NHits'])
                continue
        
        generations.append(gen)
        kbar.update(i)
    end = time.time()

    n_photons = 0

    for i in range(len(list_to_gen)):
        n_photons += list_to_gen[i]['NHits']

    print(" ")
    print("Number of tracks generated: ",len(generations))
    print("Number of photons generated: ",net.photons_generated)
    print("Number of photons resampled: ",net.photons_resampled)
    print('Percentage effect: ',net.photons_resampled * 100 / net.photons_generated)
    print("Elapsed Time: ", end - start)
    print("Time / photon: ",(end - start) / n_photons)
    print("Average time / track: ",(end - start) / len(list_to_gen))
    print(" ")
    gen_dict = {}
    gen_dict['fast_sim'] = generations
    gen_dict['truth'] = list_to_gen

    os.makedirs("Generations",exist_ok=True)
    out_folder = os.path.join("Generations",config['Inference']['fixed_point_dir'])
    os.makedirs(out_folder,exist_ok=True)
    print("Outputs can be found in " + str(out_folder))

    out_path_ = os.path.join(out_folder,str(config['method'])+f"_p_{args.momentum}_theta_{args.theta}_PID_{config['method']}_ntracks_{len(list_to_gen)}.pkl")
    
    with open(out_path_,"wb") as file:
        pickle.dump(gen_dict,file)




if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FastSim Generation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-p', '--momentum', default=6.0,type=float,help='Particle Momentum.')
    parser.add_argument('-t','--theta',default=30.0,type=float,help='Particle theta.')
    parser.add_argument('-m', '--method',default="Kaon",type=str,help='Generated particle type, Kaon or Pion.')
    parser.add_argument('-mt','--model_type',default="NF",type=str,help='Which model to use.')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args)
