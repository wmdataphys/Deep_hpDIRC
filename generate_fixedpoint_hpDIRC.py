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
import re
import sys

from models.NF.freia_models import FreiaNet
from models.OT_Flow.ot_flow import OT_Flow
from models.FlowMatching.flow_matching import FlowMatching
from models.Diffusion.resnet import ResNet
from models.Diffusion.resnet_cfg import ResNet as CFGResNet
from models.Diffusion.continuous_diffusion import ContinuousTimeGaussianDiffusion
from models.Diffusion.classifier_free_guidance import CFGDiffusion
from models.Diffusion.gaussian_diffusion import GaussianDiffusion
from models.Diffusion.gaussian_diffusion_shift import ShiftGaussianDiffusion
from models.Diffusion.shift_predictor import TabShiftPredictor

warnings.filterwarnings("ignore", message=".*weights_only.*")


def main(config,args):

    # Setup random seed
    print("Using model type: ",str(args.model_type))
    # Remove seeding, make it random.
    seed = int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

    # Setting up directory structure
    os.makedirs("Generations",exist_ok=True)
    out_folder = os.path.join("Generations",config['Inference']['fixed_point_dir'])
    os.makedirs(out_folder,exist_ok=True)
    print("Outputs can be found in " + str(out_folder))

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

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise RuntimeError("No GPU was found! Exiting the program.")
    if not args.sample_photons:
        sampler_path = None
        

    num_layers = int(config['model_'+str(args.model_type)]['num_layers'])
    input_shape = int(config['model_'+str(args.model_type)]['input_shape'])
    cond_shape = int(config['model_'+str(args.model_type)]['cond_shape'])
    hidden_nodes = int(config['model_'+str(args.model_type)]['hidden_nodes'])
    stats = config['stats']
    
    if args.model_type == "NF":
        num_blocks = int(config['model_'+str(args.model_type)]['num_blocks'])
        net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats,LUT_path=sampler_path)
    elif args.model_type == "CNF":
        num_blocks = int(config['model_'+str(args.model_type)]['num_blocks'])
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
    elif args.model_type == 'Shift':
        num_steps = int(config['model_Shift']['num_steps'])
        noise_schedule = config['model_Shift']['noise_schedule']
        shift_type = config['model_Shift']['shift_type']

        model = ResNet(input_dim=input_shape, 
                   end_dim=input_shape, 
                   cond_dim=cond_shape, 
                   mlp_dim=hidden_nodes, 
                   num_layer=num_layers
                   )
        shift_predictor = TabShiftPredictor(device=device,
                                        input_dim=input_shape, 
                                        cond_dim=cond_shape, 
                                        hidden_dim=hidden_nodes)
        net = ShiftGaussianDiffusion(
                    device=device,
                    denoise_fn=model,
                    shift_predictor=shift_predictor,
                    stats=stats,
                    timesteps=num_steps, 
                    noise_schedule=noise_schedule,
                    shift_type=shift_type)
    else:
        raise ValueError("Model type not found.")

    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    
    net.to('cuda')
    
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
        if (datapoints[i]['Theta'] == args.theta) and (datapoints[i]['P'] == args.momentum) and (datapoints[i]['Phi'] == 0.0):
            list_to_gen.append(datapoints[i])

    out_path_ = os.path.join(out_folder,str(config['method'])+f"_p_{args.momentum}_theta_{args.theta}_PID_{config['method']}_ntracks_{len(list_to_gen)}.pkl")

    # Checking for existing data files
    exists = False
    out_files = os.listdir(out_folder)
    for file in out_files:
        match = re.search(rf"{config['method']}_p_{args.momentum}_theta_{args.theta}_PID_{config['method']}_ntracks_{len(list_to_gen)}.pkl", file)
        if match:
            exists = True
            break
    
    if exists:
        print(f"Found existing .pkl files for {args.method} {args.model_type} for {len(list_to_gen)} tracks at momentum {args.momentum} and theta {args.theta}.")
        print("Skipping...\n")
        sys.exit()

    print("Generating {0} tracks, with p={1} and theta={2}.".format(len(list_to_gen),args.momentum,args.theta))
    if args.sample_photons:
        print("Using LUT photon yield sampling.")
    else:
        print("Using photon yield associated to ground truth track.")

    if args.fine_grained_prior:
        print("Using fine grained prior for PMTs. Consider disabling for increased generation speed.")
    else:
        print("Fine grained prior disabled. Consider enabling for high fidelity.")


    generations = []
    kbar = pkbar.Kbar(target=len(list_to_gen), width=20, always_stateful=False)
    start = time.time()
    truth = []
    
    for i in range(len(list_to_gen)):   
        with torch.set_grad_enabled(False):
            p = (list_to_gen[i]['P'] - stats['P_max'])  / (stats['P_max'] - stats['P_min'])
            theta = (list_to_gen[i]['Theta'] - stats['theta_max']) / (stats['theta_max'] - stats['theta_min'])
            k = torch.tensor(np.array([p,theta])).to('cuda').float()
            
            if not args.sample_photons:
                if list_to_gen[i]['NHits'] > 0 and list_to_gen[i]['NHits'] < 300:
                    gen = net.create_tracks(num_samples=list_to_gen[i]['NHits'],context=k.unsqueeze(0),fine_grained_prior=args.fine_grained_prior)
                    truth.append(list_to_gen[i])
                else:
                    #print("Error with number of hits to generate: ",list_to_gen[i]['NHits'])
                    continue
            else:
                if list_to_gen[i]['NHits'] > 0 and list_to_gen[i]['NHits'] < 300:
                    gen = net.create_tracks(num_samples=None,context=k.unsqueeze(0),p=list_to_gen[i]['P'],theta=list_to_gen[i]['Theta'],fine_grained_prior=args.fine_grained_prior)
                    truth.append(list_to_gen[i])
                else:
                    #print("Error with number of hits to generate: ",list_to_gen[i]['NHits'])
                    continue

        
        generations.append(gen)
        kbar.update(i)
    end = time.time()

    

    n_photons = 0
    n_gamma = 0

    for i in range(len(truth)):
        if truth[i]['NHits'] < 300:
            n_photons += truth[i]['NHits']

    for i in range(len(generations)):
        n_gamma += generations[i]['NHits']


    print(" ")
    print("Number of tracks generated: ",len(generations))
    print("Number of photons generated: ",net.photons_generated)
    print("Number of photons resampled: ",net.photons_resampled)
    print('Percentage effect: ',net.photons_resampled * 100 / net.photons_generated)
    print("Elapsed Time: ", end - start)
    print("Time / photon: ",(end - start) / n_gamma)
    print("Average time / track: ",(end - start) / len(truth))
    print("True photon yield: ",n_photons," Generated photon yield: ",n_gamma)
    print(" ")
    gen_dict = {}
    gen_dict['fast_sim'] = generations
    gen_dict['truth'] = truth


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
    parser.add_argument('-sp','--sample_photons', action='store_true', help="Enable verbose mode")
    parser.add_argument('-f','--fine_grained_prior',action='store_true',help="Enable fine grained prior, default False.")
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args)
