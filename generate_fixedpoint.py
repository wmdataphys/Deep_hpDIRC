import os
import json
import argparse
import torch
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from dataloader.dataloader import CreateLoaders
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from models.nflows_models import create_nflows,MAAF
from dataloader.create_data import create_dataset
from datetime import datetime
import itertools
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LogNorm
from models.freia_models import FreiaNet
import matplotlib.colors as mcolors
import pickle

def main(config,n_datasets):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    if config['method'] == "Pion":
        print("Generating for pions.")
        dicte = torch.load(config['Inference']['pion_model_path'])

    elif config['method'] == 'Kaon':
        print("Generation for kaons.")
        dicte = torch.load(config['Inference']['kaon_model_path'])
    else:
        print("Specify particle to generate in config file")
        exit()

    log_time = bool(config['log_time'])
    # Create the model
    # This will map gen -> Reco
    if config['method'] == 'Pion':
        num_layers = int(config['model']['num_layers'])
        PID = 211
    elif config['method'] == 'Kaon':
        num_layers = int(config['model']['num_layers'])
        PID = 321
    else:
        num_layers = int(config['model']['num_layers'])

    input_shape = int(config['model']['input_shape'])
    cond_shape = int(config['model']['cond_shape'])
    num_blocks = int(config['model']['num_blocks'])
    hidden_nodes = int(config['model']['hidden_nodes'])
    stats = config['stats']
    net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,log_time=log_time,stats=stats)
    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    device = torch.device('cuda')
    net.to('cuda')
    net.load_state_dict(dicte['net_state_dict'])
    n_samples = int(config['Inference']['samples'])
    p = (3.5 - stats['P_max'])  / (stats['P_max'] - stats['P_min'])
    theta = (4.0 - stats['theta_max']) / (stats['theta_max'] - stats['theta_min'])
    phi = (40.0 - stats['phi_max']) / (stats['phi_max'] - stats['phi_min'])
    k = torch.tensor(np.array([p,theta,phi])).to('cuda').float()        
    with torch.set_grad_enabled(False):
        
        #gen = net.probabalistic_sample(pre_compute_dist=3000,context=k,photon_yield=num_samples)
        gen = net.create_tracks(num_samples=10000,context=k.unsqueeze(0))

    print(" ")
    print("Number of photons generated: ",net.photons_generated)
    print("Number of photons resampled: ",net.photons_resampled)
    print('Percentage effect: ',net.photons_resampled * 100 / net.photons_generated)
    print(" ")
    with open(str(config['method'])+"_p3.5_theta4_phi40.pkl","wb") as file:
        pickle.dump(gen,file)



if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FastSim Generation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-n', '--n_datasets', default=1, type=int,
                        help='Number of time to Fast Simulate the dataset.')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args.n_datasets)
