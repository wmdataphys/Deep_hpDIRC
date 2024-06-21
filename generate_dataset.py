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
    print("Running inference")

    if not os.path.exists(config['Inference']['generation_dir']):
        os.mkdir(config['Inference']['generation_dir'])

    
    full_path = os.path.join(config['Inference']['generation_dir'],config["method"])
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    print('Generations can be found in: ' + full_path)

       # Load the dataset
    print('Creating Loaders.')
    if config['method'] == "Pion":
        print("Generating for pions.")
        data = np.load(config['dataset']['mcmc']['pion_data_path'],allow_pickle=True)
        dicte = torch.load(config['Inference']['pion_model_path'])

    elif config['method'] == 'Kaon':
        print("Generation for kaons.")
        data = np.load(config['dataset']['mcmc']['kaon_data_path'],allow_pickle=True)
        dicte = torch.load(config['Inference']['kaon_model_path'])
    else:
        print("Specify particle to generate in config file")
        exit()

    print(data.keys())
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
    net = MAAF(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats)
    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    device = torch.device('cuda')
    net.to('cuda')
    net.load_state_dict(dicte['net_state_dict'])
    n_samples = int(config['Inference']['samples'])

    #momentum_bins = [0.9,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5]
    #momentum_bins = [6.0,6.5,7.0,7.5,8.0,8.5]
    
    
    barIDs = np.array(data['BarID'])
    barX = np.array(data['X'])
    P_ = np.array(data['P'])
    print("Momentum: ",np.min(P_),np.max(P_))
    #momentum_bins = np.arange(np.min(P_),np.max(P_)+0.5,0.5) # 500 MeV bins
    #momentum_bins = np.arange(3.98,np.max(P_)+0.5,0.5)
    momentum_bins = [3.978703, 4.478703, 4.978703, 5.478703, 5.978703, 6.478703, 6.978703, 7.478703, 7.978703, 8.478703,
                    8.978703]
    print('Generating chunked dataset for {0} momentum bins.'.format(len(momentum_bins)))
    print(momentum_bins)
    print(" ")
    # Control how many times you want to generate the dataset, default is 2. See argparser.
    for n in range(n_datasets):
        print("Generating Dataset # {0}".format(n+1)+ "/" + "{0}".format(n_datasets))
        for j in range(len(momentum_bins)-1):
            p_low = momentum_bins[j]
            p_high = momentum_bins[j+1]
            print('Generating with momentum in range ({0},{1}) GeV.'.format(p_low,p_high))
            print(" ")
            generations = []
            mom_idx = np.where((P_ <= p_high) & (P_ > p_low))[0]

            if len(mom_idx) == 0:
                print(" ")
                print('No data in momentum range ({0},{1})'.format(p_low,p_high))
                print(" ")
                continue

            track_params = [data['conds'][l] for l in mom_idx]
            true_hits = np.concatenate([data['Hits'][l] for l in mom_idx])
            photon_yields = data['NHits'][mom_idx]
            bars = barIDs[mom_idx]
            xs = barX[mom_idx]
            kbar = pkbar.Kbar(target=len(photon_yields), width=20, always_stateful=False)
            start = time.time()
            for i in range(len(track_params)):
                k = torch.tensor(track_params[i][:1]).to('cuda').float()
                num_samples = int(photon_yields[i])
                

                with torch.set_grad_enabled(False):
                    #gen = net.probabalistic_sample(pre_compute_dist=3000,context=k,photon_yield=num_samples)
                    gen = net.create_tracks(num_samples=num_samples,context=k)
                    gen['BarID'] = bars[i]
                    gen['X'] = xs[i]
                    gen['PDG'] = PID
                generations.append(gen)

                kbar.update(i)
            print(" ")
            print("Number of photons generated: ",net.photons_generated)
            print("Number of photons resampled: ",net.photons_resampled)
            end = time.time()
            print(" ")
            print("Elapsed time:",end - start)
            print("Time / event:",(end - start)/len(generations))
            file_path = os.path.join(full_path,config['method'] + "_p({0:.2f},{1:.2f})_dataset{2}.pkl".format(p_low,p_high,n))
            with open(file_path,"wb") as file:
                pickle.dump(generations,file)



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
