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
    log_time = bool(config['log_time'])
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
    net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,log_time=log_time,stats=stats)
    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    device = torch.device('cuda')
    net.to('cuda')
    net.load_state_dict(dicte['net_state_dict'])
    n_samples = int(config['Inference']['samples'])


    # Control what you want to generate pair wise here:
    xs = [(-30,-20),(-20,-10),(-10,0),(0,10),(10,20),(20,30)]
    bars = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

    stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":500.00,"time_min":0.0}
    combinations = list(itertools.product(xs,bars))
    print('Generating chunked dataset for {0} combinations of BarID and x ranges.'.format(len(combinations)))
    barIDs = np.array(data['BarID'])
    barX = np.array(data['X'])

    # Control how many times you want to generate the dataset, default is 2. See argparser.
    for n in range(n_datasets):
        print("Generating Dataset # {0}".format(n+1)+ "/" + "{0}".format(n_datasets))
        for j,combination in enumerate(combinations):
            x_low = combination[0][0]
            x_high = combination[0][1]
            barID = combination[1]
            print('Generating Bar {0}, x ({1},{2})'.format(barID,x_low,x_high))
            print(" ")
            generations = []
            mom_idx = np.where((barIDs == barID) & (barX > x_low) & (barX < x_high))[0]

            if len(mom_idx) == 0:
                print(" ")
                print('No data at Bar {0}, x ({1},{2})'.format(barID,x_low,x_high))
                print(" ")
                continue

            track_params = [data['conds'][l] for l in mom_idx]
            true_hits = np.concatenate([data['Hits'][l] for l in mom_idx])
            photon_yields = data['NHits'][mom_idx]

            kbar = pkbar.Kbar(target=len(photon_yields), width=20, always_stateful=False)
            start = time.time()
            for i in range(len(track_params)):
                k = torch.tensor(track_params[i][:1]).to('cuda').float()
                num_samples = int(photon_yields[i])

                with torch.set_grad_enabled(False):
                    #gen = net.probabalistic_sample(pre_compute_dist=3000,context=k,photon_yield=num_samples)
                    gen = net.create_tracks(num_samples=num_samples,context=k)

                generations.append(gen)

                kbar.update(i)
            print(" ")
            print("Number of photons generated: ",net.photons_generated)
            print("Number of photons resampled: ",net.photons_resampled)
            end = time.time()
            #generations = np.concatenate(generations)
            #np.save(generations,'test.npy')
            print(" ")
            #print(generations.shape)
            print("Elapsed time:",end - start)
            print("Time / event:",(end - start)/len(generations))
            file_path = os.path.join(full_path,config['method'] + "_BardID_{0}_x({1},{2})_dataset{3}.pkl".format(barID,x_low,x_high,n))
            with open(file_path,"wb") as file:
                pickle.dump(generations,file)



if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FastSim Generation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-n', '--n_datasets', default=2, type=int,
                        help='Number of time to Fast Simulate the dataset.')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args.n_datasets)
