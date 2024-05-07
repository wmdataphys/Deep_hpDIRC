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

def make_plot(generations,hits,stats,barID,x_high,x_low,method=None,samples=1,folder=None):
    # Matches physical sensor
    bins_x = np.array([0,  3.,   9.,  15.,  21.,  27.,  33.,  39.,  45.,48,50, 53.,  59.,  65.,
        71.,  77.,  83.,  89.,  95., 98,100, 103., 109., 115., 121., 127., 133.,
       139., 145.,148,150, 153., 159., 165., 171., 177., 183., 189., 195.,198,200, 203.,
       209., 215., 221., 227., 233., 239., 245.,248,250, 253., 259., 265., 271.,
       277., 283., 289., 295.,298,300, 303., 309., 315., 321., 327., 333., 339.,
       345.,348,350, 353., 359., 365., 371., 377., 383., 389., 395.,398,400, 403., 409.,
       415., 421., 427., 433., 439., 445.,448,450, 453., 459., 465., 471., 477.,
       483., 489., 495.,498,500, 503., 509., 515., 521., 527., 533., 539., 545.,548,550,
       553., 559., 565., 571., 577., 583., 589., 595.,598,600, 603., 609., 615.,
       621., 627., 633., 639., 645.,648,650, 653., 659., 665., 671., 677., 683.,
       689., 695.,698,700, 703., 709., 715., 721., 727., 733., 739., 745.,748,750, 753.,
       759., 765., 771., 777., 783., 789., 795.,798,800, 803., 809., 815., 821.,
       827., 833., 839., 845.848,850, 853., 859., 865., 871., 877., 883., 889.,
       895.,898,900])
    
    bins_y = np.array([0,  3.,   9.,  15.,  21.,  27.,  33.,  39.,  45.,48,50, 53.,  59.,  65.,
        71.,  77.,  83.,  89.,  95.,98,100, 103., 109., 115., 121., 127., 133.,
       139., 145.,148,150, 153., 159., 165., 171., 177., 183., 189., 195.,198,200, 203.,
       209., 215., 221., 227., 233., 239., 245.,248,250, 253., 259., 265., 271.,
       277., 283., 289., 295.,298,300])
    
    #x_spaces = [45,95,145,195,245,295,345,395,445,495,545,595,645,695,745,795]
    #y_spaces = [45,95,145,195,245]

    fig,ax = plt.subplots(4,2,figsize=(18,16))
    ax = ax.ravel()
    # We have a 2mm offset.
    x_true = hits[:,0] - 2
    y_true = hits[:,1]- 2
    t_true = hits[:,2]

    ax[0].hist2d(x_true,y_true,density=True,bins=[bins_x,bins_y],norm=mcolors.LogNorm())
    ax[0].set_xlabel(r'X $(mm)$',fontsize=20)
    ax[0].set_ylabel(r'Y $(mm)$',fontsize=20)
    # Time PDF
    ax[2].hist(t_true,density=True,color='blue',label='Truth',bins=100,range=[0,200])
    ax[2].set_title("True Hit Time",fontsize=20)
    ax[2].set_xlabel("Hit Time (ns)",fontsize=20)
    ax[2].set_ylabel("Density",fontsize=20)
    # X PDF
    ax[4].hist(x_true,density=True,color='blue',label='Truth',bins=100,range=[0,895])
    ax[4].set_title("True X Distribution",fontsize=20)
    ax[4].set_xlabel("X (mm)",fontsize=20)
    ax[4].set_ylabel("Density",fontsize=20)
    # Y PDF
    ax[6].hist(y_true,density=True,color='blue',label='Truth',bins=100,range=[0,295])
    ax[6].set_title("True Y Distribution",fontsize=20)
    ax[6].set_xlabel("Y (mm)",fontsize=20)
    ax[6].set_ylabel("Density",fontsize=20)

    # Also a 2mm offset it seems. Probably needs correction only after so many PMTs
    x = generations[:,0].flatten() - 2
    y = generations[:,1].flatten() - 2
    t = generations[:,2].flatten() 
    ax[1].hist2d(x,y,density=True,bins=[bins_x,bins_y],norm=mcolors.LogNorm())
    ax[1].set_xlabel(r'X $(mm)$',fontsize=20)
    ax[1].set_ylabel(r'Y $(mm)$',fontsize=20)
    # Time PDF
    ax[3].hist(t,density=True,color='blue',label='Truth',bins=100,range=[0,200])
    ax[3].set_title("Generated Hit Time",fontsize=20)
    ax[3].set_xlabel("Hit Time (ns)",fontsize=20)
    ax[3].set_ylabel("Density",fontsize=20)
    # X PDF
    ax[5].hist(x,density=True,color='blue',label='Truth',bins=100,range=[0,895])
    ax[5].set_title("Generated X Distribution",fontsize=20)
    ax[5].set_xlabel("X (mm)",fontsize=20)
    ax[5].set_ylabel("Density",fontsize=20)
    # Y PDF
    ax[7].hist(y,density=True,color='blue',label='Truth',bins=100,range=[0,295])
    ax[7].set_title("Generated Y Distribution",fontsize=20)
    ax[7].set_xlabel("Y (mm)",fontsize=20)
    ax[7].set_ylabel("Density",fontsize=20)

    plt.subplots_adjust(hspace=0.5)
    if method == "Pion":
        path_ = os.path.join(folder,"Pions_BarID{0}_x({1},{2}).pdf".format(barID,x_low,x_high))
        ax[1].set_title(r'Generated ($\times {3}$) Pions: $x \in ({0},{1})$, BarID: {2}'.format(x_low,x_high,barID,samples),fontsize=20)
        ax[0].set_title(r'Pions: $x \in ({0},{1})$, BarID: {2}'.format(x_low,x_high,barID),fontsize=20)
        plt.savefig(path_,bbox_inches="tight")
    elif method == "Kaon":
        path_ = os.path.join(folder,"Kaons_BarID{0}_x({1},{2}).pdf".format(barID,x_low,x_high))
        ax[1].set_title(r'Generated ($\times {3}$) Kaons: $x \in ({0},{1})$, BarID: {2}'.format(x_low,x_high,barID,samples),fontsize=20)
        ax[0].set_title(r'Kaons: $x \in ({0},{1})$, BarID: {2}'.format(x_low,x_high,barID),fontsize=20)
        plt.savefig(path_,bbox_inches="tight")
    
    plt.close(fig)

def main(config,n_datasets):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    print("Running inference")

    base_folder = config['Inference']['gen_dir'].split("/")[0]
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)

    if not os.path.exists(config['Inference']['gen_dir']):
        print('Generations can be found in: ' + config['Inference']['gen_dir'])
        os.mkdir(config['Inference']['gen_dir'])

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
            true_hits = net._apply_mask(torch.tensor(true_hits)).numpy()
            photon_yields = data['NHits'][mom_idx]
        

            kbar = pkbar.Kbar(target=len(photon_yields), width=20, always_stateful=False)
            start = time.time()
            for i in range(len(track_params)):
                k = torch.tensor(track_params[i][:1]).to('cuda').float()
                num_samples = int(photon_yields[i])
            
                with torch.set_grad_enabled(False):
                    gen = net.create_tracks(num_samples=num_samples,context=k,plotting=True)

                generations.append(gen)

                kbar.update(i)
            end = time.time()
            generations = np.concatenate(generations)
            print(" ")
            print("Elapsed time:",end - start)
            print("Time / photon:",(end - start)/len(generations))
            print(" ")
            print("Number of photons generated: ",net.photons_generated)
            print("Number of photons resampled: ",net.photons_resampled)
            print("Effect: ",(net.photons_resampled/net.photons_generated) * 100, "%")
            folder = os.path.join(config['Inference']['gen_dir'],"BarID_{0}".format(barID))
            if not os.path.exists(folder):
                os.mkdir(folder)

            make_plot(generations,true_hits,stats,barID,x_high,x_low,config['method'],n_samples,folder)
            print(" ")
     
     



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
