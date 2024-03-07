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
from models.nflows_models import create_nflows
from dataloader.create_data import create_dataset,unscale
from datetime import datetime
import itertools
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LogNorm

def make_plot(generations,hits,stats,barID,x_high,x_low):
    fig,ax = plt.subplots(2,2,figsize=(18,8))
    ax = ax.ravel()

    x_true = unscale(hits[:,0],stats['x_max'],stats['x_min'])
    y_true = unscale(hits[:,1],stats['y_max'],stats['y_min'])
    t_true = unscale(hits[:,2],stats['time_max'],stats['time_min'])

    ax[0].hist2d(x_true,y_true,density=True,range=[(0,892),(0,268)],bins=(144,48))
    ax[0].set_xlabel(r'X $(mm)$',fontsize=20)
    ax[0].set_ylabel(r'Y $(mm)$',fontsize=20)
    ax[0].set_title(r'Kaons: $x \in ({0},{1})$, BarID: {2}'.format(x_low,x_high,barID),fontsize=20)
    ax[0].text(s=r"Truth",x=10.,y=0.0,color='White',fontsize=25)
    ax[2].hist(t_true,density=True,color='blue',label='Truth',bins=100,range=[0,200])
    ax[2].set_title("True Hit Time",fontsize=20)
    ax[2].set_xlabel("Hit Time (ns)",fontsize=20)
    ax[2].set_ylabel("Density",fontsize=20)
    x = unscale(generations[:,:,0].flatten(),stats['x_max'],stats['x_min'])
    y = unscale(generations[:,:,1].flatten(),stats['y_max'],stats['y_min'])
    t = unscale(generations[:,:,2].flatten(),stats['time_max'],stats['time_min'])


    ax[1].hist2d(x,y,density=True,range=[(0,892),(0,268)],bins=(144,48))
    ax[1].set_xlabel(r'X $(mm)$',fontsize=20)
    ax[1].set_ylabel(r'Y $(mm)$',fontsize=20)
    ax[1].set_title(r'Kaons: $x \in ({0},{1})$, BarID: {2}'.format(x_low,x_high,barID),fontsize=20)
    ax[1].text(s=r"Generated $\times 100$",x=10.,y=0.0,color='White',fontsize=25)
    ax[3].hist(t,density=True,color='blue',label='Truth',bins=100,range=[0,200])
    ax[3].set_title("Generated Hit Time",fontsize=20)
    ax[3].set_xlabel("Hit Time (ns)",fontsize=20)
    ax[3].set_ylabel("Density",fontsize=20)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig("TEST/Kaons_BarID{0}_x({1},{2}).pdf".format(barID,x_low,x_high),bbox_inches="tight")

def main(config,resume):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    print("Running inference")


       # Load the dataset
    print('Creating Loaders.')
    file_paths = [config['dataset']['training']['kaon_data_path'],
                  config['dataset']['validation']['kaon_data_path'],
                  config['dataset']['testing']['gen']['kaon_data_path']]
    hits,conds,unscaled_conds,metadata = create_dataset(file_paths)

    print(" ")
    print("Hit Statistics: ")
    print("Max: ",hits.max(0))
    print("Min: ",hits.min(0))
    print(" ")
    print("Conditional Statistics: ")
    print("Max: ",conds.max(0))
    print("Min: ",conds.min(0))
    print(" ")

    # Create the model
    # This will map gen -> Reco
    num_layers = int(config['model']['num_layers'])
    input_shape = int(config['model']['input_shape'])
    cond_shape = int(config['model']['cond_shape'])
    net = create_nflows(input_shape,cond_shape,num_layers)
    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    device = torch.device('cuda')
    net.to('cuda')
    dicte = torch.load(config['Inference']['kaon_model_path'])
    net.load_state_dict(dicte['net_state_dict'])
    n_samples = int(config['Inference']['samples'])

    xs = [(-30,-20),(-20,-10),(-10,0),(0,10),(10,20),(20,30)]
    #bars = [0,1,2,3,4,5,6,24,25,26,27,28,29,30,31]
    bars = [31,0]
    stats={"x_max": 895,"x_min":3,"y_max":295,"y_min":3,"time_max":500.00,"time_min":0.0}
    combinations = list(itertools.product(xs,bars))
    print('Generating PDFs for {0} combinations of BarID and x ranges.'.format(len(combinations)))


    for j,combination in enumerate(combinations):
        x_low = combination[0][0]
        x_high = combination[0][1]
        barID = combination[1]
        print('Generating Bar {0}, x ({1},{2})'.format(barID,x_low,x_high))
        print(" ")
        generations = []
        mom_idx = np.where((metadata[:,0] == barID) & (metadata[:,1] > x_low) & (metadata[:,1] < x_high))[0]
        kin_dataset = TensorDataset(torch.tensor(hits[mom_idx]),torch.tensor(conds[mom_idx]),torch.tensor(unscaled_conds[mom_idx]))
        kin_loader = DataLoader(kin_dataset,batch_size=1000)
        kbar = pkbar.Kbar(target=len(kin_loader), width=20, always_stateful=False)
        start = time.time()
        for i, data in enumerate(kin_loader):
            input  = data[0].to('cuda').float()
            k = data[1].to('cuda').float()

            with torch.set_grad_enabled(False):
                gen = net._sample(num_samples=n_samples,context=k)

            generations.append(gen.detach().cpu().numpy())

            kbar.update(i)
        end = time.time()
        generations = np.concatenate(generations)
        print(" ")
        print(len(generations))
        print("Elapsed time:",end - start)
        print("Time / event:",(end - start)/len(generations))

        make_plot(generations,hits[mom_idx],stats,barID,x_high,x_low)
        print(" ")






if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FastSim Generation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args.resume)
