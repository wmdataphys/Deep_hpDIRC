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
from models.freia_models import FreiaNet
import matplotlib.colors as mcolors


def make_plot(generations,hits,stats,barID,x_high,x_low,method=None):
    bins_x = np.array([  3.,   9.,  15.,  21.,  27.,  33.,  39.,  45.,  53.,  59.,  65.,
        71.,  77.,  83.,  89.,  95., 103., 109., 115., 121., 127., 133.,
       139., 145., 153., 159., 165., 171., 177., 183., 189., 195., 203.,
       209., 215., 221., 227., 233., 239., 245., 253., 259., 265., 271.,
       277., 283., 289., 295., 303., 309., 315., 321., 327., 333., 339.,
       345., 353., 359., 365., 371., 377., 383., 389., 395., 403., 409.,
       415., 421., 427., 433., 439., 445., 453., 459., 465., 471., 477.,
       483., 489., 495., 503., 509., 515., 521., 527., 533., 539., 545.,
       553., 559., 565., 571., 577., 583., 589., 595., 603., 609., 615.,
       621., 627., 633., 639., 645., 653., 659., 665., 671., 677., 683.,
       689., 695., 703., 709., 715., 721., 727., 733., 739., 745., 753.,
       759., 765., 771., 777., 783., 789., 795., 803., 809., 815., 821.,
       827., 833., 839., 845., 853., 859., 865., 871., 877., 883., 889.,
       895.])
    
    bins_y = np.array([  3.,   9.,  15.,  21.,  27.,  33.,  39.,  45.,  53.,  59.,  65.,
        71.,  77.,  83.,  89.,  95., 103., 109., 115., 121., 127., 133.,
       139., 145., 153., 159., 165., 171., 177., 183., 189., 195., 203.,
       209., 215., 221., 227., 233., 239., 245., 253., 259., 265., 271.,
       277., 283., 289., 295.])

    fig,ax = plt.subplots(4,2,figsize=(18,16))
    ax = ax.ravel()

    x_true = unscale(hits[:,0],stats['x_max'],stats['x_min'])
    y_true = unscale(hits[:,1],stats['y_max'],stats['y_min'])
    t_true = unscale(hits[:,2],stats['time_max'],stats['time_min'])

    ax[0].hist2d(x_true,y_true,density=True,bins=[bins_x,bins_y],norm=mcolors.LogNorm())
    ax[0].set_xlabel(r'X $(mm)$',fontsize=20)
    ax[0].set_ylabel(r'Y $(mm)$',fontsize=20)
    ax[0].text(s=r"Truth",x=10.,y=0.0,color='White',fontsize=25)
    # Time PDF
    ax[2].hist(t_true,density=True,color='blue',label='Truth',bins=100,range=[0,200])
    ax[2].set_title("True Hit Time",fontsize=20)
    ax[2].set_xlabel("Hit Time (ns)",fontsize=20)
    ax[2].set_ylabel("Density",fontsize=20)
    # X PDF
    ax[4].hist(x_true,density=True,color='blue',label='Truth',bins=100)
    ax[4].set_title("True X Distribution",fontsize=20)
    ax[4].set_xlabel("X (mm)",fontsize=20)
    ax[4].set_ylabel("Density",fontsize=20)
    # Y PDF
    ax[6].hist(y_true,density=True,color='blue',label='Truth',bins=100)
    ax[6].set_title("True Y Distribution",fontsize=20)
    ax[6].set_xlabel("Y (mm)",fontsize=20)
    ax[6].set_ylabel("Density",fontsize=20)


    x = generations[:,0].flatten()
    y = generations[:,1].flatten()
    t = generations[:,2].flatten()
    ax[1].hist2d(x,y,density=True,bins=[bins_x,bins_y],norm=mcolors.LogNorm())
    ax[1].set_xlabel(r'X $(mm)$',fontsize=20)
    ax[1].set_ylabel(r'Y $(mm)$',fontsize=20)
    ax[1].text(s=r"Generated $\times 100$",x=10.,y=0.0,color='White',fontsize=25)
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
        ax[1].set_title(r'Pions: $x \in ({0},{1})$, BarID: {2}'.format(x_low,x_high,barID),fontsize=20)
        ax[0].set_title(r'Pions: $x \in ({0},{1})$, BarID: {2}'.format(x_low,x_high,barID),fontsize=20)
        plt.savefig("Figures/Affine/Pions_BarID{0}_x({1},{2}).pdf".format(barID,x_low,x_high),bbox_inches="tight")
    elif method == "Kaon":
        ax[1].set_title(r'Kaons: $x \in ({0},{1})$, BarID: {2}'.format(x_low,x_high,barID),fontsize=20)
        ax[0].set_title(r'Kaons: $x \in ({0},{1})$, BarID: {2}'.format(x_low,x_high,barID),fontsize=20)
        plt.savefig("Figures/Affine/Kaons_BarID{0}_x({1},{2}).pdf".format(barID,x_low,x_high),bbox_inches="tight")


def main(config,resume):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    print("Running inference")


       # Load the dataset
    print('Creating Loaders.')
    if config['method'] == "Pion":
        print("Generating for pions.")
        file_paths = [config['dataset']['training']['pion_data_path'],
                      config['dataset']['validation']['pion_data_path'],
                      config['dataset']['testing']['gen']['pion_data_path']]
        dicte = torch.load(config['Inference']['pion_model_path'])

    elif config['method'] == 'Kaon':
        print("Generation for kaons.")
        file_paths = [config['dataset']['training']['kaon_data_path'],
                      config['dataset']['validation']['kaon_data_path'],
                      config['dataset']['testing']['gen']['kaon_data_path']]
        dicte = torch.load(config['Inference']['kaon_model_path'])
    else:
        print("Specify particle to generate in config file")
        exit()

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
    if config['method'] == 'Pion':
        num_layers = int(config['model']['num_layers'])
    elif config['method'] == 'Kaon':
        num_layers = int(config['model']['num_layers'])
    else:
        num_layers = int(config['model']['num_layers'])

    num_layers = int(config['model']['num_layers'])
    input_shape = int(config['model']['input_shape'])
    cond_shape = int(config['model']['cond_shape'])
    net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False)
    #net = create_nflows(input_shape,cond_shape,num_layers)
    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    device = torch.device('cuda')
    net.to('cuda')
    net.load_state_dict(dicte['net_state_dict'])
    n_samples = int(config['Inference']['samples'])


    # Control what you want to generate pair wise here:
    xs = [(-30,-20),(-20,-10),(-10,0),(0,10),(10,20),(20,30)]
    #bars = [0,1,2,3,4,5,6,24,25,26,27,28,29,30,31]
    bars = [31,0]
    stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":500.00,"time_min":0.0}
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
        kin_loader = DataLoader(kin_dataset,batch_size=1000,shuffle=False)
        kbar = pkbar.Kbar(target=len(kin_loader), width=20, always_stateful=False)
        start = time.time()
        for i, data in enumerate(kin_loader):
            input  = data[0].to('cuda').float()
            k = data[1].to('cuda').float()

            with torch.set_grad_enabled(False):
                gen = net._sample(num_samples=n_samples,context=k)

            generations.append(gen)

            kbar.update(i)
        end = time.time()
        generations = np.concatenate(generations)
        print(" ")
        print(generations.shape)
        print("Elapsed time:",end - start)
        print("Time / event:",(end - start)/len(generations))

        make_plot(generations,hits[mom_idx],stats,barID,x_high,x_low,config['method'])
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
