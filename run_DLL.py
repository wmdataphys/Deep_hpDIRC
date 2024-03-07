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
from dataloader.create_data import create_dataset,unscale,scale_data
from datetime import datetime
import itertools
import matplotlib.pyplot as plt
import time
import pickle
from dataloader.create_data import DIRC_Dataset
from support.utils import DIRCProbabilitySpread

def make_plot(generations,hits,stats,barID,x_high,x_low):
    fig,ax = plt.subplots(1,2,figsize=(18,4))
    ax = ax.ravel()

    x_true = unscale(hits[:,0],stats['x_max'],stats['x_min'])
    y_true = unscale(hits[:,1],stats['y_max'],stats['y_min'])
    t_true = unscale(hits[:,2],stats['time_max'],stats['time_min'])

    ax[0].hist2d(x_true,y_true,density=True,range=[(0,892),(0,268)],bins=(144,48))
    ax[0].set_xlabel(r'X $(mm)$',fontsize=20)
    ax[0].set_ylabel(r'Y $(mm)$',fontsize=20)
    ax[0].set_title(r'Kaons: $x \in ({0},{1})$, BarID: {2}'.format(x_low,x_high,barID),fontsize=20)
    ax[0].text(s=r"Truth",x=10.,y=0.0,color='White',fontsize=25)

    x = unscale(generations[:,:,0].flatten(),stats['x_max'],stats['x_min'])
    y = unscale(generations[:,:,1].flatten(),stats['y_max'],stats['y_min'])
    t = unscale(generations[:,:,2].flatten(),stats['time_max'],stats['time_min'])

    ax[1].hist2d(x,y,density=True,range=[(0,892),(0,268)],bins=(144,48))
    ax[1].set_xlabel(r'X $(mm)$',fontsize=20)
    ax[1].set_ylabel(r'Y $(mm)$',fontsize=20)
    ax[1].set_title(r'Kaons: $x \in ({0},{1})$, BarID: {2}'.format(x_low,x_high,barID),fontsize=20)
    ax[1].text(s=r"Generated $\times 100$",x=10.,y=0.0,color='White',fontsize=25)
    plt.savefig("Figures_v3/Kaons_BarID{0}_x({1},{2}).pdf".format(barID,x_low,x_high),bbox_inches="tight")

def main(config,resume):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    print("Running inference")

    pion_hits,pion_conds,pion_unscaled_conds,pion_metadata = create_dataset(config['dataset']['pion_data_path'])
    kaon_hits,kaon_conds,kaon_unscaled_conds,kaon_metadata = create_dataset(config['dataset']['kaon_data_path'])
    pion_stats = {"max":pion_unscaled_conds.max(0),"min":pion_unscaled_conds.min(0)}
    kaon_stats = {"max":kaon_unscaled_conds.max(0),"min":kaon_unscaled_conds.min(0)}

    print(pion_stats)
    print(kaon_stats)

    #DIRC_DLL = DIRCProbabilitySpread()
    # Create the model
    # This will map gen -> Reco
    num_layers = config['model']['num_layers']
    pion_net = create_nflows(pion_hits.shape[1],pion_conds.shape[1],8)
    device = torch.device('cuda')
    pion_net.to('cuda')
    dicte = torch.load(config['Inference']['pion_model_path'])
    pion_net.load_state_dict(dicte['net_state_dict'])

    kaon_net = create_nflows(kaon_hits.shape[1],kaon_conds.shape[1],8)
    device = torch.device('cuda')
    kaon_net.to('cuda')
    dicte = torch.load(config['Inference']['kaon_model_path'])
    kaon_net.load_state_dict(dicte['net_state_dict'])

    validation_pions = np.load(r"C:\Users\James-PC\James\EIC\Cherenkov_FastSim\Validation_Pions_Centered_NoSmear_Cut.pkl",allow_pickle=True)
    validation_kaons = np.load(r"C:\Users\James-PC\James\EIC\Cherenkov_FastSim\Validation_Kaons_Centered_NoSmear.pkl",allow_pickle=True)

    pion_as_pion = DIRC_Dataset(validation_pions,stats=pion_stats,method='Pion')
    kaon_as_kaon = DIRC_Dataset(validation_kaons,stats=kaon_stats,method='Kaon')
    kaon_as_pion = DIRC_Dataset(validation_kaons,stats=pion_stats,method='Pion')
    pion_as_kaon = DIRC_Dataset(validation_pions,stats=kaon_stats,method='Kaon')

    pion_as_pion = CreateLoaders(pion_as_pion,config)
    kaon_as_kaon = CreateLoaders(kaon_as_kaon,config)
    kaon_as_pion = CreateLoaders(kaon_as_pion,config)
    pion_as_kaon = CreateLoaders(pion_as_kaon,config)

    LL_Pion = []
    LL_Kaon = []
    del pion_hits,pion_conds,pion_metadata
    del kaon_hits,kaon_conds,kaon_metadata

    print('Starting DLL for Pions.')
    kbar = pkbar.Kbar(target=len(pion_as_pion), width=20, always_stateful=False)
    for i,(p_as_p,p_as_k) in enumerate(zip(pion_as_pion,pion_as_kaon)):
        h = p_as_p[0]
        n_hits = p_as_p[3].numpy()[0]
        c = p_as_p[1]

        h = h[:,:n_hits].to('cuda').float().squeeze(0)
        c = c[:,:n_hits].to('cuda').float().squeeze(0)

        PID = p_as_p[2].numpy()
        unsc = p_as_p[4].numpy()
        unsc = unsc[:,:n_hits][0][0]

        with torch.set_grad_enabled(False):
            pion_hyp_pion = pion_net.log_prob(h,context=c).sum().detach().cpu().numpy()

        h = p_as_k[0]
        n_hits = p_as_k[3].numpy()[0]
        c = p_as_k[1]


        h = h[:,:n_hits].to('cuda').float().squeeze(0)
        c = c[:,:n_hits].to('cuda').float().squeeze(0)

        with torch.set_grad_enabled(False):
            pion_hyp_kaon = kaon_net.log_prob(h,context=c).sum().detach().cpu().numpy()

            LL_Pion.append({"hyp_pion":pion_hyp_pion,"hyp_kaon":pion_hyp_kaon,"Truth":PID,"Kins":unsc})

        kbar.update(i)
        # if i == 1000:
        #     break

    print(' ')
    print('Starting DLL for Kaons')

    kbar = pkbar.Kbar(target=len(kaon_as_kaon), width=20, always_stateful=False)
    for i,(k_as_k,k_as_p) in enumerate(zip(kaon_as_kaon,kaon_as_pion)):
        h = k_as_k[0]
        n_hits = k_as_k[3].numpy()[0]
        c = k_as_k[1]

        h = h[:,:n_hits].to('cuda').float().squeeze(0)
        c = c[:,:n_hits].to('cuda').float().squeeze(0)

        PID = k_as_k[2].numpy()
        unsc = k_as_k[4].numpy()
        unsc = unsc[:,:n_hits][0][0]

        with torch.set_grad_enabled(False):
            kaon_hyp_kaon = kaon_net.log_prob(h,context=c).sum().detach().cpu().numpy()

        h = k_as_p[0]
        n_hits = k_as_p[3].numpy()[0]
        c = k_as_p[1]


        h = h[:,:n_hits].to('cuda').float().squeeze(0)
        c = c[:,:n_hits].to('cuda').float().squeeze(0)

        with torch.set_grad_enabled(False):
            kaon_hyp_pion = pion_net.log_prob(h,context=c).sum().detach().cpu().numpy()

            LL_Kaon.append({"hyp_kaon":kaon_hyp_kaon,"hyp_pion":kaon_hyp_pion,"Truth":PID,"Kins":unsc})

        kbar.update(i)
        # if i == 1000:
        #     break






    end = time.time()
    with open("Pion_DLL_Results.pkl","wb") as file:
        pickle.dump(LL_Pion,file)
    with open("Kaon_DLL_Results.pkl","wb") as file:
        pickle.dump(LL_Kaon,file)

    # print(" ")
    # print("Elapsed time:",end - start)
    # print("Time / event:",(end - start)/len(LL_kaon))

    #make_plot(generations,hits[mom_idx],stats,barID,x_high,x_low)
    print(" ")






if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FastSim Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args.resume)
