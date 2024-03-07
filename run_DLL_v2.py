import os
import json
import argparse
import torch
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from dataloader.dataloader import CreateInferenceLoader
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
from dataloader.create_data import DLL_Dataset


def run_inference_seperate(pions,kaons,pion_net,kaon_net):
    LL_Pion = []
    LL_Kaon = []
    print('Starting DLL for Pions.')
    start = time.time()
    kbar = pkbar.Kbar(target=len(pions), width=20, always_stateful=False)
    for i,data in enumerate(pions):
        h = data[0]
        n_hits = data[3].numpy()
        c = data[1]

        h = h[:,:n_hits].to('cuda').float().squeeze(0)
        c = c[:,:n_hits].to('cuda').float().squeeze(0)

        PID = data[2].numpy()
        unsc = data[4].numpy()
        unsc = unsc[:,:n_hits][0][0]

        with torch.set_grad_enabled(False):
            pion_hyp_pion = pion_net.log_prob(h,context=c).sum().detach().cpu().numpy()
            pion_hyp_kaon = kaon_net.log_prob(h,context=c).sum().detach().cpu().numpy()

        LL_Pion.append({"hyp_pion":pion_hyp_pion,"hyp_kaon":pion_hyp_kaon,"Truth":PID,"Kins":unsc})

        kbar.update(i)
        #if i == 5000:
        #    break

    end = time.time()
    print(" ")
    print("Elapsed time: ",end - start)
    print('Time / event: ',(end - start) / len(LL_Pion))

    print(' ')
    print('Starting DLL for Kaons')
    start = time.time()
    kbar = pkbar.Kbar(target=len(kaons), width=20, always_stateful=False)
    for i,data in enumerate(kaons):
        h = data[0]
        n_hits = data[3].numpy()
        c = data[1]

        h = h[:,:n_hits].to('cuda').float().squeeze(0)
        c = c[:,:n_hits].to('cuda').float().squeeze(0)

        PID = data[2].numpy()
        unsc = data[4].numpy()
        unsc = unsc[:,:n_hits][0][0]

        with torch.set_grad_enabled(False):
            kaon_hyp_kaon = kaon_net.log_prob(h,context=c).sum().detach().cpu().numpy()
            kaon_hyp_pion = pion_net.log_prob(h,context=c).sum().detach().cpu().numpy()

        LL_Kaon.append({"hyp_kaon":kaon_hyp_kaon,"hyp_pion":kaon_hyp_pion,"Truth":PID,"Kins":unsc})

        kbar.update(i)
        #if i == 5000:
        #    break

    end = time.time()
    print(" ")
    print("Elapsed time: ",end - start)
    print('Time / event: ',(end - start) / len(LL_Pion))

    return LL_Pion,LL_Kaon


def main(config,resume):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    print("Running inference")

    num_layers = int(config['model']['num_layers'])
    input_shape = int(config['model']['input_shape'])
    cond_shape = int(config['model']['cond_shape'])

    assert config["method"] in ["Combined","Pion","Kaon"]

    test_pions = DLL_Dataset(file_path=config['dataset']['testing']['DLL']['pion_data_path'])
    test_kaons = DLL_Dataset(file_path=config['dataset']['testing']['DLL']['kaon_data_path'])

    pions = CreateInferenceLoader(test_pions) # Batch size is 1 untill I figure out a better way
    kaons = CreateInferenceLoader(test_kaons)

    if config['method'] != "Combined":
        pion_net = create_nflows(input_shape,cond_shape,num_layers)
        device = torch.device('cuda')
        pion_net.to('cuda')
        dicte = torch.load(config['Inference']['pion_model_path'])
        pion_net.load_state_dict(dicte['net_state_dict'])

        kaon_net = create_nflows(input_shape,cond_shape,num_layers)
        device = torch.device('cuda')
        kaon_net.to('cuda')
        dicte = torch.load(config['Inference']['kaon_model_path'])
        kaon_net.load_state_dict(dicte['net_state_dict'])

        LL_Pion,LL_Kaon = run_inference_seperate(pions,kaons,pion_net,kaon_net)
        with open("Pion_DLL_Results.pkl","wb") as file:
            pickle.dump(LL_Pion,file)
        with open("Kaon_DLL_Results.pkl","wb") as file:
            pickle.dump(LL_Kaon,file)
    else:
        print("WIP. Functions not defined. Exiting.")
        exit()
        net = create_nflows(input_shape,cond_shape,num_layers)
        device = torch.device('cuda')
        net.to('cuda')
        dicte = torch.load(config['Inference']['kaon_model_path'])
        net.load_state_dict(dicte['net_state_dict'])




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
