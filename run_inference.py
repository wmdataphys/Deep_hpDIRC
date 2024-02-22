import os
import json
import argparse
import torch
import random
import numpy as np
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataloader.dataloader import CreateLoaders
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from models.freia_models import create_freai
from dataloader.create_data import return_data
from mmd_loss import MMD
import pandas as pd
import pickle
import time

def main(config):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # Create experiment name
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    exp_name = exp_name[:-11]
    print(exp_name)

    # Create directory structure
    output_folder = config['Inference']['out_dir']


    # Load the dataset
    print('Creating Loaders.')
    N_t= config['dataset']['N_t']
    N_v = config['dataset']['N_v']

    data,scaler = return_data(N_t=N_t,N_v=N_v,constrain=True)

    sim_herwig          = data['sim_herwig'][N_t:]
    gen_herwig          = data['gen_herwig'][N_t:]
    scaled_sim_herwig   = data['scaled_sim_herwig'][N_t:]

    gen_features        = data['gen_features']
    sim_features        = data['sim_features']

    train_gen_features = gen_features[:N_t]
    train_sim_features = sim_features[:N_t]

    val_gen_features = gen_features[N_t:]
    val_sim_features = sim_features[N_t:]

    scaled_gen_features = data['scaled_gen_features']
    scaled_sim_features = data['scaled_sim_features']

    train_gen = scaled_gen_features[:N_t]
    train_sim = scaled_sim_features[:N_t]

    val_gen = scaled_gen_features[N_t:]
    val_sim = scaled_sim_features[N_t:]

    print('Herwig: ',gen_herwig.shape)


    train_dataset = TensorDataset(torch.tensor(train_sim),torch.tensor(train_gen))
    val_dataset = TensorDataset(torch.tensor(val_sim),torch.tensor(val_gen))

    train_loader,val_loader = CreateLoaders(train_dataset,val_dataset,config)

    herwig_data = TensorDataset(torch.tensor(scaled_sim_herwig),torch.tensor(gen_herwig))
    herwig_loader = DataLoader(herwig_data,batch_size=config['dataloader']['herwig']['batch_size'],shuffle=False)


     # Load the MNF model
    net = create_freai(6,layers=config['model']['num_layers'])
    net.to('cuda')
    dict = torch.load(config['Inference']['MNF_model'])
    net.load_state_dict(dict['net_state_dict'])





    kbar = pkbar.Kbar(target=len(val_loader),width=20, always_stateful=False)
    # This performs sampling for the OT-Flow
    samples = config['Inference']['samples']
    mus = []
    sigmas = []
    truth = []
    net.eval()
    start = time.time()
    for i,data in enumerate(val_loader):
        reco = data[0]
        gen = data[1]
        truth.append(data[0].numpy())

        temp = []
        for j in range(len(reco)):
            temp.append(np.expand_dims(gen[j],0).repeat(samples,0))

        gen = torch.tensor(np.concatenate(temp)).to('cuda').float()

        with torch.no_grad():
            targets,log_det = net.forward(gen,rev=False)

        targets = scaler.inverse_transform(targets.detach().cpu().numpy())
        targets = targets.reshape(-1,samples,reco.shape[1])
        mus.append(np.mean(targets,axis=1))
        sigmas.append([np.cov(targets[i],rowvar=False) for i in range(len(targets))])

        # if i == 50:
        #     break
        kbar.update(i)
    print(" ")
    print(len(mus))
    end = time.time()
    elapsed_time = end - start
    print('Elapsed Time: ',elapsed_time)

    mus = np.concatenate(mus)
    print('Time / event: ',elapsed_time / len(mus))
    truth = np.concatenate(truth)
    uncertainty = np.concatenate(sigmas)
    obs = {'Events':mus,'Truth':truth,'Covariance':uncertainty}
    save_path = os.path.join(config['Inference']['out_dir'],config['Inference']['out_file'])
    with open(save_path, 'wb') as f:
        pickle.dump(obs,f)



if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config)
