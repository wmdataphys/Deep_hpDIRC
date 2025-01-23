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
from models.NF.freia_models import FreiaNet
from models.OT_Flow.ot_flow import OT_Flow
from models.FlowMatching.flow_matching import FlowMatching
import matplotlib.colors as mcolors
import pickle
import warnings
from make_plots import make_ratios

warnings.filterwarnings("ignore", message=".*weights_only.*")


def main(config,args):

    # Setup random seed
    print("Using model type: ",str(args.model_type))
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    config['method'] = args.method

    if config['method'] == "Pion":
        print("Generating for pions.")
        dicte = torch.load(config['Inference']['pion_model_path_'+str(args.model_type)])
        PID = 211
    elif config['method'] == 'Kaon':
        print("Generation for kaons.")
        dicte = torch.load(config['Inference']['kaon_model_path_'+str(args.model_type)])
        PID = 321
    else:
        print("Specify particle to generate in config file")
        exit()
        

    num_layers = int(config['model_'+str(args.model_type)]['num_layers'])
    input_shape = int(config['model_'+str(args.model_type)]['input_shape'])
    cond_shape = int(config['model_'+str(args.model_type)]['cond_shape'])
    num_blocks = int(config['model_'+str(args.model_type)]['num_blocks'])
    hidden_nodes = int(config['model_'+str(args.model_type)]['hidden_nodes'])
    stats = config['stats']
    
    if args.model_type == "NF":
        net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats)
    elif args.model_type == "CNF":
        alph = config['model_'+str(args.model_type)]['alph']
        train_T = bool(config['model_'+str(args.model_type)]['train_T'])
        net = net = OT_Flow(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,train_T=True,alph=alph)
    elif args.model_type == "FlowMatching":
        net = FlowMatching(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats)
    else:
        raise ValueError("Model type not found.")

    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    device = torch.device('cuda')
    net.to('cuda')
    net.load_state_dict(dicte['net_state_dict'])
    n_samples = int(config['Inference']['samples'])

    if config['method'] == 'Pion':
        print("Generating Pions over entire phase space.")
        print("Using first {0} pions for comparison.".format(args.n_particles))
        datapoints = np.load(config['dataset']['full_phase_space']["pion_data_path"],allow_pickle=True)[:args.n_particles]

    elif config['method'] == 'Kaon':
        print("Generating Kaons over entire phase space.")
        print("Using first {0} pions for comparison.".format(args.n_particles))
        datapoints = np.load(config['dataset']['full_phase_space']["kaon_data_path"],allow_pickle=True)[:args.n_particles]
            
    else:
        raise ValueError("Method not found.")

    generations = []
    kbar = pkbar.Kbar(target=len(datapoints), width=20, always_stateful=False)
    start = time.time()
    n_photons = 0
    for i in range(len(datapoints)):   
        with torch.set_grad_enabled(False):
            p = (datapoints[i]['P'] - stats['P_max'])  / (stats['P_max'] - stats['P_min'])
            theta = (datapoints[i]['Theta'] - stats['theta_max']) / (stats['theta_max'] - stats['theta_min'])
            k = torch.tensor(np.array([p,theta])).to('cuda').float()
            
            if datapoints[i]['NHits'] > 0:
                gen = net.create_tracks(num_samples=datapoints[i]['NHits'],context=k.unsqueeze(0))
                n_photons += datapoints[i]['NHits']
            else:
                print("Error with number of hits to generate: ",datapoints[i]['NHits'])
                continue
        
        generations.append(gen)
        kbar.update(i)
    end = time.time()

    print(" ")
    print("Number of tracks generated: ",len(generations))
    print("Number of photons generated: ",net.photons_generated)
    print("Number of photons resampled: ",net.photons_resampled)
    print('Percentage effect: ',net.photons_resampled * 100 / net.photons_generated)
    print("Elapsed Time: ", end - start)
    print("Time / photon: ",(end - start) / n_photons)
    print("Average time / track: ",(end - start) / len(datapoints))
    print(" ")
    gen_dict = {}
    gen_dict['fast_sim'] = generations
    gen_dict['truth'] = datapoints

    os.makedirs("Generations",exist_ok=True)
    out_folder = os.path.join("Generations",config['Inference']['full_phase_space_dir'])
    os.makedirs(out_folder,exist_ok=True)
    print("Outputs can be found in " + str(out_folder))

    out_path_ = os.path.join(out_folder,str(config['method'])+f"_ntracks_{len(datapoints)}.pkl")
    
    with open(out_path_,"wb") as file:
        pickle.dump(gen_dict,file)


    print("Making ratio plots.")
    del gen_dict,datapoints 


    make_ratios(path_=out_folder,label=args.method,momentum="1-10",outpath=os.path.join(out_folder,f"Ratios_{args.method}.pdf"))



if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FastSim Generation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-nt', '--n_particles', default=2e5,type=int,help='Number of particles to generate. Take the first n_particles. -1 is all tracks available.')
    parser.add_argument('-m', '--method',default="Kaon",type=str,help='Generated particle type, Kaon or Pion.')
    parser.add_argument('-mt','--model_type',default="NF",type=str,help='Which model to use.')
    args = parser.parse_args()

    args.n_particles = int(args.n_particles)

    config = json.load(open(args.config))

    main(config,args)
