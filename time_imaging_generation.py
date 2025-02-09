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

warnings.filterwarnings("ignore", message=".*weights_only.*")

def convert_numpy(obj):
    """Recursively convert NumPy objects to JSON-compatible types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array to list
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)  # Convert NumPy float to standard float
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)  # Convert NumPy int to standard int
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}  # Convert dict recursively
    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]  # Convert list recursively
    return obj  # Return as is if not a NumPy type

def main(config,args):

    # Setup random seed
    print("Using model type: ",str(args.model_type))
    # Remove seeding, make it random.
    seed = int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device('cuda')

    num_layers = int(config['model_'+str(args.model_type)]['num_layers'])
    input_shape = int(config['model_'+str(args.model_type)]['input_shape'])
    cond_shape = int(config['model_'+str(args.model_type)]['cond_shape'])
    num_blocks = int(config['model_'+str(args.model_type)]['num_blocks'])
    hidden_nodes = int(config['model_'+str(args.model_type)]['hidden_nodes'])
    stats = config['stats']

    if not args.sample_photons:
        sampler_path = None

    if args.model_type == "NF":
        dicte = torch.load(config['Inference']['pion_model_path_'+str(args.model_type)])
        if args.sample_photons:
            sampler_path = config['Photon_Sampler']['Pion_LUT_path']
        pion_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats,LUT_path=sampler_path)
        pion_net.to('cuda')
        pion_net.load_state_dict(dicte['net_state_dict'])
        print("Loaded pion network.")

        dicte = torch.load(config['Inference']['kaon_model_path_'+str(args.model_type)])
        if args.sample_photons:
            sampler_path = config['Photon_Sampler']['Kaon_LUT_path']
        kaon_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats,LUT_path=sampler_path)
        kaon_net.to('cuda')
        kaon_net.load_state_dict(dicte['net_state_dict'])
        print("Loaded kaon network.")

    elif args.model_type == "FlowMatching":
        dicte = torch.load(config['Inference']['pion_model_path_'+str(args.model_type)])
        if args.sample_photons:
            sampler_path = config['Photon_Sampler']['Pion_LUT_path']
        pion_net = FlowMatching(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,LUT_path=sampler_path)
        pion_net.to('cuda')
        pion_net.load_state_dict(dicte['net_state_dict'])
        print("Loaded pion network.")

        dicte = torch.load(config['Inference']['kaon_model_path_'+str(args.model_type)])
        if args.sample_photons:
            sampler_path = config['Photon_Sampler']['Kaon_LUT_path']
        kaon_net = FlowMatching(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,LUT_path=sampler_path)
        kaon_net.to('cuda')
        kaon_net.load_state_dict(dicte['net_state_dict'])
        print("Loaded kaon network.")

    else:
        raise ValueError("Model type not supported.")
    


    if args.momentum == 3.0:
        print("Loading 3GeV dataset.")
        datapoints = np.load(config['dataset']['time_imaging']["data_path_3GeV"],allow_pickle=True)
    elif args.momentum == 6.0:
        print("Loading 6GeV dataset.")
        datapoints = np.load(config['dataset']['time_imaging']["data_path_6GeV"],allow_pickle=True)
    elif args.momentum == 9.0:
        print("Loading 9GeV dataset.")
        datapoints = np.load(config['dataset']['time_imaging']["data_path_9GeV"],allow_pickle=True)
    else:
        raise ValueError("Value of momentum does correspond to a dataset. Check if the path is correct, or simulate and processes.")


    
    os.makedirs("Generations",exist_ok=True)
    out_folder = os.path.join("Generations",config['Inference']['time_imaging_dir'])
    os.makedirs(out_folder,exist_ok=True)
    print("Outputs can be found in " + str(out_folder))

    if args.sample_photons:
        print("Using LUT photon yield sampling.")
    else:
        print("Using photon yield associated to ground truth track.")

    if args.fine_grained_prior:
        print("Using fine grained prior for PMTs. Consider disabling for increased generation speed.")
    else:
        print("Fine grained prior disabled. Consider enabling for high fidelity.")

    thetas = [30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,95.,100.,105.,110.,115.,120.,125.,130.,135.,140.,145.,150.] # Bug 155 and 25?

    for theta_ in thetas:
        list_to_gen = []

        for i in range(len(datapoints)):
            if (datapoints[i]['Theta'] == theta_) and (datapoints[i]['P'] == args.momentum) and (datapoints[i]['Phi'] == 0.0):
                list_to_gen.append(datapoints[i])

        print("Generating {0} tracks, with p={1} and theta={2}.".format(len(list_to_gen),args.momentum,theta_))
        generations = []
        kbar = pkbar.Kbar(target=len(list_to_gen), width=20, always_stateful=False)
        start = time.time()
        for i in range(len(list_to_gen)):   
            with torch.set_grad_enabled(False):
                p = (list_to_gen[i]['P'] - stats['P_max'])  / (stats['P_max'] - stats['P_min'])
                theta = (list_to_gen[i]['Theta'] - stats['theta_max']) / (stats['theta_max'] - stats['theta_min'])
                k = torch.tensor(np.array([p,theta])).to('cuda').float()

                if list_to_gen[i]['PDG'] == 321:   
                    PDG = 3
                    if not args.sample_photons:
                        if list_to_gen[i]['NHits'] > 0:
                            gen = kaon_net.create_tracks(num_samples=list_to_gen[i]['NHits'],context=k.unsqueeze(0),fine_grained_prior=args.fine_grained_prior)
                        else:
                            print(" ")
                            print("Error with number of hits to generate: ",list_to_gen[i]['NHits'])
                            continue
                    else:
                        if list_to_gen[i]['NHits'] > 0:
                            gen = kaon_net.create_tracks(num_samples=None,context=k.unsqueeze(0),p=list_to_gen[i]['P'],theta=list_to_gen[i]['Theta'],fine_grained_prior=args.fine_grained_prior)
                        else:
                            print(" ")
                            print("Error with number of hits to generate: ",list_to_gen[i]['NHits'])
                            continue

                elif list_to_gen[i]['PDG'] == 211:
                    PDG = 2
                    if not args.sample_photons:
                        if list_to_gen[i]['NHits'] > 0:
                            gen = pion_net.create_tracks(num_samples=list_to_gen[i]['NHits'],context=k.unsqueeze(0),fine_grained_prior=args.fine_grained_prior)                   
                        else:
                            print(" ")
                            print("Error with number of hits to generate: ",list_to_gen[i]['NHits'])
                            continue
                    else:
                        if list_to_gen[i]['NHits'] > 0:
                            gen = pion_net.create_tracks(num_samples=None,context=k.unsqueeze(0),p=list_to_gen[i]['P'],theta=list_to_gen[i]['Theta'],fine_grained_prior=args.fine_grained_prior)
                        else:
                            print(" ")
                            print("Error with number of hits to generate: ",list_to_gen[i]['NHits'])
                            continue

                else:
                    raise ValueError("Bad PID at event: {0}".format(i))

            gen['PDG'] = PDG
            
            generations.append(convert_numpy(gen))
            kbar.update(i)
            #if i == 10:
            #    break

        end = time.time()

        
        print(" ")
        print("Elapsed Time: ", end - start)
        print("Average time / track: ",(end - start) / len(generations))
        print(" ")

        out_path_ = os.path.join(out_folder,"Mixed_PiK"+f"_p_{args.momentum}_theta_{theta_}_ntracks_{len(list_to_gen)}.json")
        print("Writing json file: ",out_path_)
        with open(out_path_,"w") as file:
            json.dump(generations,file,indent=4,default=convert_numpy)
        print(" ")




if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FastSim Generation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-p', '--momentum', default=6.0,type=float,help='Particle Momentum.')
    parser.add_argument('-mt','--model_type',default="NF",type=str,help='Which model to use.')
    parser.add_argument('-sp','--sample_photons', action='store_true', help="Enable verbose mode")
    parser.add_argument('-f','--fine_grained_prior',action='store_true',help="Enable fine grained prior, default False.")
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args)
