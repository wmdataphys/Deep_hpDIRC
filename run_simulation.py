import os
import re
import json
import time
import copy
import pkbar
import random
import pickle
import warnings
import multiprocessing
import argparse
import numpy as np

import torch

warnings.filterwarnings("ignore", message=".*weights_only.*")

from models.NF.freia_models import FreiaNet
from models.OT_Flow.ot_flow import OT_Flow
from models.FlowMatching.flow_matching import FlowMatching
from models.Diffusion.resnet import ResNet
from models.Diffusion.continuous_diffusion import ContinuousTimeGaussianDiffusion
from models.Diffusion.gaussian_diffusion import GaussianDiffusion


def main(config,args):

    # Remove seeding, make it random.
    seed = int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.method in ['Kaon','MixPiK']:
        Kdicte = torch.load(config['Inference']['kaon_model_path_'+str(args.model_type)])
        Ksampler_path = config['Photon_Sampler']['Kaon_LUT_path']
        PID = 321
    if args.method == "Pion" or "MixPiK":
        Pdicte = torch.load(config['Inference']['pion_model_path_'+str(args.model_type)])
        Psampler_path = config['Photon_Sampler']['Pion_LUT_path']
        PID = 211
    else:
        print("Specify particle to generate in config file")
        exit()

    print('------------------------ Setup ------------------------')
    print("Generating: ",args.method)
    print("Using model type: ",str(args.model_type))

    if args.dark_noise and args.model_type != "NF":
        raise ValueError("Dark noise is only currently only implemented in our DNF (most performant) model.")
    elif args.dark_noise:
        print("Adding dark noise in 100 ns window with provided rate: ",args.dark_rate)
        print("See https://github.com/rdom/eicdirc for more information on dark rates.")
    else:
        pass

    if torch.cuda.is_available() and args.use_gpu:
        print("Using GPU.")
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    else:
        print("Defaulting to CPU.")
        device = 'cpu'
        if args.num_threads is not None:
            print("Setting threads...")
            num_cores = args.num_threads
            torch.set_num_threads(num_cores)
            torch.set_num_interop_threads(num_cores)
        else:
            print("Defaulting to thread pytorch values.")
            pass

        
        print(f"PyTorch is now using {torch.get_num_threads()} threads for intra-op parallelism.")
        print(f"PyTorch is now using {torch.get_num_interop_threads()} threads for inter-op parallelism.")
        
    
    if not args.n_dump:
        print('No value found for n_dump. Setting it equal to {}'.format(args.n_tracks))
        print("Consider dumping simulation to disk in batches.")
        args.n_dump = args.n_tracks
    print('-------------------------------------------------------')

    assert args.n_dump <= args.n_tracks, "total n_tracks must be at least n_dump, the number of tracks to dump per .pkl file. Got n_tracks of {} and n_dump of {} instead.".format(args.n_tracks, args.n_dump)

    num_layers = int(config['model_'+str(args.model_type)]['num_layers'])
    input_shape = int(config['model_'+str(args.model_type)]['input_shape'])
    cond_shape = int(config['model_'+str(args.model_type)]['cond_shape'])
    hidden_nodes = int(config['model_'+str(args.model_type)]['hidden_nodes'])
    stats = config['stats']
    
    if args.model_type == "NF":
        num_blocks = int(config['model_'+str(args.model_type)]['num_blocks'])
        if args.method in ['Kaon','MixPiK']:
            kaon_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats,LUT_path=Ksampler_path,device=device)
        if args.method in ['Pion','MixPiK']:
            pion_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats,LUT_path=Psampler_path,device=device)
    elif args.model_type == "CNF":
        num_blocks = int(config['model_'+str(args.model_type)]['num_blocks'])
        alph = config['model_'+str(args.model_type)]['alph']
        train_T = bool(config['model_'+str(args.model_type)]['train_T'])
        if args.method in ['Kaon','MixPiK']:
            kaon_net = OT_Flow(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,train_T=train_T,alph=alph,LUT_path=Ksampler_path,device=device)
        if args.method in ['Pion','MixPiK']:
            pion_net = OT_Flow(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,train_T=train_T,alph=alph,LUT_path=Psampler_path,device=device)
    elif args.model_type == "FlowMatching":
        if args.method in ['Kaon','MixPiK']:
            kaon_net = FlowMatching(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,LUT_path=Ksampler_path,device=device)
        if args.method in ['Pion','MixPiK']:
            pion_net = FlowMatching(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,LUT_path=Psampler_path,device=device)
    elif args.model_type == 'Score':
        num_steps = int(config['model_Score']['num_steps'])
        noise_schedule = config['model_Score']['noise_schedule']
        learned_schedule_net_hidden_dim = int(config['model_Score']['learned_schedule_net_hidden_dim'])
        gamma = int(config['model_Score']['gamma'])
        model = ResNet(input_dim=input_shape, end_dim=input_shape, cond_dim=cond_shape, mlp_dim=hidden_nodes, num_layer=num_layers)
        if args.method in ['Kaon','MixPiK']:
            kaon_net = ContinuousTimeGaussianDiffusion(model=model, stats=stats,num_sample_steps=num_steps, noise_schedule=noise_schedule, learned_schedule_net_hidden_dim=learned_schedule_net_hidden_dim, min_snr_loss_weight = True, min_snr_gamma=gamma,LUT_path=Ksampler_path)
        if args.method in ['Pion','MixPiK']:
            pion_net = ContinuousTimeGaussianDiffusion(model=model, stats=stats,num_sample_steps=num_steps, noise_schedule=noise_schedule, learned_schedule_net_hidden_dim=learned_schedule_net_hidden_dim, min_snr_loss_weight = True, min_snr_gamma=gamma,LUT_path=Psampler_path)
    elif args.model_type == 'DDPM':
        num_steps = int(config['model_DDPM']['num_steps'])
        model = ResNet(input_dim=input_shape, end_dim=input_shape, cond_dim=cond_shape, mlp_dim=hidden_nodes, num_layer=num_layers)
        if args.method in ['Kaon','MixPiK']:
            kaon_net = GaussianDiffusion(denoise_fn=model, device=device, stats=stats, timesteps=num_steps, loss_type='l2',LUT_path=Ksampler_path)
        if args.method in ['Pion','MixPiK']:
            pion_net = GaussianDiffusion(denoise_fn=model, device=device, stats=stats, timesteps=num_steps, loss_type='l2',LUT_path=Psampler_path)
    else:
        raise ValueError("Model type not found.")
    
    if args.method in ['Kaon','MixPiK']:
        kaon_net.to(device)
    if args.method in ['Pion','MixPiK']:
        pion_net.to(device)

    print("Using LUT photon yield sampling.")
    
    if args.fine_grained_prior:
        print("Using fine grained prior for PMTs. Consider disabling for increased generation speed.")
    else:
        print("Fine grained prior disabled. Consider enabling for high fidelity.")

    if args.method in ['Kaon','MixPiK']:
        kaon_net.load_state_dict(Kdicte['net_state_dict'])
        kaon_net.eval()
        kaon_net = torch.compile(model=kaon_net,mode="max-autotune")
    if args.method in ['Pion','MixPiK']:
        pion_net.load_state_dict(Pdicte['net_state_dict'])
        pion_net.eval()
        pion_net = torch.compile(model=pion_net,mode="max-autotune")

    generations = []
    kbar = pkbar.Kbar(target=args.n_tracks, width=20, always_stateful=False)

    if re.fullmatch(r"\d+(?:\.\d+)?", args.momentum):
        p_low = float(args.momentum)
        p_high = float(args.momentum)
    elif re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", args.momentum):
        match = re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", args.momentum)
        p_low = float(match.group(1))
        p_high = float(match.group(2))
    else:
        raise ValueError('Momentum format is p1-p2, where p1, p2 are between 1 and 10 (e.g. 3.5-10).')
    
    if re.fullmatch(r"\d+(?:\.\d+)?", args.theta):
        theta_low = int(args.theta)
        theta_high = int(args.theta)
    elif re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", args.theta):
        match = re.fullmatch(r"(\d+)-(\d+)", args.theta)
        theta_low = int(match.group(1))
        theta_high = int(match.group(2))
    else:
        raise ValueError('Theta format is theta1-theta2, where theta1, theta2 are between 25 and 155 (e.g. 70.5-80)')
    
    assert 1 <= p_low <= 10 and 1 <= p_high <= 10, "momentum range must be between 0 and 10, got {}, {} instead.".format(p_low, p_high)
    assert 25 <= theta_low <= 155 and 25 <= theta_high <= 155, "theta range must be between 25 and 155, got {}, {} instead.".format(theta_low, theta_high)

    output_folder = os.path.join("outputs","Simulation")
    os.makedirs(output_folder,exist_ok=True)
    out_folder = os.path.join(output_folder,config['Inference']['simulation_dir'])
    os.makedirs(out_folder,exist_ok=True)

    print("Simulation can be found in: ",out_folder)
    start = time.time()
    running_gen = 0
    file_count = 1
    while running_gen < args.n_tracks:
        if args.method in ['Kaon','MixPiK']:
            with torch.set_grad_enabled(False):
                p = np.random.uniform(low = p_low, high = p_high, size = None)
                p_scaled = (p - stats['P_max'])  / (stats['P_max'] - stats['P_min'])

                theta = np.random.uniform(low = theta_low, high = theta_high, size = None)
                theta_scaled = theta = (theta - stats['theta_max']) / (stats['theta_max'] - stats['theta_min'])

                k = torch.tensor(np.array([p_scaled,theta_scaled])).to(device).float()

                gen = kaon_net.create_tracks(num_samples=None,context=k.unsqueeze(0),p=p,theta=theta,fine_grained_prior=args.fine_grained_prior,dark_rate=args.dark_rate)
                gen['PID']=321

            generations.append(gen)
            running_gen +=1 
            kbar.update(running_gen)

            if running_gen >= args.n_tracks:
                break

            if len(generations) >= args.n_dump:
                out_path_ = os.path.join(out_folder,str(args.method)+f"_p_{args.momentum}_theta_{args.theta}_PID_{args.method}_ntracks_{len(generations)}_{file_count}.pkl")
                random.shuffle(generations)
                with open(out_path_,"wb") as file:
                    pickle.dump(generations,file)

                # reset lists
                generations = []
                file_count += 1

        if args.method in ['Pion','MixPiK']:
            with torch.set_grad_enabled(False):
                p = np.random.uniform(low = p_low, high = p_high, size = None)
                p_scaled = (p - stats['P_max'])  / (stats['P_max'] - stats['P_min'])

                theta = np.random.uniform(low = theta_low, high = theta_high, size = None)
                theta_scaled = theta = (theta - stats['theta_max']) / (stats['theta_max'] - stats['theta_min'])

                k = torch.tensor(np.array([p_scaled,theta_scaled])).to(device).float()

                gen = pion_net.create_tracks(num_samples=None,context=k.unsqueeze(0),p=p,theta=theta,fine_grained_prior=args.fine_grained_prior,dark_rate=args.dark_rate)
                gen['PID']=211

            generations.append(gen)
            running_gen +=1 
            kbar.update(running_gen)

            if running_gen >= args.n_tracks:
                break

            if len(generations) >= args.n_dump:
                out_path_ = os.path.join(out_folder,str(args.method)+f"_p_{args.momentum}_theta_{args.theta}_PID_{args.method}_ntracks_{len(generations)}_{file_count}.pkl")
                random.shuffle(generations)
                with open(out_path_,"wb") as file:
                    pickle.dump(generations,file)

                # reset lists
                generations = []
                file_count += 1
        
    end = time.time()

    n_photons = 0
    n_gamma_kaon = 0
    n_gamma_pion = 0

    for i in range(len(generations)):
        if generations[i]['PID'] == 321:
            n_gamma_kaon += generations[i]['NHits']
        else:
            n_gamma_pion += generations[i]['NHits']
    
    time_per_track = (end - start) / args.n_tracks if args.n_tracks > 0 else r'N/A'
    
    print(" ")
    print("Sampling statistics:")
    print("Elapsed Time: ", end - start)
    print("Average time / track: ",time_per_track)
    print(" ")

    if len(generations) > 0:
        print(f'Writing final samples...')
        out_path_ = os.path.join(out_folder, str(args.method) + f"_p_{args.momentum}_theta_{args.theta}_PID_{args.method}_ntracks_{len(generations)}_{file_count}.pkl")
        random.shuffle(generations)
        with open(out_path_, "wb") as file:
            pickle.dump(generations, file)
    

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FastSim Generation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-nt', '--n_tracks', default=1e5,type=int,help='Number of particles to generate. Take the first n_tracks.')
    parser.add_argument('-nd', '--n_dump', default=None, type=int, help='Number of particles to dump per .pkl file.')
    parser.add_argument('-m', '--method',default="MixPiK",type=str,help='Generated particle type, Kaon, Pion, or MixPiK.')
    parser.add_argument('-rho','--momentum',default=3,type=str,help='Which momentum to generate for.')
    parser.add_argument('-th','--theta',default=30,type=str,help='Which theta angle to generate for.')
    parser.add_argument('-mt','--model_type',default="NF",type=str,help='Which model to use.')
    parser.add_argument('-f','--fine_grained_prior',action='store_false',help="Enable fine grained prior, default True.")
    parser.add_argument('-dn','--dark_noise',action='store_true',help='Included hits from dark noise with specific rate. Currently only implmeneted for DNF.')
    parser.add_argument('-dr','--dark_rate', default=22800,type=float,help='Dark rate value. Default 22800.')
    parser.add_argument('-ug','--use_gpu', action='store_true',help="Whether to use a GPU. Note that CPU can be faster depending on # of cores. We reccomend testing with both.")
    parser.add_argument('-nthreads','--num_threads', default=None,type=int,help="Number of CPU threads - default is all pytorch defaults (1/2).")
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args)
