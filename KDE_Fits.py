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
from cuml.neighbors import KernelDensity
import cupy as cp
from utils.KDE_utils import perform_fit_KDE,gaussian_normalized,gaussian_unnormalized,plot,FastDIRC
import math
from utils.hpDIRC import gapx,gapy,pixel_width,pixel_height,npix,npmt

warnings.filterwarnings("ignore", message=".*weights_only.*")
# This is a bandwith warning. Occurs with low number of support photons.
warnings.filterwarnings("ignore", message=".*Grid size 1 will likely result in GPU under-utilization.*")

def add_dark_noise(hits,dark_noise_pmt=28000):
    # probability to have a noise hit in 100 ns window
    prob = dark_noise_pmt * 100 / 1e9
    new_hits = []
    for p in range(npmt):
        for i in range(int(prob) + 1):
            if(i == 0) and (prob - int(prob) < np.random.uniform()):
                continue

            dn_time = 100 * np.random.uniform() # [1,100] ns
            dn_pix = int(npix * np.random.uniform())
            row = (p//6) * 16 + dn_pix//16 
            col = (p%6) * 16 + dn_pix%16
            x = 2 + col * pixel_width + (p % 6) * gapx + (pixel_width) / 2. # Center at middle
            y = 2 + row * pixel_height + (p // 6) * gapy + (pixel_height) / 2. # Center at middle
            h = [x,y,dn_time]
            new_hits.append(h)

    if new_hits:
        new_hits = np.array(new_hits)
        hits = np.vstack([hits,new_hits])

    return hits


def create_supports_geant(pions,kaons,num_support=200000):
    pmtID = np.concatenate([pion['pmtID'] for pion in pions])
    pixelID = np.concatenate([pion['pixelID'] for pion in pions])
    t = np.concatenate([pion['leadTime'] for pion in pions])

    row = (pmtID//6) * 16 + pixelID//16 
    col = (pmtID%6) * 16 + pixelID%16
    x = 2 + col * pixel_width + (pmtID % 6) * gapx + (pixel_width) / 2. # Center at middle
    y = 2 + row * pixel_height + (pmtID // 6) * gapy + (pixel_height) / 2. # Center at middle
    support_pions = np.vstack((x, y,t)).T

    pmtID = np.concatenate([kaon['pmtID'] for kaon in kaons])
    pixelID = np.concatenate([kaon['pixelID'] for kaon in kaons])
    t = np.concatenate([kaon['leadTime'] for kaon in kaons])

    row = (pmtID//6) * 16 + pixelID//16 
    col = (pmtID%6) * 16 + pixelID%16
    x = 2 + col * pixel_width + (pmtID % 6) * gapx + (pixel_width) / 2. # Center at middle
    y = 2 + row * pixel_height + (pmtID // 6) * gapy + (pixel_height) / 2. # Center at middle
    support_kaons = np.vstack((x, y,t)).T  

    support_pions,support_kaons =  support_pions[np.where((support_pions[:,2] < 100.0) & (support_pions[:,2] > 10.0))[0]],support_kaons[np.where((support_kaons[:,2] < 100.0) & (support_kaons[:,2] > 10.0))[0]]
    return support_pions[:num_support],support_kaons[:num_support]

def create_supports_fs(pions,kaons):
    pmtID = pions['pmtID']
    pixelID = pions['pixelID']
    t = pions['leadTime']

    row = (pmtID//6) * 16 + pixelID//16 
    col = (pmtID%6) * 16 + pixelID%16
    x = 2 + col * pixel_width + (pmtID % 6) * gapx + (pixel_width) / 2. # Center at middle
    y = 2 + row * pixel_height + (pmtID // 6) * gapy + (pixel_height) / 2. # Center at middle
    support_pions = np.vstack((x, y,t)).T

    pmtID = kaons['pmtID']
    pixelID = kaons['pixelID']
    t = kaons['leadTime']

    row = (pmtID//6) * 16 + pixelID//16 
    col = (pmtID%6) * 16 + pixelID%16
    x = 2 + col * pixel_width + (pmtID % 6) * gapx + (pixel_width) / 2. # Center at middle
    y = 2 + row * pixel_height + (pmtID // 6) * gapy + (pixel_height) / 2. # Center at middle
    support_kaons = np.vstack((x, y,t)).T  

    return support_pions[np.where((support_pions[:,2] < 100.0) & (support_pions[:,2] > 10.0))[0]],support_kaons[np.where((support_kaons[:,2] < 100.0) & (support_kaons[:,2] > 10.0))[0]]

def create_KDE(support_pions,support_kaons,num_support):
    rng = cp.random.RandomState(42) # Just hard code for nowresult = arr[::-1][:N]
    print("Num support: ",num_support)
    kde_p = KernelDensity(kernel='exponential', bandwidth=2*pixel_width).fit(support_pions[:num_support].astype('float32'))
    kde_k = KernelDensity(kernel='exponential', bandwidth=2*pixel_width).fit(support_kaons[:num_support].astype('float32'))
    return kde_p,kde_k

def inference(tracks,dirc_obj,support_kaons,support_pions,add_dn=False):
    DLLs = []
    tprobs_k = []
    tprobs_p = []
    kbar = pkbar.Kbar(len(tracks))
    start = time.time()
    for i,track in enumerate(tracks):
        
        pixelID = np.array(track['pixelID'])
        pmtID = np.array(track['pmtID'])
        row = (pmtID//6) * 16 + pixelID//16 
        col = (pmtID%6) * 16 + pixelID%16
        
        x = 2 + col * pixel_width + (pmtID % 6) * gapx + (pixel_width) / 2. # Center at middle
        y = 2 + row * pixel_height + (pmtID // 6) * gapy + (pixel_height) / 2. # Center at middle
        t = track['leadTime']
        
        hits = np.vstack([x,y,t]).T

        if add_dn:
            hits = add_dark_noise(hits)

        ll_k,tprob_k = dirc_obj.get_log_likelihood(hits.astype('float16'),support_kaons.astype('float16'))
        ll_p,tprob_p = dirc_obj.get_log_likelihood(hits.astype('float16'),support_pions.astype("float16"))
        
        DLLs.append(ll_k - ll_p)
        tprobs_k.append({"rvalue": tprob_k,"coords":hits})
        tprobs_p.append({"rvalue": tprob_p,"coords":hits})
        kbar.update(i)
    end = time.time()

    print(" - Time/track: ",(end - start)/len(DLLs))

    return np.array(DLLs),tprobs_k,tprobs_p


def calculate_splits(gpu_mem, total_samples):
    events_per_24GB = int(3.5e5)  # roughly 350k events for 24GB VRAM at generation
    vram_per_event = 24 / events_per_24GB
    available_events = gpu_mem / vram_per_event
    num_splits = math.ceil(total_samples / available_events)
    
    if num_splits < 1:
        num_splits = 1

    events_per_split = [int(total_samples / num_splits)] * num_splits
    events_per_split[-1] += total_samples - sum(events_per_split)

    return num_splits, events_per_split

def combine_photons(pions,kaons):
    new_pions = {"pmtID": np.concatenate([pion['pmtID'] for pion in pions]),
                "pixelID": np.concatenate([pion['pixelID'] for pion in pions]),
                "leadTime": np.concatenate([pion['leadTime'] for pion in pions])}
    new_kaons = {"pmtID": np.concatenate([kaon['pmtID'] for kaon in kaons]),
            "pixelID": np.concatenate([kaon['pixelID'] for kaon in kaons]),
            "leadTime": np.concatenate([kaon['leadTime'] for kaon in kaons])}

    return new_pions,new_kaons

def main(config,args):

    # Setup random seed
    print("Using model type: ",str(args.model_type))
    # Fix seed
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    gpu_mem, _ = torch.cuda.mem_get_info()
    gpu_mem = gpu_mem / (1024 ** 3) # GB

    print("---------------- PDF Stats ------------------")
    print("Fast Simulated Support Photons: ",args.fs_support)
    print("Geant4 Support Photons: ",args.geant_support)
    print("Important note: Generating 350k photons requires ~ 24GB of VRAM.")
    print("You currently have {0:.2f} GB of free GPU memory. Our generation procedure will scale accorindgly, KDE will not.".format(gpu_mem))
    print("---------------------------------------------")
    device = torch.device('cuda')

    if args.model_type != "NF":
        raise ValueError("Currently only implemented for Normalizing Flows.")

    num_layers = int(config['model_'+str(args.model_type)]['num_layers'])
    input_shape = int(config['model_'+str(args.model_type)]['input_shape'])
    cond_shape = int(config['model_'+str(args.model_type)]['cond_shape'])
    num_blocks = int(config['model_'+str(args.model_type)]['num_blocks'])
    hidden_nodes = int(config['model_'+str(args.model_type)]['hidden_nodes'])
    stats = config['stats']

    sampler_path = None

    dicte = torch.load(config['Inference']['pion_model_path_'+str(args.model_type)])
    pion_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats,LUT_path=sampler_path)
    pion_net.to('cuda')
    pion_net.load_state_dict(dicte['net_state_dict'])
    print("Loaded pion network.")

    dicte = torch.load(config['Inference']['kaon_model_path_'+str(args.model_type)])
    kaon_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats,LUT_path=sampler_path)
    kaon_net.to('cuda')
    kaon_net.load_state_dict(dicte['net_state_dict'])
    print("Loaded kaon network.")

    if args.momentum == 3.0:
        print("Loading 3GeV datasets.")
        geant = np.load(config['dataset']['time_imaging']["data_path_3GeV"],allow_pickle=True)
        inference_datapoints = np.load(config['dataset']['fixed_point']['pion_data_path_3GeV'],allow_pickle=True) \
                             + np.load(config['dataset']['fixed_point']['kaon_data_path_3GeV'],allow_pickle=True)
    elif args.momentum == 6.0:
        print("Loading 6GeV datasets.")
        geant = np.load(config['dataset']['time_imaging']["data_path_6GeV"],allow_pickle=True)
        inference_datapoints = np.load(config['dataset']['fixed_point']['pion_data_path_6GeV'],allow_pickle=True) \
                             + np.load(config['dataset']['fixed_point']['kaon_data_path_6GeV'],allow_pickle=True)
    elif args.momentum == 9.0:
        print("Loading 9GeV datasets.")
        geant = np.load(config['dataset']['time_imaging']["data_path_9GeV"],allow_pickle=True)
        inference_datapoints = np.load(config['dataset']['fixed_point']['pion_data_path_9GeV'],allow_pickle=True)  \
                             + np.load(config['dataset']['fixed_point']['kaon_data_path_9GeV'],allow_pickle=True)
    else:
        raise ValueError("Value of momentum does correspond to a dataset. Check if the path is correct, or simulate and processes.")


    os.makedirs("KDE_Fits",exist_ok=True)
    out_folder = config['Inference']['KDE_dir']
    os.makedirs(out_folder,exist_ok=True)
    print("Outputs can be found in " + str(out_folder))

    if args.fine_grained_prior:
        print("Using fine grained prior for PMTs. Consider disabling for increased generation speed.")
    else:
        print("Fine grained prior disabled. Consider enabling for high fidelity.")

    if args.fs_inference:
        print("Using fast simulated data as inference to KDE. There is a slight overhead here in terms of time.")
    else:
        print("Using Geant4 data as inference to KDE.")

    if args.dark_noise:
        print("Adding dark noise to inference tracks.")
    else:
        pass

    fastDIRC = FastDIRC(device='cuda')
    sigma_dict_geant = {}
    sigma_dict_fs = {}
    thetas =  [30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,95.,100.,105.,110.,115.,120.,125.,130.,135.,140.,145.,150.] 
    
    for theta_ in thetas:
        geant_support_pions = []
        geant_support_kaons = []
        inference_pions = []
        inference_kaons = []


        for i in range(len(geant)):
            if (geant[i]['Theta'] == theta_) and (geant[i]['P'] == args.momentum) and (geant[i]['Phi'] == 0.0) and (geant[i]['NHits'] < 150) and (geant[i]['NHits'] > 0):
                PDG = geant[i]['PDG']
                if PDG == 321:
                    geant_support_kaons.append(geant[i])
                elif PDG == 211:
                    geant_support_pions.append(geant[i])
                else:
                    pass

        for i in range(len(inference_datapoints)):
            if (inference_datapoints[i]['Theta'] == theta_) and (inference_datapoints[i]['P'] == args.momentum) and (inference_datapoints[i]['Phi'] == 0.0) and (inference_datapoints[i]['NHits'] < 300) and (inference_datapoints[i]['NHits'] > 0):
                PDG = inference_datapoints[i]['PDG']
                if PDG == 321:
                    if not args.fs_inference:
                        inference_kaons.append(inference_datapoints[i])
                    else:
                        with torch.no_grad():
                            p = (inference_datapoints[i]['P'] - stats['P_max'])  / (stats['P_max'] - stats['P_min'])
                            theta = (inference_datapoints[i]['Theta'] - stats['theta_max']) / (stats['theta_max'] - stats['theta_min'])
                            k = torch.tensor(np.array([p,theta])).to('cuda').float().unsqueeze(0)
                            nhits = inference_datapoints[i]['NHits']
                            inf_k_ = kaon_net.create_tracks(num_samples=nhits,context=k,fine_grained_prior=args.fine_grained_prior)
                        inference_kaons.append(inf_k_)
                elif PDG == 211:
                    if not args.fs_inference:
                        inference_pions.append(inference_datapoints[i])
                    else:
                        with torch.no_grad():
                            p = (inference_datapoints[i]['P'] - stats['P_max'])  / (stats['P_max'] - stats['P_min'])
                            theta = (inference_datapoints[i]['Theta'] - stats['theta_max']) / (stats['theta_max'] - stats['theta_min'])
                            k = torch.tensor(np.array([p,theta])).to('cuda').float().unsqueeze(0)
                            nhits = inference_datapoints[i]['NHits']
                            inf_p_ = pion_net.create_tracks(num_samples=nhits,context=k,fine_grained_prior=args.fine_grained_prior)
                        inference_pions.append(inf_p_)
                else:
                    pass


        torch.cuda.empty_cache()
        print("Running inference for momentum = {0}, theta = {1}, for {2} pions and {3} kaons.".format(args.momentum,theta_,len(inference_pions),len(inference_kaons)))

        print("--------------- Fast Simulation -----------------")
        with torch.set_grad_enabled(False):
            p = (args.momentum - stats['P_max'])  / (stats['P_max'] - stats['P_min'])
            theta = (theta_ - stats['theta_max']) / (stats['theta_max'] - stats['theta_min'])
            k = torch.tensor(np.array([p,theta])).to('cuda').float().unsqueeze(0)

            
            start = time.time()
            num_splits,batches = calculate_splits(gpu_mem, args.fs_support)
            if num_splits > 1:
                print("Batching generations into {0} runs.".format(num_splits)," ",batches)
                fs_kaons = []
                fs_pions = []
                start = time.time()
                for i in range(num_splits):
                    fs_kaons.append(kaon_net.create_tracks(num_samples=batches[i],context=k,fine_grained_prior=args.fine_grained_prior))
                    fs_pions.append(pion_net.create_tracks(num_samples=batches[i],context=k,fine_grained_prior=args.fine_grained_prior))
                    torch.cuda.empty_cache()

                fs_pions,fs_kaons = combine_photons(fs_pions,fs_kaons)

                end = time.time()
            else:
                # Essentially generate 1 track with N photons.
                start = time.time()
                fs_kaons = kaon_net.create_tracks(num_samples=args.fs_support,context=k,fine_grained_prior=args.fine_grained_prior)
                fs_pions = pion_net.create_tracks(num_samples=args.fs_support,context=k,fine_grained_prior=args.fine_grained_prior)
                end = time.time()

            torch.cuda.empty_cache() # This is a must.
            support_pions,support_kaons = create_supports_fs(fs_pions,fs_kaons)
            print("Number of pions: ",len(support_pions)," Number of Kaons: ",len(support_kaons))
            del fs_kaons,fs_pions

        np.save(os.path.join(out_folder,f"FastSim_SupportPions_{theta_}.npy"),support_pions)
        np.save(os.path.join(out_folder,f"FastSim_SupportKaons_{theta_}.npy"),support_kaons)

        print("Time to create both PDFs: ",end - start)
        print("Time / photon: {0}.\n".format((end - start)/ (2*args.fs_support)))
        print("Running inference for pions with Fast Simulated PDF.")
 
        DLL_p,tprobs_p_k,tprobs_p_p = inference(inference_pions,fastDIRC,support_kaons,support_pions,add_dn=args.dark_noise)
        torch.cuda.empty_cache()

        # See how likelihood is distributed, leaving for others.
        # with open(os.path.join(out_folder,f"FastSim_Pion_given_Kaon_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(tprobs_p_k,file)

        # with open(os.path.join(out_folder,f"FastSim_Pion_given_Pion_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(tprobs_p_p,file)  

        print(" ")
        print("Running inference for kaons with Fast Simulated PDF.")

        DLL_k,tprobs_k_k,tprobs_k_p = inference(inference_kaons,fastDIRC,support_kaons,support_pions,add_dn=args.dark_noise)
        torch.cuda.empty_cache()

        # with open(os.path.join(out_folder,f"FastSim_Kaon_given_Kaon_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(tprobs_k_k,file)

        # with open(os.path.join(out_folder,f"FastSim_Kaon_given_Pion_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(tprobs_k_p,file)  

        print("\n")
        DLL_k = DLL_k[~np.isnan(DLL_k)].astype('float32')
        DLL_p = DLL_p[~np.isnan(DLL_p)].astype('float32')
        DLL_k = DLL_k[~np.isinf(DLL_k)].astype('float32')
        DLL_p = DLL_p[~np.isinf(DLL_p)].astype('float32')
        fit_params = perform_fit_KDE(DLL_k,DLL_p,bins=200,normalized=False,momentum=args.momentum)
        plot(fit_params,DLL_p,DLL_k,"Fast Sim.",out_folder,theta_,pdf_method="Fast Sim.",bins=200,momentum=args.momentum)
        del support_pions,support_kaons
        sigma_dict_fs[theta_] = [fit_params[2],fit_params[-2]]


        # with open(os.path.join(out_folder,f"DLL_Pion_FastSim_theta_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(DLL_p,file)

        # with open(os.path.join(out_folder,f"DLL_Kaon_FastSim_theta_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(DLL_k,file)    

        

        print("------------------- Geant4 ---------------------")
        support_pions,support_kaons = create_supports_geant(geant_support_pions,geant_support_kaons,num_support=args.geant_support)
        print("Number of pions: ",len(support_pions)," Number of Kaons: ",len(support_kaons))
        np.save(os.path.join(out_folder,f"Geant_SupportPions_{theta_}.npy"),support_pions)
        np.save(os.path.join(out_folder,f"Geant_SupportKaons_{theta_}.npy"),support_kaons)
        
        print("Running inference for pions with Geant4 PDF.")
        DLL_p,tprobs_p_k,tprobs_p_p = inference(inference_pions,fastDIRC,support_kaons,support_pions,add_dn=args.dark_noise)
        torch.cuda.empty_cache()

        # with open(os.path.join(out_folder,f"Geant_Pion_given_Kaon_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(tprobs_p_k,file)

        # with open(os.path.join(out_folder,f"Geant_Pion_given_Pion_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(tprobs_p_p,file)  
        print(" ")

        print("Running inference for kaons with Geant4 PDF.")
        
        DLL_k,tprobs_k_k,tprobs_k_p = inference(inference_kaons,fastDIRC,support_kaons,support_pions,add_dn=args.dark_noise)
        torch.cuda.empty_cache()

        # with open(os.path.join(out_folder,f"Geant_Kaon_given_Kaon_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(tprobs_k_k,file)

        # with open(os.path.join(out_folder,f"Geant_Kaon_given_Pion_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(tprobs_k_p,file)  

        print("\n")
        DLL_k = DLL_k[~np.isnan(DLL_k)].astype('float32')
        DLL_p = DLL_p[~np.isnan(DLL_p)].astype('float32')
        DLL_k = DLL_k[~np.isinf(DLL_k)].astype('float32')
        DLL_p = DLL_p[~np.isinf(DLL_p)].astype('float32')
        fit_params = perform_fit_KDE(DLL_k,DLL_p,bins=200,normalized=False,momentum=args.momentum)
        plot(fit_params,DLL_p,DLL_k,"Geant4",out_folder,theta_,pdf_method="Geant4",bins=200,momentum=args.momentum)
        print("\n")
        del support_pions,support_kaons
        sigma_dict_geant[theta_] = [fit_params[2],fit_params[-2]]

        # with open(os.path.join(out_folder,f"DLL_Pion_Geant_theta_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(DLL_p,file)

        # with open(os.path.join(out_folder,f"DLL_Kaon_Geant_theta_{theta_}.pkl"),"wb") as file:
        #     pickle.dump(DLL_k,file)    



    with open(os.path.join(out_folder,"FastSim_Sigmas.pkl"),"wb") as file:
            pickle.dump(sigma_dict_fs,file)

    with open(os.path.join(out_folder,"Geant_Sigmas.pkl"),"wb") as file:
        pickle.dump(sigma_dict_geant,file)


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FastSim Generation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-p', '--momentum', default=6.0,type=float,help='Particle Momentum.')
    parser.add_argument('-mt','--model_type',default="NF",type=str,help='Which model to use.')
    parser.add_argument('-f','--fine_grained_prior',action='store_true',help="Enable fine grained prior, default False.")
    parser.add_argument('-fs','--fs_support', default=8e5,type=int,help='Number of Fast Simulated support photons.')
    parser.add_argument('-fg','--geant_support', default=2e5,type=int,help='Number of Geant4 support photons.')
    parser.add_argument('-fsi','--fs_inference',action='store_true',help="Use Fast Simulated tracks for inference as opposed to Geant4 (default).")
    parser.add_argument('-dn', '--dark_noise',action='store_true',help="Add dark noise to inference tracks.")
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args)
