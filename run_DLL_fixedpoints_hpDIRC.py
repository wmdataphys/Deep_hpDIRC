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
from dataloader.create_data import unscale,scale_data
from datetime import datetime
import itertools
import matplotlib.pyplot as plt
import time
import pickle
from dataloader.create_data import hpDIRC_DLL_Dataset
from models.NF.freia_models import FreiaNet
from models.OT_Flow.ot_flow import OT_Flow
from models.FlowMatching.flow_matching import FlowMatching
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn import metrics
from scipy.optimize import curve_fit
import glob
from PyPDF2 import PdfWriter
from scipy.stats import norm
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from scipy.special import expit
import warnings

warnings.filterwarnings("ignore", message=".*weights_only.*")

def sigmoid(x):
    x = np.float64(x)
    return expit(x)


def compute_efficiency_rejection(delta_log_likelihood, true_labels):
    thresholds = np.linspace(-4000.0, 4000.0, 20000)
    thresholds_broadcasted = np.expand_dims(thresholds, axis=1)
    predicted_labels = delta_log_likelihood > thresholds_broadcasted

    TP = np.sum((predicted_labels == 1) & (true_labels == 1), axis=1)
    FP = np.sum((predicted_labels == 1) & (true_labels == 0), axis=1)
    TN = np.sum((predicted_labels == 0) & (true_labels == 0), axis=1)
    FN = np.sum((predicted_labels == 0) & (true_labels == 1), axis=1)

    efficiencies = TP / (TP + FN)  # Efficiency (True Positive Rate)
    rejections = TN / (TN + FP)  # Rejection (True Negative Rate)
    auc = np.trapz(y=np.flip(rejections),x=np.flip(efficiencies))

    return efficiencies,rejections,auc


def merge_PDF(out_dir):
    pdf_dir = os.path.join(out_dir,'DLL')
    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, '*.pdf')))

    pdf_files.sort(key=lambda x: float(x.split('_p(')[-1].split(')')[0].split(',')[0]))
    output_pdf = PdfWriter()

    for pdf_file in pdf_files:
        with open(pdf_file, 'rb') as f:
            pdf_data = f.read()
            output_pdf.append(fileobj=f)

    # Save the combined PDF to a file
    with open(os.path.join(out_dir,'Combined_DLL.pdf'), 'wb') as f:
        output_pdf.write(f)

def perform_fit(dll_k,dll_p,bins=200):
    hist_k, bin_edges_k = np.histogram(dll_k, bins=bins, density=True)
    bin_centers_k = (bin_edges_k[:-1] + bin_edges_k[1:]) / 2
    try:
        popt_k, pcov_k = curve_fit(gaussian, bin_centers_k, hist_k, p0=[1, np.mean(dll_k), np.std(dll_k)],maxfev=1000,bounds = ([0, -np.inf, 1e-9], [np.inf, np.inf, np.inf]))
        amplitude_k, mean_k, stddev_k = popt_k
        perr_k = np.sqrt(np.diag(pcov_k))
    except RuntimeError as e:
        print('Kaon error, exiting.')
        print(e)
        exit()
        

    hist_p, bin_edges_p = np.histogram(dll_p, bins=bins, density=True)
    bin_centers_p = (bin_edges_p[:-1] + bin_edges_p[1:]) / 2
    try:
        popt_p, pcov_p = curve_fit(gaussian, bin_centers_p, hist_p, p0=[1, np.mean(dll_p), np.std(dll_p)],maxfev=1000,bounds = ([0, -np.inf, 1e-9], [np.inf, np.inf, np.inf]))
        amplitude_p, mean_p, stddev_p = popt_p
        perr_p = np.sqrt(np.diag(pcov_p))
    except RuntimeError as e:
        print('Pion error, exiting.')
        print(e)
        exit()
    
    sigma_sep = (mean_k - mean_p) / ((stddev_k + stddev_p)/2.) #np.sqrt(stddev_k**2 + stddev_p**2)
    sigma_err = (2*perr_k[1]/(stddev_k + stddev_p))** 2 + (2*perr_p[1]/(stddev_k + stddev_p))** 2 + (-2*(mean_k - mean_p) * perr_k[2] / (stddev_k + stddev_p)**2)**2 + (-2*(mean_k - mean_p) * perr_p[2] / (stddev_k + stddev_p)**2)**2
    return popt_k,popt_p,sigma_sep,bin_centers_k,bin_centers_p,np.sqrt(sigma_err)

def gaussian(x, amplitude, mean, stddev):
    return (1 / (np.sqrt(2*np.pi)*stddev))* np.exp(-((x - mean) / stddev) ** 2 / 2)

def extract_values(file_path):
    results = np.load(file_path,allow_pickle=True)
    sigmas = []
    thetas = []
    for theta, gr_value in results.items():
        thetas.append(float(theta))
        sigmas.append(float(gr_value))
        
    sorted_thetas, sorted_sigmas = zip(*sorted(zip(thetas, sigmas)))

    return list(sorted_sigmas), list(sorted_thetas)


def run_plotting(out_folder,momentum,model_type):

    kaons = np.load(os.path.join(out_folder,"Kaon_DLL_Results.pkl"),allow_pickle=True)
    pions = np.load(os.path.join(out_folder,"Pion_DLL_Results.pkl"),allow_pickle=True)

    dll_k = []
    dll_p = []
    kin_p = []
    kin_k = []
    nhits_k = []
    nhits_pi = []
    ll_p = []
    ll_k = []

    for i in range(len(kaons)):
        dll_k.append((np.array(kaons[i]['hyp_kaon']) - np.array(kaons[i]['hyp_pion'])).flatten())
        ll_k.append(np.array(kaons[i]['hyp_kaon']).flatten())
        kin_k.append(kaons[i]['Kins'])
        nhits_k.append(kaons[i]['Nhits'])

    for i in range(len(pions)):
        dll_p.append((np.array(pions[i]['hyp_kaon']) - np.array(pions[i]['hyp_pion'])).flatten())
        ll_p.append(np.array(pions[i]['hyp_pion']).flatten())
        kin_p.append(pions[i]['Kins'])
        nhits_pi.append(pions[i]['Nhits'])

    kin_p = np.concatenate(kin_p)
    kin_k = np.concatenate(kin_k)
    dll_p = np.concatenate(dll_p)
    dll_k = np.concatenate(dll_k)
    print("NaN Checks: ",np.isnan(dll_k).sum())
    print("NaN Checks: ",np.isnan(dll_p).sum())
    dll_k = np.clip(dll_k[~np.isnan(dll_k)],-99999,99999)
    dll_p = np.clip(dll_p[~np.isnan(dll_p)],-99999,99999)
    kin_k =  kin_k[~np.isnan(dll_k)]
    kin_p = kin_p[~np.isnan(dll_p)]

    idx = np.where(kin_k[:,0] == momentum)[0]
    dll_k = dll_k[idx]
    kin_k = kin_k[idx]

    idx = np.where(kin_p[:,0] == momentum)[0]
    dll_p = dll_p[idx]
    kin_p = kin_p[idx]

    print("Pion max/min: ", dll_p.max(),dll_p.min())
    print("Kaon max/min: ",dll_k.max(),dll_k.min())

    
    if momentum >= 6.0:
        bins = np.linspace(-50,50,400) 
    else:
        bins = np.linspace(-250,300,400) 

    ### Raw DLL
    plt.hist(dll_k,bins=bins,density=True,alpha=1.0,label=r'$\mathcal{K} - $'+str(model_type),color='red',histtype='step',lw=2)
    plt.hist(dll_p,bins=bins,density=True,alpha=1.0,label=r'$\pi - $'+str(model_type),color='blue',histtype='step',lw=2)
    plt.xlabel('Loglikelihood Difference',fontsize=25)
    plt.ylabel('A.U.',fontsize=25)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(r'$ \Delta \mathcal{L}_{\mathcal{K} \pi}$',fontsize=30)
    out_path_DLL = os.path.join(out_folder,"DLL_piK.pdf")
    plt.savefig(out_path_DLL,bbox_inches='tight')
    plt.close()



    thetas = [30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,95.,100.,105.,110.,115.,120.,125.,130.,135.,140.,145.,150.]
    seps = []
    sep_err = []
    seps_cnf = []
    sep_err_cnf = []

    for theta in thetas:
        k_idx = np.where(kin_k[:,1] == theta)[0]
        p_idx = np.where(kin_p[:,1] == theta)[0]
        print("Theta: ",theta, "Pions: ",len(p_idx)," Kaons: ",len(k_idx))
        popt_k_NF,popt_p_NF,sep_NF,bin_centers_k_NF,bin_centers_p_NF,se = perform_fit(dll_k[k_idx],dll_p[p_idx],bins)
        seps.append(abs(sep_NF))
        sep_err.append(se)


    if momentum == 6.0:
        path_ = "LUT_Stats/6GeV/sigma_sep.pkl"
        sigma_10mill,theta_10mill = extract_values(path_)
    elif momentum == 3.0:
        path_ = "LUT_Stats/3GeV/sigma_sep.pkl"
        sigma_10mill,theta_10mill = extract_values(path_)
    elif momentum == 9.0:
        path_ = "LUT_Stats/9GeV/sigma_sep.pkl"
        sigma_10mill,theta_10mill = extract_values(path_)
    else:
        raise ValueError("Momentum value not found.")

    fig = plt.figure(figsize=(12,6))

    plt.errorbar(thetas, seps, yerr=sep_err, color='k', lw=2, 
                label=str(model_type)+r'- DLL - $\bar{\sigma} = $' + "{0:.2f}".format(np.average(seps)), capsize=5, linestyle='--', 
                fmt='o', markersize=4)
    plt.plot(theta_10mill,sigma_10mill,color='magenta',lw=2,linestyle='--',
            label=r'LUT - $\bar{\sigma} = $' + "{0:.2f}".format(np.average(sigma_10mill)),markersize=4,marker='o')
    plt.legend(fontsize=22,ncol=2)
    plt.xlabel("Polar Angle [deg.]",fontsize=25,labelpad=15)
    plt.ylabel("Separation [s.d.]",fontsize=25,labelpad=15)
    plt.ylim(0,None)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(r"$|\vec{p}| = "+ r" {0} \; GeV$".format(momentum),fontsize=28)
    plt.savefig(os.path.join(out_folder,"Seperation_{0}_LUT_{1}GeV.pdf".format(str(model_type),int(momentum))),bbox_inches="tight")
    plt.close()


def drop_and_sum(x,batch_size):
    lls = []
    unsummed = []
    for b in range(batch_size):
        ll = x[b]
        mask = torch.isnan(ll)
        lls.append(ll[~mask].sum().detach().cpu().numpy())

    return lls


def run_inference_seperate(pions,kaons,pion_net,kaon_net):
    LL_Pion = []
    LL_Kaon = []
    print('Starting DLL for Pions.')
    start = time.time()
    n_photons = pions.dataset.n_photons
    delta_t = []
    kbar = pkbar.Kbar(target=len(pions), width=20, always_stateful=False)
    for i,data in enumerate(pions):
        h = data[0]
        b = h.shape[0]
        h = h.reshape(int(b*n_photons),3).to('cuda').float()
        n_hits = data[3].numpy()
        c = data[1].reshape(int(b*n_photons),2).to('cuda').float()
        PID = data[2].numpy()
        unsc = data[4].numpy()[:,0,:]
        LL_k_geom = data[5].numpy()
        LL_pi_geom = data[6].numpy()
 

        
        with torch.set_grad_enabled(False):
            t_ = time.time()
            pion_hyp_pion = pion_net.log_prob(h,context=c).reshape(-1,n_photons)
            pion_hyp_pion = drop_and_sum(pion_hyp_pion,pion_hyp_pion.shape[0])
            pion_hyp_kaon = kaon_net.log_prob(h,context=c).reshape(-1,n_photons)
            pion_hyp_kaon = drop_and_sum(pion_hyp_kaon,pion_hyp_kaon.shape[0])
            delta_t.append((time.time() - t_) / b)

        assert len(pion_hyp_kaon) == len(pion_hyp_pion)
        assert len(unsc) == len(pion_hyp_pion)
        LL_Pion.append({"hyp_pion":pion_hyp_pion,"hyp_kaon":pion_hyp_kaon,"Truth":PID,"Kins":unsc,"Nhits":n_hits,"hyp_pion_geom":LL_pi_geom,"hyp_kaon_geom":LL_k_geom})#,"invMass":invMass})

        kbar.update(i)
        #if i == 5000:
        #    break

    end = time.time()
    print(" ")
    print("Elapsed time: ",end - start)
    print('Time / event: ',(end - start) / len(pions.dataset))
    print("Average GPU time: ",np.average(delta_t))

    print(' ')
    print('Starting DLL for Kaons')
    start = time.time()
    delta_t = []
    kbar = pkbar.Kbar(target=len(kaons), width=20, always_stateful=False)
    for i,data in enumerate(kaons):
        h = data[0]
        b = h.shape[0]
        h = h.reshape(int(b*n_photons),3).to('cuda').float()
        n_hits = data[3].numpy()
        c = data[1].reshape(int(b*n_photons),2).to('cuda').float()
        PID = data[2].numpy()
        unsc = data[4].numpy()[:,0,:]
        LL_k_geom = data[5].numpy()
        LL_pi_geom = data[6].numpy()
        
        with torch.set_grad_enabled(False):
            t_ = time.time()
            kaon_hyp_kaon = kaon_net.log_prob(h,context=c).reshape(-1,n_photons)
            kaon_hyp_kaon = drop_and_sum(kaon_hyp_kaon,kaon_hyp_kaon.shape[0])
            kaon_hyp_pion = pion_net.log_prob(h,context=c).reshape(-1,n_photons)
            kaon_hyp_pion = drop_and_sum(kaon_hyp_pion,kaon_hyp_pion.shape[0])
            delta_t.append((time.time() - t_) / b)

        assert len(kaon_hyp_kaon) == len(kaon_hyp_pion)
        assert len(unsc) == len(kaon_hyp_kaon)
        LL_Kaon.append({"hyp_kaon":kaon_hyp_kaon,"hyp_pion":kaon_hyp_pion,"Truth":PID,"Kins":unsc,"Nhits":n_hits,"hyp_pion_geom":LL_pi_geom,"hyp_kaon_geom":LL_k_geom})#,"invMass":invMass})

        kbar.update(i)

    end = time.time()
    print(" ")
    print("Elapsed time: ",end - start)
    print('Avg GPU Time / event: ',np.average(delta_t))

    return LL_Pion,LL_Kaon


def main(config,args):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    print("Running inference")
    assert config["method"] in ["Combined","Pion","Kaon"]

    datatype = config['datatype']
    num_layers = int(config['model_'+str(args.model_type)]['num_layers'])
    input_shape = int(config['model_'+str(args.model_type)]['input_shape'])
    cond_shape = int(config['model_'+str(args.model_type)]['cond_shape'])
    num_blocks = int(config['model_'+str(args.model_type)]['num_blocks'])
    hidden_nodes = int(config['model_'+str(args.model_type)]['hidden_nodes'])
    stats = config['stats']

    if not os.path.exists("Inference"):
        os.makedirs("Inference")

    if os.path.exists(os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum),"Kaon_DLL_Results.pkl")) and os.path.exists(os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum),"Pion_DLL_Results.pkl")):
        print("Found existing inference files. Skipping inference and only plotting.")
        LL_Kaon = np.load(os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum),"Kaon_DLL_Results.pkl"),allow_pickle=True)#[:10000]
        LL_Pion = np.load(os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum),"Pion_DLL_Results.pkl"),allow_pickle=True)#[:10000]
        print('Stats:',len(LL_Kaon),len(LL_Pion))
        out_folder = os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum))
        run_plotting(out_folder,args.momentum,args.model_type)

    else:
        if args.momentum == 9.0:
            test_pions = hpDIRC_DLL_Dataset(file_path=config['dataset']['fixed_point']["pion_data_path_9GeV"],time_cuts=args.time,stats=stats)
            test_kaons = hpDIRC_DLL_Dataset(file_path=config['dataset']['fixed_point']["kaon_data_path_9GeV"],time_cuts=args.time,stats=stats)
        elif args.momentum == 6.0:
            test_pions = hpDIRC_DLL_Dataset(file_path=config['dataset']['fixed_point']["pion_data_path_6GeV"],time_cuts=args.time,stats=stats)
            test_kaons = hpDIRC_DLL_Dataset(file_path=config['dataset']['fixed_point']["kaon_data_path_6GeV"],time_cuts=args.time,stats=stats)
        elif args.momentum == 3.0:
            test_pions = hpDIRC_DLL_Dataset(file_path=config['dataset']['fixed_point']["pion_data_path_3GeV"],time_cuts=args.time,stats=stats)
            test_kaons = hpDIRC_DLL_Dataset(file_path=config['dataset']['fixed_point']["kaon_data_path_3GeV"],time_cuts=args.time,stats=stats)
        else:
            raise ValueError("Momentum value not found.")
            

        print("# of Pions: ",len(test_pions))
        print("# of Kaons: ",len(test_kaons))

        pions = CreateInferenceLoader(test_pions,config) # Batch size is 1 untill I figure out a better way
        kaons = CreateInferenceLoader(test_kaons,config)

        if args.model_type == 'NF':
            pion_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats)
            kaon_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats)
        elif args.model_type == 'CNF':
            alph = config['model_'+str(args.model_type)]['alph']
            train_T = bool(config['model_'+str(args.model_type)]['train_T'])
            pion_net = OT_Flow(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,train_T=train_T,alph=alph)
            kaon_net  = OT_Flow(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,train_T=train_T,alph=alph)
        elif args.model_type == 'FlowMatching':
            pion_net = FlowMatching(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats)
            kaon_net = FlowMatching(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats)
        else:
            raise ValueError("Model type not found.")
            
        device = torch.device('cuda')
        pion_net.to('cuda')
        dicte = torch.load(config['Inference']['pion_model_path_'+str(args.model_type)])
        pion_net.load_state_dict(dicte['net_state_dict'])

        kaon_net.to('cuda')
        dicte = torch.load(config['Inference']['kaon_model_path_'+str(args.model_type)])
        kaon_net.load_state_dict(dicte['net_state_dict'])

        LL_Pion,LL_Kaon = run_inference_seperate(pions,kaons,pion_net,kaon_net)
        
        print('Inference plots can be found in: ' + config['Inference']['out_dir_fixed'])
        os.makedirs(config['Inference']['out_dir_fixed'],exist_ok=True)
        os.makedirs(os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum)),exist_ok=True)

        pion_path = os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum),"Pion_DLL_Results.pkl")
        with open(pion_path,"wb") as file:
            pickle.dump(LL_Pion,file)

        kaon_path = os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum),"Kaon_DLL_Results.pkl")
        with open(kaon_path,"wb") as file:
            pickle.dump(LL_Kaon,file)

        out_folder = os.path.join(config['Inference']['out_dir_fixed'],str(args.momentum))
        run_plotting(out_folder,args.momentum,args.model_type)



if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='DLL at fixed kinematics.')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-t','--time',default=None,type=float,
                        help='Maximum hit time for Cherenkov photons')
    parser.add_argument('-p','--momentum',default=6.0,type=float,help='Momentum value.')
    parser.add_argument('-m','--model_type',default='NF',type=str,help='Type of model to use.')
    args = parser.parse_args()

    config = json.load(open(args.config))

    if not os.path.exists("Inference"):
        print("Making Inference Directory.")
        os.makedirs("Inference",exist_ok=True)

    main(config,args)
