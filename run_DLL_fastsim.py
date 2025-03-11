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

def efficiency_func_momentum(probabilities, labels, momentum, method, out_dir):
    print("Computing efficiency as a function of momentum for varying mis-ID rates.")
    delta = 0.5
    momentum_bins = np.arange(2.0, np.max(momentum) + delta, delta)
    bin_centers = (momentum_bins[:-1] + momentum_bins[1:]) / 2
    fprs = [0.001, 0.01, 0.1]
    effs = []

    for i in range(len(momentum_bins) - 1):
        upper = momentum_bins[i + 1]
        lower = momentum_bins[i]
        idx = np.where((momentum <= upper) & (momentum > lower))[0]
        p = probabilities[idx]
        l = labels[idx]

        fpr, tpr, thresholds = roc_curve(l, p,drop_intermediate=False)

        tpr_interp_func = interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value='extrapolate')
        tpr_at_desired_fprs = [tpr_interp_func(desired_fpr) for desired_fpr in fprs]

        effs.append(tpr_at_desired_fprs)

    effs = np.array(effs)
    fig = plt.figure(figsize=(8, 4))
    colors = ['red', 'green', 'blue']

    for i in range(3):
        eff_ = effs[:, i]
        interp_ = interp1d(bin_centers,eff_,kind='slinear',bounds_error=False,fill_value='extrapolate')
        x = np.arange(2.0, np.max(momentum)+0.0001, 0.0001)
        interp_eff = interp_(x)
        plt.plot(x, interp_eff, color=colors[i], label='{0:.1f}%'.format(fprs[i] * 100),
                 linestyle='-', linewidth=1)
        plt.ylim(0,1)

    if method == 'NF':
        plt.title("Flow Based", fontsize=25, pad=20)
    elif method == "Geometric":
        plt.title("Geometric", fontsize=25, pad=20)
    else:
        print("Method not specified, exiting.")
        exit()

    plt.ylabel(r"$\mathcal{K}$ efficiency", fontsize=24)
    plt.xlabel(r"$ p \; [GeV/c]$", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='best', fontsize=20)
    plt.savefig(os.path.join(out_dir, "Efficiency_vs_Momentum_" + str(method) + ".pdf"), bbox_inches="tight")
    plt.close(fig)

def compute_efficiency_rejection(delta_log_likelihood, true_labels):
    thresholds = np.linspace(-4000.0, 4000.0, 20000)
    thresholds_broadcasted = np.expand_dims(thresholds, axis=1)
    predicted_labels = delta_log_likelihood > thresholds_broadcasted

    TP = np.sum((predicted_labels == 1) & (true_labels == 1), axis=1)
    FP = np.sum((predicted_labels == 1) & (true_labels == 0), axis=1)
    TN = np.sum((predicted_labels == 0) & (true_labels == 0), axis=1)
    FN = np.sum((predicted_labels == 0) & (true_labels == 1), axis=1)

    efficiencies = TP / (TP + FN)  
    rejections = TN / (TN + FP)  
    auc = np.trapezoid(y=np.flip(rejections),x=np.flip(efficiencies))

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
    except RuntimeError as e:
        print('Kaon error, exiting.')
        print(e)
        exit()
        

    hist_p, bin_edges_p = np.histogram(dll_p, bins=bins, density=True)
    bin_centers_p = (bin_edges_p[:-1] + bin_edges_p[1:]) / 2
    try:
        popt_p, pcov_p = curve_fit(gaussian, bin_centers_p, hist_p, p0=[1, np.mean(dll_p), np.std(dll_p)],maxfev=1000,bounds = ([0, -np.inf, 1e-9], [np.inf, np.inf, np.inf]))
        amplitude_p, mean_p, stddev_p = popt_p
    except RuntimeError as e:
        print('Pion error, exiting.')
        print(e)
        exit()
    
    sigma_sep = (mean_k - mean_p) / ((stddev_k + stddev_p)/2.) #np.sqrt(stddev_k**2 + stddev_p**2)

    return popt_k,popt_p,sigma_sep,bin_centers_k,bin_centers_p

def gaussian(x, amplitude, mean, stddev):
    return (1 / (np.sqrt(2*np.pi)*stddev))* np.exp(-((x - mean) / stddev) ** 2 / 2)
    #A*np.exp(-(x-mu)**2/(2.*sigma_squared))


def fine_grained_DLL(dll_k,dll_p,kin_k,kin_p,out_folder,dll_k_geom,dll_p_geom,kin_k_geom,kin_p_geom,sim_type):
    print("Running fine grained DLL analysis.")

    out_DLL_folder = os.path.join(out_folder,"DLL")
    if not os.path.exists(out_DLL_folder):
        os.mkdir(out_DLL_folder)

    print(kin_k.shape,kin_p.shape)
    if sim_type == "pgun":
        bounds = list(np.arange(np.min(kin_k[:,0]),np.max(kin_k[:,0]),0.1))
        bounds = np.array(bounds + [10.0])
    elif sim_type == "decays":
        bounds = list(np.arange(1.0,10.1,0.1))

    bound_centers = []
    sigma_NF = []
    sigma_geom = []

    theta_low = 0.
    delta_theta = 9999999.
    print(kin_k.shape,kin_p.shape)
    for k in range(len(bounds)-1):
        upper = bounds[k+1]
        lower = bounds[k]
        p_idx = np.where((kin_p[:,0] >= lower) & (kin_p[:,0] < upper))[0]
        k_idx = np.where((kin_k[:,0] >= lower) & (kin_k[:,0] < upper))[0]
        p_idx_geom = np.where((kin_p_geom[:,0] >= lower) & (kin_p_geom[:,0] < upper))[0]
        k_idx_geom = np.where((kin_k_geom[:,0] >= lower) & (kin_k_geom[:,0] < upper))[0]
        print('Kaons: ',len(k_idx)," Pions: ",len(p_idx)," for |p| in ({0:.2f},{1:.2f})".format(lower,upper))
        
        if len(k_idx) < 100 or len(p_idx) < 100:
            print('Skipping due to low stats.')
            continue

        # NF Method
        min_ = np.minimum.reduce([np.min(dll_k[k_idx]), np.min(dll_p[p_idx])])
        max_ = np.maximum.reduce([np.max(dll_k[k_idx]),np.max(dll_p[p_idx])])
        min_ = max(-500,min_)
        max_ = min(500,max_)

        bins = np.linspace(min_,max_,400)
        popt_k_NF,popt_p_NF,sep_NF,bin_centers_k_NF,bin_centers_p_NF = perform_fit(dll_k[k_idx],dll_p[p_idx],bins)
        sigma_NF.append(sep_NF)
        print(upper,lower,sep_NF)
        # Geometrical  Method 
        min_ = np.minimum.reduce([np.min(dll_k_geom[k_idx_geom]),np.min(dll_p_geom[p_idx_geom])])
        max_ = np.maximum.reduce([np.max(dll_k_geom[k_idx_geom]),np.max(dll_p_geom[p_idx_geom])])
        min_ = max(-2500,min_)
        max_ = min(1000.,max_)
        bins = np.linspace(min_,max_,400)
        #popt_k_geom,popt_p_geom,sep_geom,bin_centers_k_geom,bin_centers_p_geom = perform_fit(dll_k_geom[k_idx_geom],dll_p_geom[p_idx_geom],bins)
        popt_k_geom,popt_p_geom,sep_geom,bin_centers_k_geom,bin_centers_p_geom = popt_k_NF,popt_p_NF,sep_NF,bin_centers_k_NF,bin_centers_p_NF
        sigma_geom.append(sep_geom)
        bound_centers.append((upper + lower)/2.0)

        fig,ax = plt.subplots(1,2,figsize=(12,4))
        ax = ax.ravel()
        min_ = np.minimum.reduce([np.min(dll_k[k_idx]), np.min(dll_p[p_idx])])
        max_ = np.maximum.reduce([np.max(dll_k[k_idx]),np.max(dll_p[p_idx])])
        min_ = max(-500,min_)
        max_ = min(500,max_)
        ax[0].hist(dll_k[k_idx],bins=400,density=True,alpha=1.,range=[min_,max_],label=r'$\mathcal{K}_{NF.}$',color='red',histtype='step',lw=3)
        ax[0].hist(dll_p[p_idx],bins=400,density=True,range=[min_,max_],alpha=1.0,label=r'$\pi$',color='blue',histtype='step',lw=3)
        ax[0].set_xlabel('Loglikelihood Difference',fontsize=25)
        ax[0].set_ylabel('A.U.',fontsize=25)
        ax[0].set_title(r'$ \Delta \mathcal{L}_{K \pi}$' + r'$|\vec{p}| \in $'+'({0:.2f},{1:.2f}) GeV'.format(lower,upper),fontsize=25)

        ax[1].plot(bin_centers_k_NF, gaussian(bin_centers_k_NF, *popt_k_NF),color='red', label=r"$\mathcal{K}_{NF.}$: " +r"$\mu={0:.2f}, \sigma={1:.2f}$".format(popt_k_NF[1],popt_k_NF[2]))
        ax[1].plot(bin_centers_p_NF, gaussian(bin_centers_p_NF, *popt_p_NF),color='blue', label=r"$\pi_{NF.}$: " +r"$\mu={0:.2f}, \sigma={1:.2f}$".format(popt_p_NF[1],popt_p_NF[2]))
        ax[1].set_xlabel('Fitted Loglikelihood Difference',fontsize=25)
        ax[1].set_ylabel('A.U.',fontsize=25)
        ax[1].legend(fontsize=20,loc=(1.01,0.4))
        ax[1].set_title(r'$ \Delta \mathcal{L}_{K \pi}$' + r'$|\vec{p}| \in $'+'({0:.2f},{1:.2f}) GeV'.format(lower,upper),fontsize=25)
        plt.subplots_adjust(wspace=0.3)

        out_path_DLL = os.path.join(out_DLL_folder,"DLL_piK_p({0:.2f},{1:.2f}).pdf".format(lower,upper))
        plt.savefig(out_path_DLL,bbox_inches='tight')
        plt.close()

    return sigma_NF, sigma_geom,bound_centers

def plot_DLL(kaons,pions,out_folder,datatype,sim_type,extension):
    print(out_folder)
    #     # This function is gross need to rewrite it
    #     # Kaon label = 1
    #     # Pion label = 0
    if datatype == 'Simulation':
        real = False
    elif datatype == 'Real':
        real = True
    else:
        print("Are you using simulation or real data? Specify in the config file under datatype")
        exit()

    dll_k = []
    dll_p = []
    kin_p = []
    kin_k = []
    predicted_pions = []
    predicted_kaons = []
    kaon_labels = []
    pion_labels = []
    nhits_k = []
    nhits_pi = []
    ll_p = []
    ll_k = []
    dll_k_geom = []
    dll_p_geom = []
    for i in range(len(kaons)):
        dll_k.append((np.array(kaons[i]['hyp_kaon']) - np.array(kaons[i]['hyp_pion'])).flatten())
        dll_k_geom.append((np.array(kaons[i]['hyp_kaon_geom']) - np.array(kaons[i]['hyp_pion_geom'])).flatten())
        ll_k.append(np.array(kaons[i]['hyp_kaon']).flatten())
        kin_k.append(kaons[i]['Kins'])
        nhits_k.append(kaons[i]['Nhits'])


    for i in range(len(pions)):
        dll_p.append((np.array(pions[i]['hyp_kaon']) - np.array(pions[i]['hyp_pion'])).flatten())
        dll_p_geom.append((np.array(pions[i]['hyp_kaon_geom']) - np.array(pions[i]['hyp_pion_geom'])).flatten())
        ll_p.append(np.array(pions[i]['hyp_pion']).flatten())
        kin_p.append(pions[i]['Kins'])
        nhits_pi.append(pions[i]['Nhits'])

    kin_p = np.concatenate(kin_p)
    kin_k = np.concatenate(kin_k)
    dll_p = np.concatenate(dll_p)
    dll_k = np.concatenate(dll_k)
    print(np.isnan(dll_k).sum())
    print(np.isnan(dll_p).sum())
    dll_k = np.clip(dll_k[~np.isnan(dll_k)],-99999,99999)
    dll_p = np.clip(dll_p[~np.isnan(dll_p)],-99999,99999)
    kin_k =  kin_k[~np.isnan(dll_k)]
    kin_p = kin_p[~np.isnan(dll_p)]

    dll_k_geom = np.concatenate(dll_k_geom)
    dll_p_geom = np.concatenate(dll_p_geom)
    dll_k_geom = np.clip(dll_k_geom[~np.isnan(dll_k_geom)],-99999,99999)
    dll_p_geom = np.clip(dll_p_geom[~np.isnan(dll_p_geom)],-99999,99999)
    kin_k_geom = kin_k[~np.isnan(dll_k_geom)] 
    kin_p_geom = kin_p[~np.isnan(dll_p_geom)]


    nhits_k = np.concatenate(nhits_k)
    nhits_pi = np.concatenate(nhits_pi)
    ll_k = np.concatenate(ll_k)
    ll_p = np.concatenate(ll_p)
    ll_k = ll_k[np.where((kin_k[:,0] > 3.0) & (kin_k[:,0] < 3.5))[0]]
    nhits_k = nhits_k[np.where((kin_k[:,0] > 3.0) & (kin_k[:,0] < 3.5))[0]]
    ll_p = ll_p[np.where((kin_p[:,0] > 3.0) & (kin_p[:,0] < 3.5))[0]]
    nhits_pi = nhits_pi[np.where((kin_p[:,0] > 3.0) & (kin_p[:,0] < 3.5))[0]]

    if (sim_type == 'decays') or (sim_type == 'pgun'):
        # from NF first
        idx_ = np.where((kin_k[:,0] > 1.0) & (kin_k[:,0] < 10.))
        dll_k = dll_k[idx_]
        kin_k = kin_k[idx_]
        idx_ = np.where((kin_p[:,0] > 1.0) & (kin_p[:,0] < 10.))
        dll_p = dll_p[idx_]
        kin_p = kin_p[idx_]

        # Geom - defaults to zero as of now
        idx_ = np.where((kin_k_geom[:,0] > 1.0) & (kin_k_geom[:,0] < 10.))
        dll_k_geom = dll_k_geom[idx_]
        kin_k_geom = kin_k_geom[idx_]
        idx_ = np.where((kin_p_geom[:,0] > 1.0) & (kin_p_geom[:,0] < 10))
        dll_p_geom = dll_p_geom[idx_]
        kin_p_geom = kin_p_geom[idx_]


    # DLL fits in P bins
    sep_NF, sep_geom,bound_centers = fine_grained_DLL(dll_k,dll_p,kin_k,kin_p,out_folder,dll_k_geom,dll_p_geom,kin_k_geom,kin_p_geom,sim_type)
    merge_PDF(out_folder)

    plt.plot(bound_centers,sep_NF,label=r"$\sigma_{NF.}$",color='red',marker='o')
    plt.legend(loc='upper right',fontsize=20)
    plt.xlabel("Momentum [GeV/c]",fontsize=20)
    plt.ylabel(r"$\sigma$",fontsize=20)
    plt.xticks(fontsize=18)  
    plt.yticks(fontsize=18) 
    plt.title(r"$\sigma_{sep.}$ as a function of Momentum",fontsize=20)
    plt.savefig(os.path.join(out_folder,f'Seperation_Average_{extension}.pdf'),bbox_inches='tight')
    plt.close()

    # DLL over phase space
    plt.hist(dll_k,bins=100,density=True,alpha=1.,range=[-250,250],label=r'$\mathcal{K}_{NF.}$',color='red',histtype='step',lw=2)
    plt.hist(dll_p,bins=100,density=True,range=[-250,250],alpha=1.0,label=r'$\pi_{NF.}$',color='blue',histtype='step',lw=2)
    plt.xlabel('Loglikelihood Difference',fontsize=25)
    plt.ylabel('A.U.',fontsize=25)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(r'$ \Delta \mathcal{L}_{\mathcal{K} \pi}$',fontsize=30)
    out_path_DLL = os.path.join(out_folder,f"DLL_piK_{extension}.pdf")
    plt.savefig(out_path_DLL,bbox_inches='tight')
    plt.close()

    # LL vs photon yield
    if real:
        r = [(0,250),(-500,100)]
    else:
        r = [(0,230),(-50,600)]
    plt.hist2d(nhits_k, ll_k, bins=[100,50], cmap='plasma', norm=LogNorm(), range=r)
    plt.title(r'$\log \mathcal{L}_{\mathcal{K}}$ as a function of $N_{\gamma_c}$', fontsize=30, pad=10)
    plt.xlabel(r'$N_{\gamma_c}$', fontsize=25)
    plt.ylabel(r'$\log \mathcal{L}_{\mathcal{K}}$', fontsize=25)
    text_x, text_y = 0.84, 0.12
    text = r'$\mathcal{K}^{+-}$'
    plt.text(text_x, text_y, text, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=30)
    text_bbox = plt.text(text_x, text_y, text, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=30, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    out_path_khits = os.path.join(out_folder,f"Kaons_LL_function_of_NHits_{extension}.pdf")
    plt.savefig(out_path_khits,bbox_inches="tight")
    plt.close()


    plt.hist2d(nhits_pi, ll_p, bins=[100,50], cmap='plasma', norm=LogNorm(), range=r)
    plt.title(r'$\log \mathcal{L}_{\pi}$ as a function of $N_{\gamma_c}$', fontsize=30, pad=10)
    plt.xlabel(r'$N_{\gamma_c}$', fontsize=25)
    plt.ylabel(r'$\log \mathcal{L}_{\pi}$', fontsize=25)
    text_x, text_y = 0.84, 0.12
    text = r'$\pi^{+-}$'
    plt.text(text_x, text_y, text, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=30)
    text_bbox = plt.text(text_x, text_y, text, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=30, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    out_path_phits = os.path.join(out_folder,f"Pions_DLL_function_of_NHits_{extension}.pdf")
    plt.savefig(out_path_phits,bbox_inches="tight")
    plt.close()

    print(np.max(dll_k))
    print(np.max(dll_p))
    print(np.max(dll_k_geom))
    print(np.max(dll_p_geom))
    print(np.min(dll_k))
    print(np.min(dll_p))
    print(np.min(dll_k_geom))
    print(np.min(dll_p_geom))
    print(np.isnan(dll_k).sum())
    print(np.isnan(dll_p).sum())
    print(np.isnan(dll_k_geom).sum())
    print(np.isnan(dll_p_geom).sum())
    
    true_labels = np.array(list(np.ones_like(dll_k)) + list(np.zeros_like(dll_p)))
    delta_log_likelihood = np.array(list(dll_k) + list(dll_p))
    total_conds = np.array(list(kin_k) + list(kin_p))
    print("NF:",len(true_labels))
    efficiency_func_momentum(sigmoid(delta_log_likelihood),true_labels, total_conds[:,0],"NF",out_folder)

    true_labels_geom = np.array(list(np.ones_like(dll_k_geom)) + list(np.zeros_like(dll_p_geom)))
    delta_log_likelihood_geom = np.array(list(dll_k_geom) + list(dll_p_geom))
    total_conds_geom = np.array(list(kin_k_geom) + list(kin_p_geom))
    print(len(true_labels_geom))
    efficiency_func_momentum(sigmoid(delta_log_likelihood_geom),true_labels_geom, total_conds_geom[:,0],"Geometric",out_folder)
    
    efficiencies, rejections,auc = compute_efficiency_rejection(delta_log_likelihood, true_labels)

    efficiencies_geom, rejections_geom, auc_geom = compute_efficiency_rejection(delta_log_likelihood_geom, true_labels_geom)

    dicte_ = {"eff":efficiencies,"rej":rejections,"auc":auc}

    with open(os.path.join(out_folder,f"ROC_AUC_{extension}.pkl"),"wb") as file:
        pickle.dump(dicte_,file)


    # ROC Curve
    plt.figure(figsize=(8,8))
    plt.plot(rejections,efficiencies,color='red', lw=2, label='NF-DLL. AUC = %0.3f' % auc)
    plot_swin = False
    plt.plot([0, 1], [1, 0], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(r'kaon efficiency',fontsize=30,labelpad=10)
    plt.ylabel(r'pion rejection',fontsize=30,labelpad=10) 
    plt.legend(loc="lower left",fontsize=18)
    plt.ylim(0,1)
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)
    out_path_DLL_ROC = os.path.join(out_folder,f"DLL_piK_ROC_{extension}.pdf")
    plt.savefig(out_path_DLL_ROC,bbox_inches='tight')
    plt.close()


    # ROC as a function of momentum
    if sim_type == 'pgun':
        mom_ranges = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0]
    elif sim_type == 'decays':
        mom_ranges = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5]
    else:
        print("")
        print("Please ensure sim_type is correctly set in the config file.")
        print("1. pgun")
        print("2. decays")
        exit()

    centers = [mr+0.25 for mr in mom_ranges[:-1]]
    aucs = []
    aucs_upper = []
    aucs_lower = []
    aucs_geom = []
    aucs_geom_upper = []
    aucs_geom_lower = []
    lengths = []
    n_kaons = []
    n_pions = []
    for i in range(len(mom_ranges) - 1):
        mom_low = mom_ranges[i]
        mom_high = mom_ranges[i+1]
        idx = np.where((total_conds[:,0] > mom_low) & (total_conds[:,0] < mom_high))[0]
        idx_geom = np.where((total_conds_geom[:,0] > mom_low) & (total_conds_geom[:,0] < mom_high))[0]
        p = np.array(delta_log_likelihood)[idx]
        p_geom = np.array(delta_log_likelihood_geom)[idx_geom]
        t = np.array(true_labels)[idx]
        t_geom = np.array(true_labels_geom)[idx_geom]
        print("Momentum Range: ",mom_low,"-",mom_high)
        print("# Kaons: ",len(t[t==1]))
        n_kaons.append(len(t[t==1]))
        n_pions.append(len(t[t==0]))
        print("# Pions: ",len(t[t==0]))
        lengths.append(len(p))
        eff,rej,_ = compute_efficiency_rejection(p,t)
        eff_geom,rej_geom,_= compute_efficiency_rejection(p_geom,t_geom)
        AUC = []
        AUC_geom = []
        sigma_eff = np.sqrt(eff * (1.0 - eff) / len(t[t == 1]))
        sigma_rej = np.sqrt(rej * (1.0 - rej) / len(t[t == 0]))
        sigma_eff_geom = np.sqrt(eff_geom * (1.0 - eff_geom) / len(t_geom[t_geom == 1]))
        sigma_rej_geom = np.sqrt(rej_geom * (1.0 - rej_geom) / len(t_geom[t_geom == 0]))

        for _ in range(1000):
            eff_ = np.random.normal(eff,sigma_eff)
            rej_ = np.random.normal(rej,sigma_rej)
            eff_geom_ = np.random.normal(eff_geom,sigma_eff_geom)
            rej_geom_ = np.random.normal(rej_geom,sigma_rej_geom)

            AUC.append(np.trapezoid(y=np.flip(rej_),x=np.flip(eff_)))
            AUC_geom.append(np.trapezoid(y=np.flip(rej_geom_),x=np.flip(eff_geom_)))


        aucs.append(np.mean(AUC))
        aucs_geom.append(np.mean(AUC_geom))

        aucs_upper.append(np.percentile(AUC,97.5))
        aucs_lower.append(np.percentile(AUC,2.5))

        aucs_geom_upper.append(np.percentile(AUC_geom,97.5))
        aucs_geom_lower.append(np.percentile(AUC_geom,2.5))
        print("NF-> Mean AUC: ",np.mean(AUC)," 95%",np.percentile(AUC,2.5),"-",np.percentile(AUC,97.5))
        print("Geom. -> Mean AUC: ",np.mean(AUC_geom)," 95%",np.percentile(AUC_geom,2.5),"-",np.percentile(AUC_geom,97.5))

    fig = plt.figure(figsize=(8,8))
    plt.errorbar(centers,aucs,yerr=[np.array(aucs) - np.array(aucs_lower),np.array(aucs_upper) - np.array(aucs)],label=r"$AUC_{NF-DLL.}$",color='red',marker='o',capsize=5)
    legend1 = plt.legend(loc='lower left', fontsize=24)
    legend1.get_frame().set_facecolor('white')  
    legend1.get_frame().set_edgecolor('grey')  
    legend1.get_frame().set_alpha(1.0)  
    plt.xlabel("momentum [GeV/c]",fontsize=30,labelpad=10)
    plt.ylabel("AUC",fontsize=30,labelpad=10)
    plt.xticks(fontsize=22)  
    plt.yticks(fontsize=22)  
    if np.min(aucs) < np.min(aucs_geom):
        min_aucs = np.min(aucs)
    else:
        min_aucs = np.min(aucs_geom)
    if np.max(aucs) > np.max(aucs_geom):
        max_aucs = np.max(aucs)
    else:
        max_aucs = np.max(aucs_geom)

    plt.ylim(min_aucs - 0.05,max_aucs + 0.05)

    ax2 = plt.twinx()

    # Plot bars for pions and kaons
    ax2.bar(np.array(centers) - 0.1, n_pions, width=0.2, label='Pions', color='blue', alpha=0.25)
    ax2.bar(np.array(centers) + 0.1, n_kaons, width=0.2, label='Kaons', color='green', alpha=0.25)
    ax2.set_ylabel('Counts', fontsize=30,labelpad=10)
    ax2.tick_params(axis='y', labelsize=20)
    legend2 = ax2.legend(loc='upper right', fontsize=24)
    legend2.get_frame().set_facecolor('white')  
    legend2.get_frame().set_edgecolor('grey')  
    legend2.get_frame().set_alpha(1.0)  
    out_path_AUC_func_P = os.path.join(out_folder,f"DLL_AUC_func_P_{extension}.pdf")
    plt.savefig(out_path_AUC_func_P,bbox_inches='tight')
    plt.close()


    dicte_ = {"n_pions":n_pions,"n_kaons":n_kaons,
             "aucs":aucs,"aucs_lower":aucs_lower,"aucs_upper":aucs_upper}

    with open(os.path.join(out_folder,f"DLL_AUC_func_P_{extension}.pkl"),"wb") as file:
        pickle.dump(dicte_,file)

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
    assert config["method"] in ["Pion","Kaon"]

    datatype = config['datatype']
    num_layers = int(config['model_CNF']['num_layers'])
    input_shape = int(config['model_CNF']['input_shape'])
    cond_shape = int(config['model_CNF']['cond_shape'])
    num_blocks = int(config['model_CNF']['num_blocks'])
    hidden_nodes = int(config['model_CNF']['hidden_nodes'])
    stats = config['stats']

    alph = config['model_CNF']['alph']
    train_T = bool(config['model_CNF']['train_T'])
    pion_net = OT_Flow(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,train_T=train_T,alph=alph)
    kaon_net  = OT_Flow(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,stats=stats,train_T=train_T,alph=alph)
    device = torch.device('cuda')
    pion_net.to('cuda')
    dicte = torch.load(config['Inference']['pion_model_path_CNF'])
    pion_net.load_state_dict(dicte['net_state_dict'])

    kaon_net.to('cuda')
    dicte = torch.load(config['Inference']['kaon_model_path_CNF'])
    kaon_net.load_state_dict(dicte['net_state_dict'])

    if not os.path.exists("Inference"):
        os.makedirs("Inference")

    if args.momentum == -1:
        nested_dir = "FullPhaseSpace"
    else:
        nested_dir = str(args.momentum)


    # Run for Fast Simulated Data.
    print("------------------ Fast Simulation -----------------")
    pion_exist = os.path.exists(os.path.join(config['Inference']['out_dir_fixed'],str(nested_dir),"Pion_DLL_Results_FastSim.pkl"))
    kaon_exist = os.path.exists(os.path.join(config['Inference']['out_dir_fixed'],str(nested_dir),"Kaon_DLL_Results_FastSim.pkl"))
    
    if not pion_exist and kaon_exist:
        if args.momentum == 9.0:
            test_pions = hpDIRC_DLL_Dataset(path_=config['dataset']['fixed_point_fs']["data_path_9GeV"],time_cuts=args.time,stats=stats,fast_sim_comp=True,fast_sim_type="Pion",geant=False)
            test_kaons = hpDIRC_DLL_Dataset(path_=config['dataset']['fixed_point_fs']["data_path_9GeV"],time_cuts=args.time,stats=stats,fast_sim_comp=True,fast_sim_type="Kaon",geant=False)
        elif args.momentum == 6.0:
            test_pions = hpDIRC_DLL_Dataset(path_=config['dataset']['fixed_point_fs']["data_path_6GeV"],time_cuts=args.time,stats=stats,fast_sim_comp=True,fast_sim_type="Pion",geant=False)
            test_kaons = hpDIRC_DLL_Dataset(path_=config['dataset']['fixed_point_fs']["data_path_6GeV"],time_cuts=args.time,stats=stats,fast_sim_comp=True,fast_sim_type="Kaon",geant=False)
        elif args.momentum == 3.0:
            test_pions = hpDIRC_DLL_Dataset(path_=config['dataset']['fixed_point_fs']["data_path_3GeV"],time_cuts=args.time,stats=stats,fast_sim_comp=True,fast_sim_type="Pion",geant=False)
            test_kaons = hpDIRC_DLL_Dataset(path_=config['dataset']['fixed_point_fs']["data_path_3GeV"],time_cuts=args.time,stats=stats,fast_sim_comp=True,fast_sim_type="Kaon",geant=False)
        elif args.momentum == -1:
            test_pions = hpDIRC_DLL_Dataset(path_=config['dataset']['fixed_point_fs']["data_path_full"],time_cuts=args.time,stats=stats,fast_sim_comp=True,fast_sim_type="Pion",geant=False)
            test_kaons = hpDIRC_DLL_Dataset(path_=config['dataset']['fixed_point_fs']["data_path_full"],time_cuts=args.time,stats=stats,fast_sim_comp=True,fast_sim_type="Kaon",geant=False)       
        else:
            raise ValueError("Momentum value not found.")
            
        print("# of Pions: ",len(test_pions))
        print("# of Kaons: ",len(test_kaons))

        pions = CreateInferenceLoader(test_pions,config) 
        kaons = CreateInferenceLoader(test_kaons,config)

        LL_Pion,LL_Kaon = run_inference_seperate(pions,kaons,pion_net,kaon_net)

        print('Inference data can be found in: ' + config['Inference']['out_dir_fixed'])
        os.makedirs(config['Inference']['out_dir_fixed'],exist_ok=True)
        os.makedirs(os.path.join(config['Inference']['out_dir_fixed'],str(nested_dir)),exist_ok=True)

        pion_path = os.path.join(config['Inference']['out_dir_fixed'],str(nested_dir),"Pion_DLL_Results_FastSim.pkl")
        with open(pion_path,"wb") as file:
            pickle.dump(LL_Pion,file)

        kaon_path = os.path.join(config['Inference']['out_dir_fixed'],str(nested_dir),"Kaon_DLL_Results_FastSim.pkl")
        with open(kaon_path,"wb") as file:
            pickle.dump(LL_Kaon,file)

        print(" ")

    else:
        print("Found existing inference files for fast sim.")
        pion_path = os.path.join(config['Inference']['out_dir_fixed'],str(nested_dir),"Pion_DLL_Results_FastSim.pkl")
        LL_Pion = np.load(pion_path,allow_pickle=True)
        kaon_path = os.path.join(config['Inference']['out_dir_fixed'],str(nested_dir),"Kaon_DLL_Results_FastSim.pkl")
        LL_Kaon = np.load(kaon_path,allow_pickle=True)

    if args.full_phase_space:
        sim_type = config['sim_type']
        plot_DLL(LL_Kaon,LL_Pion,config['Inference']['out_dir_fixed'],datatype,sim_type,extension="FastSim")


    print("------------------ Geant4 -----------------")
    pion_exist = os.path.exists(os.path.join(config['Inference']['out_dir_fixed'],str(nested_dir),"Pion_DLL_Results_Geant.pkl"))
    kaon_exist = os.path.exists(os.path.join(config['Inference']['out_dir_fixed'],str(nested_dir),"Kaon_DLL_Results_Geant.pkl"))
    
    if not pion_exist and kaon_exist:
        if args.momentum == 9.0:
            test_pions = hpDIRC_DLL_Dataset(path_=config['dataset']['fixed_point_fs']["data_path_9GeV"],time_cuts=args.time,stats=stats,fast_sim_comp=True,fast_sim_type="Pion",geant=True)
            test_kaons = hpDIRC_DLL_Dataset(path_=config['dataset']['fixed_point_fs']["data_path_9GeV"],time_cuts=args.time,stats=stats,fast_sim_comp=True,fast_sim_type="Kaon",geant=True)
        elif args.momentum == 6.0:
            test_pions = hpDIRC_DLL_Dataset(path_=config['dataset']['fixed_point_fs']["data_path_6GeV"],time_cuts=args.time,stats=stats,fast_sim_comp=True,fast_sim_type="Pion",geant=True)
            test_kaons = hpDIRC_DLL_Dataset(path_=config['dataset']['fixed_point_fs']["data_path_6GeV"],time_cuts=args.time,stats=stats,fast_sim_comp=True,fast_sim_type="Kaon",geant=True)
        elif args.momentum == 3.0:
            test_pions = hpDIRC_DLL_Dataset(path_=config['dataset']['fixed_point_fs']["data_path_3GeV"],time_cuts=args.time,stats=stats,fast_sim_comp=True,fast_sim_type="Pion",geant=True)
            test_kaons = hpDIRC_DLL_Dataset(path_=config['dataset']['fixed_point_fs']["data_path_3GeV"],time_cuts=args.time,stats=stats,fast_sim_comp=True,fast_sim_type="Kaon",geant=True)
        elif args.momentum == -1:
            test_pions = hpDIRC_DLL_Dataset(path_=config['dataset']['fixed_point_fs']["data_path_full"],time_cuts=args.time,stats=stats,fast_sim_comp=True,fast_sim_type="Pion",geant=True)
            test_kaons = hpDIRC_DLL_Dataset(path_=config['dataset']['fixed_point_fs']["data_path_full"],time_cuts=args.time,stats=stats,fast_sim_comp=True,fast_sim_type="Kaon",geant=True)       
        else:
            raise ValueError("Momentum value not found.")
            
        print("# of Pions: ",len(test_pions))
        print("# of Kaons: ",len(test_kaons))

        pions = CreateInferenceLoader(test_pions,config) 
        kaons = CreateInferenceLoader(test_kaons,config)

        
        LL_Pion,LL_Kaon = run_inference_seperate(pions,kaons,pion_net,kaon_net)
        
        print('Inference data can be found in: ' + config['Inference']['out_dir_fixed'])
        os.makedirs(config['Inference']['out_dir_fixed'],exist_ok=True)


        os.makedirs(os.path.join(config['Inference']['out_dir_fixed'],str(nested_dir)),exist_ok=True)

        pion_path = os.path.join(config['Inference']['out_dir_fixed'],str(nested_dir),"Pion_DLL_Results_Geant.pkl")
        with open(pion_path,"wb") as file:
            pickle.dump(LL_Pion,file)

        kaon_path = os.path.join(config['Inference']['out_dir_fixed'],str(nested_dir),"Kaon_DLL_Results_Geant.pkl")
        with open(kaon_path,"wb") as file:
            pickle.dump(LL_Kaon,file)

    else:
        print("Found existing inference file for Geant.")
        pion_path = os.path.join(config['Inference']['out_dir_fixed'],str(nested_dir),"Pion_DLL_Results_Geant.pkl")
        LL_Pion = np.load(pion_path,allow_pickle=True)
        kaon_path = os.path.join(config['Inference']['out_dir_fixed'],str(nested_dir),"Kaon_DLL_Results_Geant.pkl")
        LL_Kaon = np.load(kaon_path,allow_pickle=True)

    if args.full_phase_space:
        sim_type = config['sim_type']
        plot_DLL(LL_Kaon,LL_Pion,config['Inference']['out_dir_fixed'],datatype,sim_type,extension="Geant")

        


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='DLL at fixed kinematics.')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-t','--time',default=None,type=float,
                        help='Maximum hit time for Cherenkov photons')
    parser.add_argument('-p','--momentum',default=-1,type=float,help='Momentum value.')
    parser.add_argument('-f','--full_phase_space', action='store_false', help="DLL over continuous phase space.")
    args = parser.parse_args()

    config = json.load(open(args.config))

    if not os.path.exists("Inference"):
        print("Making Inference Directory.")
        os.makedirs("Inference",exist_ok=True)

    main(config,args)
