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
from models.freia_models import FreiaNet
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn import metrics
from scipy.optimize import curve_fit
import glob
from PyPDF2 import PdfWriter
from scipy.stats import norm
from matplotlib.colors import LogNorm
from sklearn.neighbors import KernelDensity
from FastDIRC_utils.FastDIRC import FastDIRC

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


def fine_grained_DLL(dll_k,dll_p,kin_k,kin_p,out_folder):
    print("Running fine grained DLL analysis.")

    def gaussian(x, amplitude, mean, stddev):
        return (1 / (np.sqrt(2*np.pi)*stddev))* np.exp(-((x - mean) / stddev) ** 2 / 2)
        #A*np.exp(-(x-mu)**2/(2.*sigma_squared))

    out_DLL_folder = os.path.join(out_folder,"DLL")
    if not os.path.exists(out_DLL_folder):
        os.mkdir(out_DLL_folder)

    bins = 100
    bounds = list(np.arange(np.min(kin_k[:,0]),np.max(kin_k[:,0]),0.1))
    bounds = np.array(bounds + [8.5])
    bound_centers = []
    sigma_seps = []

    for k in range(len(bounds)-1):
        upper = bounds[k+1]
        lower = bounds[k]
        p_idx = np.where((kin_p[:,0] >= lower) & (kin_p[:,0] < upper))[0]
        k_idx = np.where((kin_k[:,0] >= lower) & (kin_k[:,0] < upper))[0]
        print('Kaons: ',len(k_idx)," Pions: ",len(p_idx)," for |p| in ({0:.2f},{1:.2f})".format(lower,upper))
        
        if len(k_idx) < 100 or len(p_idx) < 100:
            print('Skipping due to low stats.')
            continue


        hist_k, bin_edges_k = np.histogram(dll_k[k_idx], bins=bins, density=True,range=[-50,50])
        bin_centers_k = (bin_edges_k[:-1] + bin_edges_k[1:]) / 2
        try:
            popt_k, pcov_k = curve_fit(gaussian, bin_centers_k, hist_k, p0=[1, np.mean(dll_k[k_idx]), np.std(dll_k[k_idx])],maxfev=1000,bounds = ([0, -np.inf, 1e-9], [np.inf, np.inf, np.inf]))
            amplitude_k, mean_k, stddev_k = popt_k
        except RuntimeError as e:
            print('Kaon error, skipping.')
            print(e)
            continue

        hist_p, bin_edges_p = np.histogram(dll_p[p_idx], bins=bins, density=True,range=[-50,50])
        bin_centers_p = (bin_edges_p[:-1] + bin_edges_p[1:]) / 2
        try:
            popt_p, pcov_p = curve_fit(gaussian, bin_centers_p, hist_p, p0=[1, np.mean(dll_p[p_idx]), np.std(dll_p[p_idx])],maxfev=1000,bounds = ([0, -np.inf, 1e-9], [np.inf, np.inf, np.inf]))
            amplitude_p, mean_p, stddev_p = popt_p
        except RuntimeError as e:
            print('Pion error, skipping.')
            print(e)
            continue
        
        sigma_sep = (mean_k - mean_p) / ((stddev_k + stddev_p)/2.) #np.sqrt(stddev_k**2 + stddev_p**2)
        sigma_seps.append(sigma_sep)
        bound_centers.append((upper + lower)/2.0)

        fig,ax = plt.subplots(1,2,figsize=(12,4))
        ax = ax.ravel()
        ax[0].hist(dll_k[k_idx],bins=bins,density=True,alpha=1.,range=[-50,50],label=r'$\mathcal{K}$',color='red',histtype='step',lw=3)
        ax[0].hist(dll_p[p_idx],bins=bins,density=True,range=[-50,50],alpha=1.0,label=r'$\pi$',color='blue',histtype='step',lw=3)
        ax[0].set_xlabel('Loglikelihood Difference',fontsize=25)
        ax[0].set_ylabel('A.U.',fontsize=25)
        ax[0].legend(fontsize=20)
        ax[0].set_title(r'$ \Delta \mathcal{L}_{K \pi}$',fontsize=30)

        ax[1].plot(bin_centers_k, gaussian(bin_centers_k, *popt_k),color='red', label=r"$\mathcal{K}$: " +r"$\mu={0:.2f}, \sigma={1:.2f}$".format(mean_k,stddev_k))
        ax[1].plot(bin_centers_p, gaussian(bin_centers_p, *popt_p),color='blue', label=r"$\pi$: " +r"$\mu={0:.2f}, \sigma={1:.2f}$".format(mean_p,stddev_p))
        ax[1].set_xlabel('Fitted Loglikelihood Difference',fontsize=25)
        ax[1].set_ylabel('A.U.',fontsize=25)
        ax[1].legend(fontsize=20,loc=(1.01,0.5))
        ax[1].set_title(r'$ \Delta \mathcal{L}_{K \pi}$',fontsize=30)
        ax[1].text(85, 0.013, r'$\sigma_{sep.} =$'+'{0:.2f}'.format(sigma_sep), fontsize=18, ha='center', va='center')
        ax[1].text(100,0.005,  r'$|\vec{p}| \in $'+'({0:.2f},{1:.2f}) GeV'.format(lower,upper), fontsize=18, ha='center', va='center')
        plt.subplots_adjust(wspace=0.3)

        out_path_DLL = os.path.join(out_DLL_folder,"DLL_piK_p({0:.2f},{1:.2f}).pdf".format(lower,upper))
        plt.savefig(out_path_DLL,bbox_inches='tight')
        plt.close()

    plt.plot(bound_centers,sigma_seps,label=r"$\sigma_{sep.}$",color='red',marker='o')
    plt.legend(loc='upper right',fontsize=20)
    plt.xlabel("Momentum [GeV/c]",fontsize=20)
    plt.ylabel(r"$\sigma$",fontsize=20)
    plt.xticks(fontsize=18)  # adjust fontsize as needed
    plt.yticks(fontsize=18)  # adjust fontsize as needed
    plt.title("$\sigma_{sep.}$ as a function of Momentum",fontsize=20)
    plt.savefig(os.path.join(out_folder,'Seperation_Average.pdf'),bbox_inches='tight')
    plt.close()

def plot_DLL(kaons,pions,out_folder,datatype):
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
    nhits_k = np.array(nhits_k)
    nhits_pi = np.array(nhits_pi)
    ll_k = np.concatenate(ll_k)
    ll_p = np.concatenate(ll_p)
    ll_k = ll_k[np.where((kin_k[:,0] > 3.0) & (kin_k[:,0] < 3.5))[0]]
    nhits_k = nhits_k[np.where((kin_k[:,0] > 3.0) & (kin_k[:,0] < 3.5))[0]]
    ll_p = ll_p[np.where((kin_p[:,0] > 3.0) & (kin_p[:,0] < 3.5))[0]]
    nhits_pi = nhits_pi[np.where((kin_p[:,0] > 3.0) & (kin_p[:,0] < 3.5))[0]]

    # DLL fits in P bins
    fine_grained_DLL(dll_k,dll_p,kin_k,kin_p,out_folder)
    merge_PDF(out_folder)

    # DLL over phase space
    plt.hist(dll_k,bins=100,density=True,alpha=1.,range=[-50,50],label=r'$\mathcal{K}$',color='red',histtype='step',lw=3)
    plt.hist(dll_p,bins=100,density=True,range=[-50,50],alpha=1.0,label=r'$\pi$',color='blue',histtype='step',lw=3)
    plt.xlabel('Loglikelihood Difference',fontsize=25)
    plt.ylabel('A.U.',fontsize=25)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(r'$ \Delta \mathcal{L}_{\mathcal{K} \pi}$',fontsize=30)
    out_path_DLL = os.path.join(out_folder,"DLL_piK.pdf")
    plt.savefig(out_path_DLL,bbox_inches='tight')
    plt.close()

    # LL vs photon yield
    if real:
        r = [(0,250),(-500,100)]
    else:
        r = [(0,250),(-150,100)]
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
    out_path_khits = os.path.join(out_folder,"Kaons_LL_function_of_NHits.pdf")
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
    out_path_phits = os.path.join(out_folder,"Pions_DLL_function_of_NHits.pdf")
    plt.savefig(out_path_phits,bbox_inches="tight")
    plt.close()

    def sigmoid(x):
        return 1.0 / (1 + np.exp(-np.array(x)))

    sk = sigmoid(dll_k)
    sp = sigmoid(dll_p)

    delta_log_likelihood = list(sk) + list(sp)
    true_labels = list(np.ones_like(sk)) + list(np.zeros_like(sp))
    total_conds = np.array(list(kin_k) + list(kin_p))

    fpr, tpr, thresholds = roc_curve(true_labels, delta_log_likelihood)
    roc_auc = auc(fpr, tpr)

    # ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=25)
    plt.ylabel('True Positive Rate',fontsize=25)
    plt.title('Receiver Operating Characteristic (ROC) Curve',fontsize=25,pad=20)
    plt.legend(loc="lower right",fontsize=18)
    plt.ylim(0,1)
    plt.xticks(fontsize=18)  # adjust fontsize as needed
    plt.yticks(fontsize=18)  # adjust fontsize as needed
    out_path_DLL_ROC = os.path.join(out_folder,"DLL_piK_ROC.pdf")
    plt.savefig(out_path_DLL_ROC,bbox_inches='tight')
    plt.close()
    #plt.show()

    # ROC as a function of momentum
    mom_ranges = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5]
    centers = [mr+0.25 for mr in mom_ranges[:-1]]
    aucs = []
    lengths = []
    aucs_upper = []
    aucs_lower = []
    n_kaons = []
    n_pions = []
    for i in range(len(mom_ranges) - 1):
        mom_low = mom_ranges[i]
        mom_high = mom_ranges[i+1]
        idx = np.where((total_conds > mom_low) & (total_conds < mom_high))[0]
        p = np.array(delta_log_likelihood)[idx]
        t = np.array(true_labels)[idx]
        print("Momentum Range: ",mom_low,"-",mom_high)
        print("# Kaons: ",len(t[t==1]))
        n_kaons.append(len(t[t==1]))
        n_pions.append(len(t[t==0]))
        print("# Pions: ",len(t[t==0]))
        lengths.append(len(p))
        fpr,tpr,thresholds = roc_curve(t,p)
        AUC = []
        sigma_tpr = np.sqrt(tpr * (1.0 - tpr) / len(t))
        sigma_fpr = np.sqrt(fpr * (1.0 - fpr) / len(t))
        #print('FPR: ',fpr,'+-',sigma_fpr, " TPR: ",tpr,"+-",sigma_tpr)

        for _ in range(1000):
            fpr_ = np.random.normal(fpr,sigma_fpr)
            tpr_ = np.random.normal(tpr,sigma_tpr)

            AUC.append(np.trapz(y=tpr_,x=fpr_))


        aucs.append(np.mean(AUC))
        aucs_upper.append(np.percentile(AUC,97.5))
        aucs_lower.append(np.percentile(AUC,2.5))
        print("Mean AUC: ",np.mean(AUC)," 95%",np.percentile(AUC,2.5),"-",np.percentile(AUC,97.5))

    fig = plt.figure(figsize=(10,10))
    plt.errorbar(centers,aucs,yerr=[np.array(aucs) - np.array(aucs_lower),np.array(aucs_upper) - np.array(aucs)],label="AUC",color='red',marker='o',capsize=5)
    plt.legend(loc='upper left',fontsize=20)
    plt.xlabel("Momentum [GeV/c]",fontsize=20)
    plt.ylabel("AUC",fontsize=20)
    plt.xticks(fontsize=18)  # adjust fontsize as needed
    plt.yticks(fontsize=18)  # adjust fontsize as needed
    plt.title("AUC as function of Momentum - Analytic",fontsize=20)
    plt.ylim(np.min(aucs) - 0.05,np.max(aucs) + 0.05)

    ax2 = plt.twinx()

    # Plot bars for pions and kaons
    ax2.bar(np.array(centers) - 0.1, n_pions, width=0.2, label='Pions', color='blue', alpha=0.25)
    ax2.bar(np.array(centers) + 0.1, n_kaons, width=0.2, label='Kaons', color='green', alpha=0.25)
    ax2.set_ylabel('Counts', fontsize=20)
    ax2.tick_params(axis='y', labelsize=18)
    ax2.legend(loc='upper right', fontsize=20)
    out_path_AUC_func_P = os.path.join(out_folder,"DLL_piK_AUC_func_P.pdf")
    plt.savefig(out_path_AUC_func_P,bbox_inches='tight')
    plt.close()

def drop_and_sum(x,batch_size):
    lls = []
    for b in range(batch_size):
        ll = x[b]
        #mask = torch.isinf(ll)
        mask = torch.isnan(ll)
       # print('inside',ll[~mask].shape)
        lls.append(ll[~mask].sum().detach().cpu().numpy())
    return lls


def run_inference_seperate(pions,kaons,pion_net,kaon_net):
    LL_Pion = []
    LL_Kaon = []
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    FD = FastDIRC(device)
    print('Starting DLL for Pions.')
    start = time.time()
    n_photons = pions.dataset.n_photons
    delta_t = []
    kbar = pkbar.Kbar(target=len(pions), width=20, always_stateful=False)
    for i,data in enumerate(pions):
        h = data[0].numpy()[0]
        mask = np.isinf(h)[:,0]
        mask = np.where(mask == False)[0]
        h = h[mask]
        n_hits = data[3].numpy()[0]

        assert len(h) == n_hits

        c = data[1][0][:1].to('cuda').float()
        PID = data[2].numpy()
        unsc = data[4].numpy()[:,0,:]
        #invMass = data[5].numpy()
        photon_yield = 10000
        t_ = time.time()
        with torch.set_grad_enabled(False):
            generated_pions = pion_net._sample(num_samples=photon_yield,context=c)
            # Check for bad generated and remove if needed
            mask = np.isnan(generated_pions) | np.isinf(generated_pions)
            rows_to_remove = mask.any(axis=1)
            generated_pions = generated_pions[~rows_to_remove]
            #kde = KernelDensity(kernel='gaussian', bandwidth=6).fit(generated_pions) # 6 mm is one pixel
            #pion_hyp_pion = kde.score_samples(h).sum()
            pion_hyp_pion = FD.get_log_likelihood(generated_pions,h)
            
            generated_kaons = kaon_net._sample(num_samples=photon_yield,context=c)
            # Check for bad generations and remove if needed
            mask = np.isnan(generated_kaons) | np.isinf(generated_kaons)
            rows_to_remove = mask.any(axis=1)
            generated_kaons = generated_kaons[~rows_to_remove]
            #kde = KernelDensity(kernel='gaussian', bandwidth=6).fit(generated_kaons) # 6 mm is one pixel
            #pion_hyp_kaon = kde.score_samples(h).sum()
            pion_hyp_kaon = FD.get_log_likelihood(generated_kaons,h)
            delta_t.append(time.time() - t_)

        #assert len(pion_hyp_kaon) == len(pion_hyp_pion)
        #assert len(unsc) == len(pion_hyp_pion)
        LL_Pion.append({"hyp_pion":pion_hyp_pion,"hyp_kaon":pion_hyp_kaon,"Truth":PID,"Kins":unsc,"Nhits":n_hits})#,"invMass":invMass})

        kbar.update(i)
        #if i == 5000:
        #    break

    end = time.time()
    print(" ")
    print("Elapsed time: ",end - start)
    print('Time / event: ',(end - start) / len(pions.dataset))

    print(' ')
    print('Starting DLL for Kaons')
    start = time.time()
    delta_t = []
    kbar = pkbar.Kbar(target=len(kaons), width=20, always_stateful=False)
    for i,data in enumerate(kaons):
        h = data[0].numpy()[0]
        mask = np.isinf(h)[:,0]
        mask = np.where(mask == False)[0]
        h = h[mask]
        n_hits = data[3].numpy()[0]

        assert len(h) == n_hits

        c = data[1][0][:1].to('cuda').float()
        PID = data[2].numpy()
        unsc = data[4].numpy()[:,0,:]
        photon_yield = 10000
        t_ = time.time()
        with torch.set_grad_enabled(False):
            generated_pions = pion_net._sample(num_samples=photon_yield,context=c)
            # Check for bad generations and remove if needed
            mask = np.isnan(generated_pions) | np.isinf(generated_pions)
            rows_to_remove = mask.any(axis=1)
            generated_pions = generated_pions[~rows_to_remove]
            #kde = KernelDensity(kernel='gaussian', bandwidth=6).fit(generated_pions) # 6 mm is one pixel
            #kaon_hyp_pion = kde.score_samples(h).sum()
            kaon_hyp_pion = FD.get_log_likelihood(generated_pions,h)

            generated_kaons = kaon_net._sample(num_samples=photon_yield,context=c)
            # Check for bad generations and remove if needed
            mask = np.isnan(generated_kaons) | np.isinf(generated_kaons)
            rows_to_remove = mask.any(axis=1)
            generated_kaons = generated_kaons[~rows_to_remove]
            #kde = KernelDensity(kernel='gaussian', bandwidth=6).fit(generated_kaons) # 6 mm is one pixel
            #kaon_hyp_kaon = kde.score_samples(h).sum()
            kaon_hyp_kaon = FD.get_log_likelihood(generated_kaons,h)

        delta_t.append(time.time() - t_)

        #assert len(kaon_hyp_kaon) == len(kaon_hyp_pion)
        #assert len(unsc) == len(kaon_hyp_kaon)
        LL_Kaon.append({"hyp_kaon":kaon_hyp_kaon,"hyp_pion":kaon_hyp_pion,"Truth":PID,"Kins":unsc,"Nhits":n_hits})#,"invMass":invMass})

        kbar.update(i)
        #if i == 5000:
        #    break

    end = time.time()
    print(" ")
    print("Elapsed time: ",end - start)
    print('Avg GPU Time / event: ',np.average(delta_t))

    return LL_Pion,LL_Kaon


def main(config,resume):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    print("Running inference")
    datatype = config['datatype']
    num_layers = int(config['model']['num_layers'])
    input_shape = int(config['model']['input_shape'])
    cond_shape = int(config['model']['cond_shape'])
    num_blocks = int(config['model']['num_blocks'])
    hidden_nodes = int(config['model']['hidden_nodes'])

    assert config["method"] in ["Combined","Pion","Kaon"]
    log_time = config['log_time']
    stats = config['stats']

    test_pions = DLL_Dataset(file_path=config['dataset']['testing']['DLL']['pion_data_path'],time_cuts=args.time,log_time=log_time,stats=stats)
    test_kaons = DLL_Dataset(file_path=config['dataset']['testing']['DLL']['kaon_data_path'],time_cuts=args.time,log_time=log_time,stats=stats)

    print("# of Pions: ",len(test_pions))
    print("# of Kaons: ",len(test_kaons))
    print('Setting batch size to 1.')
    config['dataloader']['test']['batch_size'] = 1
    pions = CreateInferenceLoader(test_pions,config) # Batch size is 1 untill I figure out a better way
    kaons = CreateInferenceLoader(test_kaons,config)

    if config['method'] != "Combined":
        pion_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats)
        #pion_net = create_nflows(input_shape,cond_shape,num_layers)
        device = torch.device('cuda')
        pion_net.to('cuda')
        dicte = torch.load(config['Inference']['pion_model_path'])
        pion_net.load_state_dict(dicte['net_state_dict'])

        kaon_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks,stats=stats)
        #kaon_net = create_nflows(input_shape,cond_shape,num_layers)
        device = torch.device('cuda')
        kaon_net.to('cuda')
        dicte = torch.load(config['Inference']['kaon_model_path'])
        kaon_net.load_state_dict(dicte['net_state_dict'])

        LL_Pion,LL_Kaon = run_inference_seperate(pions,kaons,pion_net,kaon_net)
        
        if not os.path.exists(config['Inference']['out_dir']):
            print('Inference plots can be found in: ' + config['Inference']['out_dir'])
            os.mkdir(config['Inference']['out_dir'])

        pion_path = os.path.join(config['Inference']['out_dir'],"Pion_DLL_Results.pkl")
        with open(pion_path,"wb") as file:
            pickle.dump(LL_Pion,file)

        kaon_path = os.path.join(config['Inference']['out_dir'],"Kaon_DLL_Results.pkl")
        with open(kaon_path,"wb") as file:
            pickle.dump(LL_Kaon,file)


        plot_DLL(LL_Kaon,LL_Pion,config['Inference']['out_dir'],datatype)
        
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
    parser.add_argument('-t','--time',default=None,type=float,
                        help='Maximum hit time for Cherenkov photons')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args.resume)
