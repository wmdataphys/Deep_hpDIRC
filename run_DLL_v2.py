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

def plot_DLL(kaons,pions,out_folder):
    #     # This function is gross need to rewrite it
    #     # Kaon label = 1
    #     # Pion label = 0
    dll_k = []
    dll_p = []
    kin_p = []
    kin_k = []
    predicted_pions = []
    predicted_kaons = []
    kaon_labels = []
    pion_labels = []
    for i in range(len(kaons)):
        dll_k.append((np.array(kaons[i]['hyp_kaon']) - np.array(kaons[i]['hyp_pion'])).flatten())
        kin_k.append(kaons[i]['Kins'][:,0,:])

    for i in range(len(pions)):
        dll_p.append((np.array(pions[i]['hyp_kaon']) - np.array(pions[i]['hyp_pion'])).flatten())
        kin_p.append(pions[i]['Kins'][:,0,:])

    kin_p = np.concatenate(kin_p)
    kin_k = np.concatenate(kin_k)
    dll_p = np.concatenate(dll_p)
    dll_k = np.concatenate(dll_k)

    plt.hist(dll_k,bins=100,density=True,alpha=1.,range=[-50,50],label='Kaons',color='red',histtype='step',lw=3)
    plt.hist(dll_p,bins=100,density=True,range=[-50,50],alpha=1.0,label='Pions',color='blue',histtype='step',lw=3)
    plt.xlabel('Loglikelihood Difference',fontsize=25)
    plt.ylabel('A.U.',fontsize=25)
    plt.legend(fontsize=20)
    plt.title(r'$ \Delta \mathcal{L}_{K \pi}$',fontsize=30)
    out_path_DLL = os.path.join(out_folder,"DLL_piK.pdf")
    plt.savefig(out_path_DLL,bbox_inches='tight')
    plt.close()

    def sigmoid(x):
        return 1.0 / (1 + np.exp(-np.array(x)))

    sk = sigmoid(dll_k)
    sp = sigmoid(dll_p)

    delta_log_likelihood = list(sk) + list(sp)
    true_labels = list(np.ones_like(sk)) + list(np.zeros_like(sp))
    total_conds = np.array(list(kin_k) + list(kin_p))

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, delta_log_likelihood)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
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
    plt.ylim(0.75,0.95)

    ax2 = plt.twinx()

    # Plot bars for pions and kaons
    ax2.bar(np.array(centers) - 0.1, n_pions, width=0.2, label='Pions', color='blue', alpha=0.25)
    ax2.bar(np.array(centers) + 0.1, n_kaons, width=0.2, label='Kaons', color='green', alpha=0.25)
    ax2.set_ylabel('Counts', fontsize=20)
    ax2.tick_params(axis='y', labelsize=18)
    ax2.legend(loc='upper right', fontsize=20)
    #plt.ylim(0.5,1.0)
    out_path_AUC_func_P = os.path.join(out_folder,"DLL_piK_AUC_func_P.pdf")
    plt.savefig(out_path_AUC_func_P,bbox_inches='tight')
    plt.close()

def drop_and_sum(x,batch_size):
    lls = []
    for b in range(batch_size):
        ll = x[b]
        mask = torch.isinf(ll)
       # print('inside',ll[~mask].shape)
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
        c = data[1].reshape(int(b*n_photons),3).to('cuda').float()
        PID = data[2].numpy()
        unsc = data[4].numpy()
        t_ = time.time()
        with torch.set_grad_enabled(False):
            pion_hyp_pion = pion_net.log_prob(h,context=c).reshape(-1,n_photons)
            pion_hyp_pion = drop_and_sum(pion_hyp_pion,pion_hyp_pion.shape[0])
            pion_hyp_kaon = kaon_net.log_prob(h,context=c).reshape(-1,n_photons)
            pion_hyp_kaon = drop_and_sum(pion_hyp_kaon,pion_hyp_kaon.shape[0])
            delta_t.append((time.time() - t_) / b)
        LL_Pion.append({"hyp_pion":pion_hyp_pion,"hyp_kaon":pion_hyp_kaon,"Truth":PID,"Kins":unsc})

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
        h = data[0]
        b = h.shape[0]
        h = h.reshape(int(b*n_photons),3).to('cuda').float()
        n_hits = data[3].numpy()
        c = data[1].reshape(int(b*n_photons),3).to('cuda').float()
        PID = data[2].numpy()
        unsc = data[4].numpy()
        t_ = time.time()
        with torch.set_grad_enabled(False):
            kaon_hyp_kaon = kaon_net.log_prob(h,context=c).reshape(-1,n_photons)
            kaon_hyp_kaon = drop_and_sum(kaon_hyp_kaon,kaon_hyp_kaon.shape[0])
            kaon_hyp_pion = pion_net.log_prob(h,context=c).reshape(-1,n_photons)
            kaon_hyp_pion = drop_and_sum(kaon_hyp_pion,kaon_hyp_pion.shape[0])
        delta_t.append((time.time() - t_) / b)
        LL_Kaon.append({"hyp_kaon":kaon_hyp_kaon,"hyp_pion":kaon_hyp_pion,"Truth":PID,"Kins":unsc})

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

    num_layers = int(config['model']['num_layers'])
    input_shape = int(config['model']['input_shape'])
    cond_shape = int(config['model']['cond_shape'])
    num_blocks = int(config['model']['num_blocks'])
    hidden_nodes = int(config['model']['hidden_nodes'])

    assert config["method"] in ["Combined","Pion","Kaon"]

    test_pions = DLL_Dataset(file_path=config['dataset']['testing']['DLL']['pion_data_path'],time_cuts=args.time)
    test_kaons = DLL_Dataset(file_path=config['dataset']['testing']['DLL']['kaon_data_path'],time_cuts=args.time)

    print("# of Pions: ",len(test_pions))
    print("# of Kaons: ",len(test_kaons))

    pions = CreateInferenceLoader(test_pions,config) # Batch size is 1 untill I figure out a better way
    kaons = CreateInferenceLoader(test_kaons,config)

    if config['method'] != "Combined":
        pion_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks)
        #pion_net = create_nflows(input_shape,cond_shape,num_layers)
        device = torch.device('cuda')
        pion_net.to('cuda')
        dicte = torch.load(config['Inference']['pion_model_path'])
        pion_net.load_state_dict(dicte['net_state_dict'])

        kaon_net = FreiaNet(input_shape,num_layers,cond_shape,embedding=False,hidden_units=hidden_nodes,num_blocks=num_blocks)
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


        plot_DLL(LL_Kaon,LL_Pion,config['Inference']['out_dir'])
        
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
