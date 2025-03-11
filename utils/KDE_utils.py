import numpy as np
import itertools
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit
import glob
from scipy.stats import norm
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from scipy.special import expit
import matplotlib.pyplot as plt
import os
import torch
from scipy.integrate import simpson
from scipy.integrate import quad
from scipy.interpolate import interp1d
from sklearn.utils import resample
from utils.hpDIRC import bins_x,bins_y,gapx,gapy,pixel_width,pixel_height

t_bins = np.arange(0.0,100.0,0.1)

def perform_fit_KDE(dll_k,dll_p,bins=200,normalized=False,momentum=6.0):
    if normalized:
        gaussian = gaussian_normalized
    else:
        gaussian = gaussian_unnormalized

    if momentum == 6.0:
        bins_ = np.linspace(-10.0,10.0,bins)
    elif momentum == 9.0:
        bins_ = np.linspace(-5,5,bins)
    elif momentum == 3.0:
        bins_ = np.linspace(-50,50,int(2*bins))
    else:
        raise ValueError("Momentum value not found.")

    hist_k, bin_edges_k = np.histogram(dll_k, bins=bins_, density=normalized)
    bin_centers_k = (bin_edges_k[:-1] + bin_edges_k[1:]) / 2
    try:
        popt_k, pcov_k = curve_fit(gaussian, bin_centers_k, hist_k, p0=[1, np.mean(dll_k), np.std(dll_k)],maxfev=10000,bounds = ([0, -np.inf, 1e-9], [np.inf, np.inf, np.inf]))
        amplitude_k, mean_k, stddev_k = popt_k
        perr_k = np.sqrt(np.diag(pcov_k))
    except RuntimeError as e:
        print('Kaon error, exiting.')
        print(e)
        exit()
        

    hist_p, bin_edges_p = np.histogram(dll_p, bins=bins_, density=normalized)
    bin_centers_p = (bin_edges_p[:-1] + bin_edges_p[1:]) / 2
    try:
        popt_p, pcov_p = curve_fit(gaussian, bin_centers_p, hist_p, p0=[1, np.mean(dll_p), np.std(dll_p)],maxfev=10000,bounds = ([0, -np.inf, 1e-9], [np.inf, np.inf, np.inf]))
        amplitude_p, mean_p, stddev_p = popt_p
        perr_p = np.sqrt(np.diag(pcov_p))
    except RuntimeError as e:
        print('Pion error, exiting.')
        print(e)
        exit()
    
    sigma_sep = abs(mean_k - mean_p) / ((stddev_k + stddev_p)/2.) #np.sqrt(stddev_k**2 + stddev_p**2)
    sigma_err = (2*perr_k[1]/(stddev_k + stddev_p))** 2 + (2*perr_p[1]/(stddev_k + stddev_p))** 2 + (-2*(mean_k - mean_p) * perr_k[2] / (stddev_k + stddev_p)**2)**2 + (-2*(mean_k - mean_p) * perr_p[2] / (stddev_k + stddev_p)**2)**2
    return (popt_k,popt_p,sigma_sep,bin_centers_k,bin_centers_p,np.sqrt(sigma_err), normalized)

def gaussian_normalized(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) / (np.sqrt(2 * np.pi) * stddev)

def gaussian_unnormalized(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def plot(fit_params,DLL_p,DLL_k,support_method,out_folder,theta,pdf_method,bins=200,momentum=6.0):
    popt_k,popt_p,sigma_sep,bin_centers_k,bin_centers_p,sigma_err, normalized = fit_params
    if normalized:
        gaussian = gaussian_normalized
    else:
        gaussian = gaussian_unnormalized

    fig = plt.figure(figsize=(6,4))
    if momentum == 6.0:
        bins_ = np.linspace(-10.,10.,bins)
    elif momentum == 9.0:
        bins_ = np.linspace(-5,5,bins)
    elif momentum == 3.0:
        bins_ = np.linspace(-50,50,int(2*bins))
    else:
        raise ValueError("Momentum value not found.")
        
    plt.plot(bin_centers_k, gaussian(bin_centers_k, *popt_k),color='blue', label=r"$\mathcal{K}$")
    plt.plot(bin_centers_p, gaussian(bin_centers_p, *popt_p),color='red', label=r"$\pi$")
    plt.hist(DLL_p,bins=bins_,density=normalized,color='red',histtype='step',lw=3)
    plt.hist(DLL_k,bins=bins_,density=normalized,color='blue',histtype='step',lw=3)
    plt.legend(fontsize=18) 
    plt.xlabel(r"$Ln \, L(\mathcal{K}) - Ln \, L(\pi)$",fontsize=18)
    plt.ylabel("entries [#]",fontsize=18)
    plt.title(r"$\sigma$ = {0:.2f} +- {1:.2f} (PDF - {2})".format(sigma_sep,sigma_err,support_method),fontsize=20)
    plt.savefig(os.path.join(out_folder,"Gauss_fit_theta_{0}_{1}.pdf".format(theta,pdf_method)),bbox_inches="tight")
    plt.close()



class FastDIRC():
    def __init__(self,device):
        self.log_mult = 20
        self.weight = 1
        self.device = device
        self.bins_x = torch.tensor(np.array(bins_x),dtype=torch.float16)
        self.bins_y = torch.tensor(np.array(bins_y),dtype=torch.float16)
        self.t_bins = torch.tensor(t_bins,dtype=torch.float16)

    def radius_spread_function(self, r2):
        sigma2inv = 1.0
        sigma2 = 1.0
        threshold = 5 * sigma2  
        kernel_values = torch.where(r2 < threshold, torch.exp(-r2 * sigma2inv), torch.zeros_like(r2))
        return kernel_values 

    def get_log_likelihood(self, inpoints,support,eps=0.1,weighted=False):
        # Not useful
        if weighted:
            hist, edges = np.histogramdd(support, bins=[bins_x, bins_y,t_bins], density=True)
            x_idx = np.digitize(support[:, 0], edges[0]) - 1  
            y_idx = np.digitize(support[:, 1], edges[1]) - 1
            t_idx = np.digitize(support[:, 2], edges[2]) - 1
            x_idx = np.clip(x_idx, 0, len(bins_x) - 2)
            y_idx = np.clip(y_idx, 0, len(bins_y) - 2)
            t_idx = np.clip(t_idx, 0, len(t_bins) - 2)
            weights = torch.tensor(hist[x_idx, y_idx,t_idx],device=self.device,dtype=torch.float16)

            
        with torch.no_grad():
            support_tensor = torch.tensor(support, device=self.device, dtype=torch.float16)
            inpoints_tensor = torch.tensor(inpoints, device=self.device, dtype=torch.float16)

            # Broadcast, all subtractions at once - (N_gamma,support)
            dx = inpoints_tensor[:, 0].unsqueeze(1) - support_tensor[:, 0] #+ eps*torch.randn_like(support_tensor[:,0])
            dy = inpoints_tensor[:, 1].unsqueeze(1) - support_tensor[:, 1] #+ eps*torch.randn_like(support_tensor[:,1])
            dt = inpoints_tensor[:, 2].unsqueeze(1) - support_tensor[:, 2] 
            x_sig2inv = 1.0 / (1*3.3125) ** 2 # pixel width
            y_sig2inv = 1.0 / (1*3.3125) ** 2 # pixel height 
            t_sig2inv = 1.0 / (0.05*2) ** 2  # 0.05 ns time res 
            distance_squared = dx ** 2 * x_sig2inv + dy ** 2 * y_sig2inv + dt ** 2 * t_sig2inv
    

            spread = self.radius_spread_function(distance_squared) 
            tprob = spread.sum(dim=1) / len(support)

        rval = self.weight * self.log_mult * torch.log(torch.sum(tprob) + 1e-50)
        return rval.detach().cpu().numpy(),tprob.detach().cpu().numpy()
