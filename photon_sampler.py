import numpy as np
from scipy.ndimage import gaussian_filter1d
import pickle
import time
import os
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def extract_data(data,stats):
    p = []
    theta = []
    nhits = []

    for i in range(len(data)):
        if int(data[i]['NHits']) >= 300 or int(data[i]['NHits']) == 0: 
            continue 

        if float(data[i]['P']) > stats['P_max'] or float(data[i]['P']) < stats['P_min']: 
            continue

        if float(data[i]['Theta']) > stats['theta_max'] or float(data[i]['Theta']) < stats['theta_min']: 
            continue

        if float(data[i]['Phi']) != 0:  # Given current simulation.
            continue
        
        p.append(float(data[i]['P']))
        theta.append(float(data[i]['Theta']))
        nhits.append(int(data[i]['NHits']))

    p = np.array(p)
    theta = np.array(theta)
    nhits = np.array(nhits)

    return p,theta,nhits


def plots(dicte,p,theta,nhits,PID):
    print("Running tests.")
    global_values = dicte['global_values']
    p_points = np.array(list(dicte.keys())[:-1]) 
    theta_points = np.array(list(dicte[p_points[0]].keys()))
    times = []
    sampled_hits = []

    for i in range(len(p)):
        p_value = p[i]
        theta_value = theta[i]

        start = time.time()
        closest_p_idx = np.argmin(np.abs(p_points - p_value))
        closest_p = float(p_points[closest_p_idx])
        
        closest_theta_idx = np.argmin(np.abs(theta_points - theta_value))
        closest_theta = float(theta_points[closest_theta_idx])

        sh = int(np.random.choice(global_values,p=dicte[closest_p][closest_theta]))
        times.append(time.time() - start)

        sampled_hits.append(sh)

        if i % 1000000 == 0:
            print("Event: ",i)

    sampled_hits = np.array(sampled_hits)

    print("First 10 true yield: ",nhits[:10])
    print("First 10 sampled yield: ",sampled_hits[:10])
    print("First 10 momentum: ",p[:10])
    print("First 10 theta: ",theta[:10])

    fig = plt.figure(figsize=(8,6))
    plt.hist(nhits,bins=300,histtype='step',color='red',label='True Photon Yield',range=[0,300],density=False)
    plt.hist(sampled_hits,bins=300,histtype='step',color='k',label='Sampled Photon Yield',range=[0,300],density=False)
    plt.legend()
    plt.title(str(PID),fontsize=25)
    plt.savefig(f"Photon_Yield/{PID}_global.pdf",bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,6))
    plt.hist2d(theta,nhits,bins=[np.arange(25, 155, 0.06),300],density=True,norm=LogNorm(),range=[[25,155],[0,300]])
    plt.xlabel("Polar Angle [Degrees]",fontsize=25)
    plt.ylabel("Photon Yield",fontsize=25)
    plt.title("True {0}".format(str(PID)),fontsize=25)
    plt.savefig(f"Photon_Yield/{PID}_polar_angle.pdf",bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,6))
    plt.hist2d(p,nhits,bins=[100,300],density=True,norm=LogNorm(),range=[[1,10],[0,300]])
    plt.xlabel("Momentum [GeV/c]",fontsize=25)
    plt.ylabel("Photon Yield",fontsize=25)
    plt.title("True {0}".format(str(PID)),fontsize=25)
    plt.savefig(f"Photon_Yield/{PID}_momentum.pdf",bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,6))
    plt.hist2d(theta,sampled_hits,bins=[np.arange(25, 155, 0.06),300],density=True,norm=LogNorm(),range=[[25,155],[0,300]])
    plt.xlabel("Polar Angle [Degrees]",fontsize=25)
    plt.ylabel("Photon Yield",fontsize=25)
    plt.title("Sampled {0}".format(PID),fontsize=25)
    plt.savefig(f"Photon_Yield/{PID}_sampled_polar_angle.pdf",bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,6))
    plt.hist2d(p,sampled_hits,bins=[100,300],density=True,norm=LogNorm(),range=[[1,10],[0,300]])
    plt.xlabel("Momentum [GeV/c]",fontsize=25)
    plt.ylabel("Photon Yield",fontsize=25)
    plt.title("Sampled {0}".format(PID),fontsize=25)
    plt.savefig(f"Photon_Yield/{PID}_sampled_momentum.pdf",bbox_inches="tight")
    plt.close(fig)

def create_LUT(file_path,PID,stats):
    data = np.load(file_path,allow_pickle=True)
    p,theta,nhits = extract_data(data,stats)
    delta_p = 0.1
    delta_theta = 1.0
    p_bins = np.arange(0.5,10.0 + delta_p,delta_p)
    theta_bins = np.arange(25.0,155.0 + delta_theta,delta_theta)
    global_values = np.arange(np.min(nhits),np.max(nhits)+1,1)
    sigma = 2.0
    dicte = {}


    for i in range(len(p_bins) - 1):
        p_low,p_high = p_bins[i].astype('float32'),p_bins[i+1].astype('float32')
        print("At momentum bin: ",p_low,"-",p_high)
        for j in range(len(theta_bins)-1):
            theta_low,theta_high = theta_bins[j],theta_bins[j+1]
            idx = np.where((p > p_low) & (p <= p_high) & (theta > theta_low) & (theta <theta_high))[0]
            photon_yield = nhits[idx]
            photon_yield = photon_yield[np.argsort(photon_yield)]
            bin_indices = np.digitize(photon_yield, global_values) - 1  
            frequency = np.bincount(bin_indices, minlength=len(global_values)) 
            p_mid = (p_low + p_high) / 2.0
            theta_mid = (theta_low + theta_high)/2.0
            p_mid = p_mid.astype('float32')
            theta_mid = theta_mid.astype('float32')

            smoothed_frequency = gaussian_filter1d(frequency.astype(float), sigma=sigma)
            smoothed_prob = smoothed_frequency / smoothed_frequency.sum()


            if frequency.sum() == 0:
                print(p_mid, theta_mid)
            
            p_mid_rounded = round(float(p_mid), 2)  
            theta_mid_rounded = round(float(theta_mid), 2) 
            
            if p_mid_rounded not in dicte:
                dicte[p_mid_rounded] = {}
            
            dicte[p_mid_rounded][theta_mid_rounded] = smoothed_prob

    os.makedirs("Photon_Yield",exist_ok=True)
    outpath = f"Photon_Yield/{PID}_Photon_Yield.pkl"

    dicte['global_values'] = global_values
    with open(outpath,"wb") as file:
        pickle.dump(dicte,file)

    print("Wrote file: ",outpath)
    print("Update config file.")

    return dicte,p,theta,nhits



def main(config):
    stats = config['stats']

    print("Creating Photon Yield LUT for Pions.")
    PID = "Pion"
    file_path = config['Photon_Sampler'][f"{PID}_dataset"]
    dicte,p,theta,nhits = create_LUT(file_path,PID,stats)
    plots(dicte,p,theta,nhits,PID)

    print("Creating Photon Yield LUT for Kaons.")
    PID = "Kaon"
    file_path = config['Photon_Sampler'][f"{PID}_dataset"]
    dicte,p,theta,nhits = create_LUT(file_path,PID,stats)
    plots(dicte,p,theta,nhits,PID)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Photon Yield LUT Creation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')

    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config)




        