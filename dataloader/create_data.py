import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
import torch
import os

def unscale(x,max_,min_):
    return x*0.5*(max_ - min_) + min_ + (max_-min_)/2


def scale_data(hits,stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":380.0,"time_min":0.0}):
    x = hits[:,0]
    y = hits[:,1]
    time = hits[:,2]

    x = 2.0 * (x - stats['x_min'])/(stats['x_max'] - stats['x_min']) - 1.0
    y = 2.0 * (y - stats['y_min'])/(stats['y_max'] - stats['y_min']) - 1.0
    time = 2.0 * (time - stats['time_min'])/(stats['time_max'] - stats['time_min']) - 1.0

    return np.concatenate([np.c_[x],np.c_[y],np.c_[time]],axis=1)

class hpDIRC_DLL_Dataset(Dataset):
    def __init__(self,path_,stats={"x_max": 350.0,"x_min":2.0,"y_max":230.1,"y_min":2.0,"time_max":157.00,"time_min":0.0,"P_max":10.0 ,"P_min":0.5 ,"theta_max": 160.0,"theta_min": 25.0},time_cuts=None,n_photons=300,n_particles=200000,fast_sim_comp=False,fast_sim_type=None,geant=True):
        self.stats = stats
        self.geant = geant
        if not fast_sim_comp: # path_ is a .pkl file here
            data = np.load(path_,allow_pickle=True)[:n_particles] # Useful for testing
            self.data = []
            print(len(data))
            for i in range(len(data)):
                theta__ = data[i]['Theta']
                p__ = data[i]['P']
                n_hits = data[i]['NHits']
                if ((theta__ > self.stats['theta_min']) and (theta__ < self.stats['theta_max']) and (p__ > self.stats['P_min']) and (p__ < self.stats['P_max']) and (n_hits > 0)):
                    self.data.append(data[i])

        elif fast_sim_comp and fast_sim_type is not None and not geant: # Use fast sim file structure, provide PID, dont use geant4 data
            data = []
            files = os.listdir(path_) # path_ is a directory here
            for file in files:
                if ".pkl" in file:
                    if fast_sim_type in file:
                        print("Loading file: ",file)
                        d_ = np.load(os.path.join(path_,file),allow_pickle=True)
                        for i in range(len(d_['fast_sim'])):
                            d_['fast_sim'][i]['PDG'] = d_['truth'][i]['PDG']
            
                        data += d_['fast_sim']
                else:
                    continue
            
            self.data = data[:n_particles]
            del data
            print(len(self.data))

        elif fast_sim_comp and fast_sim_type is not None and geant: # Use fast sim file structure, provide PID, use geant
            data = []
            files = os.listdir(path_) # path_ is a directory here
            for file in files:
                if ".pkl" in file:
                    if fast_sim_type in file:
                        print("Loading file: ",file)
                        d_ = np.load(os.path.join(path_,file),allow_pickle=True)
                        data += d_['truth']
                else:
                    continue
            
            self.data = data[:n_particles]
            del data
            print(len(self.data))      

        else:
            raise ValueError("Fast Sim Comp is set true, but have not provided the type. Set as either Pion or Kaon. Check geant arg.")


        self.n_photons = n_photons
        self.conditional_maxes = np.array([self.stats['P_max'],self.stats['theta_max']])
        self.conditional_mins = np.array([self.stats['P_min'],self.stats['theta_min']])
        self.time_cuts = time_cuts
        self.gapx =  1.89216111455965 + 4.
        self.gapy = 1.3571428571428572 + 4.
        self.pixel_width = 3.3125
        self.pixel_height = 3.3125
        if self.time_cuts is not None:
            print('Rejecting photons with time > {0}'.format(self.time_cuts))

    def __len__(self):
        return len(self.data)

    def scale_data(self,hits,stats):
        x = hits[:,0]
        y = hits[:,1]
        time = hits[:,2]
        x = 2.0 * (x - stats['x_min'])/(stats['x_max'] - stats['x_min']) - 1.0
        y = 2.0 * (y - stats['y_min'])/(stats['y_max'] - stats['y_min']) - 1.0
        time = 2.0 * (time - stats['time_min'])/(stats['time_max'] - stats['time_min']) - 1.0
        return np.concatenate([np.c_[x],np.c_[y],np.c_[time]],axis=1)


    def __getitem__(self, idx):

        data = self.data[idx]
        PID = data['PDG']
        LL_k = 0.0
        LL_pi = 0.0
        pmtID = np.array(data['pmtID'])
        if data['NHits'] == 0:
            print("Stop.",NHits)

        if self.geant:

            pixelID = np.array(data['pixelID'])

            row = (pmtID//6) * 16 + pixelID//16 
            col = (pmtID%6) * 16 + pixelID%16
            
            x = 2 + col * self.pixel_width + (pmtID % 6) * self.gapx + (self.pixel_width) / 2. # Center at middle
            y = 2 + row * self.pixel_height + (pmtID // 6) * self.gapy + (self.pixel_height) / 2. # Center at middle

        else:
            x = data['x']
            y = data['y']

        time = np.array(data['leadTime'])      

        pos_time = np.where((time > 0) & (time < self.stats['time_max']))[0]
        time = time[pos_time]
        x = x[pos_time]
        y = y[pos_time]

        if len(time) == 0:
            print("Hey, wrong.",idx)
        
        if len(time) > self.n_photons:

            time_idx = np.argsort(time)[:self.n_photons]
            time = time[time_idx]
            x = x[time_idx]
            y = y[time_idx]

        assert len(x) == len(time)
        assert len(y) == len(time)

        hits = np.concatenate([np.c_[x],np.c_[y],np.c_[time]],axis=1)
        hits = self.scale_data(hits,self.stats)
        conds = np.array([data['P'],data['Theta']])
        conds = conds.reshape(1,-1).repeat(len(x),0)
        unscaled_conds = conds.copy()
        n_hits = len(hits)

        conds = (conds - self.conditional_maxes) / (self.conditional_maxes - self.conditional_mins)


        if len(hits) > self.n_photons:
            #usually argsort in time
            hits = hits[np.argsort(time)]
            hits = hits[:self.n_photons]
            conds = conds[:self.n_photons]
            unscaled_conds = unscaled_conds[:self.n_photons]
            time = time[np.argsort(time)]
            time = time[:self.n_photons]

        elif len(hits) < self.n_photons:
            n_needed = self.n_photons - len(hits)
            hits = np.pad(hits,((0,n_needed),(0,0)),mode='constant',constant_values=-np.inf)
            conds = np.pad(conds,((0,n_needed),(0,0)),mode='constant',constant_values=-np.inf)
            unscaled_conds = np.pad(unscaled_conds,((0,n_needed),(0,0)),mode='constant',constant_values=-np.inf)
        else: # Already taken care of
            pass

        return hits,conds,PID,n_hits,unscaled_conds,LL_k,LL_pi



class hpDIRCCherenkovPhotons(Dataset):
    def __init__(self,kaon_path=None,pion_path=None,mode=None,combined=False,inference=False,stats={"x_max": 350.0,"x_min":2.0,"y_max":230.1,"y_min":2.0,"time_max":157.00,"time_min":0.0,"P_max":10.0 ,"P_min":0.5 ,"theta_max": 160.0,"theta_min": 25.0}):
        if mode is None:
            print("Please select one of the following modes:")
            print("1. Pion")
            print("2. Kaon")
            print("3. Combined")
            exit()
        self.inference = inference
        self.combined = combined
        self.stats = stats
        self.kaon_path = kaon_path
        self.pion_path = pion_path
        self.conditional_maxes = np.array([self.stats['P_max'],self.stats['theta_max']])
        self.conditional_mins = np.array([self.stats['P_min'],self.stats['theta_min']])

        if mode == "Kaon":
            columns=["x","y","time","P","theta"]
            if not self.inference:
                print('Using training mode.')
                df = pd.read_feather(kaon_path)
                df = df[(df.time < self.stats['time_max']) & (df.theta > self.stats['theta_min']) & (df.theta < self.stats['theta_max']) & (df.P > self.stats['P_min']) & (df.P < self.stats['P_max'])]
                self.data = df[columns].to_numpy()
                del df
                print('Max Time: ',self.stats['time_max'])
                print(self.data.max(0))
                print(self.data.min(0))

            else:
                print("Using inference mode.")
                df = pd.read_feather(kaon_path)
                df = df[(df.time < self.stats['time_max']) & (df.theta > self.stats['theta_min']) & (df.theta < self.stats['theta_max']) & (df.P > self.stats['P_min']) & (df.P < self.stats['P_max'])]
                self.data = df[columns].to_numpy()

            self.data = np.concatenate([self.data,np.c_[np.ones_like(self.data[:,0])]],axis=1)

        elif mode == "Pion":
            columns=["x","y","time","P","theta"]
            if not self.inference:
                print('Using training mode.')
                df = pd.read_feather(pion_path)
                print(df.shape)
                df = df[(df.time < self.stats['time_max']) & (df.theta > self.stats['theta_min']) & (df.theta < self.stats['theta_max']) & (df.P > self.stats['P_min']) & (df.P < self.stats['P_max']) & (df.phi == 0)]
                self.data = df[columns].to_numpy()
                del df
                print('Max Time: ',self.stats['time_max'])
                print(self.data.max(0))
                print(self.data.min(0))
            else:
                print("Using inference mode.")
                df = pd.read_feather(pion_path)
                df = df[(df.time < self.stats['time_max']) & (df.theta > self.stats['theta_min']) & (df.theta < self.stats['theta_max']) & (df.P > self.stats['P_min']) & (df.P < self.stats['P_max']) & (df.phi == 0.0)]
                self.data = df[columns].to_numpy()

            self.data = np.concatenate([self.data,np.c_[np.zeros_like(self.data[:,0])]],axis=1)
            
        elif mode == "Combined":
            pions = pd.read_csv(pion_path,sep=',',index_col=None)
            pions['PID'] = np.zeros(len(pions))
            kaons = pd.read_csv(kaon_path,sep=',',index_col=None)
            kaons['PID'] = np.ones(len(kaons))
            self.data = pd.concat([pions,kaons],axis=0).to_numpy()
            random.shuffle(self.data)
            del pions,kaons
        else:
            print("Error in dataset creation. Exiting")
            exit()

        self.stats = stats

    def __len__(self):
        return len(self.data)

    def scale_data(self,hits,stats):
        x = hits[0]
        y = hits[1]
        time = hits[2]
        x = 2.0 * (x - stats['x_min'])/(stats['x_max'] - stats['x_min']) - 1.0
        y = 2.0 * (y - stats['y_min'])/(stats['y_max'] - stats['y_min']) - 1.0
        time = 2.0 * (time - stats['time_min'])/(stats['time_max'] - stats['time_min']) - 1.0
        return np.array([x,y,time])

    def __getitem__(self, idx):
        # Get the sample
        data = self.data[idx]
        hits = data[:3]

        hits = self.scale_data(hits,self.stats)
        conds = np.array(data[3:-1])

        unscaled_conds = conds.copy()
        metadata = unscaled_conds.copy()

        #conds = 2*(conds - self.conditional_maxes) / (self.conditional_maxes - self.conditional_mins) - 1.0
        conds = (conds - self.conditional_maxes) / (self.conditional_maxes - self.conditional_mins)
        PID = data[-1]

        return hits,conds,PID,metadata,unscaled_conds