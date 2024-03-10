import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset

def create_dataset(file_paths):
    df1 = pd.read_csv(file_paths[0],sep=',',index_col=None)
    df2 = pd.read_csv(file_paths[1],sep=',',index_col=None)
    df3 = pd.read_csv(file_paths[2],sep=',',index_col=None)
    df = pd.concat([df1,df2,df3],axis=0)
    df = df.to_numpy()
    print(len(df))
    random.shuffle(df)
    hits = df[:,:3]
    hits = scale_data(hits)
    conds = df[:,3:6]
    unscaled_conds = conds.copy()
    metadata = df[:,6:]
    conditional_maxes = np.array([8.5,11.63,175.5])
    conditional_mins = np.array([0.95,0.90,-176.])
    conds = (conds - conditional_maxes) / (conditional_maxes - conditional_mins)
    PID = np.ones_like(conds[:,0])
    return hits,conds,unscaled_conds,metadata


def scale_data(hits,stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":500.00,"time_min":0.0}):
    x = hits[:,0]
    y = hits[:,1]
    time = hits[:,2]
    x = 2.0 * (x - stats['x_min'])/(stats['x_max'] - stats['x_min']) - 1.0
    y = 2.0 * (y - stats['y_min'])/(stats['y_max'] - stats['y_min']) - 1.0
    time = 2.0 * (time - stats['time_min'])/(stats['time_max'] - stats['time_min']) - 1.0

    return np.concatenate([np.c_[x],np.c_[y],np.c_[time]],axis=1)

def unscale(x,max_,min_):
    return x*0.5*(max_ - min_) + min_ + (max_-min_)/2


class CherenkovPhotons(Dataset):

    def __init__(self,kaon_path=None,pion_path=None,mode=None,combined=False,inference=False,stats={"x_max": 895,"x_min":0,"y_max":298,"y_min":0,"time_max":500.00,"time_min":0.0}):
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
        self.conditional_maxes = np.array([8.5,11.63,175.5])
        self.conditional_mins = np.array([0.95,0.90,-176.])

        if mode == "Kaon":
            columns=["x","y","time","P","theta","phi"]
            self.data = pd.read_csv(kaon_path,sep=',',index_col=None).to_numpy()
            #df = pd.read_csv(kaon_path,sep=',',index_col=None)
            #self.data = df[df.time < 150].to_numpy()
            self.data = np.concatenate([self.data,np.c_[np.ones_like(self.data[:,0])]],axis=1)

        elif mode == "Pion":
            self.data = pd.read_csv(pion_path,sep=',',index_col=None).to_numpy()
            #df = pd.read_csv(pion_path,sep=',',index_col=None)
            #self.data = df[df.time < 150].to_numpy()
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
      
        # When we perform inference, we take the centered mapping.
        if not self.inference:
            hits[0] = hits[0] + np.clip(np.random.normal(0,1),-3,3) # Add noise, clip due to sensor size
            hits[1] = hits[1] + np.clip(np.random.normal(0,1),-3,3) # Add noise, clip due to sensor size

        hits = self.scale_data(hits,self.stats)
        conds = data[3:6]

        unscaled_conds = conds.copy()
        metadata = data[6:-1]

        conds = (conds - self.conditional_maxes) / (self.conditional_maxes - self.conditional_mins)

        PID = data[-1]

        return hits,conds,PID,metadata,unscaled_conds



class DLL_Dataset(Dataset):

    def __init__(self,file_path,stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":500.00,"time_min":0.0}):
        self.data = np.load(file_path,allow_pickle=True)
        self.n_photons = 1000
        self.stats = stats
        self.conditional_maxes = np.array([8.5,11.63,175.5])
        self.conditional_mins = np.array([0.95,0.90,-176.])

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

        particle = self.data[idx]
        hits = np.array(particle['hits'])
        conds = np.array(particle['conditions'])
        unscaled_conds = conds.copy()
        PID = np.array(particle['PID'])
        n_hits = len(hits)

        hits = self.scale_data(hits,self.stats)
        conds = (conds - self.conditional_maxes) / (self.conditional_maxes - self.conditional_mins)

        if len(hits) > self.n_photons:
            hits = hits[:self.n_photons]

        elif len(hits) < self.n_photons:
            n_needed = self.n_photons - len(hits)
            hits = np.pad(hits,((0,n_needed),(0,0)),mode='constant')
            conds = np.pad(conds,((0,n_needed),(0,0)),mode='constant')
        else:
            pass

        return hits,conds,PID,n_hits,unscaled_conds
