import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
import torch

def unscale(x,max_,min_):
    return x*0.5*(max_ - min_) + min_ + (max_-min_)/2

# def create_dataset(file_paths,stats):
#     if len(file_paths) > 1:
#         df1 = pd.read_csv(file_paths[0],sep=',',index_col=None)#,nrows=10000)
#         df2 = pd.read_csv(file_paths[1],sep=',',index_col=None)#,nrows=10000)
#         df3 = pd.read_csv(file_paths[2],sep=',',index_col=None)#,nrows=10000)
#         df = pd.concat([df1,df2,df3],axis=0)                   # Useful for debugging
#     else:
#         df = pd.read_csv(file_paths[0],sep=',',index_col=None)

#     top_row = df[(df.y > 248) & (df.x > 348)]
#     middle = df[(df.y < 248) & (df.y > 48)]
#     bottom_row = df[(df.y < 48) & (df.x > 548)]
#     df = pd.concat([top_row,middle,bottom_row])

#     df = df.to_numpy()
#     print(len(df))
#     hits = df[:,:3]
#     conds = df[:,3:6]
#     unscaled_conds = conds.copy()
#     metadata = df[:,6:]
#     conditional_maxes = np.array([8.5,11.63,175.5])
#     conditional_mins = np.array([0.95,0.90,-176.])
#     conds = (conds - conditional_maxes) / (conditional_maxes - conditional_mins)
#     PID = np.ones_like(conds[:,0])
#     return hits,conds,unscaled_conds,metadata


def scale_data(hits,stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":380.0,"time_min":0.0}):
    x = hits[:,0]
    y = hits[:,1]
    time = hits[:,2]

    x = 2.0 * (x - stats['x_min'])/(stats['x_max'] - stats['x_min']) - 1.0
    y = 2.0 * (y - stats['y_min'])/(stats['y_max'] - stats['y_min']) - 1.0
    time = 2.0 * (time - stats['time_min'])/(stats['time_max'] - stats['time_min']) - 1.0

    return np.concatenate([np.c_[x],np.c_[y],np.c_[time]],axis=1)


class CherenkovPhotons(Dataset):
    def __init__(self,kaon_path=None,pion_path=None,mode=None,combined=False,inference=False,stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":500.00,"time_min":0.0}):
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
        self.conditional_maxes = np.array([self.stats['P_max'],self.stats['theta_max'],self.stats['phi_max']])
        self.conditional_mins = np.array([self.stats['P_min'],self.stats['theta_min'],self.stats['phi_min']])

        if mode == "Kaon":
            columns=["x","y","time","P","theta","phi"]
            if not self.inference:
                print('Using training mode.')
                df = pd.read_csv(kaon_path,sep=',',index_col=None)
                #top_row = df[(df.y > 248) & (df.x > 348)]
                #middle = df[(df.y < 248) & (df.y > 48)]
                #bottom_row = df[(df.y < 48) & (df.x > 548)]
                #df = pd.concat([top_row,middle,bottom_row])
                self.data = df[df.time < self.stats['time_max']].to_numpy()
                print('Max Time: ',self.stats['time_max'])
                print(self.data.max(0))
                print(self.data.min(0))
                #self.data = self.modify_tails(data)
            else:
                print("Using inference mode.")
                df = pd.read_csv(kaon_path,sep=',',index_col=None)
                self.data = df[df.time < self.stats['time_max']].to_numpy()
                #top_row = df[(df.y > 248) & (df.x > 348)]
                #middle = df[(df.y < 248) & (df.y > 48)]
                #bottom_row = df[(df.y < 48) & (df.x > 548)]
                #df = pd.concat([top_row,middle,bottom_row])

            self.data = np.concatenate([self.data,np.c_[np.ones_like(self.data[:,0])]],axis=1)

        elif mode == "Pion":
            if not self.inference:
                print('Using training mode.')
                df = pd.read_csv(pion_path,sep=',',index_col=None)
                #top_row = df[(df.y > 248) & (df.x > 348)]
                #middle = df[(df.y < 248) & (df.y > 48)]
                #bottom_row = df[(df.y < 48) & (df.x > 548)]
                #df = pd.concat([top_row,middle,bottom_row])
                self.data = df[df.time < self.stats['time_max']].to_numpy()
                print('Max Time: ',self.stats['time_max'])
                print(self.data.max(0))
                print(self.data.min(0))
                #data = data[data.time < self.stats['time_max']]
                print('Max Time: ',self.stats['time_max'])
                #self.data = self.modify_tails(data)
            else:
                print("Using inference mode.")
                df = pd.read_csv(pion_path,sep=',',index_col=None)
                self.data = df[df.time < self.stats['time_max']].to_numpy()
                #top_row = df[(df.y > 248) & (df.x > 348)]
                #middle = df[(df.y < 248) & (df.y > 48)]
                #bottom_row = df[(df.y < 48) & (df.x > 548)]
                #df = pd.concat([top_row,middle,bottom_row])
                #self.data = df.to_numpy()

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

    def modify_tails(self,data):
        print('Modifying your tails quickly.')
        list_of_data = []
        # Without Time cuts
        #time_bins = [0,80] + list(np.arange(80,376,1))
        #N = len(np.arange(80,376,1))
        # With time cuts < 150ns
        time_bins = [0,80] + list(np.arange(80,150,1))
        N = len(np.arange(80,150,1))
        # modify right tail
        for i in range(len(time_bins) - 1):
            temp_data = data[(data.time > time_bins[i]) & (data.time < time_bins[i+1])]
            if i > 1:
                #f = len(temp_data) / (len(temp_data) * np.sqrt(i/2.))
                f = np.exp(-i**1.2 / N)
                temp_data = temp_data.sample(frac=f)
            list_of_data.append(temp_data)

        d = pd.concat(list_of_data)

        list_of_data = []
        # modify left tail
        # Without time cuts
        #time_bins = list(np.arange(0,10,1)) + [150,380]
        #N = len(np.arange(0,10,1))
        # With time cuts
        time_bins = list(np.arange(0,15,1)) + [150]
        N = len(list(np.arange(0,15,1)))
        for i in range(len(time_bins) - 1):
            temp_data = d[(d.time > time_bins[i]) & (d.time < time_bins[i+1])]
            #if i < 9 :
            if i < 15:
                #f = len(temp_data) / (len(temp_data) *np.sqrt(i+50))
                f = 1.0 - np.exp(-(i+1)/N)
                temp_data = temp_data.sample(frac=f)

            list_of_data.append(temp_data)

        dd = pd.concat(list_of_data)

        del d,list_of_data

        return dd.to_numpy()

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
        #hits = hits[:2]
        conds = data[3:6]

        unscaled_conds = conds.copy()
        metadata = data[6:-1]

        conds = (conds - self.conditional_maxes) / (self.conditional_maxes - self.conditional_mins)

        PID = data[-1]

        return hits,conds,PID,metadata,unscaled_conds


class DLL_Dataset(Dataset):
    # Real Max: 376.82, Min: 1.901769999999999e-05
    def __init__(self,file_path,stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":500.00,"time_min":0.0},time_cuts=None):
        self.data = np.load(file_path,allow_pickle=True)#[:10000] # Useful for testing
        self.n_photons = 250
        self.stats = stats
        self.conditional_maxes = np.array([self.stats['P_max'],self.stats['theta_max'],self.stats['phi_max']])
        self.conditional_mins = np.array([self.stats['P_min'],self.stats['theta_min'],self.stats['phi_min']])
        self.time_cuts = time_cuts
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

    def _apply_mask(self, hits):
        # Time > 0 
        #mask = torch.where(hits[:,2] > 0)
        #hits = hits[mask]
        # Outter bounds
        mask = torch.where((hits[:, 0] > 0) & (hits[:, 0] < 898) & (hits[:, 1] > 0) & (hits[:, 1] < 298))[0] # Acceptance mask
        hits = hits[mask]
        # PMTs OFF
        top_row_mask = torch.where(~((hits[:, 1] > 249) & (hits[:, 0] < 350)))[0] # rejection mask (keep everything not identified)
        hits = hits[top_row_mask]
        # PMTs OFF
        bottom_row_mask = torch.where(~((hits[:, 1] < 50) & (hits[:, 0] < 550)))[0] # rejection mask (keep everything not identified)
        hits = hits[bottom_row_mask]

        return hits


    def create_noise(self,n):
        bins_x = np.array([0,  3.,   9.,  15.,  21.,  27.,  33.,  39.,  45.,48,50, 53.,  59.,  65.,
                    71.,  77.,  83.,  89.,  95., 98,100, 103., 109., 115., 121., 127., 133.,
                    139., 145.,148,150, 153., 159., 165., 171., 177., 183., 189., 195.,198,200, 203.,
                    209., 215., 221., 227., 233., 239., 245.,248,250, 253., 259., 265., 271.,
                    277., 283., 289., 295.,298,300, 303., 309., 315., 321., 327., 333., 339.,
                    345.,348,350, 353., 359., 365., 371., 377., 383., 389., 395.,398,400, 403., 409.,
                    415., 421., 427., 433., 439., 445.,448,450, 453., 459., 465., 471., 477.,
                    483., 489., 495.,498,500, 503., 509., 515., 521., 527., 533., 539., 545.,548,550,
                    553., 559., 565., 571., 577., 583., 589., 595.,598,600, 603., 609., 615.,
                    621., 627., 633., 639., 645.,648,650, 653., 659., 665., 671., 677., 683.,
                    689., 695.,698,700, 703., 709., 715., 721., 727., 733., 739., 745.,748,750, 753.,
                    759., 765., 771., 777., 783., 789., 795.,798,800, 803., 809., 815., 821.,
                    827., 833., 839., 845.848,850, 853., 859., 865., 871., 877., 883., 889.,
                    895.,898,900])
    
        bins_y = np.array([0,  3.,   9.,  15.,  21.,  27.,  33.,  39.,  45.,48,50, 53.,  59.,  65.,
                    71.,  77.,  83.,  89.,  95.,98,100, 103., 109., 115., 121., 127., 133.,
                    139., 145.,148,150, 153., 159., 165., 171., 177., 183., 189., 195.,198,200, 203.,
                    209., 215., 221., 227., 233., 239., 245.,248,250, 253., 259., 265., 271.,
                    277., 283., 289., 295.,298,300])

        x = np.random.choice(bins_x,size=1000)
        y = np.random.choice(bins_y,size=1000)
        time = np.random.uniform(0,self.stats['time_max'],size=1000)

        noise_hits = np.concatenate([np.c_[x],np.c_[y],np.c_[time]],axis=1)
        noise_hits = self._apply_mask(torch.tensor(noise_hits)).numpy()[:n] # Take n first hits, random doesnt matter.
        return noise_hits

    def __getitem__(self, idx):

        particle = self.data[idx]
        LL_k = particle['LikelihoodKaon']
        LL_pi = particle['LikelihoodPion']
        pmtID = np.array(particle['pmtID'])
        o_box = pmtID//108
        if o_box[0] == 1:
            pmtID -= 108

        pixelID = np.array(particle['pixelID'])

        row = (pmtID//18) * 8 + pixelID//8
        col = (pmtID%18) * 8 + pixelID%8

        time = np.array(particle['leadTime'])

        pos_time = np.where((time > 0) & (time < self.stats['time_max']))[0]
        row = row[pos_time]
        col = col[pos_time]
        time = time[pos_time]

        pmtID = pmtID[pos_time]

        assert len(row) == len(time)

        x = col * 6. + (pmtID % 18) * 2. + 3.
        y = row * 6. + (pmtID // 18) * 2. + 3.

        hits = np.concatenate([np.c_[x],np.c_[y],np.c_[time]],axis=1)
        conds = np.array([particle['P'],particle['Theta'],particle['Phi']])

        if self.time_cuts is not None:
            idx_ = np.where(hits[:,2] < self.time_cuts)[0]
            hits = hits[idx_]

        PID = np.array(particle['PDG'])
        n_hits = len(hits)

        # n_noise = int(len(hits) * 0.20)
        # noise_hits = self.create_noise(n_noise)
        # hits = np.concatenate([hits,noise_hits],axis=0)
        # time = hits[:,2]

        hits = self.scale_data(hits,self.stats)

        conds = conds.reshape(1,-1).repeat(len(hits[:,0]),0)
        unscaled_conds = conds.copy()

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


def transform(pmtID,pixelID,time):
        o_box = pmtID//108
        if o_box[0] == 1:
            pmtID -= 108

        pixelID = np.array(particle['pixelID'])

        row = (pmtID//18) * 8 + pixelID//8
        col = (pmtID%18) * 8 + pixelID%8
        pmtID = pmtID[pos_time]

        assert len(row) == len(time)

        x = col * 6. + (pmtID % 18) * 2. + 3.
        y = row * 6. + (pmtID // 18) * 2. + 3.

        return np.concatenate([np.c_[x],np.c_[y],np.c_[time]],axis=1)