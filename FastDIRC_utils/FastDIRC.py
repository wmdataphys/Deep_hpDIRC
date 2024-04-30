import numpy as np
import torch

class FastDIRC():
    def __init__(self,device):
        self.log_mult = 4
        self.weight = 1
        self.device = device

    def radius_spread_function(self,r2):
        sigma2inv = 1.0
        sigma2 = 1.0
        idx = torch.where(r2 > 5*sigma2)
        r2[idx] = 0.0
        #if (r2 < 5*sigma2):

        return torch.exp(-r2*sigma2inv)

    def support_spread_function(self,support,test):
        dx2 = (test[:,0] - np.repeat(support[0],len(test)))**2
        dy2 = (test[:,1] - np.repeat(support[1],len(test)))**2
        dt2 = (test[:,2] - np.repeat(support[2],len(test)))**2
        x_sig2inv = 1.0/6.0 ** 2 # 6 mm sensors
        y_sig2inv = 1.0/6.0 **2 # 6mm sensors
        t_sig2inv = 1.0/1.0 ** 2 # Timing resolution is 1ns
        return self.radius_spread_function(dx2*x_sig2inv+dy2*y_sig2inv+dt2*t_sig2inv).sum()

    def get_log_likelihood(self, support, inpoints):
        support_tensor = torch.tensor(support, device=self.device, dtype=torch.float)
        inpoints_tensor = torch.tensor(inpoints, device=self.device, dtype=torch.float)

        dx = inpoints_tensor[:, 0].unsqueeze(1) - support_tensor[:, 0]
        dy = inpoints_tensor[:, 1].unsqueeze(1) - support_tensor[:, 1]
        dt = inpoints_tensor[:, 2].unsqueeze(1) - support_tensor[:, 2]
        x_sig2inv = 1.0 / 6.0 ** 2  # 6 mm sensors
        y_sig2inv = 1.0 / 6.0 ** 2  # 6mm sensors
        t_sig2inv = 1.0 / 1.0 ** 2  # Timing resolution is 1ns
        distance_squared = dx ** 2 * x_sig2inv + dy ** 2 * y_sig2inv + dt ** 2 * t_sig2inv

        spread = self.radius_spread_function(distance_squared)
        tprob = spread.sum(dim=1) / len(support)

        sub = torch.tensor(len(inpoints),device=self.device,dtype=torch.float)

        rval = self.weight * self.log_mult * torch.log(torch.sum(tprob) + 1e-50) - torch.log(sub)
        return rval.detach().cpu().numpy()