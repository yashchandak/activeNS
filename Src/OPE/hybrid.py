import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import tensor
from Src.OPE.Regression import Simple, Linear, fit


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        std = 1.0 / np.sqrt((fan_in + fan_out))
        # m.weight.data.normal_(0.0, std)
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
        # m.weight.data.fill_(0)
        # m.bias.data.fill_(0)

class Base(object):
    def __init__(self):
        pass

    def get_feats(self, X, p):
        n = len(X)
        feats = np.ones((n-p+1,p))  # Pyotrch adds one for bias correction automatically
        # feats = np.zeros((n-p+1,p))  # No term for bias correction
        for idx in range(n-p+1):
            feats[idx,:p] = X[idx:idx+p, 0]
        return feats

    def get_rho_perf(self, data):
        """
        Assumes data structure in array with following dimensions

        #Traj x #Horizon length x 2 (i.e., rho and reward)
        """
        per_step_rho        = data[:,:,0]   # All the importance ratios
        per_step_rewards    = data[:,:,1]   # All the rewards

        rho = np.prod(per_step_rho, axis=1, keepdims=True)  # NxH => Nx1
        # IMP: Ensure Gamma = 1 everywhere (Collected data + Eval data)
        perf = np.sum(per_step_rewards, axis=1, keepdims=True)   

        # rho = np.clip(rho, 0, 1)
        return rho, perf / self.max_G               
        
class Naive(Base):
    # Bias reduced: importance weighted instrument variable regression

    def __init__(self, p=100) -> None:
        self.S1 = Linear(p)
        # self.S1 = Simple(p)
        self.p = p

    def singleSLS(self, data):
        rho, perf = self.get_rho_perf(data)
        # rho = np.vstack(data[:, 0])            # Nx1
        # perf = np.vstack(data[:, 1])        # Nx1
        p  =self.p

        X = rho * perf                                           # Nx1
        X_feats = self.get_feats(X, p)                              # (N-p+1)xp
        self.S1.fit(x=X_feats[:-1], y=rho[p-1:-1]*X[p:])
      

        return X, X_feats


    def predict(self, data, k):
        X, X_feats = self.singleSLS(data)
        
        p  =self.p
        # temp array contains the predictions
        # First p values are copied over to allow auto-regressive prediction
        temp=np.zeros(k+p)
        temp[:p] = X_feats[-1][:p]
        
        # This loops predicts p+1th item
        # Then uses that (along with preivous values)
        # to predict p+2th item, and so on.
        for idx in range(k):
            feats = np.ones(p)
            feats[:p] = temp[idx: idx+p]
            temp[idx+p] = self.S1(feats).item()
        
        return temp[p:]


class OPEN_1(Base):
    # Bias reduced: importance weighted instrument variable regression

    def __init__(self, p=100) -> None:
        self.S1 = Linear(p)
        self.S2 = Linear(p)
        # self.S1 = Simple(2*p)
        # self.S2 = Simple(p)
        self.p = p

    def twoSLS(self, data):
        rho, perf = self.get_rho_perf(data)      # Nx1, Nx1
        p = self.p

        # Denoising
        # X_bar contains the denoised estimates
        X = rho * perf                                              # Nx1 x Nx1 => Nx1
        X_feats = self.get_feats(X, p)                              # Nx1 => (N-p+1)xp
        self.S1.fit(x=X_feats[:-1], y=X[p:])
        X_bar = self.S1(X_feats[:-1]).detach().numpy()              # (N-p)x1

        # Forecasting
        X_bar_feats = self.get_feats(X_bar, p)                      # (N-2p+1)xp
        
        rho2 = rho[2*p-1:-1] * rho[2*p:]            # Active        # (N-2p)x1
        # rho2 =  rho[2*p:]                         # Passive
        
        self.S2.fit(x=X_bar_feats[:-1], y=rho2*perf[2*p:])

        return X, X_bar, X_bar_feats


    def predict(self, data, k, past=False):
        X, X_bar, X_bar_feats = self.twoSLS(data)
        p = self.p

        # temp array contains the predictions
        # First p values are copied over to allow auto-regressive prediction
        temp=np.zeros(k+p)
        temp[:p] = X_bar_feats[-1][:p]
        
        # This loops predicts p+1th item
        # Then uses that (along with preivous values)
        # to predict p+2th item, and so on.
        for idx in range(k):
            feats = np.ones(p)
            feats[:p] = temp[idx: idx+p]
            temp[idx+p] = self.S2(feats).item()
        
        if past:
            return temp[p:], X_bar.reshape(-1)
        else:            
            return temp[p:]



class OPEN_2(Base):
    # Bias AND variance reduced: importance weighted instrument variable regression

    def __init__(self, 
        hyper={
        'zs':2,
        'p_frac':0.2, 
        'weight_decay':1e-3,
        'lr':1e-4,
        'split':0},
         p=100, passive=False, norm=1) -> None:
        # super().__init__(p)
        self.S1 = Linear(hyper['zs']*p)
        self.S2 = Linear(p)
        # self.S1 = Simple(2*p)
        # self.S2 = Simple(p)
        self.p = p
        self.passive = passive
        self.hyper=hyper

        # self.S1.apply(weight_init)
        # self.S2.apply(weight_init)

        self.max_G = norm

    def twoSLS(self, data):
        rho, perf = self.get_rho_perf(data)
        p = self.p
        N = rho.shape[0]

        # Denoising
        # X_bar contains the denoised estimates
        
        X = perf                                           # Nx1
        X1 = perf
            
        if self.hyper['zs'] == 1:
            X_feats = self.get_feats(X1, p)
        elif self.hyper['zs'] == 2:
            X2 = (rho / N*(np.cumsum(rho)/np.arange(1, N+1)).reshape(-1,1)) * perf
            X_feats = np.hstack((self.get_feats(X1, p), self.get_feats(X2, p)))              # (N-p+1)x2p
        else:
            raise ValueError
        
        fit(self.hyper, self.S1, x=X_feats[:-1], y=perf[p:], weights=rho[p:])
        X_bar = self.S1(X_feats[:-1]).detach().numpy()              # (N-p)x1

        # Forecasting
        X_bar_feats = self.get_feats(X_bar, p)                      # (N-2p+1)xp
        
        if self.passive: 
            rho2 =  rho[2*p:]                         # Passive
        else:
            # rho2 = np.clip(rho[2*p-1:-1] * rho[2*p:], 0,1)
            # rho2 = rho[2*p-1:-1]                        # Only MDP transition        # (N-2p)x1
            rho2 = rho[2*p-1:-1] * rho[2*p:]            # Active        # (N-2p)x1
            # rho2 =  np.ones(rho[2*p:].shape)          # No-IS
        

        fit(self.hyper, self.S2, x=X_bar_feats[:-1], y=perf[2*p:], weights=rho2)

        return X, X_bar, X_bar_feats

    def predict(self, data, k, past=False):
        X, X_bar, X_bar_feats = self.twoSLS(data)
        p = self.p

        # temp array contains the predictions
        # First p values are copied over to allow auto-regressive prediction
        temp=np.zeros(k+p)
        temp[:p] = X_bar_feats[-1][:p]
        
        # This loops predicts p+1th item
        # Then uses that (along with preivous values)
        # to predict p+2th item, and so on.
        for idx in range(k):
            feats = np.ones(p)
            feats[:p] = temp[idx: idx+p]
            temp[idx+p] = self.S2(feats).item()
        
        if past:
            return temp[p:], X_bar.reshape(-1)
        else:            
            return temp[p:]
