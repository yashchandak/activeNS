import numpy as np
import torch as torch

class WLS:
    # Pro_WLS algorithm by Chandak et al. (2020)
    
    def __init__(self, basis_type='Fourier', p=7, norm=1):
        self.basis_type = basis_type
        self.p = p  # number of terms in Fourier series

        if self.basis_type == 'Fourier':
            weights = torch.from_numpy(np.arange(p).reshape([1, p]))    # 1xp
        elif self.basis_type == 'Linear':
            weights = torch.from_numpy(np.ones([1, 2]))                 # 1x2
        elif self.basis_type == 'Poly':
            weights = torch.from_numpy(np.arange(p))                    # 1xp
        else:
            return ValueError

        self.max_G = norm
        self.basis_weights = weights.type(torch.FloatTensor).requires_grad_(False)#.to(self.config.device)

    def get_basis(self, x):

        if self.basis_type == 'Fourier':
            basis = x * self.basis_weights  # Broadcast multiplication B*1 x 1*k => Bxk
            basis = torch.cos(basis * np.pi)

        elif self.basis_type == 'Linear':
            basis = x * self.basis_weights  # Broadcast multiplication B*1 x 1*2 => Bx2
            basis[:, 1] = 1                 # set the second column to be 1

        elif self.basis_type == 'Poly':
            basis = x ** self.basis_weights  # Broadcast power B*1 xx k => Bxk

        return basis

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


    def predict(self, data, k):
        N = data.shape[0]    
        rho, perf = self.get_rho_perf(data)     # Nx1, Nx1

        const = 1.0 * (N + k + int(0.05 * N))   # For normalizing to 0-1
        x = torch.arange(N)
        x = x.float().view(-1, 1) / const       # Nx1
        x_pred = torch.from_numpy(np.arange(N, N + k).reshape([-1, 1])/const).float()  # kx1

        phi_x = self.get_basis(x)               # Nxp
        phi_x_pred = self.get_basis(x_pred)     # kxp 

        # TODO: might want to add a small noise to avoid inversion of singular matrix
        diag = torch.diag(torch.FloatTensor(rho).view(-1))                             # NxN

        phi_xt = phi_x.transpose(1, 0)                                  # pxN
        inv = torch.inverse(phi_xt.mm(diag).mm(phi_x))                  # (pxN x NxN x Nxp)^{-1} -> pxp
        w = inv.mm(phi_xt).mm(diag).mm(torch.FloatTensor(perf))                               # pxp x pxN x NxN x Nx1 -> px1

        hat_y = phi_x_pred.mm(w)                                        # kxp x px1 -> kx1

        return hat_y.view(-1).numpy()

