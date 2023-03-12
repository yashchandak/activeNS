import numpy as np

class WIS:
    # Weighted importance sampling 
    # Precup (2000)

    def __init__(self, norm=1, p=-1):
        self.p = p
        self.max_G = norm

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
        return rho, perf /self.max_G  

    def predict(self, data, k):
        rho, perf = self.get_rho_perf(data)     # Nx1, Nx1
        
        # Use p for sliding-window
        # where only last-p episodes are used for prediction
        if self.p>0:
            rho = rho[-self.p:]
            perf = perf[-self.p:]

        rho /= np.sum(rho)                      # Nx1 
        estimate = np.sum(rho*perf)                    # sum(Nx1 * Nx1) => 1

        return np.ones(k) * estimate