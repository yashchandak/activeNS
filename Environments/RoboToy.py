from __future__ import print_function
import numpy as np
from Src.Utils.utils import Space, stablesoftmax
import matplotlib.pyplot as plt
from copy import deepcopy

class Active(object):
    def __init__(self, speed) -> None:
        self.rewards = np.array([10.0, 8.0])
        
        if speed == 0: # statioanry
            self.decay = 1
        elif speed == 1:
            self.decay = 0.999
            # self.decay = 1.001
        elif speed == 2:
            self.decay = 0.99
        else:
            raise NotImplementedError

    def step(self, action):
        assert action ==0 or action==1

        r = self.rewards[action]     
        all = self.rewards.copy()
        # 0Th action is bad and decays the rewards
        if action == 0:
            # self.rewards -= 0.005
            # self.rewards += 0.005
            self.rewards *= self.decay

        return r, all


class Passive(object):
    def __init__(self, speed) -> None:
        self.rewards = np.array([10.0, -5.0])
        self.ctr = 0
        self.speed = speed

    def step(self, action):
        assert action ==0 or action==1
        
        if self.speed == 0:
            all = self.rewards
        else:
            all = np.sin(self.speed * self.ctr/100) * self.rewards
        # self.rewards -= 0.1

        self.ctr += 1

        r = all[action] 
        return r, all


class Hybrid(object):
    def __init__(self, speed) -> None:
        self.rewards = np.array([10.0, -5.0])
        self.ctr = 0
        self.speed = speed

    def step(self, action):
        raise NotImplementedError



# collect_data()
class CollectData(object):
    def __init__(self) -> None:
        # self.pi = np.array([0.5, 0.5])
        # self.pi = np.array([0.95, 0.05])
        # self.pi = np.array([0.9, 0.1])
        self.pi = np.array([0.9, 0.1])
        # self.beta = np.array([0.5, 0.5])
        self.beta = np.array([0.1, 0.9])
        # self.beta = np.array([0.95, 0.05])
        # self.beta = np.array([0.05, 0.95])

    def collect(self, env, eps=1000, fore=100, trial=0, speed=0, tag='active'):
        path = 'Experiments_2/RoboToy/ActorCritic/DataDefault/0/Results/'
        data = np.zeros((eps, 1, 3)) # Episodes x Horizon x (rho, r, true_perf)

        for idx in range(eps):
            action = np.random.choice([0, 1], p=self.beta)
            reward, all = env.step(action)
            rho = self.pi[action]/self.beta[action]
        
            data[idx, 0] = (rho, reward, np.dot(all, self.pi))


        np.save("{}{}_beta_rho_data_{}_{}_{}".format(path, tag, speed, 'x', trial) , data)


        # Evaluation is for a non-stationary domain, i.e., stochastic sequence of POMDPs
        # Do multiple rollouts of future to compute the true _expected_ future performance
        # This will average out stochasticity in both intra and inter POMDP transitions.
        n_trials = 30
        pi_data = np.zeros((n_trials, fore))
        
        for inner_trial in range(n_trials):
            env_copy = deepcopy(env)
            returns = np.zeros(fore)
            for idx in range(fore):
                action = np.random.choice([0, 1], p=self.pi)
                reward, all = env_copy.step(action)
                returns[idx] = np.dot(all, self.pi)
            
            pi_data[inner_trial, :] = returns

        np.save("{}{}_eval_data_{}_{}_{}".format(path, tag, speed, 'x', trial) , np.mean(pi_data, axis=0))



def collect_all():
    eps = 2000
    fore = 500
    speeds = [0, 1, 2]
    n_trials = 30

    for speed in speeds:
        for trial in range(n_trials):                
            # env = Active(speed=speed)
            env = Passive(speed=speed)
            collecter = CollectData()
            collecter.collect(env, eps=eps, fore=fore, trial=trial, speed=speed, tag='passive') 
        print(speed)

if __name__=="__main__":
    collect_all()