
import numpy as np
from scipy.stats import beta
from Src.Utils.utils import Space

"""
Compared to the MEDEVAC_old 
here there is only a single lifelong episode (which induces non-statioanrity)
This is done by not resetting the state between episodes
Each episode is created by chunking seqeunce of N steps
Passive non-stationarity is induced alongside by altering the rate of arrival
"""

class MEDEVAC(object):
    """
    Routing of Medical Air Ambulances (MEDEVACs)

    Robbins, Matthew J., et al. 
    "Approximate dynamic programming for the aeromedical evacuation dispatching problem:
    Value function approximation utilizing multiple level aggregation."
    Omega 91 (2020): 102020.
    """


    def __init__(self,
                 speed=0,
                 debug=True):    
        
        gamma_term=0.95
        self.max_horizon = 30

        self.debug = debug
        self.M_n = 4
        self.Z_n = 34
        self.K_n = 3
        self.state = np.zeros((self.M_n + self.K_n, self.Z_n), dtype=np.bool)
        
        self.n_actions = self.M_n + 1
        self.action_space = Space(size=self.n_actions)

        self.low = np.zeros(self.state.size + self.M_n + 1)    # Flattened state representation + valid actions vector
        self.high = np.ones(self.state.size + self.M_n + 1)
        self.observation_space = Space(low=self.low, high=self.high, dtype=np.float32)

        # For arrival rates
        self.pZ = np.genfromtxt('Environments/IoBT/requests.csv', delimiter=' ')
        self.pK = np.genfromtxt('Environments/IoBT/kpriority.csv', delimiter=' ')   #UNUSED
        self.partial_lmd = self.pK * self.pZ.reshape(-1,1)
        self.base_lmd = 1.0/30
        print(np.shape(self.partial_lmd))
        assert np.shape(self.partial_lmd) == (self.Z_n, self.K_n)

        # For completion times
        self.ST = np.genfromtxt('Environments/IoBT/completion.csv', delimiter=' ')
        self.mu_zm = 1.0 / self.ST
        self.mu = np.sum(np.max(self.mu_zm[:, idx]) for idx in range(self.M_n))
        assert np.shape(self.mu_zm) == (self.Z_n, self.M_n)

        # Rewards for serving specific zones by specific MEDEVACs
        self.RT = np.genfromtxt('Environments/IoBT/returns.csv', delimiter=' ')
        assert np.shape(self.RT) == (self.Z_n, self.M_n)
        
        self.w_k = np.array([100.0, 10.0, 1.0])
        self.w_k /= np.sum(self.w_k)
        
        self.R_zm = {}
        self.R_zm[0] = self.w_k[0] * beta(3,5).cdf((150 - self.RT)/150.0)
        self.R_zm[1] = self.w_k[1] * beta(1,1).cdf((400 - self.RT)/400.0)
        self.R_zm[2] = self.w_k[2] * beta(1,1).cdf((2400 - self.RT)/2400.0)
        self.gamma_term = gamma_term

        # Various variables to keep track of how the interaction evolves
        self.request = np.zeros(2, dtype=int) -1                                   # Initialized to (-1, -1)
        self.active = np.zeros((self.M_n, 2))                           # No MEDEVACs active initially
        self.valid_actions = np.ones(self.n_actions, dtype=np.bool)     # All actions valid initially

        # Variables for inducing non-stationarity
        self.episodes = 0
        self.speed = speed
        self.deploy_count = np.zeros(self.M_n)
        self.reset()

    def reset(self):
        """
        Sets the environment to default conditions
        :return: state
        """
        # (Passive) Change arrival rates for indivdiual priorities
        self.pK_high = np.array([0.9, 0.05, 0.05])
        self.pk_low = np.array([0.1, 0.3, 0.6])
        alpha = (np.sin(self.speed * self.episodes/200) + 1)/2
        self.pK = (1-alpha) * self.pK_high + alpha*self.pk_low

        self.partial_lmd = self.pK * self.pZ.reshape(-1,1)
        assert np.shape(self.partial_lmd) == (self.Z_n, self.K_n)
        self.lmd = self.base_lmd
        
        # (Active) Decay the service rate based on how many times a MEDVAC has serviced already
        self.mu_zm = (1 - 0.00012 * self.speed * self.deploy_count) * (1.0 / self.ST)
        self.mu = np.sum(np.max(self.mu_zm[:, idx]) for idx in range(self.M_n))
        assert np.shape(self.mu_zm) == (self.Z_n, self.M_n)


        self.lmd_zk = self.lmd * self.partial_lmd
        self.varphi = self.lmd + self.mu

        # Additionally, To induce active-non-statioanrity state variables are not reset
        # Instead they are carried over from previous interaction to induce 'one-long-sequence of interaction'
        # self.state.fill(0)
        # self.request.fill(-1)
        # self.active.fill(0)
        # self.valid_actions.fill(1)

        self.episodes += 1     
        self.ctr = 0       

        # Execute the first no-op action
        state, _ = self._transition(self.M_n, True)
        valid_actions = self._get_valid_actions()

        return np.hstack((state.reshape(-1), valid_actions)).astype(float)        

    def seed(self, seed):
        self.seed = seed

    def render(self):
        raise NotImplementedError
        
    def _get_valid_actions(self):
        return self.valid_actions.copy()

    def _get_state(self):
        return self.state.copy()

    def _is_terminal(self):
        # Simulate continuing setting in the episodic manner
        # by using probabilistic termination criteria
        return self.ctr > self.max_horizon
        # return np.random.rand() > self.gamma_term or self.ctr>self.max_horizon

    def _transition(self, action, skip):
        # print(self.valid_actions, action)
        assert self.valid_actions[action]

        if action != self.M_n:
            self.deploy_count[action] += 1 


        # Whether any action was taken or not,
        # previous request needs to be removed from the state
        k, z = self.request
        if k>-1 and z>-1:
            self.state[self.M_n + k, z] = False
            self.request.fill(-1)

        reward = 0

        # If a valid MEDEVAC action is taken
        # then allocate MEDEVAC to that zone
        # Note: action = M_n is for no-op
        if action < self.M_n and z>-1:
            self.valid_actions[action] = False
            self.state[action, z] = True
            self.active[action, :] = [z, self.mu_zm[z, action]]
            reward += self.R_zm[k][z, action]

        flag = True
        while flag:
            # If skipping is disabled, don't loop
            # If skipping is enabled, loop till a new action is needed
            flag = skip

            # sum of service completion rates for the active MEDEVACs
            mu = np.sum(self.active[:,1])


            # Dsiabling skip in the next if statement
            # such that number of events that occur in an episode can change
            # with different rates of arrival
            self.ctr += 1

            # Some event occurs
            # Note: Can get rid of this loop when skip is enabled
            # if skip or np.random.rand() < (self.lmd + mu)/self.varphi: 
            if np.random.rand() < (self.lmd + mu)/self.varphi: 
            # if True: 
                
                if np.random.rand() < mu/(self.lmd + mu):
                    # A service finishes
                    probs = self.active[:,1] / mu
                    M = np.random.choice(self.M_n, p=probs)

                    # Update the state variables
                    self.state[M, int(self.active[M,0])] = False
                    self.valid_actions[M] = True

                    # Note: This marks zone 0 as active for M, but with completion rate 0. SHould be fine
                    self.active[M, :] = 0     

                else:
                    # A new request arrives 
                    # Following is equivalent to sampling from λ_zk/λ
                    z = np.random.choice(self.Z_n, p=self.pZ)
                    k = np.random.choice(self.K_n, p=self.pK)

                    # Update the state variables
                    self.state[self.M_n + k, z] = True
                    self.request[:] = [k, z] 

                    # Break out of loop; An action is needed
                    flag = False
            
            if self.ctr > self.max_horizon:
                break
    
        return self._get_state(), reward

    def step(self, action, skip=True):
        next_state, reward = self._transition(action, skip)   
        valid_actions = self._get_valid_actions()
        done = self._is_terminal()

        reward  = reward * 10       # Multiplying with positive scalar does not change the optimal policy
        return np.hstack((next_state.reshape(-1), valid_actions)).astype(float), reward, done, {'No Info Implemented yet'}



def Myopic(env):
    _, zone = env.request

    if zone == -1:
        print('weird! This zone resquest sahould not have occured')
        action = env.M_n

    service_rate = env.mu_zm[zone, :]
    valid_actions = env._get_valid_actions()[:-1]

    best_m, best_t = env.M_n, -1
    for idx in range(env.M_n):
        if valid_actions[idx]:
            if service_rate[idx] > best_t:
                best_t = service_rate[idx]
                best_m = idx
    
    # print(valid_actions, best_m)
    return best_m


def Random(env, state):
    valid = state[-env.n_actions:]
    probs = valid/ np.sum(valid)
    action = np.random.choice(env.n_actions, p=probs)

    return action


def smooth(ar, alpha=0.99):
    new_ar = [ar[0]]
    for idx in range(1, len(ar)):
        temp = new_ar[-1]*alpha + (1-alpha)*ar[idx]
        new_ar.append(temp)

    return new_ar


if __name__=="__main__":
    # Random Agent
    rewards_list = []
    eps = []
    env = MEDEVAC(speed=0, debug=True)
    for i in range(2000):
        rewards = 0
        steps=  0
        done = False
        state = env.reset()
        while not done:
            # action = Random(env, state)
            action = Myopic(env)
            # print(valid, probs)
            next_state, r, done, _ = env.step(action)
            rewards += r
            state = next_state
            steps += 1
        rewards_list.append(rewards)
        eps.append(steps)

    print("Average random rewards: ", np.mean(rewards_list), np.sum(rewards_list))

    import matplotlib.pyplot as plt
    plt.plot(rewards_list)
    plt.plot(eps)
    plt.plot(smooth(rewards_list))
    plt.show()