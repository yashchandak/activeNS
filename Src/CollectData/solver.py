#!~miniconda3/envs/pytorch/bin python
# from __future__ import print_function
# from memory_profiler import profile


import argparse
from datetime import datetime

import numpy as np
import Src.Utils.utils as utils
from Src.CollectData.config import Config
from time import time
import matplotlib.pyplot as plt
from copy import deepcopy

class Solver:
    def __init__(self, config):
        # Initialize the required variables

        self.config = config
        self.env = self.config.env
        self.state_dim = np.shape(self.env.reset())[0]

        if len(self.env.action_space.shape) > 0:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n
        print("Actions space: {} :: State space: {}".format(self.action_dim, self.state_dim))

        self.model = config.algo(config=config)


    def train(self, max_episodes):
        # Learn the model on the environment
        return_history = []

        ckpt = self.config.save_after
        rm_history, rm, start_ep = [], 0, 0

        steps = 0
        t0 = time()
        for episode in range(start_ep, max_episodes):
            # Reset both environment and model before a new episode

            state = self.env.reset()
            self.model.reset()

            step, total_r = 0, 0
            done = False
            while not done:
                # self.env.render(mode='human')
                action, dist = self.model.get_action(state)
                new_state, reward, done, info = self.env.step(action=action)
                self.model.update(state, action, dist, reward, new_state, done)
                state = new_state

                # Tracking intra-episode progress
                total_r += reward
                step += 1

            # track inter-episode progress
            # returns.append(total_r)
            steps += step
            if episode == 0:
                rm = total_r
            else:
                rm = 0.9*rm + 0.1*total_r
            # rm = total_r
            if episode%ckpt == 0 or episode == self.config.max_episodes-1:
                rm_history.append(rm)
                return_history.append(total_r)
                print("{} :: Rewards {:.3f} :: steps: {:.2f} :: Time: {:.3f}({:.5f}/step) :: Entropy : {:.3f} :: Grads : {}".
                      format(episode, rm, steps/ckpt, (time() - t0)/ckpt, (time() - t0)/steps, self.model.entropy, self.model.get_grads()))

                self.model.save()
                # utils.save_plots(rm_history, config=self.config, name='{}_rewards'.format(self.config.seed))
                utils.save_plots(return_history, config=self.config, name='{}_rewards'.format(self.config.seed))

                t0 = time()
                steps = 0


    def eval(self, max_episodes):
        self.model.load()
        temp = max_episodes/100

        # Evaluation is for a non-stationary domain, i.e., stochastic sequence of POMDPs
        # Do multiple rollouts of future to compute the true expected performance
        # This will average out stochasticity in both intra and inter POMDP transitions.
        n_trials = 30
        all_returns = np.zeros((n_trials, max_episodes))

        for trial in range(n_trials):
            env = deepcopy(self.env)
            returns = []
            for episode in range(max_episodes):
                # Reset both environment and model before a new episode
                state = env.reset()
                self.model.reset()

                step, total_r = 0, 0
                done = False
                while not done:                                
                    action, dist = self.model.get_action(state)
                    new_state, reward, done, info = env.step(action=action)
                    state = new_state

                    # Tracking intra-episode progress
                    # total_r += self.config.gamma**step * reward
                    total_r += reward  # Doesnt make much sense for our setup to use gamma != 1.
                    step += 1

                returns.append(total_r)

            all_returns[trial, :] = returns

            if trial % temp == 0 or trial == n_trials-1:
                print("Eval Collected {}/{} :: Mean return {}".format(trial, n_trials, np.mean(all_returns[trial, :])))
            
                np.save("{}eval_data_{}_{}_{}".format(self.config.paths['results'], self.config.speed,
                                                  self.config.alpha, self.config.seed) , np.mean(all_returns, axis=0))


    def collect(self, max_episodes):
        self.model.load()
        temp = max_episodes/100

        rho_trajectories = []
        SAR_trajectories = []
        for episode in range(max_episodes):
            # Reset both environment and model before a new episode
            state = self.env.reset()
            self.model.reset()

            rho_traj = []
            SAR_traj = []
            step, total_r = 0, 0
            done = False
            while not done:                
                action, rho = self.model.get_action(state, behavior=True)
                new_state, reward, done, info = self.env.step(action=action)
                state = new_state

                # Track importance ratio of current action, and current reward
                rho_traj.append((rho, reward))
                # SAR_traj.append((state, action, reward))

                step += 1
                # if step >= self.config.max_steps:
                #     break

            # Make the length of all trajectories the same.
            # Make rho = 1 and reward = 0, which corresponds to a self loop in the terminal state
            for i in range(step, self.env.max_horizon):
                rho_traj.append((1, 0))
                SAR_traj.append((new_state, action, 0))

            rho_trajectories.append(rho_traj)
            SAR_trajectories.append(SAR_traj)

            if episode % temp == 0 or episode == max_episodes-1:
                print("Beta Collected {}/{} :: Average return {}".format(episode, max_episodes, 
                                                                         np.sum(np.array(rho_traj)[:,1])))

                np.save("{}beta_rho_data_{}_{}_{}".format(self.config.paths['results'], self.config.speed,
                                                          self.config.alpha, self.config.seed) , rho_trajectories)
                np.save("{}beta_SAR_data_{}_{}_{}".format(self.config.paths['results'], self.config.speed,
                                                          self.config.alpha, self.config.seed) , SAR_trajectories)