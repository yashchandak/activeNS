from __future__ import print_function
import numpy as np
import matplotlib.pyplot  as plt
from Src.Utils.utils import Space


class MountainCar(object):
    """
    Implementation modified from OpenAI gym 
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
    
    ### Description
    The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
    at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
    that can be applied to the car in either direction. The goal of the MDP is to strategically
    accelerate the car to reach the goal state on top of the right hill. There are two versions
    of the mountain car domain in gym: one with discrete actions and one with continuous.
    This version is the one with discrete actions.
    This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)
    ```
    @TECHREPORT{Moore90efficientmemory-based,
        author = {Andrew William Moore},
        title = {Efficient Memory-based Learning for Robot Control},
        institution = {University of Cambridge},
        year = {1990}
    }
    ```
    ### Observation Space
    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:
    | Num | Observation                                                 | Min                | Max    | Unit |
    |-----|-------------------------------------------------------------|--------------------|--------|------|
    | 0   | position of the car along the x-axis                        | -Inf               | Inf    | position (m) |
    | 1   | velocity of the car                                         | -Inf               | Inf  | position (m) |
    ### Action Space
    There are 3 discrete deterministic actions:
    | Num | Observation                                                 | Value   | Unit |
    |-----|-------------------------------------------------------------|---------|------|
    | 0   | Accelerate to the left                                      | Inf    | position (m) |
    | 1   | Don't accelerate                                            | Inf  | position (m) |
    | 2   | Accelerate to the right                                     | Inf    | position (m) |
    ### Transition Dynamics:
    Given an action, the mountain car follows the following transition dynamics:
    *velocity<sub>t+1</sub> = velocity<sub>t</sub> + (action - 1) * force - cos(3 * position<sub>t</sub>) * gravity*
    *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*
    where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall. The position is clipped to the range `[-1.2, 0.6]` and velocity is clipped to the range `[-0.07, 0.07]`.
    ### Reward:
    The goal is to reach the flag placed on top of the right hill as quickly as possible, as such the agent is penalised with a reward of -1 for each timestep it isn't at the goal and is not penalised (reward = 0) for when it reaches the goal.
    ### Starting State
    The position of the car is assigned a uniform random value in *[-0.6 , -0.4]*. The starting velocity of the car is always assigned to 0.
    ### Episode Termination
    The episode terminates if either of the following happens:
    1. The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)
    2. The length of the episode is 200.
    """

    def __init__(self,
                 speed=0,
                 debug=True,
                 repeat=10,          # 10
                 max_steps=300):    # 200

        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = 0
        self.force = 0.001
        self.gravity = 0.0025

        self.disp_flag = False
        self.debug = debug

        self.n_actions = 3
        self.action_space = Space(size=self.n_actions)

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)
        self.observation_space = Space(low=self.low, high=self.high, dtype=np.float32)
        
        self.max_horizon = int(max_steps / repeat)
        self.repeat = repeat
        self.step_reward = -1
        self.randomness = 0

        # NS Specific settings
        # self.frequency = speed * 0.001
        self.ns_factor = 0.075 * speed

        self.speed = speed
        self.episode = 0
        self.ep_vel = 0
        self.steps_taken = 0

        self.reset()

    def seed(self, seed):
        self.seed = seed

    def render(self):
        raise NotImplementedError
        
    def reset(self):
        """
        Sets the environment to default conditions
        :return: state
        """
        # Decay force based on average velocity in the previous episode
        self.force = self.force * (1 - self.ns_factor * self.ep_vel/(self.steps_taken+1))

        self.episode += 1
        self.steps_taken = 0
        self.ep_vel = 0
        self.state = (np.random.uniform(low=-0.6, high=-0.4), 0)

        return  np.array(self.state, dtype=np.float32)


    def step(self, action):
        assert action in (0,1,2)

        true_action = action
        self.steps_taken += 1
        
        position, velocity = self.state        
        reward  = 0
        done = False

        for i in range(self.repeat):
            if np.random.rand() < self.randomness:
                action = np.random.randint(self.n_actions)
            else:
                action = true_action
            
            velocity += (action - 1) * self.force + np.cos(3 * position) * (-self.gravity)
            velocity = np.clip(velocity, -self.max_speed, self.max_speed)

            position += velocity
            position = np.clip(position, self.min_position, self.max_position)

            if position == self.min_position and velocity < 0:
                velocity = 0

            reward = reward -1.0

            done = bool(position >= self.goal_position and velocity >= self.goal_velocity) 
            if done:
                break

        done = done or (self.steps_taken >= self.max_horizon)
        self.state = (position, velocity)
        self.ep_vel += velocity
        
        return np.array(self.state, dtype=np.float32), reward, done, {'No INFO implemented yet'}




if __name__=="__main__":
    # Random Agent
    return_list = []
    env = MountainCar(debug=True)
    for i in range(1000):
        returns = 0
        done = False
        env.reset()
        while not done:
            action = np.random.randint(env.n_actions)
            next_state, r, done, _ = env.step(action)
            returns += r
        return_list.append(returns)

    import matplotlib.pyplot as plt
    print("Average random rewards: ", np.mean(return_list), np.sum(return_list))
    plt.plot(return_list)
    plt.show()