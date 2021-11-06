import numpy as np

import gym
import gym.spaces
from gym.utils import seeding

class LQREnv(gym.Env):
    """
    Simple LQR learning envioronment using numpy
    
    x is nx1, A is nxn, 
    B is nxm, u is mx1 

    x+ = A@x + B@u
    r = x'x

    """

    def __init__(self,
                 ep_length = 100,
                 obs_size = 3,
                 act_size = 3,
                 A = None,
                 B = None
                 ):

        self.ep_length = ep_length

        if A is None:
            A = np.random.standard_normal((obs_size,obs_size))
        if B is None:
            B = np.random.standard_normal((obs_size, act_size))

        self.A = A
        self.B = B
        self.obs_size = A.shape[0]
        self.act_size = B.shape[1]


        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=(self.obs_size,))
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=(self.act_size,))

        self.x = self.reset()

    def reset(self):
        self.x = np.random.standard_normal((self.obs_size,1))
        self.cur_step = 0

        return self.x.squeeze()

    def step(self, act):
        act = act.reshape(-1,1)
        self.x = self.A@self.x + self.B@act
        reward = -(self.x.T@self.x).item()

        self.cur_step +=1

        done = False

        if np.any(self.x > 1e6):
            done = True

        if self.cur_step > self.ep_length:
            done = True

        return self.x.squeeze(), reward, done, {}
            
