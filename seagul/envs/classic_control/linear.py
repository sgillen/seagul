import numpy as np
from numpy.random import default_rng
import gym


class LinearEnv(gym.Env):
    """
    A linear system of the form x+ = A*x + B*u , where B is ones.

    """

    def __init__(self, A, num_steps = 100, act_limit = 1, reset_range=5, seed = None, Q = None, R = None):
        self.A = A
        self.act_limit = act_limit
        self.num_steps = num_steps
        self.reset_range = reset_range

        self.rng = default_rng(seed)

        self.num_obs = A.shape[0]

        if Q is None:
            Q = np.identity(self.num_obs)
        self.Q = Q

        if R is None:
            R = np.zeros((self.num_obs, self.num_obs))
        self.R = R

        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.num_obs,))
        self.action_space = gym.spaces.Box(low=-act_limit, high=act_limit, shape=(self.num_obs,))
        self.X = np.zeros((1,self.num_obs))

    def seed(self, seed):
        self.rng = default_rng(seed)
    def reset(self):
        self.X = self.rng.uniform(low=-self.reset_range,high=self.reset_range,size=(1,self.num_obs))
        self.cur_step = 0
        return self.X.squeeze()

    def step(self, u):
        u = np.asarray(u)
        u = u.reshape(1,-1)
        u = np.clip(u, -self.act_limit, self.act_limit)

        self.X = self.X@self.A + u

        reward = -(self.X@self.Q@self.X.transpose() + u@self.R@u.transpose()).item()

        done = False
        if self.cur_step > self.num_steps:
            done = True
            
        self.cur_step += 1

        
        
        return self.X.squeeze(), reward, done, {}
        
        
                                                
        
