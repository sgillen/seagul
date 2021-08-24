import numpy as np
import gym
import os
from numpy import pi, sin, cos
from seagul.integration import rk4, euler, wrap


class PlanarQuadCopter(gym.Env):
    """
    A simple quadcopter, going from here: https://hybrid-robotics.berkeley.edu/publications/ACC2016_Safety_Control_Planar_Quadrotor.pdf
    q[0] = x
    q[1] = y
    q[2] = theta
    q[3] = x dot
    q[4] = y dot 
    q[5] = theta dot

    """

    def __init__(self, num_steps=100, m = .25, J = .25, g = 9.8, max_F = 5, max_M = 5, dt = .01, xtarg = 2, ytarg = 2, theta_targ=0):
        self.num_steps = num_steps
        self.cur_step = 0

        self.m = m
        self.J = J
        self.g = g
        self.max_F = max_F
        self.max_M = max_M
        self.dt = dt

        self.xtarg = xtarg
        self.ytarg = xtarg
        self.theta_targ = theta_targ

        self.act_hold = 1
        
        obs_high = np.ones(6)*100
        act_high = np.array([max_F, max_M])
        
        self.observation_space = gym.spaces.Box(low=-obs_high, high=obs_high)
        self.action_space = gym.spaces.Box(low=-act_high, high=act_high)
        self.state = np.zeros(6)
        
    def _get_obs(self):
        return np.array(self.state)

    def reset(self):
        self.state = np.zeros(6)
        self.cur_step = 0
        return self._get_obs()

    def step(self, action):
        done = False
        action = np.asarray(action).squeeze()

        action[0] = np.clip(action[0], -self.max_F, self.max_F)
        action[1] = np.clip(action[1], -self.max_M, self.max_M)

        for _ in range(self.act_hold):
            ns = rk4(self._derivs, action, 0, self.dt, self.state)
            self.state[0] = ns[0]
            self.state[1] = ns[1]
            self.state[2] = wrap(ns[2], -2*pi, 2*pi) # We might not need to wrap ...

            self.state[3] = ns[3]
            self.state[4] = ns[4]
            self.state[5] = ns[5]

        xpen = -0.01*np.clip((self.state[0] - self.xtarg)**2,-4,4)
        ypen = -0.01*np.clip((self.state[1] - self.ytarg)**2,-4,4)
        thpen = -0.01*np.clip((self.state[2]- self.theta_targ)**2, -2,2)
            
        reward = xpen + ypen + thpen

        self.cur_step += 1
        if self.cur_step > self.num_steps:
            done = True

        return self._get_obs(), reward, done, {}


    def _derivs(self, t, q, u):
        dqdt = np.zeros_like(q)

        # Directly set the thrust and moment, but can derive these if we want. 
        F = u[0]
        M = u[1]

        dqdt[0] = q[3]
        dqdt[1] = q[4]
        dqdt[2] = q[5]
        
        dqdt[3] = F*sin(q[2])/self.m
        dqdt[4] = (F*cos(q[2]) - self.m*self.g)/self.m
        dqdt[5] = -M/self.J

        return dqdt
        

    def render(self, mode=None):
        pass

    def plot_episode(self, obs):
        import matplotlib.pyplot as plt
        plt.plot(obs[:,0], obs[:,1],'o', alpha=.5)
        plt.plot(obs[0,0], obs[0,1], 'o', color='red')

        arrow_length = np.max(np.abs(np.concatenate([obs[:,0],obs[:,1]])))*.05

        for i,o in enumerate(obs):
            if i % 5 == 0:
                plt.arrow(o[0], o[1], arrow_length*sin(o[3]), arrow_length*cos(o[3]), width=0.01)

        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
