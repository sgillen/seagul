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

    def __init__(self, num_steps=1000, m = 1, J = 1, g = 9.8, B = 2, L = 5, dt = .01):
        
        
        self.num_steps = 1000
        self.cur_step = 0

        self.m = m
        self.J = J
        self.g = g
        self.B = B
        self.L = L
        self.dt = dt

        self.act_hold = 1

        self.state = np.zeros(6)
        


    def reset(self):
        self.state = np.zeros(6)
        self.cur_step = 0
        return self.state.copy()
    

    def step(self, action):
        done = False

        action = np.clip(action, -self.L, self.L)

        for _ in range(self.act_hold):
            ns = rk4(self._derivs, action, 0, self.dt, self.state)
            self.state[0] = ns[0]
            self.state[1] = ns[1]
            self.state[2] = wrap(ns[2], -2*pi, 2*pi) # We might not need to wrap ...

            self.state[3] = ns[3]
            self.state[4] = ns[4]
            self.state[5] = ns[5]

        reward = 1

        self.cur_step += 1
        if self.cur_step > self.num_steps:
            done = True

        return self.state.copy(), reward, done, {}


    def _derivs(self, t, q, u):
        dqdt = np.zeros_like(q)

        # Directly set the thrust and moment, but can derive these if we want. 
        F = u[0]
        M = u[1]

        dqdt[0] = q[1]
        dqdt[1] = q[2]
        dqdt[2] = q[3]
        
        dqdt[3] = F*sin(q[2])/self.m
        dqdt[4] = (F*cos(q[2]) - self.m*self.g)/self.m
        dqdt[5] = -M/self.J

        return dqdt
        
